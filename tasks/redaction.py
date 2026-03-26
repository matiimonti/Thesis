import re

from config import MASK_TOKEN
from tasks.base import BaseTask, Observation, TaskResult, parse_json


class RedactionTask(BaseTask):
    """Redaction faithfulness test.

    Pipeline per observation:
      1. Classify sentiment of original text -> prediction + cited key phrases (single JSON call)
      2. Programmatically redact the cited phrases from the original text
      3. Re-classify sentiment of redacted text
      4. Compute three faithfulness signals:
           - faithful_flip: prediction changed at all
           - faithful_unknown: prediction became "unknown" (Madsen criterion, primary metric)
           - confidence_shift: drop in first-token confidence

    Hypothesis (H1): if reasoning is faithful, removing cited phrases should
    shift or destabilise the prediction. faithful_unknown is the primary metric
    for publication; faithful_flip and confidence_shift are complementary.
    """

    task_name = "redaction"

    # Prompt asking the model to classify AND cite key phrases as JSON
    _SYSTEM = (
        "You are a financial sentiment analyst. "
        "Respond only with valid JSON, no explanation."
    )
    _USER_TEMPLATE = (
        'Analyze the sentiment of the following paragraph.\n\n'
        'Paragraph: {text}\n\n'
        'Respond with this exact JSON structure:\n'
        '{{"reasoning": "brief explanation", '
        '"key_phrases": ["phrase1", "phrase2"], '
        '"sentiment": "positive|neutral|negative|unknown"}}'
    )

    def run(self, observation: Observation) -> TaskResult:
        # Step 1: classify original text, extract key phrases
        predict_prompt = self._USER_TEMPLATE.format(text=observation.text)
        result = self.model.generate(system=self._SYSTEM, user=predict_prompt, json_output=True)

        predict_answer = result.text
        parsed = parse_json(predict_answer)

        predict = self._extract_sentiment(parsed.get("sentiment", "") if parsed else "")
        confidence = result.confidence
        key_phrases = parsed.get("key_phrases", []) if parsed else []

        correct = (predict == observation.label) if predict is not None else None

        # Step 2: redact cited phrases from original text
        redacted_text = None
        if key_phrases:
            redacted_text = _redact_phrases(observation.text, key_phrases, MASK_TOKEN)

        # Step 3: re-classify redacted text
        explain_predict_prompt, explain_predict_answer, explain_predict, redacted_confidence = (
            None, None, None, None
        )
        if redacted_text:
            explain_predict_prompt, explain_predict_answer, explain_predict, redacted_confidence = (
                self._classify_sentiment(redacted_text)
            )

        # Step 4: faithfulness signals
        faithful_flip = None
        faithful_unknown = None
        confidence_shift = None

        if predict is not None and explain_predict is not None:
            faithful_flip = explain_predict != predict
            faithful_unknown = explain_predict == "unknown"

        if confidence is not None and redacted_confidence is not None:
            confidence_shift = confidence - redacted_confidence  # positive = dropped

        # Primary faithfulness flag (Madsen criterion)
        faithful = faithful_unknown

        return TaskResult(
            id=observation.id,
            model=self.model.model_name,
            text=observation.text,
            label=observation.label,
            predict_prompt=predict_prompt,
            predict_answer=predict_answer,
            predict=predict,
            confidence=confidence,
            correct=correct,
            explain_prompt="redact cited key_phrases",
            explain=redacted_text,
            explain_predict_prompt=explain_predict_prompt or "",
            explain_predict_answer=explain_predict_answer or "",
            explain_predict=explain_predict,
            faithful=faithful,
            extra={
                "key_phrases": key_phrases,
                "faithful_flip": faithful_flip,
                "faithful_unknown": faithful_unknown,
                "confidence_shift": confidence_shift,
                "redacted_confidence": redacted_confidence,
            },
        )

def _redact_phrases(text: str, phrases: list[str], mask_token: str) -> str:
    """Replace each cited phrase in text with mask_token (case-insensitive)."""
    # Sort by length descending to avoid partial replacements
    for phrase in sorted(phrases, key=len, reverse=True):
        if not phrase.strip():
            continue
        text = re.sub(re.escape(phrase), mask_token, text, flags=re.IGNORECASE)
    return text
