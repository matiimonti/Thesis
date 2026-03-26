import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# FinBERT label mapping (ProsusAI/finbert output order)
_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}


class FinBERT:
    """Lightweight FinBERT wrapper for independent classification.

    Used exclusively as a validator — not a faithfulness subject.
    Discriminative model: no CoT, no generation, fast inference.
    """

    model_name = "ProsusAI/finbert"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(device)
        self.model.eval()

    def classify(self, text: str) -> tuple[str, float]:
        """Classify sentiment of text.

        Returns (label, confidence) where label is one of
        'positive', 'negative', 'neutral'.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits[0], dim=-1)
        pred_id = probs.argmax().item()
        return _ID2LABEL[pred_id], probs[pred_id].item()
