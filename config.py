from pathlib import Path

# PATHS 
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# DATASET
DATASET_NAME = "financial_phrasebank"
DATASET_SPLIT = "sentences_allagree"   # highest-agreement subset
SAMPLE_SIZE = None                     # None = use full dataset
RANDOM_SEED = 42

# MODELS

# General-purpose LLMs (faithfulness subjects)
GENERAL_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-2-2b-it",
}

# Domain-specialised LLMs (faithfulness subjects, for H2)
DOMAIN_MODELS = {
    "fingpt": "FinGPT/fingpt-sentiment_llama2-13b-lora",
}

# Accuracy baseline only — not a faithfulness subject (no CoT)
BASELINE_CLASSIFIER = "ProsusAI/finbert"

ALL_FAITHFULNESS_MODELS = {**GENERAL_MODELS, **DOMAIN_MODELS}

# GENERATION
GENERATION = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True,
    "return_dict_in_generate": True,   # needed for confidence scores
    "output_scores": True,
}

# MASK TOKEN (for redaction)
MASK_TOKEN = "[REDACTED]"

# H3 — PROMPT PARAPHRASES
# Three semantically equivalent prompts for stability testing
SENTIMENT_PROMPTS = {
    "v1": "What is the sentiment of the following paragraph?",
    "v2": "How would you classify the tone of the following text?",
    "v3": "Analyze the emotional orientation of the following passage.",
}

# OUTPUT FILES
RESULT_FILES = {
    "baseline": RESULTS_DIR / "baseline.jsonl",
    "redaction": RESULTS_DIR / "redaction.jsonl",
    "counterfactual": RESULTS_DIR / "counterfactual.jsonl",
    "cot_intervention": RESULTS_DIR / "cot_intervention.jsonl",
    "stability": RESULTS_DIR / "stability.jsonl",
}

# AUTO-SAVE INTERVAL
SAVE_EVERY = 50
