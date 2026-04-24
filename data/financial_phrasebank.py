import json
import random
import requests
from io import BytesIO
from zipfile import ZipFile

from tasks.base import Observation
from config import DATA_DIR, RANDOM_SEED, SAMPLE_SIZE


_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/financial_phrasebank"
    "/resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
_ZIP_PATH = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
_CACHE_FILE = DATA_DIR / "financial_phrasebank.jsonl"


def _download() -> list[dict]:
    response = requests.get(_DOWNLOAD_URL, timeout=30)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as zf:
        with zf.open(_ZIP_PATH) as f:
            lines = f.read().decode("iso-8859-1").strip().split("\n")

    records = []
    for line in lines:
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.strip()
        if label in ("positive", "neutral", "negative"):
            records.append({"text": sentence.strip(), "label": label})

    return records


def _load_cached() -> list[dict]:
    with open(_CACHE_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_cache(records: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        for i, r in enumerate(records):
            # Stable ID assigned once at cache time — never changes across runs
            f.write(json.dumps({"id": i, **r}) + "\n")


def load_dataset(sample_size: int | None = SAMPLE_SIZE) -> list[Observation]:
    """Load Financial PhraseBank (sentences_allagree subset).

    Downloads once and caches to data/financial_phrasebank.jsonl.
    Returns a list of Observation objects ready for task consumption.
    """
    if _CACHE_FILE.exists():
        records = _load_cached()
    else:
        print("Downloading Financial PhraseBank...")
        records = _download()
        _save_cache(records)
        print(f"Saved {len(records)} sentences to {_CACHE_FILE}")
        records = _load_cached()

    if sample_size is not None:
        rng = random.Random(RANDOM_SEED)
        records = rng.sample(records, min(sample_size, len(records)))

    observations = [
        Observation(id=r["id"], text=r["text"], label=r["label"])
        for r in records
    ]

    label_counts = {}
    for o in observations:
        label_counts[o.label] = label_counts.get(o.label, 0) + 1
    print(f"Loaded {len(observations)} observations: {label_counts}")

    return observations
