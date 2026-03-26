import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from models.base import BaseModel, GenerationResult


class FinGPTModel(BaseModel):
    model_name = "FinGPT/fingpt-sentiment_llama2-7b-lora"
    _base_model_name = "meta-llama/Llama-2-7b-hf"

    def __init__(self, device: str = "cuda"):
        # FinGPT is a LoRA adapter — load base model first, then apply adapter.
        # We override __init__ entirely to skip BaseModel's from_pretrained call.
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, self.model_name)
        self.model.eval()

    def _build_messages(self, system: str, user: str) -> list[dict]:
        # Llama-2 chat template supports system role
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
