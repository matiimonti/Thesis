from models.base import BaseModel


class LlamaModel(BaseModel):
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    def _build_messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
