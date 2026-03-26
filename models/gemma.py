from models.base import BaseModel


class GemmaModel(BaseModel):
    model_name = "google/gemma-2-2b-it"

    def _build_messages(self, system: str, user: str) -> list[dict]:
        # Gemma does not support a system role — merge it into the user turn
        return [
            {"role": "user", "content": f"{system}\n\n{user}"},
        ]
