from models.llama import LlamaModel
from models.gemma import GemmaModel
from models.fingpt import FinGPTModel

ALL_MODELS = {
    "llama": LlamaModel,
    "gemma": GemmaModel,
    "fingpt": FinGPTModel,
}
