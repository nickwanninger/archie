from dataclasses import dataclass
import tiktoken
import torch

from archie.config import Config
from archie.model import ArchieModel


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 200


class InferenceEngine:
    def __init__(
        self,
        model: ArchieModel,
        tokenizer: tiktoken.Encoding,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config

        self.model.eval()

    def stream(self, prompt: str, gen_config=GenerationConfig()):

        device = self.config.device

        input_ids = self.tokenizer.encode(prompt)
        current = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(gen_config.max_tokens):
                logits = self.model(current[:, -self.config.block_size :])
                logits = logits[0, -1, :] / gen_config.temperature

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                prob = probs[next_token].item()

                current = torch.cat(
                    [current, torch.tensor([[next_token]], device=device)], dim=1
                )
                text = self.tokenizer.decode([next_token])
                yield (text, prob)

    def generate(self, prompt: str, gen_config=GenerationConfig()):
        text = ""
        for text, prob in self.stream(prompt, gen_config):
            text += text
        return text
