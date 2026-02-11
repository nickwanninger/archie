import archie
import torch
from torch.utils.data import DataLoader

import archie.training

import math


def main():
    # Grab a default config
    config = archie.Config(device="cuda")

    model, tokenizer = archie.load_model_from_checkpoint(config)

    # 1. Rescale output projections
    scale = 1.0 / math.sqrt(2 * config.n_layers)

    for layer in model.layers:
        layer.attention.wo.weight.data *= scale
        layer.feed_forward.w2.weight.data *= scale

    # tokenizer = archie.get_tokenizer()

    # dataset = archie.training.get_datasets()
    # loader = archie.training.TextDataset(dataset, tokenizer, config)
    # loader = DataLoader(loader, batch_size=config.batch_size, num_workers=4)

    # for x, y in loader:
    #     print(x.shape, y.shape)
    #     break

    for i in range(10):
        print("-----------------------")
        engine = archie.InferenceEngine(model, tokenizer)
        prompt = "The capital of France is"
        print(prompt, end="", flush=True)
        for text, prob in engine.stream(
            prompt, gen_config=archie.GenerationConfig(max_tokens=100, temperature=0.1)
        ):
            print(text, end="", flush=True)
        print()
    # print(f"{prob:.4f} {text}")

    # # model = ArchieModel(config).to(config.device).to(torch.bfloat16)

    # x = torch.zeros(1, config.block_size, dtype=torch.long).to(config.device)
    # print(x)
    # y = model(x)
    # print(y.shape)


if __name__ == "__main__":
    main()
