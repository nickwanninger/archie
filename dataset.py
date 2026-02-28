import archie
import archie.training
import archie.config
from torch.utils.data import DataLoader


config = archie.config.flicker

tokenizer = archie.get_tokenizer()

dataset = archie.training.get_datasets()
text_dataset = archie.training.TextDataset(dataset, tokenizer, config)
train_loader = DataLoader(
    text_dataset,
    batch_size=1,
    num_workers=4,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
)


for i, (x, y) in enumerate(train_loader):
    print("Batch:")
    print("x:", tokenizer.decode(x[0].cpu().tolist()))
    print("y:", tokenizer.decode(y[0].cpu().tolist()))
    print()
    if i > 10:
        break
