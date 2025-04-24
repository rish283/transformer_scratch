from src.dataset import TinyShakespeareTokenizer, TinyShakespeareDataset
from src.config import *
from torch.utils.data import DataLoader, random_split, Subset
from src.model import MiniTransformer
from src.train import train_model
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import neptune
import os
import random
import numpy as np

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = TinyShakespeareTokenizer(text)
print(f"Vocab size: {len(tokenizer)}")
fulldataset = TinyShakespeareDataset(text, tokenizer, SEQ_LEN)

# Fix seed
torch.manual_seed(42)
random.seed(42)

# Create shuffled index split
dataset_len = len(fulldataset)
indices = list(range(dataset_len))
random.shuffle(indices)

test_size = int(dataset_len * 0.1)
test_idx = indices[-test_size:]
train_val_idx = indices[:-test_size]

# Save for reproducibility
np.save("splits/test_idx.npy", test_idx)
np.save("splits/train_val_idx.npy", train_val_idx)

# Create subsets
test_dataset = Subset(fulldataset, test_idx)
train_val_dataset = Subset(fulldataset, train_val_idx)

# Now split train_val into train/val
val_size = int(len(train_val_dataset) * 0.1)
train_size = len(train_val_dataset) - val_size

train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

x, y = next(iter(train_loader))

model = MiniTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size = SEQ_LEN,
    embed_dim = EMBED_DIM,
    n_heads = N_HEADS,
    n_layers = N_LAYERS
)

os.makedirs(CHEKPOINT_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version PyTorch built with:", torch.version.cuda)
print("CUDA version available from system:", torch.version.cuda)
model = model.to(device)
#define loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

run = neptune.init_run(
    project="Rishabh-Singh/Transformer-Scratch",
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)

run["config/SEQ_LEN"] = SEQ_LEN
run["config/BATCH_SIZE"] = BATCH_SIZE
run["config/EMBED_DIM"] = EMBED_DIM
run["config/N_HEADS"] = N_HEADS
run["config/N_LAYERS"] = N_LAYERS
run["config/EPOCHS"] = EPOCHS

train_model(model, EPOCHS, train_loader, val_loader, loss_fn, optimizer, device, run)
