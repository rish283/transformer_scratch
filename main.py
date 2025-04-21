from src.dataset import TinyShakespeareTokenizer, TinyShakespeareDataset
from src.config import SEQ_LEN, BATCH_SIZE, DATA_PATH, LEARNING_RATE, EPOCHS, CHEKPOINT_PATH
from torch.utils.data import DataLoader, random_split
from src.model import MiniTransformer
from src.train import train_model
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import neptune
import os

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = TinyShakespeareTokenizer(text)
print(f"Vocab size: {len(tokenizer)}")
fulldataset = TinyShakespeareDataset(text, tokenizer, SEQ_LEN)

val_pct = 0.1
val_size = int(len(fulldataset) * val_pct)
train_size = len(fulldataset) - val_size

train_dataset, val_dataset = random_split(fulldataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

x, y = next(iter(train_loader))

model = MiniTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size = SEQ_LEN,
    embed_dim = 128,
    n_heads = 4,
    n_layers = 4,
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

train_model(model, EPOCHS, train_loader, val_loader, loss_fn, optimizer, device, run)
