import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from src.model import MiniTransformer
from src.dataset import TinyShakespeareTokenizer, TinyShakespeareDataset
from src.eval import evaluate
from src.config import *
import neptune
import os
import random

# Setup Neptune
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

# Load text
with open(DATA_PATH, "r") as f:
    text = f.read()

tokenizer = TinyShakespeareTokenizer(text)
fulldataset = TinyShakespeareDataset(text, tokenizer, SEQ_LEN)

# Load saved test indices
test_idx = np.load("splits/test_idx.npy")
test_dataset = Subset(fulldataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model and checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size=SEQ_LEN,
    embed_dim=EMBED_DIM,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
)

checkpoint_path = f"{CHEKPOINT_PATH}/model_epoch_100.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)

# Run evaluation
metrics = evaluate(model, test_loader, tokenizer, device, run=run)
print("\n--- Evaluation Metrics ---")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
