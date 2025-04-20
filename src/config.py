# src/config.py
import os

SEQ_LEN = 128
BATCH_SIZE = 64
EMBED_DIM = 256
N_HEADS = 4
N_LAYERS = 4
EPOCHS = 20
LEARNING_RATE = 3e-4
SAVE_EVERY = 5

DATA_PATH = "data/input.txt"
CHEKPOINT_PATH = "outputs/checkpoints"
os.makedirs(CHEKPOINT_PATH, exist_ok=True)