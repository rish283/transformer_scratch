import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from src.model import MiniTransformer
from src.config import *
import numpy as np
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)


def top_k_accuracy(logits, targets, k):
    top_k = logits.topk(k, dim=-1).indices  # [B, T, k]
    correct = top_k.eq(targets.unsqueeze(-1))  # [B, T, k]
    return correct.any(dim=-1).float().mean().item()


def decode_sequence(seq, tokenizer):
    return tokenizer.decode(seq.tolist()).strip()


def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    bleu_scores, rouge_scores = [], []
    top1_scores, top5_scores = [], []

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            top1 = top_k_accuracy(logits, y, k=1)
            top5 = top_k_accuracy(logits, y, k=5)

            top1_scores.append(top1)
            top5_scores.append(top5)

            preds = torch.argmax(logits, dim=-1)  # [B, T]
            for pred_seq, target_seq in zip(preds, y):
                pred_text = decode_sequence(pred_seq, tokenizer)
                target_text = decode_sequence(target_seq, tokenizer)

                # BLEU
                pred_tokens = nltk.word_tokenize(pred_text)
                target_tokens = [nltk.word_tokenize(target_text)]
                bleu = sentence_bleu(target_tokens, pred_tokens, smoothing_function=smooth)
                bleu_scores.append(bleu)

                # ROUGE
                rouge_score = rouge.score(pred_text, target_text)
                rouge_scores.append(rouge_score["rougeL"].fmeasure)

    metrics = {
        "BLEU": np.mean(bleu_scores),
        "ROUGE-L": np.mean(rouge_scores),
        "Top-1 Acc": np.mean(top1_scores),
        "Top-5 Acc": np.mean(top5_scores),
    }
    return metrics
