import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def top_k_accuracy(logits, targets, k):
    top_k = logits.topk(k, dim=-1).indices  # [B, T, k]
    correct = top_k.eq(targets.unsqueeze(-1))  # [B, T, k]
    return correct.any(dim=-1).float()  # [B, T]


def decode_sequence(seq, tokenizer):
    return tokenizer.decode(seq.tolist()).strip()


def log_lineplot_to_neptune(x, y, run, key, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    run[key].upload(buf)
    plt.close()


def evaluate(model, dataloader, tokenizer, device, run=None):
    model.eval()

    top1_per_token, top5_per_token = [], []
    bleu_per_sequence, rouge_per_sequence = [], []

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            # Token-level accuracy
            top1 = top_k_accuracy(logits, y, k=1)
            top5 = top_k_accuracy(logits, y, k=5)
            top1_flat = top1.view(-1).tolist()
            top5_flat = top5.view(-1).tolist()

            top1_per_token.extend(top1_flat)
            top5_per_token.extend(top5_flat)

            if run:
                for i, score in enumerate(top1_flat):
                    if i % 10 == 0:
                        run["eval/per_token/top1"].append(score)
                for i, score in enumerate(top5_flat):
                    if i % 10 == 0:
                        run["eval/per_token/top5"].append(score)

            # Sequence-level BLEU/ROUGE
            preds = torch.argmax(logits, dim=-1)  # [B, T]
            for pred_seq, target_seq in zip(preds, y):
                pred_text = decode_sequence(pred_seq, tokenizer)
                target_text = decode_sequence(target_seq, tokenizer)

                pred_tokens = pred_text.split()
                target_tokens = [target_text.split()]

                bleu = sentence_bleu(target_tokens, pred_tokens, smoothing_function=smooth)
                rouge_score = rouge.score(pred_text, target_text)["rougeL"].fmeasure

                bleu_per_sequence.append(bleu)
                rouge_per_sequence.append(rouge_score)

                if run:
                    run["eval/per_sequence/bleu"].append(bleu)
                    run["eval/per_sequence/rougeL"].append(rouge_score)

    # Aggregate metrics
    metrics = {
        "Top-1 Acc (avg per token)": np.mean(top1_per_token),
        "Top-5 Acc (avg per token)": np.mean(top5_per_token),
        "BLEU (avg per sequence)": np.mean(bleu_per_sequence),
        "ROUGE-L (avg per sequence)": np.mean(rouge_per_sequence),
    }

    if run:
        # Aggregate scalars
        run["eval/aggregate/top1_token_acc"] = metrics["Top-1 Acc (avg per token)"]
        run["eval/aggregate/top5_token_acc"] = metrics["Top-5 Acc (avg per token)"]
        run["eval/aggregate/bleu"] = metrics["BLEU (avg per sequence)"]
        run["eval/aggregate/rougeL"] = metrics["ROUGE-L (avg per sequence)"]

        # Plots
        log_lineplot_to_neptune(
            x=list(range(len(top1_per_token))),
            y=top1_per_token,
            run=run,
            key="plots/token_level/top1_acc",
            title="Top-1 Accuracy vs Token Index",
            xlabel="Token Index",
            ylabel="Top-1 Accuracy"
        )
        log_lineplot_to_neptune(
            x=list(range(len(top5_per_token))),
            y=top5_per_token,
            run=run,
            key="plots/token_level/top5_acc",
            title="Top-5 Accuracy vs Token Index",
            xlabel="Token Index",
            ylabel="Top-5 Accuracy"
        )
        log_lineplot_to_neptune(
            x=list(range(len(bleu_per_sequence))),
            y=bleu_per_sequence,
            run=run,
            key="plots/sequence_level/bleu",
            title="BLEU vs Sequence Index",
            xlabel="Sequence Index",
            ylabel="BLEU Score"
        )
        log_lineplot_to_neptune(
            x=list(range(len(rouge_per_sequence))),
            y=rouge_per_sequence,
            run=run,
            key="plots/sequence_level/rougeL",
            title="ROUGE-L vs Sequence Index",
            xlabel="Sequence Index",
            ylabel="ROUGE-L Score"
        )

    return metrics
