"""plots.py — Plotting functions for individual experiment runs and comparison groups.

Generates:
    - figures/<exp_id>.png: 3 subplots (Loss, Metrics, Gradient Norm)
    - figures/compare_<group>.png: Comparison across runs in a topic
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg = result["cfg"]
    hist = result["history"]
    sum_res = result["summary"]

    epochs = hist["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    best_ep = sum_res.get("best_epoch", -1)

    # (1) Loss
    ax1 = axes[0]
    ax1.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", lw=2)
    ax1.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    if best_ep > 0 and best_ep in epochs:
        ax1.axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_ep})")
    ax1.set_title("Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # (2) Accuracy & Macro-F1
    ax2 = axes[1]
    ax2.plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", lw=2)
    ax2.axhline(0.4876, color="black", linestyle=":", alpha=0.5, label="Majority Baseline (0.4876)")
    if best_ep > 0 and best_ep in epochs:
        ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.7)
    ax2.set_title(f"Val Metrics (Best F1: {sum_res.get('val_macro_f1', 0):.4f})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # (3) Grad Norm
    ax3 = axes[2]
    ax3.plot(epochs, hist["grad_norm"], label="Grad Norm (L2, pre-clip)", color="#9467bd", lw=2)
    ax3.set_title("Average Gradient Norm vs. Epoch", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("L2 Norm")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Tiêu đề tổng quát với cấu hình chính
    suptitle = (
        f"[{cfg.get('exp_id', '')}] {cfg.get('description', '')}\n"
        f"Opt: {cfg.get('optimizer')} | lr: {cfg.get('lr')} | batch: {cfg.get('batch')} | "
        f"init: {cfg.get('init')} | dropout: {cfg.get('dropout')} | clip: {cfg.get('clip_norm')} | "
        f"prec: {cfg.get('precision')}"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.03)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một đồ thị."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    metric_labels = {
        "val_loss": "Validation Loss",
        "train_loss": "Train Loss (eval mode)",
        "val_acc": "Validation Accuracy",
        "val_macro_f1": "Validation Macro-F1",
        "grad_norm": "Gradient Norm (pre-clip)",
    }
    label_y = metric_labels.get(metric, metric)

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "")
        if metric in hist and hist[metric]:
            ax.plot(hist["epoch"], hist[metric], label=f"{exp_id} ({cfg.get('optimizer')}, lr={cfg.get('lr')})", lw=1.8)

    ax.set_title(title if title else f"Comparison of {label_y}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
