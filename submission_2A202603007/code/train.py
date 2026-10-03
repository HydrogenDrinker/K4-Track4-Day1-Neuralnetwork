"""train.py — Training pipeline, evaluation, seed setting, and experiment execution.

All evaluation metrics (accuracy, macro-F1, confusion matrix) strictly align with scripts/evaluate.py.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base (He init, SGD+mom 0.9, lr=0.05, batch 512, 20 epochs)",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn bằng tập val
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc số thực (ví dụ 1.0)
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Đúng chuẩn công thức trong scripts/evaluate.py.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (lấy trung bình trên mọi phần tử).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_one_hot, reduction="mean")
    else:
        raise ValueError(f"Hàm mất mát không được hỗ trợ: {loss_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1, cm) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = len(X)
    all_preds = []
    all_targets = []

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += loss.item() * len(xb)
        p = torch.argmax(logits, dim=1)
        total_correct += (p == yb).sum().item()
        all_preds.append(p.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    mean_loss = total_loss / n
    acc = total_correct / n

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(mean_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "cm": cm,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)
    """
    set_seed(cfg["seed"])
    device = data["X_tr"].device

    hidden = tuple(cfg["hidden"])
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số không khớp bảng: có {count_params(model)}, kỳ vọng {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    precision = cfg.get("precision", "fp32")
    is_cuda = "cuda" in str(device)
    scaler = torch.amp.GradScaler("cuda") if (precision == "fp16" and is_cuda) else None

    # Step 0 loss trên val TRƯỚC bước cập nhật đầu tiên
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    diverged = False

    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch"])
    clip_norm = cfg.get("clip_norm")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision == "fp16" and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()

            elif precision == "bf16" and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            else:  # fp32
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            epoch_grad_norms.append(gn)

        if is_cuda:
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        if diverged:
            print(f"[{cfg['exp_id']}] Diverged at epoch {epoch} (NaN/inf loss)!")
            break

        # Đánh giá cuối epoch ở chế độ eval() (dùng 50 000 mẫu cố định của train theo GUIDE để đo nhanh và chuẩn)
        X_tr_eval = data["X_tr"][:50000] if len(data["X_tr"]) > 50000 else data["X_tr"]
        y_tr_eval = data["y_tr"][:50000] if len(data["y_tr"]) > 50000 else data["y_tr"]
        tr_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        avg_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)

        # Lưu best checkpoint theo val_loss thấp nhất
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_acc = val_eval["acc"]
            best_val_macro_f1 = val_eval["macro_f1"]

    peak_mem_MB = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if is_cuda else 0.0
    )
    time_per_epoch_s = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss) if not diverged else float("nan"),
        "best_epoch": int(best_epoch) if not diverged else 0,
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(time_per_epoch_s),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    df = pd.DataFrame({
        "row_id": row_id.astype(int),
        "pred": preds.astype(int),
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Đã ghi {len(df)} dòng dự đoán vào {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg["hidden"])
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
