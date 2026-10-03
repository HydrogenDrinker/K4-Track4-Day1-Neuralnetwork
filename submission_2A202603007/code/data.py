"""data.py — Data loading, splitting, standardization, and batching.

CoverType dataset:
    X : float32, shape (N, 54) — 10 continuous numeric features, 44 binary (one-hot)
    y : int64,   shape (N,)    — labels 0..6 (converted from original 1..7)
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # First 10 columns are continuous numeric features


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    train_path = Path(processed_dir) / "train.npz"
    eval_path = Path(processed_dir) / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy train.npz hoặc eval.npz trong {processed_dir}. "
            f"Hãy chạy 'python scripts/split_data.py' trước."
        )

    tr_data = np.load(train_path)
    ev_data = np.load(eval_path)

    X_train_full = tr_data["X"].astype(np.float32)
    y_train_full = tr_data["y"].astype(np.int64)

    X_eval = ev_data["X"].astype(np.float32)
    y_eval = ev_data["y"].astype(np.int64)
    eval_row_id = ev_data["row_id"].astype(np.int64)

    assert X_train_full.shape == (464809, 54), f"Shape X_train_full bất thường: {X_train_full.shape}"
    assert y_train_full.shape == (464809,), f"Shape y_train_full bất thường: {y_train_full.shape}"
    assert X_eval.shape == (116203, 54), f"Shape X_eval bất thường: {X_eval.shape}"
    assert y_eval.shape == (116203,), f"Shape y_eval bất thường: {y_eval.shape}"
    assert eval_row_id.shape == (116203,), f"Shape eval_row_id bất thường: {eval_row_id.shape}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Tuyệt đối không tính trên val hay eval để tránh rò rỉ dữ liệu (data leakage).
    """
    numeric_data = X_tr[:, :N_NUMERIC]
    mean = np.mean(numeric_data, axis=0)
    std = np.std(numeric_data, axis=0)
    # Tránh chia cho 0 nếu phương sai bằng 0
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval (tensors trên device)
        eval_row_id (mảng numpy)
    """
    X_train_full, y_train_full, X_eval_raw, y_eval_raw, eval_row_id = load_split(processed_dir)
    X_tr_raw, y_tr_raw, X_val_raw, y_val_raw = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # Chuẩn hoá 10 đặc trưng số liên tục chỉ dựa trên thống kê train
    mean, std = fit_standardizer(X_tr_raw)
    X_tr_norm = apply_standardizer(X_tr_raw, mean, std)
    X_val_norm = apply_standardizer(X_val_raw, mean, std)
    X_eval_norm = apply_standardizer(X_eval_raw, mean, std)

    # Đưa tensor lên device
    dev = torch.device(device)
    X_tr = torch.tensor(X_tr_norm, dtype=torch.float32, device=dev)
    y_tr = torch.tensor(y_tr_raw, dtype=torch.int64, device=dev)
    X_val = torch.tensor(X_val_norm, dtype=torch.float32, device=dev)
    y_val = torch.tensor(y_val_raw, dtype=torch.int64, device=dev)
    X_eval = torch.tensor(X_eval_norm, dtype=torch.float32, device=dev)
    y_eval = torch.tensor(y_eval_raw, dtype=torch.int64, device=dev)

    # In thông tin kiểm tra
    print(f"Dataset summary (on device: {device}):")
    print(f"  Train: X {X_tr.shape}, y {y_tr.shape}")
    print(f"  Val  : X {X_val.shape}, y {y_val.shape}")
    print(f"  Eval : X {X_eval.shape}, y {y_eval.shape}")

    val_counts = torch.bincount(y_val, minlength=7)
    majority_class = torch.argmax(val_counts).item()
    majority_acc = val_counts[majority_class].item() / len(y_val)
    print(f"  Majority class on Val: class {majority_class} with accuracy {majority_acc:.4f} (~0.4876)")

    # Kiểm tra chuẩn hoá trên X_tr
    mean_tr = X_tr[:, :N_NUMERIC].mean(dim=0).cpu().numpy()
    std_tr = X_tr[:, :N_NUMERIC].std(dim=0).cpu().numpy()
    print(f"  Train numeric mean max abs: {np.abs(mean_tr).max():.5f} (≈ 0)")
    print(f"  Train numeric std max diff vs 1: {np.abs(std_tr - 1.0).max():.5f} (≈ 0)")

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối nếu nhỏ hơn batch_size vẫn được trả về đầy đủ.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
