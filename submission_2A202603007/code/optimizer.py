"""optimizer.py — Optimizer creation, learning rate schedulers, and gradient clipping.

Optimizers supported:
    - sgd:           w <- w - lr * g
    - sgd_momentum:  v <- mu * v + g ; w <- w - lr * v
    - adam:          Adaptive moment estimation with L2 penalty coupled into gradients
    - adamw:         Adam with decoupled weight decay
"""
from __future__ import annotations

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer theo tên cấu hình."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu '{name}' không nằm trong {OPTIMIZERS}")

    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch tốc độ học (tuỳ chọn)."""
    if name is None:
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Không hỗ trợ scheduler: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi dùng mixed precision FP16 + GradScaler: scaler.unscale_(optimizer) phải được gọi
    trước khi gọi hàm này để chuẩn gradient phản ánh đúng độ lớn thực tế.
    """
    params_list = list(params)
    if max_norm is None or max_norm <= 0:
        total_norm = clip_grad_norm_(params_list, float("inf"))
    else:
        total_norm = clip_grad_norm_(params_list, max_norm)
    return float(total_norm)
