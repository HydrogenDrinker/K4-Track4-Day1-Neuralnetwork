"""model.py — MLP model definition, initialization, parameter counting, and activation statistics.

Architecture:
    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Rules:
    - Raw logits at the output (NO softmax inside model).
    - Dropout only placed after ReLU of hidden layers.
    - All Linear layers have bias.
    - No BatchNorm, no residual connections.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
        in_features: số đặc trưng đầu vào (mặc định 54)
        num_classes: số lớp đầu ra (mặc định 7)
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout_rate = dropout
        self.init = init

        layers = []
        curr_in = in_features
        for h in self.hidden:
            layers.append(nn.Linear(curr_in, h, bias=True))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            curr_in = h

        # Lớp đầu ra (linear, không ReLU, không dropout, không softmax)
        layers.append(nn.Linear(curr_in, num_classes, bias=True))
        self.net = nn.Sequential(*layers)

        # Khởi tạo trọng số
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0).

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out))
        "he"      : nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")  (Var = 2/n_in)
        "default" : giữ khởi tạo mặc định của nn.Linear (Kaiming uniform)
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            if init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif init == "default":
                pass
            else:
                raise ValueError(f"Khởi tạo không hợp lệ: '{init}'. Chọn zeros|normal|xavier|he|default")


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp nn.Linear ở bước 0 trên một lô val.

    Dùng cho thí nghiệm so sánh các phương pháp khởi tạo tham số.
    """
    model.eval()
    stds = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            stds.append(float(h.std().item()))
    return stds
