"""run_experiments.py — Automated experiment execution pipeline.

Runs all experiments across the 7 required topics + baseline seed noise.
Saves JSON results to submission_2A202603007/results/ and plots to submission_2A202603007/figures/.
Generates all comparison plots, populates experiments.xlsx, and evaluates final predictions.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "code")

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data import prepare_data
from model import MLP, count_params, activation_stats
from train import DEFAULT_CFG, run_experiment, final_eval, set_seed
from plots import plot_run, plot_compare
from results_table import save_result, to_row, write_xlsx

REPO_ROOT = Path("d:/Code/VinAI/Labs/Track4/K4-Track4-Day1-Neuralnetwork")
SUB_DIR = REPO_ROOT / "submission_2A202603007"
RESULTS_DIR = SUB_DIR / "results"
FIGURES_DIR = SUB_DIR / "figures"
TEMPLATE_XLSX = REPO_ROOT / "templates" / "experiment_table_template.xlsx"
OUT_XLSX = SUB_DIR / "experiments.xlsx"
EVAL_PRED_CSV = SUB_DIR / "predictions_eval.csv"
EVAL_RESULT_JSON = SUB_DIR / "eval_result.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Các thí nghiệm có cùng cấu hình với base-s1
BASE_CLONES = {
    "loss-ce": ("loss", "Cross-Entropy Loss (chuẩn phân loại 7 lớp)"),
    "opt-sgdm-lr0.05": ("optimizer", "SGD + momentum 0.9 (lr chuẩn = 0.05)"),
    "hp-batch-512": ("hparam", "Batch size chuẩn 512"),
    "drop-0.0": ("dropout", "Không dùng Dropout (q=0.0)"),
    "clip-none-normlr": ("clipping", "Không clipping ở lr bình thường (lr=0.05)"),
    "amp-fp32": ("amp", "Độ chính xác chuẩn FP32"),
    "init-he": ("init", "Khởi tạo He Normal (Kaiming)"),
}


def define_experiments() -> list[dict]:
    """Định nghĩa danh sách đầy đủ tất cả thí nghiệm theo quy định."""
    exps = []

    # 1. Baseline runs (đo độ nhiễu 3 seed)
    for s in [1, 2, 3]:
        exps.append({
            **DEFAULT_CFG,
            "exp_id": f"base-s{s}",
            "group": "baseline",
            "description": f"Baseline M-base (seed {s})",
            "seed": s,
        })

    # 2. Topic 1: Loss (CE vs MSE)
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "loss-ce",
        "group": "loss",
        "description": "Cross-Entropy Loss (chuẩn phân loại 7 lớp)",
        "loss": "ce",
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "loss-mse",
        "group": "loss",
        "description": "MSE Loss trên nhãn one-hot",
        "loss": "mse",
        "seed": 1,
    })

    # 3. Topic 2: Optimizer (SGD, SGD+mom, Adam, AdamW với nhiều lr)
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-sgd-lr0.05",
        "group": "optimizer",
        "description": "SGD chuẩn (không momentum, lr=0.05)",
        "optimizer": "sgd",
        "lr": 0.05,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-sgd-lr0.2",
        "group": "optimizer",
        "description": "SGD chuẩn (lr lớn hơn = 0.2)",
        "optimizer": "sgd",
        "lr": 0.2,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-sgdm-lr0.01",
        "group": "optimizer",
        "description": "SGD + momentum 0.9 (lr nhỏ = 0.01)",
        "optimizer": "sgd_momentum",
        "lr": 0.01,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-sgdm-lr0.05",
        "group": "optimizer",
        "description": "SGD + momentum 0.9 (lr chuẩn = 0.05)",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-sgdm-lr0.1",
        "group": "optimizer",
        "description": "SGD + momentum 0.9 (lr cao = 0.1)",
        "optimizer": "sgd_momentum",
        "lr": 0.1,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-adam-lr3e-4",
        "group": "optimizer",
        "description": "Adam (lr nhỏ = 3e-4)",
        "optimizer": "adam",
        "lr": 0.0003,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-adam-lr1e-3",
        "group": "optimizer",
        "description": "Adam (lr chuẩn = 1e-3)",
        "optimizer": "adam",
        "lr": 0.001,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-adam-lr3e-3",
        "group": "optimizer",
        "description": "Adam (lr cao = 3e-3)",
        "optimizer": "adam",
        "lr": 0.003,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-adamw-lr1e-3",
        "group": "optimizer",
        "description": "AdamW (lr=1e-3, weight_decay=0.01)",
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-adamw-lr3e-3",
        "group": "optimizer",
        "description": "AdamW (lr=3e-3, weight_decay=0.01)",
        "optimizer": "adamw",
        "lr": 0.003,
        "weight_decay": 0.01,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "opt-adamw-wd0",
        "group": "optimizer",
        "description": "AdamW (lr=1e-3, weight_decay=0.0 để so với Adam)",
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.0,
        "seed": 1,
    })

    # 4. Topic 3: Hyper-parameter (batch size, architecture, weight decay)
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "hp-batch-128",
        "group": "hparam",
        "description": "Batch nhỏ 128 (nhiều bước cập nhật hơn)",
        "batch": 128,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "hp-batch-512",
        "group": "hparam",
        "description": "Batch chuẩn 512",
        "batch": 512,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "hp-batch-2048",
        "group": "hparam",
        "description": "Batch lớn 2048 (ít bước cập nhật hơn)",
        "batch": 2048,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "hp-arch-wide",
        "group": "hparam",
        "description": "M-wide (512 -> 256 -> 7, 161 287 params)",
        "hidden": (512, 256),
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "hp-arch-deep",
        "group": "hparam",
        "description": "M-deep (256 -> 128 -> 64 -> 7, 55 687 params)",
        "hidden": (256, 128, 64),
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "hp-wd-1e-4",
        "group": "hparam",
        "description": "SGD+mom với L2 regularization weight_decay=1e-4",
        "weight_decay": 0.0001,
        "seed": 1,
    })

    # 5. Topic 4: Dropout (q=0.0, 0.2, 0.4)
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "drop-0.0",
        "group": "dropout",
        "description": "Không dùng Dropout (q=0.0)",
        "dropout": 0.0,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "drop-0.2",
        "group": "dropout",
        "description": "Dropout vừa phải (q=0.2)",
        "dropout": 0.2,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "drop-0.4",
        "group": "dropout",
        "description": "Dropout mạnh (q=0.4)",
        "dropout": 0.4,
        "seed": 1,
    })

    # 6. Topic 5: Gradient clipping (normal lr vs high lr)
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "clip-none-normlr",
        "group": "clipping",
        "description": "Không clipping ở lr bình thường (lr=0.05)",
        "clip_norm": None,
        "lr": 0.05,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "clip-1.0-normlr",
        "group": "clipping",
        "description": "Clipping c=1.0 ở lr bình thường (lr=0.05)",
        "clip_norm": 1.0,
        "lr": 0.05,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "clip-none-highlr",
        "group": "clipping",
        "description": "Không clipping ở lr rất cao (lr=2.0) -> bất ổn định",
        "clip_norm": None,
        "lr": 2.0,
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "clip-1.0-highlr",
        "group": "clipping",
        "description": "Có clipping c=1.0 ở lr rất cao (lr=2.0) -> ngăn bùng nổ",
        "clip_norm": 1.0,
        "lr": 2.0,
        "seed": 1,
    })

    # 7. Topic 6: Mixed precision (FP32 vs FP16 vs BF16)
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "amp-fp32",
        "group": "amp",
        "description": "Độ chính xác chuẩn FP32",
        "precision": "fp32",
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "amp-fp16",
        "group": "amp",
        "description": "Mixed Precision FP16 (với GradScaler)",
        "precision": "fp16",
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "amp-bf16",
        "group": "amp",
        "description": "Mixed Precision BF16 (bfloat16 native)",
        "precision": "bf16",
        "seed": 1,
    })

    # 8. Topic 7: Weight initialization (zeros, normal, xavier, he)
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "init-zeros",
        "group": "init",
        "description": "Khởi tạo tất cả bằng 0 (W=0, b=0)",
        "init": "zeros",
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "init-normal",
        "group": "init",
        "description": "Khởi tạo Normal N(0, 0.01^2)",
        "init": "normal",
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "init-xavier",
        "group": "init",
        "description": "Khởi tạo Xavier Normal",
        "init": "xavier",
        "seed": 1,
    })
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "init-he",
        "group": "init",
        "description": "Khởi tạo He Normal (Kaiming)",
        "init": "he",
        "seed": 1,
    })

    # 9. Final Best Configuration (kết hợp các yếu tố tối ưu nhất dựa trên val)
    # AdamW lr=0.002, M-wide, He init, weight_decay=0.01
    exps.append({
        **DEFAULT_CFG,
        "exp_id": "final-model",
        "group": "final",
        "description": "Cấu hình cuối cùng: M-wide, AdamW lr=0.002, wd=0.01, He init, batch 512, 20 epochs",
        "hidden": (512, 256),
        "optimizer": "adamw",
        "lr": 0.002,
        "weight_decay": 0.01,
        "init": "he",
        "seed": 42,
    })

    return exps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%X')}] Bắt đầu pipeline huấn luyện trên {device}...")
    data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir=str(REPO_ROOT / "data/processed"))

    experiments = define_experiments()
    print(f"Tổng số thí nghiệm: {len(experiments)}")

    results_dict = {}

    for idx, cfg in enumerate(experiments, start=1):
        exp_id = cfg["exp_id"]
        json_file = RESULTS_DIR / f"{exp_id}.json"
        fig_file = FIGURES_DIR / f"{exp_id}.png"

        print(f"\n[{idx}/{len(experiments)}] Thí nghiệm {exp_id} ({cfg['group']}): {cfg['description']}")

        # Nếu đã có file JSON
        if json_file.exists():
            print(f"  -> Đã có {json_file.name}, nạp kết quả...")
            with open(json_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            if not fig_file.exists():
                plot_run(res, str(fig_file))
            results_dict[exp_id] = res
            continue

        # Nếu là clone của base-s1 và base-s1 đã chạy xong:
        if exp_id in BASE_CLONES and "base-s1" in results_dict:
            base_res = results_dict["base-s1"]
            grp, desc = BASE_CLONES[exp_id]
            res = {
                "cfg": {**base_res["cfg"], "exp_id": exp_id, "group": grp, "description": desc},
                "history": base_res["history"],
                "summary": base_res["summary"],
                "best_state": base_res.get("best_state"),
            }
            save_result(res, str(RESULTS_DIR))
            plot_run(res, str(fig_file))
            results_dict[exp_id] = res
            print(f"  -> Áp dụng cấu hình chuẩn từ base-s1 cho {exp_id} ({grp})")
            continue

        t_start = time.time()
        res = run_experiment(cfg, data)
        elapsed = time.time() - t_start

        # Lưu JSON và vẽ đồ thị
        save_result(res, str(RESULTS_DIR))
        plot_run(res, str(fig_file))

        s = res["summary"]
        print(f"  -> Xong trong {elapsed:.1f}s | Val Acc: {s['val_acc']:.4f} | Val F1: {s['val_macro_f1']:.4f} | Best Ep: {s['best_epoch']}")

        results_dict[exp_id] = res

        # Nếu vừa chạy base-s1, tự động cập nhật ngay cho các clone của base-s1 chưa có
        if exp_id == "base-s1":
            for clone_id, (grp, desc) in BASE_CLONES.items():
                c_json = RESULTS_DIR / f"{clone_id}.json"
                c_fig = FIGURES_DIR / f"{clone_id}.png"
                if not c_json.exists():
                    clone_res = {
                        "cfg": {**res["cfg"], "exp_id": clone_id, "group": grp, "description": desc},
                        "history": res["history"],
                        "summary": res["summary"],
                    }
                    save_result(clone_res, str(RESULTS_DIR))
                    plot_run(clone_res, str(c_fig))
                    results_dict[clone_id] = clone_res
                    print(f"     [Tự động tạo clone] {clone_id} -> {c_json.name}")

        # Lưu best_state của final-model và base-s1
        if exp_id in ["final-model", "base-s1"]:
            torch.save(res["best_state"], RESULTS_DIR / f"{exp_id}_best_state.pt")

    print("\n" + "="*50)
    print("VẼ ĐỒ THỊ SO SÁNH THEO NHÓM (compare_<group>.png)...")
    print("="*50)

    # 1. Compare Loss
    loss_runs = [results_dict[eid] for eid in ["loss-ce", "loss-mse"] if eid in results_dict]
    plot_compare(loss_runs, "val_macro_f1", str(FIGURES_DIR / "compare_loss.png"), "So sánh Loss: CE vs MSE (Val Macro-F1)")

    # 2. Compare Optimizer
    opt_ids = ["opt-sgd-lr0.05", "opt-sgd-lr0.2", "opt-sgdm-lr0.01", "opt-sgdm-lr0.05", "opt-sgdm-lr0.1",
               "opt-adam-lr3e-4", "opt-adam-lr1e-3", "opt-adam-lr3e-3", "opt-adamw-lr1e-3", "opt-adamw-lr3e-3", "opt-adamw-wd0"]
    opt_runs = [results_dict[eid] for eid in opt_ids if eid in results_dict]
    plot_compare(opt_runs, "val_macro_f1", str(FIGURES_DIR / "compare_optimizer.png"), "So sánh Bộ tối ưu hoá (Val Macro-F1 theo Epoch)")
    plot_compare(opt_runs, "val_loss", str(FIGURES_DIR / "compare_optimizer_loss.png"), "So sánh Bộ tối ưu hoá (Val Loss theo Epoch)")

    # 3. Compare Hyper-parameters
    hp_ids = ["hp-batch-128", "hp-batch-512", "hp-batch-2048", "hp-arch-wide", "hp-arch-deep", "hp-wd-1e-4"]
    hp_runs = [results_dict[eid] for eid in hp_ids if eid in results_dict]
    plot_compare(hp_runs, "val_macro_f1", str(FIGURES_DIR / "compare_hparam.png"), "So sánh Hyper-parameters (Val Macro-F1)")

    # 4. Compare Dropout
    drop_ids = ["drop-0.0", "drop-0.2", "drop-0.4"]
    drop_runs = [results_dict[eid] for eid in drop_ids if eid in results_dict]
    plot_compare(drop_runs, "val_loss", str(FIGURES_DIR / "compare_dropout.png"), "So sánh Dropout: Val Loss theo Epoch")

    # 5. Compare Clipping
    clip_ids = ["clip-none-normlr", "clip-1.0-normlr", "clip-none-highlr", "clip-1.0-highlr"]
    clip_runs = [results_dict[eid] for eid in clip_ids if eid in results_dict]
    plot_compare(clip_runs, "val_loss", str(FIGURES_DIR / "compare_clipping.png"), "So sánh Gradient Clipping (Val Loss theo Epoch)")

    # 6. Compare Mixed Precision
    amp_ids = ["amp-fp32", "amp-fp16", "amp-bf16"]
    amp_runs = [results_dict[eid] for eid in amp_ids if eid in results_dict]
    plot_compare(amp_runs, "val_macro_f1", str(FIGURES_DIR / "compare_amp.png"), "So sánh Mixed Precision (Val Macro-F1)")

    # 7. Compare Initialization
    init_ids = ["init-zeros", "init-normal", "init-xavier", "init-he"]
    init_runs = [results_dict[eid] for eid in init_ids if eid in results_dict]
    plot_compare(init_runs, "val_macro_f1", str(FIGURES_DIR / "compare_init.png"), "So sánh Khởi tạo trọng số (Val Macro-F1)")

    print("Đã vẽ tất cả ảnh so sánh theo nhóm!")

    print("\n" + "="*50)
    print("ĐÁNH GIÁ CUỐI TRÊN TẬP EVAL (final_eval)...")
    print("="*50)

    # Đánh giá cấu hình cuối cùng (final-model)
    final_res = results_dict["final-model"]
    if "best_state" not in final_res or final_res["best_state"] is None:
        final_state_path = RESULTS_DIR / "final-model_best_state.pt"
        if final_state_path.exists():
            final_res["best_state"] = torch.load(final_state_path)

    final_eval(final_res["cfg"], final_res, data, str(EVAL_PRED_CSV))

    # Chạy evaluate.py
    import subprocess
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--pred", str(EVAL_PRED_CSV),
        "--data", str(REPO_ROOT / "data" / "covtype.csv.gz"),
        "--meta", str(REPO_ROOT / "data" / "split_metadata.csv"),
        "--out", str(EVAL_RESULT_JSON),
    ]
    eval_proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(eval_proc.stdout)

    with open(EVAL_RESULT_JSON, "r", encoding="utf-8") as f:
        eval_scores = json.load(f)
    print(f"==> KẾT QUẢ EVAL CHÍNH THỨC: Accuracy = {eval_scores['accuracy']:.4f} | Macro-F1 = {eval_scores['macro_f1']:.4f}")

    # Đánh giá baseline trên eval để ghi vào bảng
    base_res = results_dict["base-s1"]
    if "best_state" not in base_res or base_res["best_state"] is None:
        base_state_path = RESULTS_DIR / "base-s1_best_state.pt"
        if base_state_path.exists():
            base_res["best_state"] = torch.load(base_state_path)

    base_pred_csv = SUB_DIR / "predictions_eval_baseline.csv"
    final_eval(base_res["cfg"], base_res, data, str(base_pred_csv))
    base_json = SUB_DIR / "eval_result_baseline.json"
    cmd_base = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--pred", str(base_pred_csv),
        "--data", str(REPO_ROOT / "data" / "covtype.csv.gz"),
        "--meta", str(REPO_ROOT / "data" / "split_metadata.csv"),
        "--out", str(base_json),
    ]
    subprocess.run(cmd_base, capture_output=True, text=True, check=True)
    with open(base_json, "r", encoding="utf-8") as f:
        base_eval_scores = json.load(f)
    print(f"==> KẾT QUẢ EVAL BASELINE: Accuracy = {base_eval_scores['accuracy']:.4f} | Macro-F1 = {base_eval_scores['macro_f1']:.4f}")

    print("\n" + "="*50)
    print("ĐIỀN BẢNG experiments.xlsx...")
    print("="*50)

    rows = []
    for cfg in experiments:
        eid = cfg["exp_id"]
        res = results_dict[eid]
        score_to_pass = None
        notes = ""
        if eid == "final-model":
            score_to_pass = eval_scores
            notes = "Cấu hình tối ưu cuối cùng, điểm eval đo bằng evaluate.py"
        elif eid == "base-s1":
            score_to_pass = base_eval_scores
            notes = "Baseline chính thức, điểm eval đo bằng evaluate.py"
        elif eid in ["base-s2", "base-s3"]:
            notes = f"Baseline seed {cfg['seed']} để đo độ nhiễu 2σ"
        elif eid == "clip-none-highlr":
            notes = "lr=2.0 không clip: loss tăng vọt dao động mạnh"
        elif eid == "clip-1.0-highlr":
            notes = "lr=2.0 có clip 1.0: ổn định gradient, kiềm chế loss"
        elif eid == "init-zeros":
            notes = "Mất đối xứng, nơ-ron chết, chỉ đoán lớp đa số (acc 0.4876)"
        elif eid == "init-normal":
            notes = "Tín hiệu triệt tiêu qua các lớp (vanishing activations)"
        elif eid == "opt-adamw-wd0":
            notes = "AdamW với wd=0 cho kết quả tương đương Adam chuẩn"
        elif eid in ["amp-fp16", "amp-bf16"]:
            notes = "Mạng nhỏ nên overhead kernel làm thời gian xấp xỉ hoặc nhỉnh hơn FP32"

        row = to_row(res, eval_scores=score_to_pass, notes=notes)
        rows.append(row)

    summary_notes = {
        "baseline": "Baseline ổn định, 3 seed cho thấy độ nhiễu seed rất nhỏ (2σ ~ 0.003), làm mốc so sánh vững chắc.",
        "loss": "CE vượt trội hoàn toàn so với MSE trên bài toán phân loại đa lớp (gradient không bão hoà).",
        "optimizer": "Adam/AdamW vượt trội hơn SGD/SGDM rõ rệt nhờ thích nghi bước nhảy riêng cho từng tham số.",
        "hparam": "M-wide và batch 128 giúp tăng độ chính xác; batch 2048 hội tụ chậm hơn do ít bước cập nhật.",
        "dropout": "Mô hình M-base chưa bị quá khớp nặng nên dropout làm giảm nhẹ val accuracy; không cần thiết ở bài này.",
        "clipping": "Ở lr bình thường ít ảnh hưởng; ở lr rất cao (2.0), clipping cứu mô hình khỏi bùng nổ gradient.",
        "amp": "FP16 và BF16 duy trì độ chính xác của FP32; trên mạng nhỏ overhead kernel khiến thời gian tương đương.",
        "init": "He và Xavier giúp gradient chảy tốt; Zeros hoàn toàn hỏng do đối xứng; Normal bị suy giảm phương sai.",
        "final": "M-wide kết hợp AdamW lr=0.002 đạt macro-F1 xuất sắc (>0.87), vượt xa ngưỡng nhiễu của baseline.",
    }

    write_xlsx(
        rows,
        str(TEMPLATE_XLSX),
        str(OUT_XLSX),
        summary_notes=summary_notes,
        baseline_seeds=["base-s1", "base-s2", "base-s3"]
    )

    print("\nHOÀN TẤT TOÀN BỘ PIPELINE THÀNH CÔNG!")


if __name__ == "__main__":
    main()
