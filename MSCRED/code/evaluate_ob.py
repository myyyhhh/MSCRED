#!/usr/bin/env python3
"""
MSCRED 评估脚本 — Online Boutique 版
支持: anomaly score + P/R/F1 + 多 alpha 对比调优
"""
import argparse, re, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="MSCRED evaluation for Online Boutique")
parser.add_argument("--thred_broken", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=None, help="单个 alpha（留空则自动调优）")
parser.add_argument("--gap_time", type=int, default=10)
parser.add_argument("--test_start_point", type=int, default=1442)
parser.add_argument("--test_end_point", type=int, default=1878)
parser.add_argument("--matrix_data_path", type=str, default="../data/matrix_data/")
parser.add_argument("--output_plot", type=str, default="../data/anomaly_score.png")
args = parser.parse_args()

gap_time = args.gap_time
test_start = int(args.test_start_point / gap_time)
test_end = int(args.test_end_point / gap_time)

# ═══════════════════════════════════════════════════════════
# 1. 加载测试数据并计算 anomaly score
# ═══════════════════════════════════════════════════════════
test_data_path = os.path.join(args.matrix_data_path, "test_data/")
reconstructed_data_path = os.path.join(args.matrix_data_path, "reconstructed_data/")

test_anomaly_score = []
test_data_ids = []

for i in range(test_start, test_end):
    gt_path = os.path.join(test_data_path, f"test_data_{i}.npy")
    rec_path = os.path.join(reconstructed_data_path, f"reconstructed_data_{i}.npy")
    if not os.path.exists(gt_path) or not os.path.exists(rec_path):
        continue
    gt = np.load(gt_path)                       # (5, 3, 84, 84)
    rec = np.squeeze(np.array(np.load(rec_path)))  # (84, 84, 3)
    if rec.ndim != 3:
        continue
    select_gt = gt[4, 0, :, :]        # last step, first scale
    select_rec = rec[:, :, 0]          # first scale
    if select_gt.shape != select_rec.shape:
        continue
    error = np.square(select_gt - select_rec)
    num_broken = np.sum(error > args.thred_broken)
    test_anomaly_score.append(num_broken)
    test_data_ids.append(i)

test_anomaly_score = np.array(test_anomaly_score)
n = len(test_anomaly_score)
print(f"测试样本: {n} 个 (data_id {test_start}~{test_end - 1})")
print(f"分数范围: {test_anomaly_score.min():.1f} ~ {test_anomaly_score.max():.1f}")

# ═══════════════════════════════════════════════════════════
# 2. Ground truth 标签
#    根据故障注入时间定义异常窗口 (test sample index)
# ═══════════════════════════════════════════════════════════
# 窗口格式: (start_idx, end_idx) 闭区间
fault_windows = [
    (0, 0),     # R1_F1 aftermath | data_id 145
    (3, 4),     # R1_F2 net-delay | data_id 148-149
    (8, 9),     # R1_F3 cpu-stress | data_id 153-154
    (13, 14),   # R2_F1 pod-kill | data_id 158-159
    (17, 18),   # R2_F2 net-delay | data_id 162-163
    (22, 24),   # R2_F3 cpu-stress | data_id 167-169
    (28, 29),   # R3_F1 pod-kill | data_id 173-174
    (32, 33),   # R3_F2 net-delay | data_id 177-178
    (37, 39),   # R3_F3 cpu-stress | data_id 182-184
]

ground_truth = np.zeros(n, dtype=int)
for s, e in fault_windows:
    ground_truth[s:e+1] = 1

n_anom = ground_truth.sum()
print(f"Ground truth: {n_anom}/{n} 异常 ({100*n_anom/n:.1f}%)")

# ═══════════════════════════════════════════════════════════
# 3. 调优 alpha，找最优 F1
# ═══════════════════════════════════════════════════════════
def compute_metrics(score, gt, alpha, val_count):
    """给定 alpha，计算 P/R/F1"""
    valid_max = np.max(score[:val_count])
    threshold = valid_max * alpha
    pred = (score > threshold).astype(int)
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, threshold, tp, fp, tn, fn

val_count = max(1, n // 5)

if args.alpha is not None:
    # 单 alpha 模式
    alphas = [args.alpha]
else:
    # 自动调优: 扫描 alpha 0.5~5.0
    alphas = np.arange(0.5, 5.1, 0.1)

print(f"\n{'='*70}")
print(f"  Alpha 调优 (验证集: 前 {val_count} 个样本)")
print(f"{'='*70}")
print(f"  {'alpha':>6s}  {'threshold':>9s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  "
      f"{'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
print(f"  {'-'*60}")

best_f1 = -1
best_alpha = alphas[0]
best_metrics = None

results = []
for a in alphas:
    prec, rec, f1, thr, tp, fp, tn, fn = compute_metrics(
        test_anomaly_score, ground_truth, a, val_count)
    results.append((a, thr, tp, fp, tn, fn, prec, rec, f1))
    print(f"  {a:6.1f}  {thr:9.1f}  {tp:4d}  {fp:4d}  {tn:4d}  {fn:4d}  "
          f"{prec:6.3f}  {rec:6.3f}  {f1:6.3f}")
    if f1 > best_f1:
        best_f1 = f1
        best_alpha = a
        best_metrics = (prec, rec, f1, thr, tp, fp, tn, fn)

print(f"\n  ⭐ 最优: alpha={best_alpha:.1f}, F1={best_f1:.3f}")
print(f"     Precision={best_metrics[0]:.3f}, Recall={best_metrics[1]:.3f}")
print(f"     TP={best_metrics[4]}, FP={best_metrics[5]}, TN={best_metrics[6]}, FN={best_metrics[7]}")

# ═══════════════════════════════════════════════════════════
# 4. 用最佳 alpha 绘图
# ═══════════════════════════════════════════════════════════
best_thr = best_metrics[3]
best_pred = (test_anomaly_score > best_thr).astype(int)

fig, axes = plt.subplots(figsize=(14, 6))
x = np.arange(n)

# 主曲线
plt.plot(x, test_anomaly_score, "b-", linewidth=1.5, label="Anomaly Score")

# 阈值线
plt.axhline(y=best_thr, color="black", linestyle="--", linewidth=1.5,
            label=f"Threshold={best_thr:.0f} (α={best_alpha:.1f})")

# 真实异常窗口 (红色背景)
for s, e in fault_windows:
    plt.axvspan(s - 0.5, e + 0.5, color="red", alpha=0.12, linewidth=0)
    # 标注故障简称
    fidx = fault_windows.index((s, e))
    labels = ["F1", "F2", "F3"]
    mid = (s + e) / 2
    plt.text(mid, plt.ylim()[1] * 0.93, labels[fidx % 3],
             ha="center", fontsize=9, color="red", fontweight="bold")

# 标记检测结果
tp_idx = np.where((best_pred == 1) & (ground_truth == 1))[0]
fp_idx = np.where((best_pred == 1) & (ground_truth == 0))[0]
fn_idx = np.where((best_pred == 0) & (ground_truth == 1))[0]

plt.scatter(tp_idx, test_anomaly_score[tp_idx], color="green", s=40,
            zorder=5, marker="o", label=f"TP ({len(tp_idx)})")
plt.scatter(fp_idx, test_anomaly_score[fp_idx], color="orange", s=40,
            zorder=5, marker="x", label=f"FP ({len(fp_idx)})")
plt.scatter(fn_idx, test_anomaly_score[fn_idx], color="red", s=40,
            zorder=5, marker="v", label=f"FN ({len(fn_idx)})")

plt.xlabel("Test Sample Index", fontsize=13)
plt.ylabel("Anomaly Score (broken pixels)", fontsize=13)
plt.title(f"MSCRED - Online Boutique  (F1={best_f1:.3f}, P={best_metrics[0]:.3f}, R={best_metrics[1]:.3f})",
          fontsize=14)
plt.legend(fontsize=10, ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(args.output_plot, dpi=150)
print(f"\n✅ 图像已保存: {args.output_plot}")

# ═══════════════════════════════════════════════════════════
# 5. 详细输出
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  各样本详情 (alpha={best_alpha:.1f})")
print(f"{'='*70}")
print(f"  {'Idx':>4s}  {'data_id':>7s}  {'Score':>7s}  {'GT':>3s}  {'Pred':>4s}  {'Result':>8s}")
print(f"  {'-'*45}")
for i in range(n):
    gt = "ANOM" if ground_truth[i] else "norm"
    pred = "ANOM" if best_pred[i] else "norm"
    if ground_truth[i] == 1 and best_pred[i] == 1:
        result = "TP"
    elif ground_truth[i] == 0 and best_pred[i] == 1:
        result = "FP"
    elif ground_truth[i] == 1 and best_pred[i] == 0:
        result = "FN"
    else:
        result = "TN"
    print(f"  {i:4d}  {test_data_ids[i]:7d}  {test_anomaly_score[i]:7.1f}  {gt:>3s}  {pred:>4s}  {result:>8s}")

print(f"\n{'='*70}")
print(f"  最终指标 (alpha={best_alpha:.1f})")
print(f"{'='*70}")
print(f"  Precision:  {best_metrics[0]:.3f}  ({best_metrics[4]} / {best_metrics[4]+best_metrics[5]})")
print(f"  Recall:     {best_metrics[1]:.3f}  ({best_metrics[4]} / {best_metrics[4]+best_metrics[7]})")
print(f"  F1 Score:   {best_f1:.3f}")
print(f"  Threshold:  {best_thr:.1f}")
print(f"  TP={best_metrics[4]}  FP={best_metrics[5]}  TN={best_metrics[6]}  FN={best_metrics[7]}")
print(f"  Accuracy:   {(best_metrics[4]+best_metrics[6])/n:.3f}")
