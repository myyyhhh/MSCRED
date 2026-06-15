#!/usr/bin/env python3
"""
MSCRED Input Builder — 将 Prometheus CSV 转成 MSCRED 格式的传感器矩阵

流程:
  1. 读取 full_timeseries 目录下所有 CSV
  2. 每个 KPI 按 pod/service 透视成宽表
  3. 合并所有 KPI + service → 传感器矩阵 (rows=sensors, cols=timestamps)
  4. 确定 train/test 分割点
  5. Min-Max 归一化
  6. 输出为 MSCRED 可读的 CSV (无 header)
"""
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ─── 配置 ───────────────────────────────────────────────────
DATA_DIR = "D:/Code/stm/OnlineBoutique_data/data/full_timeseries"
OUTPUT_DIR = "D:/Code/stm/MSCRED/data"
TIMESTAMP_LOG = "D:/Code/stm/OnlineBoutique_data/data/experiment_timestamps.txt"

MSCRED_RAW_CSV = os.path.join(OUTPUT_DIR, "online_boutique_sensors.csv")

# 忽略的 KPI（node 级网络指标，没有 pod 维度）
IGNORE_KPIS = {"network_rx_rate", "network_tx_rate",
               "container_waiting_reason", "container_terminated_reason"}

# 排除的冗余服务（基础设施组件，非故障目标）
IGNORE_SERVICES = {"loadgenerator", "redis-cart"}


def extract_service(pod_name):
    """从 pod 名提取 service 名: frontend-759775d795-48t6g → frontend"""
    # K8s pod: <service>-<deployment_hash>-<pod_hash>
    parts = pod_name.rsplit("-", 2)
    if len(parts) >= 3 and len(parts[-1]) == 5:
        return parts[0]
    return pod_name


def build_sensor_matrix():
    """主函数: 读取所有 CSV 并构建 MSCRED 传感器矩阵"""
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"发现 {len(csv_files)} 个 CSV 文件\n")

    # 1. 读取每个 KPI 文件，透视成 (timestamp × service) 格式
    kpi_wide_dfs = {}

    for fname in sorted(csv_files):
        # 解析 KPI 名: full_timeseries_cpu_usage_rate.csv → cpu_usage_rate
        kpi = re.sub(r"^full_timeseries_", "", fname).replace(".csv", "")
        if kpi in IGNORE_KPIS:
            print(f"  跳过 {kpi}: 无 pod 维度")
            continue

        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(fpath)

        if "pod" not in df.columns or df["pod"].isna().all():
            print(f"  跳过 {kpi}: 无 pod 列")
            continue

        # 提取 service 名
        df["service"] = df["pod"].apply(extract_service)
        # 过滤冗余服务
        before = df["service"].nunique()
        df = df[~df["service"].isin(IGNORE_SERVICES)]
        after = df["service"].nunique()
        if before != after:
            print(f"  过滤 {kpi}: {before - after} 个冗余服务移除")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # 透视: 行=timestamp, 列=service, 值=value(均值)
        # 一个 service 可能有多个 pod 副本 → 取均值
        pivot = df.pivot_table(
            index="timestamp",
            columns="service",
            values="value",
            aggfunc="mean"
        )

        # 重命名列为 {kpi}:{service}
        pivot.columns = [f"{kpi}:{col}" for col in pivot.columns]
        kpi_wide_dfs[kpi] = pivot
        print(f"  {kpi:30s}: {len(pivot)} 行, {pivot.shape[1]} services")

    # 2. 合并所有 KPI — 外连接对齐时间戳
    print("\n合并所有 KPI...")
    merged = None
    for kpi, df in kpi_wide_dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="outer")

    # 按时间排序，填充缺失值（小间隙插值）
    merged = merged.sort_index()
    print(f"\n合并后矩阵: {merged.shape[0]} 时间步 × {merged.shape[1]} 传感器")
    print(f"  时间范围: {datetime.fromtimestamp(merged.index[0])} → "
          f"{datetime.fromtimestamp(merged.index[-1])}")
    print(f"  NaN 数量: {merged.isna().sum().sum()}")

    # 插值填充小间隙
    na_before = merged.isna().sum().sum()
    merged = merged.interpolate(method="linear", limit=6).ffill().bfill()
    merged = merged.fillna(0.0)
    print(f"  填充后 NaN: {merged.isna().sum().sum()}")

    # 3. 确定 train/test 分割点
    baseline_end_epoch = None
    with open(TIMESTAMP_LOG, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\[(\d+)\]\s+BASELINE_END", line)
            if m:
                baseline_end_epoch = int(m.group(1))

    if baseline_end_epoch is None:
        print("⚠️  未找到 BASELINE_END 时间戳，使用 70% 自动分割")
        split_idx = int(len(merged) * 0.7)
    else:
        # 找到最接近 BASELINE_END 的索引
        timestamps = merged.index.values
        split_idx = np.argmin(np.abs(timestamps - baseline_end_epoch))
        print(f"\nBASELINE_END 时间戳: {datetime.fromtimestamp(baseline_end_epoch)}")
        print(f"  对应矩阵索引: {split_idx}/{len(merged)}")

    train_start = 0
    train_end = split_idx
    test_start = split_idx
    test_end = len(merged)

    print(f"  训练集: {train_start} → {train_end} ({train_end - train_start} 步)")
    print(f"  测试集: {test_start} → {test_end} ({test_end - test_start} 步)")

    # 4. 转置为 MSCRED 格式: rows=sensors, cols=timestamps
    # MSCRED 期望: data[sensor_idx, time_idx]
    sensor_matrix = merged.values.T  # (n_sensors, n_timestamps)
    sensor_names = list(merged.columns)
    n_sensors = len(sensor_names)
    n_timestamps = sensor_matrix.shape[1]

    print(f"\n传感器矩阵: {n_sensors} sensors × {n_timestamps} timestamps")
    print(f"  传感器列表:")
    for i, name in enumerate(sensor_names):
        print(f"    [{i:3d}] {name}")

    # 5. Min-Max 归一化（按行，即每个 sensor 独立归一化）
    max_vals = np.max(sensor_matrix, axis=1, keepdims=True)
    min_vals = np.min(sensor_matrix, axis=1, keepdims=True)
    sensor_matrix_norm = (sensor_matrix - min_vals) / (max_vals - min_vals + 1e-6)

    # 6. 保存为 MSCRED CSV 格式 (无 header)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savetxt(MSCRED_RAW_CSV, sensor_matrix_norm, delimiter=",", fmt="%.6f")
    print(f"\n✅ 已保存: {MSCRED_RAW_CSV}")
    print(f"   文件大小: {os.path.getsize(MSCRED_RAW_CSV) / 1024 / 1024:.1f} MB")

    # 7. 同时保存传感器名称映射表
    names_path = os.path.join(OUTPUT_DIR, "sensor_names.txt")
    with open(names_path, "w") as f:
        for i, name in enumerate(sensor_names):
            f.write(f"{i},{name}\n")
    print(f"✅ 传感器映射: {names_path}")

    # 8. 输出 matrix_generator.py 参数建议
    print("\n" + "=" * 60)
    print("matrix_generator.py 推荐参数:")
    print("=" * 60)
    print(f"""
python matrix_generator.py \\
    --sensor_n {n_sensors} \\
    --min_time 0 \\
    --max_time {n_timestamps} \\
    --train_start_point {train_start} \\
    --train_end_point {train_end} \\
    --test_start_point {test_start} \\
    --test_end_point {test_end} \\
    --raw_data_path ../data/online_boutique_sensors.csv \\
    --save_data_path ../data/
""")

    return {
        "n_sensors": n_sensors,
        "n_timestamps": n_timestamps,
        "train_end": train_end,
        "test_end": test_end,
        "sensor_names": sensor_names,
    }


if __name__ == "__main__":
    info = build_sensor_matrix()
