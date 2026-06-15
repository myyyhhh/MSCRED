# MSCRED — Online Boutique 微服务异常检测复现

AAAI 2019 MSCRED 论文的复现实验：在 [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) 微服务系统上，基于 Prometheus 多维时序指标进行无监督异常检测。

| | |
|:---|:---|
| **论文** | [A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data (AAAI 2019)](https://doi.org/10.1609/aaai.v33i01.33011409) |
| **实验** | MSCRED 在 84 个监控传感器 × 1878 时间步上，注入 9 次 Chaos Mesh 故障 |
| **最佳 F1** | **0.522**（精确率 0.444，召回率 0.632） |

## 仓库结构

```
MSCRED/           ← MSCRED 模型核心：代码、数据、实验报告
  code/           ← 模型实现（TensorFlow 1.x）
  data/           ← 传感器矩阵、异常标签、签名矩阵
  MSCRED_model/   ← 训练好的模型检查点（已 gitignore，可重建）
  AAAI19-MSCRED.pdf
  实验报告.md      ← 详细实验报告

ob_etl/           ← Prometheus 监控数据 ETL 流水线
  etl_from_prometheus.py
  build_kpi_matrix.py
  preprocess.py
```

## 快速复现

```bash
# 1. 生成签名矩阵（多尺度）
python MSCRED/code/matrix_generator.py \
    --raw_data_path MSCRED/data/online_boutique_sensors.csv \
    --save_data_path MSCRED/data/

# 2. 训练（5 个 epoch 为最优）
python MSCRED/code/MSCRED_TF.py \
    --train_test_label 1 --sensor_n 84 --training_iters 5

# 3. 测试 + 评估
python MSCRED/code/MSCRED_TF.py --train_test_label 0
python MSCRED/code/evaluate_ob.py
```

详细说明请见 [MSCRED/README.md](MSCRED/README.md)。

## 数据来源

原始 Prometheus 指标通过 minikube 部署的 Online Boutique 系统采集，使用 Chaos Mesh 注入故障。预处理后的 84 维传感器矩阵（`online_boutique_sensors.csv`）已包含在仓库中，无需额外下载。
