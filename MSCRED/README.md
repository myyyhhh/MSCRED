# MSCRED — Reproduction on Online Boutique Microservices

**M**ulti-**S**cale **C**onvolutional **R**ecurrent **E**ncoder-**D**ecoder

A reproduction of the AAAI 2019 paper: *"A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data"*, applied to the [Google Cloud Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) microservices system.

![MSCRED Architecture](https://github.com/7fantasysz/MSCRED/raw/master/framework.png)

---

## 📄 Paper Reference

> Zhang, C., Song, D., Chen, Y., Feng, X., Lumezanu, C., Cheng, W., Ni, J., Zong, B., Chen, H. & Chawla, N.V. (2019).  
> **A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data.**  
> *Proceedings of the AAAI Conference on Artificial Intelligence, 33*(01), 1409-1416.  
> [DOI: 10.1609/aaai.v33i01.33011409](https://doi.org/10.1609/aaai.v33i01.33011409)

---

## 🧪 Experiment Overview

### Objective

Deploy MSCRED on a real-world 12-microservice system (Online Boutique) to detect anomalies from Prometheus monitoring metrics (CPU, memory, disk I/O, network, pod status), with fault injection via Chaos Mesh.

### Setup

| Component | Details |
|:----------|:--------|
| **System** | Online Boutique (12 microservices) on minikube (Docker driver) |
| **Metrics** | 7 KPI types × 12 services = 84 sensors (15s scrape interval) |
| **Duration** | ~7.8 hours total (6h baseline + 1.8h fault injection) |
| **Faults** | 3 types × 3 rounds = 9 fault events via Chaos Mesh |
| **Training** | 134 signature matrix samples (normal only) |
| **Testing** | 42 samples (19 anomalous, 23 normal) |

### Fault Injection

| Fault | Target Service | Tool | Duration |
|:------|:--------------|:-----|:---------|
| Pod Kill | `frontend` | PodChaos | Instant |
| Network Delay (200ms) | `checkoutservice` | NetworkChaos | 5 min |
| CPU Stress (80%) | `recommendationservice` | StressChaos | 5 min |

---

## 📁 Repository Structure

```
MSCRED/
├── AAAI19-MSCRED.pdf          # Original paper (PDF)
├── README.md                  # This file
├── 实验报告.md                 # Experiment report (Chinese, detailed)
├── code/
│   ├── MSCRED_TF.py           # Main MSCRED model (TensorFlow)
│   ├── matrix_generator.py    # Signature matrix generation
│   ├── build_mscred_input.py  # Sensor matrix construction
│   ├── evaluate.py            # Evaluation on synthetic data (original)
│   ├── evaluate_ob.py         # Evaluation on Online Boutique data
│   └── readme.csv             # Original paper's instructions
├── data/
│   ├── online_boutique_sensors.csv    # 84-sensor matrix (Online Boutique)
│   ├── online_boutique_anomaly.csv    # Ground truth anomaly labels
│   ├── sensor_names.txt               # Sensor name mapping
│   ├── synthetic_data_with_anomaly-s-1.csv  # Original paper's synthetic data
│   └── test_anomaly.csv               # Original paper's test labels
├── MSCRED_model/              # Trained model checkpoints (see .gitignore)
└── data/matrix_data/          # Generated signature matrices (see .gitignore)

ob_etl/
├── etl_from_prometheus.py     # Prometheus CSV → KPI matrix
├── build_kpi_matrix.py        # KPI matrix builder (per-service)
├── preprocess.py              # KPI matrix → sliding window .seq files
├── requirements.txt           # Python dependencies (PyTorch, etc.)
└── __init__.py
```

---

## ⚙️ Requirements

### For MSCRED (TensorFlow 1.x)

- Python 3.7
- TensorFlow 1.x (`tensorflow-gpu==1.15` or `tensorflow==1.15`)
- NumPy, Pandas, SciPy, Matplotlib

**Note:** MSCRED uses TensorFlow 1.x with `tf.contrib`. Use Python 3.7 to avoid compatibility issues.

### For ETL Pipeline (PyTorch 2.x)

If you want to re-run the data processing pipeline:

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

See `ob_etl/requirements.txt`.

---

## 🔄 How to Reproduce

### Step 1: Generate signature matrices

```bash
python code/matrix_generator.py \
    --min_time 0 --max_time 1878 \
    --train_start_point 0 --train_end_point 1442 \
    --test_start_point 1442 --test_end_point 1878 \
    --raw_data_path ../data/online_boutique_sensors.csv \
    --save_data_path ../data/
```

This generates multi-scale signature matrices (windows 10, 30, 60) in `data/matrix_data/`.

### Step 2: Train the model

```bash
python code/MSCRED_TF.py \
    --train_test_label 1 \
    --sensor_n 84 \
    --training_iters 5 \
    --train_start_id 11 \
    --train_end_id 145 \
    --model_path ../MSCRED_model/
```

- `sensor_n=84`: 7 KPIs × 12 services
- `training_iters=5`: optimal epoch count (more leads to overfitting)

### Step 3: Test and evaluate

```bash
python code/MSCRED_TF.py --train_test_label 0
python code/evaluate_ob.py
```

The evaluation script outputs precision, recall, and F1 score across different thresholds.

### (Optional) Full ETL pipeline with raw Prometheus data

If you have the raw Prometheus CSV exports, you can re-run the full pipeline:

```bash
# 1. Build per-service KPI matrix
python ob_etl/build_kpi_matrix.py

# 2. Build the 84-sensor unified matrix
python code/build_mscred_input.py

# 3. Generate signature matrices
python code/matrix_generator.py \
    --raw_data_path ../data/online_boutique_sensors.csv \
    --save_data_path ../data/
```

---

## 📊 Results

| Metric | Value |
|:-------|:-----:|
| **F1 Score** | **0.522** |
| **Precision** | 0.444 (12/27) |
| **Recall** | 0.632 (12/19) |
| **TP / FP / TN / FN** | 12 / 15 / 8 / 7 |

### Key Findings

1. **5 epochs optimal** — More training leads to overfitting: the model learns to reconstruct anomaly patterns, losing discriminative power
2. **More sensors ≠ worse** — Including "redundant" services (loadgenerator) captures cascading failure signals, improving F1 (0.522 vs 0.410 with 70 sensors)
3. **Metric-based detection is harder than log-based** — Same system with LogFormer achieves F1=0.81 on application logs vs MSCRED's 0.52 on metrics

### Threshold Sweep

| α | Threshold | Precision | Recall | F1 |
|:-:|:---------:|:---------:|:------:|:--:|
| 0.5 | 46.5 | 0.444 | 0.632 | **0.522** |
| 1.0 | 93.0 | 1.000 | 0.105 | 0.190 |
| 1.5 | 139.5 | — | 0.000 | 0.000 |

---

## 📝 Notes

- This implementation is modified from the [author's demo code](https://github.com/7fantasysz/MSCRED), adapted for Python 3 / Windows compatibility
- The original paper used synthetic data with 30 sensors and 8000+ training samples; this experiment uses real microservice metrics with 84 sensors and 134 training samples
- For detailed analysis, see [实验报告.md](%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.md) (in Chinese)

---

## 📖 Citation

```bibtex
@inproceedings{zhang2019deep,
  title={A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data},
  author={Zhang, Chuxu and Song, Dongjin and Chen, Yuncong and Feng, Xinyang and Lumezanu, Cristian and Cheng, Wei and Ni, Jingchao and Zong, Bo and Chen, Haifeng and Chawla, Nitesh V},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={1409--1416},
  year={2019}
}
```
