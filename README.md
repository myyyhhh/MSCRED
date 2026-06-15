# MSCRED Anomaly Detection on Online Boutique

Reproduction of the AAAI 2019 MSCRED paper — unsupervised anomaly detection on multivariate time series from the [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) microservices system.

| | |
|:---|:---|
| **Paper** | [A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data (AAAI 2019)](https://doi.org/10.1609/aaai.v33i01.33011409) |
| **Experiment** | MSCRED on 84 monitoring sensors × 1878 time steps with 9 Chaos Mesh fault injections |
| **Best F1** | **0.522** (Precision 0.444, Recall 0.632) |

## Structure

```
MSCRED/           ← Core MSCRED model, data, and experiment report
  code/           ← Model implementation (TensorFlow 1.x)
  data/           ← Sensor matrix, anomaly labels, signature matrices
  MSCRED_model/   ← Trained checkpoints (generated)
  AAAI19-MSCRED.pdf
  实验报告.md      ← Detailed experiment report (Chinese)

ob_etl/           ← ETL pipeline for Prometheus monitoring data
  etl_from_prometheus.py
  build_kpi_matrix.py
  preprocess.py
```

## Quick Start

```bash
# Generate signature matrices
python MSCRED/code/matrix_generator.py \
    --raw_data_path MSCRED/data/online_boutique_sensors.csv \
    --save_data_path MSCRED/data/

# Train (5 epochs)
python MSCRED/code/MSCRED_TF.py \
    --train_test_label 1 --sensor_n 84 --training_iters 5

# Test
python MSCRED/code/MSCRED_TF.py --train_test_label 0
python MSCRED/code/evaluate_ob.py
```

See [MSCRED/README.md](MSCRED/README.md) for full details.

## Data Source

Raw Prometheus metrics were collected from a minikube deployment of Online Boutique with Chaos Mesh fault injection. The preprocessed 84-sensor matrix (`online_boutique_sensors.csv`) is included in this repository — no external data download required.
