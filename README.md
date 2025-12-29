# 🛠️ Industrial Predictive Maintenance System
> **High-Recall Failure Prediction Pipeline for Zero-Downtime Manufacturing**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![XGBoost](https://img.shields.io/badge/XGBoost-black?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Maintenance](https://img.shields.io/badge/Industry-4.0-blue)

## 🎯 Project Overview
Unplanned downtime in manufacturing costs thousands of dollars per hour. This project implements an end-to-end Machine Learning pipeline using **XGBoost** to predict mechanical failures before they occur. 

By prioritizing **Recall over Precision**, this system acts as a high-sensitivity early warning system, ensuring that 88% of all impending failures are flagged for maintenance.

---

## 🚀 Final Model Performance
The model was optimized using a **Custom Decision Threshold** to minimize False Negatives (Missed Failures), which are the most expensive errors in an industrial context.

### **Classification Report**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Healthy (0)** | 1.00 | 0.96 | 0.98 | 1932 |
| **Failure (1)** | 0.42 | **0.88** | 0.57 | 68 |
| **Overall Accuracy** | **95%** | **95%** | **95%** | 2000 |

### **Business Impact Analysis**
* **False Negatives (8):** Only 8 failures were missed. These typically correlate with **Random Failures (RNF)** which lack detectable sensor signatures.
* **High Sensitivity:** The model successfully captured **60 out of 68** total failure events.
* **Operational Safety:** By catching 88% of failures, the system provides a significant safety buffer for factory operations.



---

## ⚙️ Engineering & Logic
* **Feature Engineering:** Developed custom features like **Mechanical Power** (`Torque * Speed`) and **Thermal Delta** (`Process Temp - Air Temp`) to isolate failure signatures.
* **Class Imbalance:** Leveraged `scale_pos_weight` in XGBoost to handle the rare occurrence of failure events (only 3.4% of data).
* **Threshold Tuning:** Shifted the classification threshold to prioritize the detection of critical failures, reducing catastrophic risk by over 50% compared to standard models.

---

## 🛠️ Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Run Training Pipeline:**
    ```bash
    python src/main.py

3. **Run Real-Time Inference:**
    ```bash
    python src/predict.py
    
## 📂 Repository Structure

```text
predictive_maintenance/
├── data/
│   └── Dataset.csv            # Raw industrial sensor data
├── models/
│   ├── maintenance_model.json # Serialized XGBoost model
│   └── scaler.joblib          # Saved StandardScaler for inference
├── src/
│   ├── main.py                # Training & Feature Engineering pipeline
│   └── predict.py             # Real-time inference script
├── requirements.txt           # Production dependencies (pip install -r)
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```
👨‍💻 Author: Vishal Naikar

Tech Stack: Python, Machine Learning, MLOps

License: MIT
