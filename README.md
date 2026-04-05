# 🛡️ Vigilant Lens — AI Fraud Intelligence System

> **Detect. Explain. Prevent.**
> An intelligent fraud detection and transaction flagging system built using machine learning and behavioral analysis.

---

## 🚀 Overview

**Vigilant Lens** is a fintech + cybersecurity solution that analyzes financial transactions in real-time and flags suspicious activity using machine learning.

Instead of just predicting fraud, the system provides:

* 📊 **Risk scoring**
* 🚨 **Transaction flagging**
* 🧠 **Behavior-based insights**

---

## 🎯 Key Features

* 🔍 **Fraud Risk Scoring**
  Assigns a probability score to each transaction.

* 🚩 **Smart Flagging System**
  Categorizes transactions into:

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk

* ⚙️ **Feature Engineering Pipeline**
  Uses behavioral and transactional patterns:

  * Transaction amount patterns
  * Balance changes
  * Time-based activity
  * Transaction type anomalies

* 📁 **CSV Upload System**
  Upload bulk transactions and analyze instantly.

* 📊 **Interactive Dashboard**

  * Total transactions
  * Flagged transactions
  * Risk distribution
  * Highest risk cases

---

## 🧠 Machine Learning Approach

The model is trained on a financial transactions dataset (PaySim-style) and uses:

* Random Forest Classifier
* Feature-engineered inputs for better pattern detection

### 📌 Key Engineered Features:

* `amount_log` → scaled transaction magnitude
* `orig_balance_diff` → sender balance change
* `dest_balance_diff` → receiver balance anomaly
* `hour` → time of transaction
* `is_night` → unusual activity detection
* `transaction type encoding`

---

## ⚠️ Fraud Detection Logic

Transactions are flagged based on patterns such as:

* 💸 **Unusually high transaction amount**
* 🏦 **Account nearly drained**
* 🌙 **Night-time transactions**
* 🔁 **High-risk transaction types (TRANSFER / CASH_OUT)**
* ❓ **Suspicious destination balance behavior**

---

## 📊 Example Output

| Metric       | Value |
| ------------ | ----- |
| Transactions | 8     |
| Flagged      | 5     |
| High Risk    | 3     |
| Medium Risk  | 2     |
| Low Risk     | 3     |
| Max Risk     | ~46%  |

---

## 🛠️ Tech Stack

* **Python**
* **Pandas / NumPy**
* **Scikit-learn**
* **Streamlit (Frontend)**
* **Joblib (Model Serialization)**

---

## 📂 Project Structure

```
├── app.py                # Streamlit frontend
├── predict.py            # Inference + scoring logic
├── train_model.py        # Model training script
├── paysim_model.pkl      # Trained ML model
├── requirements.txt
├── sample_input.csv
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/vigilant-lens.git
cd vigilant-lens

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then:

1. Upload a CSV file
2. View flagged transactions
3. Analyze risk insights

---

## 🧪 Sample Input Format

```csv
step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
743,TRANSFER,985000,990500,5500,0,0
```

---

## 🧠 Future Improvements

* 🔍 SHAP-based explainability
* 🤖 Hybrid model (XGBoost + Autoencoder)
* 🌍 Multi-dataset support (PaySim + Credit Card data)
* 🔐 Cybersecurity signals (device/IP anomalies)
* 📈 User trust scoring system

---

## 🏆 Why This Project Stands Out

* Combines **FinTech + Cybersecurity**
* Focuses on **real-world fraud patterns**
* Includes **feature engineering (not just model training)**
* Provides **interpretable insights**
* Designed for **scalability and real-time use**

---

## 👨‍💻 Author

Built by **Husain Ahmed**
Machine Learning Enthusiast | Deep Learning from Scratch

---

## 📜 License

This project is open-source and available under the MIT License.
