# 🛡️ Fraud Detection System

<p align="center">
  <b>Hybrid machine learning system for transaction fraud analysis</b><br>
  Combines <b>transaction-level anomaly detection</b> and <b>behavioral fraud analysis</b>
</p>

---

## ✨ Overview

This project is a **hybrid fraud detection system** built using two complementary machine learning models:

- 💸 **PaySim Model** — analyzes transaction flow and balance anomalies
- 🧠 **Kartik Model** — analyzes behavioral patterns such as amount, time, category, and location

The system generates:

- 📈 **Fraud probability**
- 🚨 **Fraud flag**
- 🏷️ **Risk label** (`LOW`, `MEDIUM`, `HIGH`)

---

## 🚀 Key Features

- 🤖 **Dual-model fraud detection**
- 📊 **Risk-based scoring instead of only binary output**
- 📁 **CSV batch prediction support**
- ⚖️ **Handles imbalanced fraud datasets**
- 🧩 **Combines transaction logic with behavioral intelligence**

---

## 🧠 Model Pipeline

### 🔹 PaySim Model
Detects suspicious transaction logic using features such as:

- transaction type
- amount
- sender balance
- receiver balance
- balance difference

### 🔹 Kartik Model
Detects suspicious user behavior using features such as:

- transaction amount
- category
- time of transaction
- night activity
- location distance
- merchant and user behavior trends

### 🔗 Final Risk Score

```python
final_score = 0.3 * paysim_prob + 0.7 * kartik_prob