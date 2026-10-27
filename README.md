# Bank Deposit Imbalance Benchmark

An empirical study investigating how class imbalance affects performance evaluation metrics for predicting bank term-deposit subscriptions using the UCI Bank Marketing Dataset (~45k records).

## 📊 Overview

Standard accuracy is often misleading under heavy class imbalance (the *Accuracy Paradox*). This project systematically varies the imbalance ratio across 5 controlled levels (from 1:20 to 1:1) and analyzes how evaluation metrics (Accuracy, Precision, Recall, F1-Score, PR Curves, AUC-PR) behave across three classifiers:
- **Logistic Regression**
- **Decision Tree** (regularized)
- **Random Forest**

## 🚀 Key Highlights & Research Protocol
- **Target Leakage Prevention**: Dropped the `duration` feature to reflect real-world pre-call telemarketing conditions.
- **Sentinel Treatment**: Correctly handled `pdays = -1` (sentinel code for uncontacted clients).
- **Controlled Prevalence Shift**: Train and test sets share matched prevalence ratios to evaluate metric sensitivity fairly.
- **Multi-Seed Stability**: All experiments run across 5 random seeds to report $\text{Mean} \pm \text{Std}$.

## 📁 Repository Structure

```text
├── data/
│   └── raw/               # Downloaded & extracted UCI dataset (gitignored)
├── src/
│   └── download_data.py   # Automated download script
├── main.ipynb             # Full experimental workflow & benchmarks
├── plan.md                # Research design and methodology specifications
├── problem_statement.md   # Assignment requirements and guidelines
├── .gitignore
└── README.md
```

## 🛠️ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/CodeCrafterAdi2006/bank-deposit-imbalance-benchmark.git
   cd bank-deposit-imbalance-benchmark
   ```
2. Open and run `main.ipynb` (or execute `python src/download_data.py` to fetch data automatically).
