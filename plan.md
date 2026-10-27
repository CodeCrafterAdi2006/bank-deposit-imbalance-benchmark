# Research & Implementation Plan: Comparative Study on Class Imbalance in Bank Term-Deposit Subscription Prediction

## Executive Overview
This plan establishes a research-grade experimental methodology and paper-writing framework adhering to IEEE standards to address the assignment prompt:
1. Investigating how class imbalance affects classification evaluation metrics (Accuracy, Confusion Matrix, Precision, Recall, F1, PR curves, AUC-PR).
2. Determining which metric provides the most meaningful assessment across varying imbalance levels.
3. Identifying which classifier (Logistic Regression, Decision Tree, Random Forest) provides the highest practical value in a telemarketing environment.

---

## 1. Experimental Methodology & Rigorous Protocol

### 1.1 Data Ingestion & Critical Preprocessing
* **Dataset**: UCI Bank Marketing Dataset (`bank-full.csv`, 45,211 rows, target `y` ∈ {yes, no}).
* **Addressing Target Leakage (`duration`)**:
  - **Primary Protocol**: Exclude the `duration` feature. In realistic pre-call telemarketing targeting, call length is unknown prior to dialing. Including it causes severe target leakage that distorts marketing realism.
  - **Ablation Study (Bonus Subsection in Paper)**: Conduct a concise comparative run with vs. without `duration` to explicitly demonstrate the leakage phenomenon and explain why tree models falsely exhibit near-perfect metrics when it is included.
* **Sentinel Value Treatment (`pdays`)**:
  - `pdays = -1` indicates a client was never contacted in a prior campaign.
  - Create a binary feature: `previously_contacted = (pdays != -1).astype(int)`.
  - For clients contacted previously, retain a normalized/binned log recency `log_pdays = log1p(pdays)`, setting non-contacted to `0`.
* **Categorical Encoding**:
  - Retain `'unknown'` as an explicit category (representing missing information as an informative signal).
  - Use One-Hot Encoding (`OneHotEncoder(drop='first', handle_unknown='ignore')`) for nominal features.
* **Feature Scaling**:
  - Apply `RobustScaler` / `StandardScaler` strictly on numeric features within cross-validation folds / training splits to prevent data leakage.

---

### 1.2 Imbalance Variation Protocol (Controlled Prevalence Shift)
To study how evaluation metrics behave across varying imbalance levels, the test set must reflect the controlled positive prevalence:
* **Prevalence / Imbalance Ratios ($r = \frac{N_+}{N_-}$)**:
  1. **Severe Imbalance (1:20)**: ~4.76% positive prevalence ($P \approx 1,900$, $N \approx 38,000$).
  2. **Moderate-High Imbalance (1:10)**: ~9.09% positive prevalence.
  3. **Natural Dataset Imbalance (1:7.5)**: ~11.70% positive prevalence (natural baseline).
  4. **Moderate Imbalance (1:3)**: ~25.00% positive prevalence.
  5. **Balanced (1:1)**: ~50.00% positive prevalence.
* **Variance Reduction & Multi-Seed Replication**:
  - Run **5 independent stratified random subsamples / seeds** (e.g., seeds: 42, 101, 2024, 777, 999) for each imbalance ratio.
  - Report all metrics as **$\text{Mean} \pm \text{Standard Deviation}$**.
* **Dual-Evaluation Perspective (Addressed in Paper Section V)**:
  - **Primary Design**: Matched train-test prevalence to isolate metric behavior under genuine population prevalence shifts.
  - **Secondary Benchmark**: Model trained on resampled distributions evaluated on the fixed natural test set (demonstrating the operational effect of training undersampling).

---

### 1.3 Classifier Configuration & Regularization
* **Logistic Regression (`LogisticRegression`)**:
  - Regularization: L2 (`C=1.0`), solver `lbfgs`, max iterations 1000.
  - Provides calibrated linear baseline probabilities.
* **Decision Tree (`DecisionTreeClassifier`)**:
  - Regularized with `max_depth=7`, `min_samples_leaf=25`, `min_samples_split=50`.
  - *Rationale*: Eliminates degenerate 0/1 leaf outputs, producing smooth, continuous probability rankings essential for reliable PR curves.
* **Random Forest (`RandomForestClassifier`)**:
  - Config: `n_estimators=200`, `max_depth=12`, `min_samples_leaf=10`, `n_jobs=-1`.
  - Captures complex non-linear feature interactions with low variance via bootstrap aggregation.

---

### 1.4 Metric Computation & Curve Interpolation
For each model and ratio across the 5 seeds:
1. **Confusion Matrix**: True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).
2. **Threshold-Dependent Metrics (at default $\tau = 0.5$)**:
   - $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
   - $\text{Precision} = \frac{TP}{TP + FP}$
   - $\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$
   - $\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
3. **Threshold-Independent Ranking Metrics**:
   - **Average Precision / AUC-PR**: Computed via `sklearn.metrics.average_precision_score` (step-wise interpolation per Davis & Goadrich, 2006, avoiding trapezoidal over-optimism).
   - **Precision-Recall (PR) Curves**: Plotted across varying thresholds $\tau \in [0, 1]$.
   - **Baseline PR line**: Explicitly plotted at $y = \text{Prevalence} = \frac{P}{P + N}$.

---

### 1.5 Practical Telemarketing Cost-Benefit Analysis (Section VI)
To answer *"which classifier provides the most useful predictions from a practical marketing perspective?"*:
* **Call Center Economic Model**:
  - Fixed cost per phone contact: $c = \$5$.
  - Expected customer lifetime value per deposit conversion: $v = \$100$.
  - Net Profit Equation:
    $$\text{Expected Net Profit} = v \cdot TP - c \cdot (TP + FP)$$
* **Optimal Operating Threshold ($\tau^*$)**:
  - $\tau^* = \frac{c}{v} = 0.05$ (meaning any customer with $>5\%$ probability is commercially viable to call).
* **Cumulative Gains & Lift at Deciles (Top 10% / Top 20% Call Capacity)**:
  - When telemarketing capacity is capped at calling only the top $K\%$ highest-risk/highest-propensity clients, what fraction of total subscribers are captured?

---

## 2. Directory Structure & Execution Roadmap

```text
assignment_1/
├── problem_statement.md
├── plan.md                               <-- [This File]
├── src/
│   ├── download_data.py                  <-- Automated fetch & verification of bank-full.csv
│   ├── data_pipeline.py                  <-- Cleaning, pdays feature engineering, encoding
│   ├── run_experiments.py               <-- Multi-seed benchmark across 5 ratios & 3 models
│   ├── plot_generators.py                <-- High-DPI IEEE-ready plots (PR curves, metric trends, lift)
│   └── evaluate_economics.py             <-- Profit curves, threshold optimization, lift charts
├── data/
│   ├── raw/                              <-- bank.zip, bank-full.csv
│   └── processed/
├── results/
│   ├── metrics_summary.csv               <-- Aggregated mean ± std for all models and ratios
│   ├── figures/                          <-- Vector PDFs and high-res PNGs for LaTeX inclusion
│   │   ├── pr_curves_comparison.png
│   │   ├── metric_sensitivity_plot.png
│   │   ├── confusion_matrices_grid.png
│   │   ├── cumulative_gains_lift.png
│   │   └── profit_curve.png
│   └── tables/                           <-- LaTeX formatted tables (.tex snippets)
└── paper/
    ├── main.tex                          <-- IEEE Conference template format
    ├── references.bib                    <-- Fully populated bibliography
    ├── sections/
    │   ├── 00_abstract.tex
    │   ├── 01_introduction.tex
    │   ├── 02_related_work.tex
    │   ├── 03_methodology.tex
    │   ├── 04_experimental_setup.tex
    │   ├── 05_results_and_analysis.tex
    │   ├── 06_discussion.tex
    │   └── 07_conclusion.tex
    └── figures/                          <-- Copied figures ready for Overleaf export
```

---

## 3. Detailed IEEE Paper Outline

1. **Title**: Empirical Analysis of Class Imbalance on Classifier Performance: Evaluating Bank Term Deposit Subscription with Precision-Recall Dynamics and Cost-Sensitive Utility
2. **Abstract**:
   - Synthesizes the core inquiry: how class imbalance distorts traditional accuracy metrics.
   - States dataset (~45.2k records), models (LR, DT, RF), and systematic ratio variation (1:20 to 1:1 across 5 seeds).
   - Highlights the Accuracy Paradox (Accuracy stable >95% while Recall collapses to <10%).
   - Confirms AUC-PR as the definitive threshold-independent metric and Random Forest as the optimal telemarketing model under cumulative gains.
3. **Keywords**: Class Imbalance, Precision-Recall Curve, AUC-PR, Accuracy Paradox, Bank Telemarketing, Cost-Sensitive Learning, Random Forest.
4. **I. Introduction**:
   - Real-world motivation of bank marketing campaigns.
   - Definition of class imbalance and the inherent risks of flawed metric selection.
   - Core research questions and paper contributions.
5. **II. Related Work / Literature Review**:
   - Survey of class imbalance in tabular ML (He & Garcia, Chawla et al. on SMOTE).
   - The Accuracy Paradox and the superiority of Precision-Recall over ROC under rare events (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015).
   - Telemarketing prediction benchmarks on the UCI Bank Marketing dataset (Moro et al., 2014).
6. **III. Proposed Methodology / System**:
   - End-to-end experimental framework: Leakage prevention (dropping `duration`), `pdays` sentinel handling, encoding.
   - Controlled multi-seed prevalence subsampling pipeline.
   - Algorithmic formulation of Logistic Regression, regularized Decision Tree, and Random Forest.
7. **IV. Experimental Setup**:
   - Dataset statistics, class distribution breakdown.
   - Hyperparameter configurations and regularization constraints.
   - Definition of evaluation metrics and step-wise average precision integration.
   - Economic cost-benefit parameters ($c = \$5, v = \$100, \tau^* = 0.05$).
8. **V. Results and Analysis**:
   - **Table I**: Comprehensive Performance Table across 5 Imbalance Ratios (Mean $\pm$ Std).
   - **Figure 1**: Metric Divergence vs. Imbalance Severity (Accuracy vs. F1/Recall/Precision).
   - **Figure 2**: Multi-panel Precision-Recall Curves across 1:20, natural 1:7.5, and 1:1 ratios.
   - **Figure 3**: Confusion Matrix Transitions across imbalance levels.
   - **Ablation Insight**: Impact of the `duration` feature on metric inflation.
9. **VI. Discussion**:
   - Direct answer to Question 1: Mathematical proof and empirical evidence of why Accuracy is deceptive and why AUC-PR is the premier metric.
   - Direct answer to Question 2: Practical marketing utility: Lift at deciles (Top 10% / 20%), threshold adjustment, and expected profit maximization.
10. **VII. Conclusion and Future Work**:
    - Concluding guidelines for practitioners handling imbalanced customer targeting.
    - Future directions: Cost-sensitive gradient boosting (LightGBM/XGBoost) and reinforcement-learning-based dynamic calling schedules.
11. **References**:
    - Comprehensive BibTeX list including UCI repository, Moro et al. (2014), Davis & Goadrich (2006), Provost et al. (1998), He & Garcia (2009).

---

## 4. Verification & Validation Checklist

- [ ] Automated download and checksum verification of UCI `bank-full.csv`.
- [ ] Verification that `duration` is absent in primary feature matrix to eliminate target leakage.
- [ ] Correct handling of `pdays = -1` via indicator and log-transform.
- [ ] Verification that Decision Tree probability output is smooth and not binary step functions.
- [ ] Calculation of AUC-PR via `average_precision_score`.
- [ ] 5-seed repetition to produce error bars ($\pm \sigma$) for all tabular results.
- [ ] Generation of high-resolution vector/PNG plots with proper font sizing and legends.
- [ ] Complete LaTeX compilation validation for Overleaf upload (clean `.tex` and `.bib`).
