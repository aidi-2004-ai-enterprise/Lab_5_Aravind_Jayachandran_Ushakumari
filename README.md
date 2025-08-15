# AIDI-2004-02 · AI in Enterprise  
## **Lab 5 – Bankruptcy Prediction (Binary Classification)**

Continuation of **Lab 4** using the **same modeling decisions**:
- **Models**: Logistic Regression (baseline), **XGBoost**, **LightGBM**  
- **Metrics**: **ROC-AUC, PR-AUC, F1**
- **Preprocessing**: StandardScaler for **LR/XGB (numeric only)**, **no scaling for LGBM**, OHE for categoricals  
- **Imbalance**: SMOTE (train-only) + class/pos weighting  
- **Feature Selection**: RandomForest feature importance (Top-20) → **VIF pruning ≤ 10**  
- **Explainability**: SHAP on best boosted model  
- **Stability**: PSI (train vs test) plots + CSV snapshot

---

## 1) Quickstart

### Prereqs
- **Python 3.10+** (3.13 OK; you may see benign warnings)
- Dataset CSV at `data/data.csv` with target column **`Bankrupt?`** (0/1)
- Windows PowerShell commands shown; use bash equivalents on macOS/Linux

### Create & activate venv
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python training_pipeline.py --data data/data.csv --target "Bankrupt?"


Repo layout
.
├── data/
│   └── data.csv
├── artifacts/
│   ├── eda/
│   ├── models/
│   ├── plots/
│   ├── psi/
│   ├── shap/
│   ├── report.md
│   ├── requirements_frozen.txt
│   └── ruff.toml
├── training_pipeline.py
├── requirements.txt
├── README.md
└── .ruff.toml
