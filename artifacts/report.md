# Lab 5 Report - Company Bankruptcy Prediction

## Jot Notes (per component)

### EDA
- Checked imbalance; guided SMOTE + class_weight and stratified CV.
- Skewness/outliers -> IQR winsorization; no row drops.
- Correlation heatmap used for awareness (selection stays simple).
- Numeric dataset; OHE reserved if categoricals appear.

### Preprocessing
- Median impute + IQR capping (numeric only).
- Scaler: LR/XGB numeric; LGBM unscaled. OHE for categoricals.
- SMOTE (train-only) and weighting per Lab 4.
- PSI (train vs test) validates split stability; seeds fixed.

### Feature Selection
- RandomForest importance -> keep Top-20 (simple, per Lab 4).
- VIF pruning if VIF>10 to reduce multicollinearity.
- Shared final feature list across models (fair comparison).
- Drivers are financial ratios -> interpretable.

### Tuning
- RandomizedSearchCV (Stratified 5-fold), scoring=ROC-AUC.
- Bounded grids for speed/stability; imbalance handled in-pipeline.
- Persist best params per model.
- Keep it simple; no fragile early-stopping callbacks.

### Training
- Three models: LR (baseline), XGBoost, LightGBM.
- Pipelines apply SMOTE only after preprocessing.
- Persisted models to artifacts/models/.
- Reproducible via global seeds.

### Evaluation
- Metrics: ROC-AUC, PR-AUC, F1@0.5 (train & test).
- Overlays: ROC / PR / Calibration (train vs test).
- Classification reports included.
- Best picked by Test PR-AUC then ROC-AUC.

### SHAP
- TreeExplainer summary plot for best boosted model.
- Feature attributions align with finance ratios.
- Sample cap for runtime control.
- Supports regulatory explainability.

### PSI
- Per-feature PSI; top-15 chart + CSV snapshot.
- Thresholds: <0.1 stable; 0.1-0.2 monitor; >0.2 drift.
- If drift high on key drivers -> retrain/re-sample.
- Monitors generalization to test distribution.

## EDA Visuals
- class_dist: D:\Sem 2\2004 Ai in enterprise systems Bipin\Lab_5\Lab_5_Aravind_Jayachandran_Ushakumari\artifacts\eda\class_distribution.png
- corr: D:\Sem 2\2004 Ai in enterprise systems Bipin\Lab_5\Lab_5_Aravind_Jayachandran_Ushakumari\artifacts\eda\corr_heatmap.png
- skew: D:\Sem 2\2004 Ai in enterprise systems Bipin\Lab_5\Lab_5_Aravind_Jayachandran_Ushakumari\artifacts\eda\skewness_bar.png
- box_top_skew: D:\Sem 2\2004 Ai in enterprise systems Bipin\Lab_5\Lab_5_Aravind_Jayachandran_Ushakumari\artifacts\eda\boxplots_top_skewed.png

## Feature Selection (RandomForest Top-K + VIF)
### RandomForest Importances (Top 20)
| feature | importance |
| --- | --- |
|  Persistent EPS in the Last Four Seasons | 0.04529158824729681 |
|  Borrowing dependency | 0.04482492858284566 |
|  Continuous interest rate (after tax) | 0.04029805594542892 |
|  Total debt/Total net worth | 0.03860984420485683 |
|  Net Income to Total Assets | 0.03756641679508865 |
|  Retained Earnings to Total Assets | 0.03471061251983363 |
|  Debt ratio % | 0.03453701092752637 |
|  Total income/Total expense | 0.033160942327268655 |
|  Net worth/Assets | 0.03231454879489309 |
|  Equity to Liability | 0.030543055957431186 |
|  Net profit before tax/Paid-in capital | 0.027402402760458276 |
|  Liability to Equity | 0.025810456296254 |
|  After-tax net Interest Rate | 0.023743905487199415 |
|  ROA(B) before interest and depreciation after tax | 0.021678953388770378 |
|  ROA(C) before interest and depreciation before interest | 0.019830247366733184 |
|  Degree of Financial Leverage (DFL) | 0.018768239021930942 |
|  Quick Ratio | 0.01864875455525828 |
|  Net Income to Stockholder's Equity | 0.018435456804177817 |
|  Interest Expense Ratio | 0.016093419466152715 |
|  Net Value Growth Rate | 0.01602351217634687 |

### Final Selected Numeric Features
 Borrowing dependency,  Continuous interest rate (after tax),  Total debt/Total net worth,  Net Income to Total Assets,  Retained Earnings to Total Assets,  Debt ratio %,  Total income/Total expense,  Equity to Liability,  Net profit before tax/Paid-in capital,  ROA(C) before interest and depreciation before interest,  Degree of Financial Leverage (DFL),  Quick Ratio,  Net Income to Stockholder's Equity,  Interest Expense Ratio,  Net Value Growth Rate

## Multicollinearity (VIF) - Final Snapshot (Top 10)
| feature | vif |
| --- | --- |
|  Net Income to Total Assets | 7.935629217109681 |
|  ROA(C) before interest and depreciation before interest | 6.011239702735002 |
|  Borrowing dependency | 3.409369348144134 |
|  Net Income to Stockholder's Equity | 3.21711068566823 |
|  Retained Earnings to Total Assets | 2.7714024216124784 |
|  Debt ratio % | 2.4882356736326563 |
|  Net profit before tax/Paid-in capital | 2.4564651784235028 |
|  Equity to Liability | 2.0764118450143196 |
|  Total debt/Total net worth | 1.2075749351867953 |
|  Net Value Growth Rate | 1.0837339762932523 |

## Tuning Summary (best hyperparameters)
### LogisticRegression
- Best ROC-AUC (CV): **0.9354**

```json
{
  "clf__C": 0.014618962793704957,
  "clf__penalty": "l2"
}
```
### XGBoost
- Best ROC-AUC (CV): **0.9208**

```json
{
  "clf__colsample_bytree": 0.9714548519562596,
  "clf__learning_rate": 0.012250254554691817,
  "clf__max_depth": 3,
  "clf__n_estimators": 429,
  "clf__reg_alpha": 0.01232789160545079,
  "clf__reg_lambda": 2.9084251176062836,
  "clf__scale_pos_weight": 29.994318181818183,
  "clf__subsample": 0.9812236474710556
}
```
### LightGBM
- Best ROC-AUC (CV): **0.9189**

```json
{
  "clf__colsample_bytree": 0.5257393756249946,
  "clf__learning_rate": 0.00999263542679435,
  "clf__n_estimators": 474,
  "clf__num_leaves": 31,
  "clf__reg_alpha": 0.003013864904679803,
  "clf__reg_lambda": 0.6112310526245793,
  "clf__scale_pos_weight": 29.994318181818183,
  "clf__subsample": 0.7447263801387816
}
```

## Model Comparison (Train vs Test)
|  | ('ROC-AUC', 'Test') | ('ROC-AUC', 'Train') | ('PR-AUC', 'Test') | ('PR-AUC', 'Train') | ('F1', 'Test') | ('F1', 'Train') |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.9272 | 0.9845 | 0.42 | 0.573 | 0.2553 | 0.3232 |
| LogisticRegression | 0.9299 | 0.9392 | 0.3492 | 0.3749 | 0.2824 | 0.2844 |
| XGBoost | 0.9238 | 0.9659 | 0.4007 | 0.4748 | 0.1751 | 0.1764 |

## Classification Reports
### LogisticRegression — Train
```
              precision    recall  f1-score   support

           0      0.995     0.856     0.921      5279
           1      0.170     0.881     0.284       176

    accuracy                          0.857      5455
   macro avg      0.582     0.868     0.602      5455
weighted avg      0.969     0.857     0.900      5455

```
### LogisticRegression — Test
```
              precision    recall  f1-score   support

           0      0.994     0.863     0.924      1320
           1      0.170     0.841     0.282        44

    accuracy                          0.862      1364
   macro avg      0.582     0.852     0.603      1364
weighted avg      0.967     0.862     0.903      1364

```
### XGBoost — Train
```
              precision    recall  f1-score   support

           0      1.000     0.689     0.816      5279
           1      0.097     1.000     0.176       176

    accuracy                          0.699      5455
   macro avg      0.548     0.844     0.496      5455
weighted avg      0.971     0.699     0.795      5455

```
### XGBoost — Test
```
              precision    recall  f1-score   support

           0      0.996     0.717     0.834      1320
           1      0.097     0.909     0.175        44

    accuracy                          0.724      1364
   macro avg      0.546     0.813     0.505      1364
weighted avg      0.967     0.724     0.813      1364

```
### LightGBM — Train
```
              precision    recall  f1-score   support

           0      1.000     0.860     0.925      5279
           1      0.193     1.000     0.323       176

    accuracy                          0.865      5455
   macro avg      0.596     0.930     0.624      5455
weighted avg      0.974     0.865     0.906      5455

```
### LightGBM — Test
```
              precision    recall  f1-score   support

           0      0.993     0.847     0.914      1320
           1      0.151     0.818     0.255        44

    accuracy                          0.846      1364
   macro avg      0.572     0.833     0.585      1364
weighted avg      0.966     0.846     0.893      1364

```

## PSI (Drift) — Train vs Test
| feature | psi |
| --- | --- |
|  No-credit Interval | 0.03871095356991455 |
|  Long-term Liability to Current Assets | 0.03552281251865563 |
|  Interest-bearing debt interest rate | 0.03279770256459854 |
|  Inventory/Current Liability | 0.023556189492256987 |
|  Quick Assets/Total Assets | 0.02071593441936666 |
|  Net Income to Total Assets | 0.01834356776760204 |
|  Net profit before tax/Paid-in capital | 0.017891228514733227 |
|  Fixed Assets to Assets | 0.017749595081910843 |
|  Per Share Net profit before tax (Yuan ¥) | 0.01684083041184302 |
|  Persistent EPS in the Last Four Seasons | 0.01577114720623811 |
|  Realized Sales Gross Margin | 0.015092441870729065 |
|  Tax rate (A) | 0.014560632510713701 |
|  Quick Ratio | 0.014209402362755343 |
|  Long-term fund suitability ratio (A) | 0.014176484058968201 |
|  Allocation rate per person | 0.013892332904165753 |
- PSI bar: D:\Sem 2\2004 Ai in enterprise systems Bipin\Lab_5\Lab_5_Aravind_Jayachandran_Ushakumari\artifacts\psi\psi_bar.png

## SHAP Summary
- Generated for best tree model (if applicable): D:\Sem 2\2004 Ai in enterprise systems Bipin\Lab_5\Lab_5_Aravind_Jayachandran_Ushakumari\artifacts\shap\shap_summary.png

## Q&A
**Challenges & fixes**
- Class imbalance -> SMOTE (train-only) + class/pos weights.
- Multicollinearity -> VIF pruning after Top-20 selection.
- Calibration & thresholds -> reliability curves; F1@0.5 reported.
- Version quirks -> avoided fragile early-stopping callbacks.


**Lab 4 -> Lab 5 influence**
- Preprocessing (StandardScaler for LR/XGB numeric; OHE for categoricals; no scaling for LGBM).
- Models (LR, XGBoost, LightGBM); RF used only for simple FS.
- Metrics focus (ROC-AUC, PR-AUC, F1) and SHAP explainability.
- Stability checks via PSI & stratified splitting.


**Deployment recommendation**
- **LightGBM** — Choose based on best Test PR-AUC and competitive ROC-AUC with reasonable calibration.