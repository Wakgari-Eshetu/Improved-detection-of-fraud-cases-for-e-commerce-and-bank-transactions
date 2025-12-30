# Task 3 — Model Explainability (SHAP)

## Objective
Interpret the best model's predictions using SHAP to understand what drives fraud detection and provide actionable business recommendations.

## How to run

1. Ensure dependencies are installed: `pip install -r requirements.txt` (project includes `shap` and `xgboost` in `requirements.txt`).
2. Run the explain script (uses processed data by default):

```bash
python scripts/explain_shap.py --data data/processed/Fraud_Data_processed.csv --output outputs/explainability
```

## Outputs (saved to `outputs/explainability` by default)
- `feature_importance_top10.png` — Top 10 built-in feature importances from the model
- `shap_summary.png` — Global SHAP summary plot (feature impact and direction)
- `shap_force_tp.html` — Force plot for a true positive example
- `shap_force_fp.html` — Force plot for a false positive example
- `shap_force_fn.html` — Force plot for a false negative example

## Interpretation notes
- The top 5 drivers (programmatically extracted) are listed in `docs/TASK_3_SHAP.md` after running the script.
- The `shap_force_*.html` files are interactive and intended for analysts to inspect per-instance contributions.

## Business recommendations (example template)
1. Add extra verification for transactions where `feature_X` is strongly positive in SHAP (these increase fraud probability).
2. Flag high-risk combinations (e.g., `feature_A` + `feature_B`) identified by SHAP for manual review.
3. Use SHAP-driven thresholds (e.g., transactions initiated close to signup with other high-impact features) to introduce friction only when risk is high.

Please review the generated `docs/TASK_3_SHAP.md` and the saved figures for specifics on the top drivers and suggested rules.