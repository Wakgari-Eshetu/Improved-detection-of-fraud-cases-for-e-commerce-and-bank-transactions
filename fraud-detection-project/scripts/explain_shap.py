"""scripts/explain_shap.py
Generates feature importance and SHAP explainability artifacts for the best model.

Outputs:
 - feature_importance_top10.png
 - shap_summary.png
 - shap_force_tp.html, shap_force_fp.html, shap_force_fn.html
 - docs/TASK_3_SHAP.md (summary report)

Usage:
 python scripts/explain_shap.py --data data/processed/Fraud_Data_processed.csv --output outputs/explainability
If no model is found in modelsave/, an XGBoost model will be trained and saved to modelsave/best_model.joblib
"""
import os
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.utils.config import Config
from src.explainability.shap_explainer import SHAPExplainer


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    return df


def train_or_load_model(X_train, y_train, model_path=None):
    # Try to load if exists
    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
        return model, False

    # Otherwise train an XGBoost classifier as an ensemble-style model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    if model_path:
        joblib.dump(model, model_path)
    return model, True


def plot_feature_importance(model, feature_names, output_path, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model has no feature importances or coef_")

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances[::-1], align='center')
    plt.yticks(y_pos, top_features[::-1])
    plt.xlabel('Importance')
    plt.title('Top {} Feature Importances'.format(top_n))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(args):
    data_path = args.data or os.path.join(Config.PROCESSED_DATA_DIR, 'Fraud_Data_processed.csv')
    output_dir = args.output or os.path.join('outputs', 'explainability')
    model_save = args.model or os.path.join('modelsave', 'best_model.joblib')

    ensure_dir(output_dir)
    ensure_dir('modelsave')

    print(f"Loading data from {data_path}")
    df = load_data(data_path)

    target_col = args.target or 'class'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create a reproducible split and keep test set for explainability
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train or load model
    model, trained = train_or_load_model(X_train, y_train, model_save)
    if trained:
        print(f"Trained and saved model to {model_save}")
    else:
        print(f"Loaded model from {model_save}")

    # Built-in feature importance
    feat_names = list(X.columns)
    feat_imp_path = os.path.join(output_dir, 'feature_importance_top10.png')
    plot_feature_importance(model, feat_names, feat_imp_path, top_n=10)
    print(f"Saved feature importance plot to {feat_imp_path}")

    # SHAP analysis
    explainer = SHAPExplainer(model, feat_names)

    # Use a subset of test set for speed if necessary
    X_shap = X_test.copy()

    print("Computing SHAP values... this may take a moment")
    shap_values = explainer.explain(X_shap)

    # Global summary plot
    summary_path = os.path.join(output_dir, 'shap_summary.png')
    explainer.plot_summary_save(shap_values, X_shap, summary_path)
    print(f"Saved SHAP summary plot to {summary_path}")

    # Predict and identify TP, FP, FN
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    tp_idx = next((i for i in range(len(y_test)) if (y_test.iloc[i] == 1 and y_pred[i] == 1)), None)
    fp_idx = next((i for i in range(len(y_test)) if (y_test.iloc[i] == 0 and y_pred[i] == 1)), None)
    fn_idx = next((i for i in range(len(y_test)) if (y_test.iloc[i] == 1 and y_pred[i] == 0)), None)

    if tp_idx is None or fp_idx is None or fn_idx is None:
        print("Warning: unable to find one of TP/FP/FN in test set; saving available examples only.")

    # Force plots (saved as HTML due to JS interactivity)
    if tp_idx is not None:
        path = os.path.join(output_dir, 'shap_force_tp.html')
        explainer.plot_force_save(shap_values, X_shap, tp_idx, path)
        print(f"Saved SHAP force plot (TP) to {path}")

    if fp_idx is not None:
        path = os.path.join(output_dir, 'shap_force_fp.html')
        explainer.plot_force_save(shap_values, X_shap, fp_idx, path)
        print(f"Saved SHAP force plot (FP) to {path}")

    if fn_idx is not None:
        path = os.path.join(output_dir, 'shap_force_fn.html')
        explainer.plot_force_save(shap_values, X_shap, fn_idx, path)
        print(f"Saved SHAP force plot (FN) to {path}")

    # Extract top drivers and write a small report
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
    top5 = [(feat_names[i], float(mean_abs_shap[i])) for i in top5_idx]

    report_path = os.path.join('docs', 'TASK_3_SHAP.md')
    ensure_dir('docs')
    with open(report_path, 'w') as fh:
        fh.write('# Task 3 - Model Explainability (SHAP)\n\n')
        fh.write('## Top 5 drivers (by mean |SHAP|)\n')
        for f, v in top5:
            fh.write(f'- **{f}**: mean(|SHAP|) = {v:.6f}\n')
        fh.write('\n')
        fh.write('## Business recommendations\n')
        fh.write('1. Investigate transactions with high positive SHAP values for the top features (e.g., X). Add extra verification for these conditions.\n')
        fh.write('2. Consider blocking or flagging transactions when feature A and feature B co-occur with high contribution to fraud predictions.\n')
        fh.write('3. Use SHAP-driven feature thresholds (e.g., transactions within X hours of signup) to design rule-based checks.\n')

    print(f"Written report to {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to processed data CSV')
    parser.add_argument('--model', type=str, help='Path to saved model (joblib)')
    parser.add_argument('--output', type=str, help='Output directory for explainability artifacts')
    parser.add_argument('--target', type=str, help='Target column name (default: class)')
    args = parser.parse_args()
    main(args)