import os
import tempfile
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from src.explainability.shap_explainer import SHAPExplainer


def test_shap_explainer_basic_flow():
    # Create tiny synthetic dataset
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        'f1': rng.normal(size=50),
        'f2': rng.normal(size=50),
        'f3': rng.randint(0, 2, size=50)
    })
    y = (X['f1'] + 0.5 * X['f3'] + rng.normal(scale=0.5, size=50) > 0).astype(int)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X, y)

    # SHAP explainer
    expl = SHAPExplainer(model, list(X.columns))
    shap_vals = expl.explain(X)

    assert shap_vals.shape[0] == X.shape[0]
    assert shap_vals.shape[1] == X.shape[1]

    # Save summary and a force plot to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = os.path.join(tmpdir, 'summary.png')
        force_path = os.path.join(tmpdir, 'force.html')
        expl.plot_summary_save(shap_vals, X, summary_path)
        expl.plot_force_save(shap_vals, X, 0, force_path)
        assert os.path.exists(summary_path)
        assert os.path.exists(force_path)
