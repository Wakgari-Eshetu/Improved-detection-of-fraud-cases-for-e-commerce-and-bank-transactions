from shap import TreeExplainer
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class SHAPExplainer:
    """Lightweight SHAP utility for generating and saving common SHAP plots.

    Methods added:
    - explain: returns shap_values
    - plot_summary_save: saves a global summary plot as PNG
    - plot_force_save: saves a force plot as HTML (interactive)
    """
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = TreeExplainer(model)

    def explain(self, X):
        """Compute SHAP values for dataset X.
        Returns a numpy array of shap values with shape (n_samples, n_features).
        """
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def plot_summary_save(self, shap_values, X, out_path, plot_type='dot'):
        """Create and save a SHAP summary plot as PNG."""
        # shap.summary_plot creates a matplotlib figure
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type=plot_type, show=False)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def plot_force_save(self, shap_values, X, index, out_path):
        """Generate a SHAP force plot for a single index and save as HTML for interactivity."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # shap.force_plot returns an HTML-able object; save as HTML
        force = shap.force_plot(self.explainer.expected_value, shap_values[index], X.iloc[index], feature_names=self.feature_names, matplotlib=False)
        shap.save_html(out_path, force)

    def plot_decision_save(self, shap_values, X, index, out_path):
        """Save a decision plot as a PNG."""
        plt.figure(figsize=(10, 6))
        shap.decision_plot(self.explainer.expected_value, shap_values[index], X.iloc[index], feature_names=self.feature_names, show=False)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()