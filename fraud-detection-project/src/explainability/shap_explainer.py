from shap import TreeExplainer
import shap
import numpy as np
import pandas as pd

class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = TreeExplainer(model)

    def explain(self, X):
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def plot_summary(self, shap_values, X):
        shap.summary_plot(shap_values, X, feature_names=self.feature_names)

    def plot_force(self, shap_values, index):
        shap.force_plot(self.explainer.expected_value, shap_values[index], X.iloc[index])

    def plot_decision(self, shap_values, index):
        shap.decision_plot(self.explainer.expected_value, shap_values[index], X.iloc[index])