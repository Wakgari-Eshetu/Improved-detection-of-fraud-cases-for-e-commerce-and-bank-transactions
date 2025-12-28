import argparse
import pandas as pd
import shap
import lime
import lime.lime_tabular
from src.models.trainer import load_model
from src.data.load_data import load_data

def explain_model_predictions(model_path, data_path, explainer_type='shap'):
    model = load_model(model_path)
    data = load_data(data_path)

    if explainer_type == 'shap':
        explainer = shap.Explainer(model)
        shap_values = explainer(data)
        shap.summary_plot(shap_values, data)
    elif explainer_type == 'lime':
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=data.values,
            feature_names=data.columns,
            class_names=['Not Fraud', 'Fraud'],
            mode='classification'
        )
        for i in range(len(data)):
            exp = explainer.explain_instance(data.iloc[i].values, model.predict_proba)
            exp.show_in_notebook(show_table=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explain model predictions using SHAP or LIME.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data for explanation.')
    parser.add_argument('--explainer_type', type=str, choices=['shap', 'lime'], default='shap', help='Type of explainer to use.')
    
    args = parser.parse_args()
    
    explain_model_predictions(args.model_path, args.data_path, args.explainer_type)