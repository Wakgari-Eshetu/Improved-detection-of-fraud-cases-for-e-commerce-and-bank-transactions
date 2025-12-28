from lime.lime_tabular import LimeTabularExplainer
import numpy as np

class LimeExplainer:
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = LimeTabularExplainer(
            training_data=np.array([]),  # Placeholder, should be set with training data
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

    def explain_instance(self, instance, num_features=10):
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )
        return explanation

    def explain_batch(self, instances, num_features=10):
        explanations = []
        for instance in instances:
            explanation = self.explain_instance(instance, num_features)
            explanations.append(explanation)
        return explanations

    def set_training_data(self, training_data):
        self.explainer.training_data = training_data