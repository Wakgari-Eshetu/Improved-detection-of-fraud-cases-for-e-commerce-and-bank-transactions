#src/data/dataset.py file

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

class FraudDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_res = None
        self.y_train_res = None

    def load_data(self):
        """Load CSV into a DataFrame"""
        self.data = pd.read_csv(self.data_path)
        return self.data

    def preprocess_data(self, numeric_features=None, categorical_features=None, drop_columns=None):
        """
        Handle missing values, drop unnecessary columns, scale numeric features,
        encode categorical features.
        """
        # Drop missing values
        self.data.dropna(inplace=True)

        # Drop unused columns
        if drop_columns:
            self.data.drop(columns=drop_columns, inplace=True, errors='ignore')

        # Scaling numeric features
        if numeric_features:
            scaler = StandardScaler()
            self.data[numeric_features] = scaler.fit_transform(self.data[numeric_features])

        # Encoding categorical features
        if categorical_features:
            self.data = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)

        return self.data

    def split_data(self, target_column, test_size=0.2, random_state=42, stratify=True):
        """Split into train/test sets"""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        stratify_param = y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def apply_smote(self):
        """Apply SMOTE to the training set"""
        smote = SMOTE(random_state=42)
        self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train)
        return self.X_train_res, self.y_train_res

    def merge_datasets(self, other_dataset):
        """Merge with another FraudDataset"""
        self.data = pd.concat([self.data, other_dataset.data], ignore_index=True)
        return self.data

    def get_summary(self):
        """Return descriptive statistics"""
        return self.data.describe()
