import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """Preprocess wearable device data for federated learning"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fitted = False
    
    def fit_transform(self, df, feature_cols, label_col=None):
        """Fit scaler on data and transform"""
        df = df.copy()
        
        df[feature_cols] = self.feature_scaler.fit_transform(df[feature_cols])
        
        if label_col and label_col in df.columns:
            df[label_col] = self.label_encoder.fit_transform(df[label_col])
        
        self.fitted = True
        return df
    
    def transform(self, df, feature_cols, label_col=None):
        """Transform using fitted scaler"""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df = df.copy()
        df[feature_cols] = self.feature_scaler.transform(df[feature_cols])
        
        if label_col and label_col in df.columns:
            df[label_col] = self.label_encoder.transform(df[label_col])
        
        return df
    
    def get_num_classes(self):
        """Get number of unique classes"""
        return len(self.label_encoder.classes_) if self.fitted else 0


def remove_missing_values(df):
    """Remove rows with missing values"""
    return df.dropna()


def handle_outliers(df, feature_cols, n_std=3):
    """Remove outliers using z-score method"""
    df = df.copy()
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - n_std * std) & (df[col] <= mean + n_std * std)]
    return df
