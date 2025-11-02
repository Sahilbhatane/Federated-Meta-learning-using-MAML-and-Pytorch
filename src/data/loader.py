import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder


class WearableDataset(Dataset):
    """
    PyTorch Dataset for wearable device data
    
    Based on Phase 2 analysis:
    - 9 input features
    - Binary or multi-class classification
    - Support for few-shot learning splits
    """
    
    def __init__(self, df, feature_cols, label_col=None, scaler=None, label_encoder=None):
        """
        Args:
            df: DataFrame with health monitoring data
            feature_cols: List of feature column names
            label_col: Target column name
            scaler: Fitted StandardScaler (or None to fit new)
            label_encoder: Fitted LabelEncoder (or None to fit new)
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        
        # Fit or transform features
        if scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(df[feature_cols].values)
        else:
            self.scaler = scaler
            features = self.scaler.transform(df[feature_cols].values)
        
        self.features = torch.FloatTensor(features)
        
        # Encode labels if provided
        if label_col is not None:
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform(df[label_col].values)
            else:
                self.label_encoder = label_encoder
                labels = self.label_encoder.transform(df[label_col].values)
            
            self.labels = torch.LongTensor(labels)
            self.num_classes = len(self.label_encoder.classes_)
        else:
            self.labels = None
            self.label_encoder = None
            self.num_classes = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


def load_federated_data(url: str = None) -> pd.DataFrame:
    """
    Load dataset from Hugging Face parquet file
    
    Uses pandas fallback method from Phase 2 to avoid PyTorch DLL issues.
    
    Args:
        url: Hugging Face parquet URL (defaults to train split)
    
    Returns:
        DataFrame with 140 samples, 13 columns
    """
    if url is None:
        url = "https://huggingface.co/datasets/SahilBhatane/Federated_Meta-learning_on_wearable_devices/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    
    print(f"Loading dataset from Hugging Face...")
    df = pd.read_parquet(url)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    
    return df


def partition_by_user(df: pd.DataFrame, 
                     user_col: str = 'Sensor_ID', 
                     num_clients: int = None) -> Dict[int, pd.DataFrame]:
    """
    Partition data by user ID for non-IID federated learning
    
    Based on Phase 2 insights:
    - 4 users (Sensor_ID: 1, 2, 3, 4)
    - Partition sizes: 30-42 samples per client
    - High heterogeneity (label variance: 83.84)
    
    Args:
        df: Full dataset
        user_col: Column name for user ID (default: 'Sensor_ID' from Phase 2)
        num_clients: Number of clients (default: all unique users)
    
    Returns:
        Dictionary mapping client_id -> user DataFrame
    """
    unique_users = sorted(df[user_col].unique())
    print(f"Found {len(unique_users)} unique users: {unique_users}")
    
    if num_clients is None or num_clients > len(unique_users):
        num_clients = len(unique_users)
    
    client_data = {}
    for i, user in enumerate(unique_users[:num_clients]):
        user_df = df[df[user_col] == user].copy()
        client_data[i] = user_df
        print(f"Client {i} (User {user}): {len(user_df)} samples")
    
    return client_data


def create_fewshot_splits(df: pd.DataFrame, 
                         k_shot: int = 5, 
                         n_query: int = 10,
                         label_col: str = 'Target_Health_Status') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create support and query sets for few-shot meta-learning
    
    Args:
        df: User's DataFrame
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        label_col: Target column name
    
    Returns:
        (support_df, query_df) for MAML adaptation
    """
    support_indices = []
    query_indices = []
    
    for label in df[label_col].unique():
        label_indices = df[df[label_col] == label].index.tolist()
        
        # Take k_shot for support, n_query for query
        if len(label_indices) >= k_shot + n_query:
            support_indices.extend(label_indices[:k_shot])
            query_indices.extend(label_indices[k_shot:k_shot + n_query])
        else:
            # If not enough samples, use all for support
            support_indices.extend(label_indices)
    
    support_df = df.loc[support_indices].copy()
    query_df = df.loc[query_indices].copy()
    
    return support_df, query_df


def create_client_loaders(client_partitions: Dict[int, pd.DataFrame], 
                         feature_cols: List[str], 
                         label_col: str = 'Target_Health_Status',
                         batch_size: int = 8,
                         train_split: float = 0.7,
                         k_shot: int = None) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """
    Create train/test DataLoaders for each client
    
    Based on Phase 2 insights:
    - Small batch size (8) for limited data (30-42 samples)
    - Support few-shot splits if k_shot is specified
    
    Args:
        client_partitions: Dictionary of client DataFrames
        feature_cols: List of feature columns
        label_col: Target column
        batch_size: Batch size (small for limited data)
        train_split: Train/test split ratio
        k_shot: If specified, use few-shot splits instead
    
    Returns:
        Dictionary mapping client_id -> (train_loader, test_loader)
    """
    client_loaders = {}
    
    # Fit global scaler and label encoder on all data
    all_data = pd.concat(client_partitions.values())
    global_scaler = StandardScaler()
    global_scaler.fit(all_data[feature_cols].values)
    
    global_label_encoder = LabelEncoder()
    global_label_encoder.fit(all_data[label_col].values)
    
    print(f"\nLabel classes: {global_label_encoder.classes_}")
    print(f"Number of classes: {len(global_label_encoder.classes_)}")
    
    for client_id, client_df in client_partitions.items():
        # Use few-shot or regular split
        if k_shot is not None:
            train_df, test_df = create_fewshot_splits(client_df, k_shot=k_shot, label_col=label_col)
        else:
            train_size = int(len(client_df) * train_split)
            train_df = client_df.iloc[:train_size]
            test_df = client_df.iloc[train_size:]
        
        # Create datasets with shared scaler/encoder
        train_dataset = WearableDataset(train_df, feature_cols, label_col, 
                                       scaler=global_scaler, label_encoder=global_label_encoder)
        test_dataset = WearableDataset(test_df, feature_cols, label_col,
                                      scaler=global_scaler, label_encoder=global_label_encoder)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        client_loaders[client_id] = (train_loader, test_loader)
        print(f"Client {client_id}: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    return client_loaders, global_scaler, global_label_encoder
