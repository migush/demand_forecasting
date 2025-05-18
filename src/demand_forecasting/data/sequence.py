import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os
import json

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data. This dataset loads sequences on-demand
    rather than pre-materializing all sequences, drastically reducing memory usage.
    """
    def __init__(self, data_path, feature_columns, product_ids=None, lookback=30, stats_path=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the prepared data CSV
            feature_columns: List of feature column names to use
            product_ids: Optional list of product IDs to include (for train/test split)
            lookback: Number of time steps to look back
            stats_path: Path to feature statistics JSON for normalization
        """
        self.data = pd.read_csv(data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.feature_columns = feature_columns
        self.lookback = lookback
        
        # Filter by product IDs if provided
        if product_ids is not None:
            self.data = self.data[self.data['unique_id'].isin(product_ids)]
        
        # Group data by product
        self.product_groups = self.data.groupby('unique_id')
        
        # Create an index mapping for accessing sequences
        self._create_index_mapping()
        
        # Load statistics for normalization if provided
        self.stats = None
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
    
    def _create_index_mapping(self):
        """
        Create a mapping from index to (product_id, sequence_start_idx).
        This enables accessing sequences by index for PyTorch DataLoader.
        """
        self.index_mapping = []
        
        for product_id, group in self.product_groups:
            group_size = len(group)
            if group_size > self.lookback:
                # Sort by date to ensure correct sequence order
                group = group.sort_values('date')
                
                # Add valid sequence indices
                for i in range(group_size - self.lookback):
                    self.index_mapping.append((product_id, i))
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        """
        Get a sequence by index.
        
        Args:
            idx: Index of the sequence
        
        Returns:
            X: Input sequence
            y: Target value
        """
        product_id, start_idx = self.index_mapping[idx]
        
        # Get product data
        product_data = self.product_groups.get_group(product_id).sort_values('date')
        
        # Extract sequence and target
        sequence = product_data[self.feature_columns].iloc[start_idx:(start_idx + self.lookback)].values
        target = product_data['sales'].iloc[start_idx + self.lookback]
        
        # Normalize if stats are available
        if self.stats:
            sequence = self._normalize(sequence)
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def _normalize(self, sequence):
        """
        Normalize a sequence using pre-computed statistics.
        
        Args:
            sequence: Sequence to normalize
        
        Returns:
            Normalized sequence
        """
        normalized = sequence.copy()
        
        for i, col in enumerate(self.feature_columns):
            if col in self.stats:
                # Z-score normalization
                normalized[:, i] = (normalized[:, i] - self.stats[col]['mean']) / self.stats[col]['std']
        
        return normalized

def get_data_loaders(data_path='data/processed/prepared_sales_data.csv', 
                     stats_path='data/processed/feature_stats.json',
                     lookback=30, batch_size=32, test_size=0.2, val_size=0.2):
    """
    Create PyTorch DataLoaders for train, validation, and test sets using the on-demand Dataset.
    
    Args:
        data_path: Path to prepared data CSV
        stats_path: Path to feature statistics JSON
        lookback: Number of time steps to look back
        batch_size: Batch size for DataLoader
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
    
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each dataset split
    """
    # Read data
    data = pd.read_csv(data_path)
    
    # Get feature columns
    feature_columns = [col for col in data.columns if col not in ['date', 'unique_id', 'sales']]
    
    # Get unique product IDs
    product_ids = data['unique_id'].unique()
    
    # Split products into train/val/test sets
    # This ensures no product appears in multiple sets
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(product_ids)
    
    n_products = len(product_ids)
    n_test = int(n_products * test_size)
    n_train_val = n_products - n_test
    n_val = int(n_train_val * val_size)
    
    test_ids = product_ids[:n_test]
    val_ids = product_ids[n_test:n_test+n_val]
    train_ids = product_ids[n_test+n_val:]
    
    # Create datasets using on-demand approach
    train_dataset = TimeSeriesDataset(
        data_path=data_path,
        feature_columns=feature_columns,
        product_ids=train_ids,
        lookback=lookback,
        stats_path=stats_path
    )
    
    val_dataset = TimeSeriesDataset(
        data_path=data_path,
        feature_columns=feature_columns,
        product_ids=val_ids,
        lookback=lookback,
        stats_path=stats_path
    )
    
    test_dataset = TimeSeriesDataset(
        data_path=data_path,
        feature_columns=feature_columns,
        product_ids=test_ids,
        lookback=lookback,
        stats_path=stats_path
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader 