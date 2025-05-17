import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from tqdm import tqdm
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

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

def create_sequences_for_product(product_data, lookback, feature_columns):
    """
    Create sequences for a single product.
    
    Args:
        product_data: DataFrame containing data for a single product
        lookback: Number of time steps to look back
        feature_columns: List of feature columns to use
    
    Returns:
        X: Input sequences
        y: Target values
    """
    X, y = [], []
    
    # Sort data by date to ensure correct sequence order
    product_data = product_data.sort_values('date')
    
    # Create sequences using vectorized operations
    for i in range(len(product_data) - lookback):
        sequence = product_data[feature_columns].iloc[i:(i + lookback)].values
        target = product_data['sales'].iloc[i + lookback]
        
        X.append(sequence)
        y.append(target)
    
    return np.array(X), np.array(y)

def process_product_chunk(product_ids, data, lookback, feature_columns):
    """
    Process a chunk of products in parallel.
    
    Args:
        product_ids: List of product IDs to process
        data: Full DataFrame
        lookback: Number of time steps to look back
        feature_columns: List of feature columns to use
    
    Returns:
        X_chunk: Input sequences for the chunk
        y_chunk: Target values for the chunk
    """
    X_chunk, y_chunk = [], []
    
    for product_id in product_ids:
        product_data = data[data['unique_id'] == product_id]
        if len(product_data) > lookback:  # Only process if enough data points
            X, y = create_sequences_for_product(product_data, lookback, feature_columns)
            X_chunk.extend(X)
            y_chunk.extend(y)
    
    return np.array(X_chunk), np.array(y_chunk)

def prepare_lstm_data(data_path='data/processed/prepared_sales_data.csv', lookback=30, test_size=0.2, n_jobs=None):
    """
    Prepare data for LSTM model using parallel processing.
    
    Args:
        data_path: Path to prepared data CSV
        lookback: Number of time steps to look back
        test_size: Proportion of data to use for testing
        n_jobs: Number of parallel jobs (default: number of CPU cores)
    
    Returns:
        X_train, X_test: Training and testing input sequences
        y_train, y_test: Training and testing target values
    """
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {n_jobs} parallel processes")
    
    # Read prepared data
    print("Reading prepared data...")
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    # Get all feature columns except the target
    feature_columns = [col for col in data.columns if col not in ['date', 'unique_id', 'sales']]
    
    # Get unique product IDs
    product_ids = data['unique_id'].unique()
    
    # Split products into chunks for parallel processing
    chunk_size = max(1, len(product_ids) // n_jobs)
    product_chunks = [product_ids[i:i + chunk_size] for i in range(0, len(product_ids), chunk_size)]
    
    # Process chunks in parallel
    print("Creating sequences in parallel...")
    X_chunks, y_chunks = [], []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        process_func = partial(process_product_chunk, data=data, lookback=lookback, feature_columns=feature_columns)
        results = list(tqdm(executor.map(process_func, product_chunks), total=len(product_chunks)))
    
    # Combine results
    for X_chunk, y_chunk in results:
        X_chunks.append(X_chunk)
        y_chunks.append(y_chunk)
    
    X = np.concatenate(X_chunks)
    y = np.concatenate(y_chunks)
    
    # Split into train and test sets
    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Convert to float32 to save memory
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Save processed data
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    return X_train, X_test, y_train, y_test

def get_data_loaders(data_path='data/processed/prepared_sales_data.csv', 
                     stats_path='data/processed/feature_stats.json',
                     lookback=30, batch_size=32, test_size=0.2, val_size=0.2):
    """
    Create PyTorch DataLoaders for train, validation, and test sets using the on-demand Dataset.
    
    Args:
        data_path: Path to the prepared data CSV
        stats_path: Path to feature statistics JSON
        lookback: Number of time steps to look back
        batch_size: Batch size for DataLoader
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    # Read data to get feature columns and product IDs
    data = pd.read_csv(data_path)
    feature_columns = [col for col in data.columns if col not in ['date', 'unique_id', 'sales']]
    product_ids = data['unique_id'].unique()
    
    # Split product IDs into train and test sets
    test_count = int(len(product_ids) * test_size)
    train_product_ids = product_ids[:-test_count]
    test_product_ids = product_ids[-test_count:]
    
    # Split train into train and validation
    val_count = int(len(train_product_ids) * val_size)
    val_product_ids = train_product_ids[-val_count:]
    train_product_ids = train_product_ids[:-val_count]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        data_path=data_path,
        feature_columns=feature_columns,
        product_ids=train_product_ids,
        lookback=lookback,
        stats_path=stats_path
    )
    
    val_dataset = TimeSeriesDataset(
        data_path=data_path,
        feature_columns=feature_columns,
        product_ids=val_product_ids,
        lookback=lookback,
        stats_path=stats_path
    )
    
    test_dataset = TimeSeriesDataset(
        data_path=data_path,
        feature_columns=feature_columns,
        product_ids=test_product_ids,
        lookback=lookback,
        stats_path=stats_path
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train loader: {len(train_loader.dataset)} sequences")
    print(f"Validation loader: {len(val_loader.dataset)} sequences")
    print(f"Test loader: {len(test_loader.dataset)} sequences")
    
    return train_loader, val_loader, test_loader

def main():
    # Either use the old approach (materializing all sequences)
    X_train, X_test, y_train, y_test = prepare_lstm_data()
    
    # Or use the new streaming approach
    train_loader, val_loader, test_loader = get_data_loaders()
    
    print("Data preparation completed!")

if __name__ == "__main__":
    main() 