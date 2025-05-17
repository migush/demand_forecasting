import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from tqdm import tqdm
import os

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

def prepare_lstm_data(data_path='data/prepared_sales_data.csv', lookback=30, test_size=0.2, n_jobs=None):
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

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_lstm_data() 