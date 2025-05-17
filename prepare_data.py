import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import gc
from tqdm import tqdm

def process_chunk(chunk, calendar_df, scaler=None, fit_scaler=False):
    """
    Process a chunk of sales data.
    
    Args:
        chunk: DataFrame chunk of sales data
        calendar_df: Calendar DataFrame
        scaler: Optional MinMaxScaler for numerical features
        fit_scaler: Whether to fit the scaler on this chunk
    
    Returns:
        Processed DataFrame chunk
    """
    # Convert date columns to datetime
    chunk['date'] = pd.to_datetime(chunk['date'])
    
    # Add time features
    chunk['year'] = chunk['date'].dt.year
    chunk['month'] = chunk['date'].dt.month
    chunk['day'] = chunk['date'].dt.day
    chunk['dayofweek'] = chunk['date'].dt.dayofweek
    chunk['quarter'] = chunk['date'].dt.quarter
    chunk['is_weekend'] = chunk['dayofweek'].isin([5, 6]).astype(int)
    
    # Add cyclical encoding for time features
    chunk['month_sin'] = np.sin(2 * np.pi * chunk['month']/12)
    chunk['month_cos'] = np.cos(2 * np.pi * chunk['month']/12)
    chunk['dayofweek_sin'] = np.sin(2 * np.pi * chunk['dayofweek']/7)
    chunk['dayofweek_cos'] = np.cos(2 * np.pi * chunk['dayofweek']/7)
    
    # Merge with calendar data
    chunk = pd.merge(
        chunk,
        calendar_df[['date', 'warehouse', 'holiday', 'holiday_name', 'shops_closed', 
                    'winter_school_holidays', 'school_holidays']],
        on=['date', 'warehouse'],
        how='left'
    )
    
    # Fill missing calendar values with 0
    calendar_columns = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']
    chunk[calendar_columns] = chunk[calendar_columns].fillna(0)
    
    # Create holiday type features
    chunk['is_holiday'] = chunk['holiday'].astype(int)
    chunk['is_shop_closed'] = chunk['shops_closed'].astype(int)
    chunk['is_school_holiday'] = (chunk['winter_school_holidays'] | chunk['school_holidays']).astype(int)
    
    # Drop unnecessary columns
    chunk.drop(columns=['warehouse', 'holiday_name'], inplace=True)
    
    # Scale numerical features if scaler is provided
    if scaler is not None:
        numerical_columns = ['sales', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos']
        if fit_scaler:
            chunk[numerical_columns] = scaler.fit_transform(chunk[numerical_columns])
        else:
            chunk[numerical_columns] = scaler.transform(chunk[numerical_columns])
    
    return chunk

def prepare_sales_data(chunk_size=100000):
    """
    Prepare sales data using chunked processing to optimize memory usage.
    
    Args:
        chunk_size: Number of rows to process at once
    """
    print("Reading calendar data...")
    calendar_df = pd.read_csv('data/calendar.csv')
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Create output directory if it doesn't exist
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process sales data in chunks
    print("Processing sales data in chunks...")
    chunk_files = []
    
    # Read and process sales data in chunks
    for i, chunk in enumerate(tqdm(pd.read_csv('data/sales_train.csv', chunksize=chunk_size))):
        # Process chunk - fit scaler on first chunk only
        processed_chunk = process_chunk(chunk, calendar_df, scaler, fit_scaler=(i == 0))
        
        # Save chunk to temporary file
        chunk_file = f'{output_dir}/chunk_{i}.csv'
        processed_chunk.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)
        
        # Clear memory
        del processed_chunk
        gc.collect()
    
    # Combine all chunks
    print("Combining processed chunks...")
    with open(f'{output_dir}/prepared_sales_data.csv', 'w') as outfile:
        # Write header
        pd.read_csv(chunk_files[0], nrows=0).to_csv(outfile, index=False)
        
        # Append each chunk
        for chunk_file in tqdm(chunk_files):
            chunk = pd.read_csv(chunk_file)
            chunk.to_csv(outfile, header=False, index=False)
            os.remove(chunk_file)  # Remove temporary file
    
    # Create a symlink in the main data directory for backward compatibility
    if not os.path.exists('data/prepared_sales_data.csv'):
        os.symlink(f'{output_dir}/prepared_sales_data.csv', 'data/prepared_sales_data.csv')
    
    print("Data preparation completed!")
    
    return pd.read_csv(f'{output_dir}/prepared_sales_data.csv')

if __name__ == "__main__":
    prepare_sales_data() 