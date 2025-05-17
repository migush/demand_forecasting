import pandas as pd
import os
from pathlib import Path
import logging
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import psutil
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_chunk(chunk_data, output_dir, chunk_idx):
    """
    Process a single chunk of data and convert it to Parquet.
    
    Args:
        chunk_data: DataFrame chunk to process
        output_dir: Directory to write temporary files
        chunk_idx: Index of the chunk
    
    Returns:
        Path to the temporary Parquet file
    """
    # Convert date column to proper datetime if it exists
    if 'date' in chunk_data.columns:
        chunk_data['date'] = pd.to_datetime(chunk_data['date'])
    
    # Convert to PyArrow Table
    table = pa.Table.from_pandas(chunk_data)
    
    # Write chunk to temporary file
    temp_file = os.path.join(output_dir, f'chunk_{chunk_idx}.parquet')
    pq.write_table(table, temp_file, compression='snappy')
    
    return temp_file

def get_optimal_chunk_size(file_size, num_cores):
    """
    Calculate optimal chunk size based on available memory and CPU cores.
    
    Args:
        file_size: Size of input file in bytes
        num_cores: Number of CPU cores available
    
    Returns:
        Optimal chunk size in number of rows
    """
    # Get available memory (use 70% of available memory)
    available_memory = psutil.virtual_memory().available * 0.7
    
    # Estimate memory per row (assuming 1KB per row as a starting point)
    estimated_memory_per_row = 1024
    
    # Calculate maximum rows that can fit in memory
    max_rows_in_memory = int(available_memory / estimated_memory_per_row)
    
    # Divide by number of cores to get chunk size
    chunk_size = max_rows_in_memory // num_cores
    
    # Ensure chunk size is not too small
    return max(chunk_size, 100000)

def convert_csv_to_parquet(
    input_file='data/prepared_sales_data.csv',
    output_file=None,
    chunk_size=None
):
    """
    Convert a large CSV file to Parquet format using parallel processing.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output Parquet file
        chunk_size: Number of rows to process at once (if None, calculated automatically)
    """
    input_path = Path(input_file)
    
    # Determine output path if not provided
    if output_file is None:
        output_file = input_path.with_suffix('.parquet')
    else:
        output_file = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Create temporary directory for chunks
    temp_dir = output_file.parent / 'temp_parquet_chunks'
    temp_dir.mkdir(exist_ok=True)
    
    logging.info(f"Converting {input_path} to {output_file}")
    
    # Get file size to estimate progress
    file_size = input_path.stat().st_size
    logging.info(f"Input file size: {file_size / (1024*1024):.2f} MB")
    
    # Determine number of CPU cores to use (leave one core free for system)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(file_size, num_cores)
    
    logging.info(f"Using {num_cores} CPU cores with chunk size of {chunk_size:,} rows")
    
    # Read and process in chunks using parallel processing
    total_rows = 0
    schema = None
    temp_files = []
    
    # Create a partial function with fixed arguments
    process_chunk_partial = partial(process_chunk, output_dir=str(temp_dir))
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit chunks for processing
        future_to_chunk = {}
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
            total_rows += len(chunk)
            future = executor.submit(process_chunk_partial, chunk, i)
            future_to_chunk[future] = i
            
            # Get schema from first chunk
            if schema is None:
                schema = pa.Table.from_pandas(chunk).schema
        
        # Collect results as they complete
        for future in future_to_chunk:
            try:
                temp_file = future.result()
                temp_files.append(temp_file)
                logging.info(f"Completed processing chunk {future_to_chunk[future] + 1}")
            except Exception as e:
                logging.error(f"Error processing chunk {future_to_chunk[future] + 1}: {str(e)}")
    
    # Sort temp files by chunk index
    temp_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Combine all chunks into final file
    logging.info("Combining chunks into final Parquet file...")
    
    # Read all chunks and write to final file
    with pq.ParquetWriter(output_file, schema, compression='snappy') as writer:
        for temp_file in temp_files:
            table = pq.read_table(temp_file)
            writer.write_table(table)
            # Remove temporary file
            os.remove(temp_file)
    
    # Remove temporary directory
    temp_dir.rmdir()
    
    logging.info(f"Conversion completed. Total rows: {total_rows:,}")
    
    # Get output file size
    output_size = output_file.stat().st_size
    logging.info(f"Output file size: {output_size / (1024*1024):.2f} MB")
    logging.info(f"Compression ratio: {output_size / file_size:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet format")
    parser.add_argument(
        "--input", 
        default="data/prepared_sales_data.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", 
        default=None,
        help="Path to output Parquet file (default: same as input with .parquet extension)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=None,
        help="Number of rows to process at once (default: calculated automatically)"
    )
    
    args = parser.parse_args()
    
    convert_csv_to_parquet(
        input_file=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size
    ) 