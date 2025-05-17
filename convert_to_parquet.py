import pandas as pd
import os
from pathlib import Path
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_csv_to_parquet(
    input_file='data/prepared_sales_data.csv',
    output_file=None,
    chunk_size=1000000
):
    """
    Convert a large CSV file to Parquet format using chunking to handle memory constraints.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output Parquet file (default: same as input with .parquet extension)
        chunk_size: Number of rows to process at once
    """
    input_path = Path(input_file)
    
    # Determine output path if not provided
    if output_file is None:
        output_file = input_path.with_suffix('.parquet')
    else:
        output_file = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    logging.info(f"Converting {input_path} to {output_file}")
    
    # Get file size to estimate progress
    file_size = input_path.stat().st_size
    logging.info(f"Input file size: {file_size / (1024*1024):.2f} MB")
    
    # Read and process in chunks
    first_chunk = True
    total_rows = 0
    
    # Use pandas read_csv with chunking
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        total_rows += len(chunk)
        
        # For the first chunk, we write with mode 'w' to create or overwrite the file
        # For subsequent chunks, we append with mode 'a'
        mode = 'w' if first_chunk else 'a'
        
        # Convert date column to proper datetime if it exists
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'])
        
        # Write to Parquet
        chunk.to_parquet(
            output_file,
            engine='pyarrow',
            index=False,
            compression='snappy',
            mode=mode
        )
        
        first_chunk = False
        logging.info(f"Processed chunk {i+1} ({len(chunk)} rows, total: {total_rows} rows)")
    
    logging.info(f"Conversion completed. Total rows: {total_rows}")
    
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
        default=1000000,
        help="Number of rows to process at once"
    )
    
    args = parser.parse_args()
    
    convert_csv_to_parquet(
        input_file=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size
    ) 