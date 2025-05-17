import os
import shutil
from pathlib import Path
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cleanup_processed_files(keep_original=True):
    """
    Clean up the materialized processed data files that are no longer needed
    with the streaming approach.
    
    Args:
        keep_original: Whether to keep the prepared_sales_data.csv file
    """
    processed_dir = Path('data/processed')
    
    # Check if directory exists
    if not processed_dir.exists():
        logging.info(f"Directory {processed_dir} does not exist. Nothing to clean up.")
        return
    
    # Files to delete
    files_to_delete = [
        'X_train.npy',
        'X_test.npy',
        'y_train.npy',
        'y_test.npy'
    ]
    
    # We'll keep the normalization parameters as they might be useful for inference
    files_to_keep = [
        'X_mean.npy',
        'X_std.npy'
    ]
    
    # Delete files
    for filename in files_to_delete:
        file_path = processed_dir / filename
        if file_path.exists():
            os.remove(file_path)
            logging.info(f"Deleted: {file_path}")
    
    # Log information about files being kept
    for filename in files_to_keep:
        file_path = processed_dir / filename
        if file_path.exists():
            logging.info(f"Keeping: {file_path}")
            
    # If not keeping the original file
    if not keep_original:
        original_csv = Path('data/prepared_sales_data.csv')
        if original_csv.exists() and original_csv.is_file():
            os.remove(original_csv)
            logging.info(f"Deleted: {original_csv}")
            
    logging.info("Cleanup completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up materialized processed data files")
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the original prepared_sales_data.csv file"
    )
    
    args = parser.parse_args()
    
    cleanup_processed_files(keep_original=args.keep_original) 