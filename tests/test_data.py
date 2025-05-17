import os
import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from demand_forecasting.data.sequence import TimeSeriesDataset

@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
    # Create a temporary directory for test data
    os.makedirs("tests/data", exist_ok=True)
    
    # Generate synthetic data
    dates = pd.date_range(start='2020-01-01', end='2020-03-31')
    products = [1001, 1002, 1003]
    
    data = []
    for product in products:
        for date in dates:
            # Generate some features
            sales = np.random.randint(10, 100)
            price = np.random.uniform(10, 50)
            
            data.append({
                'unique_id': product,
                'date': date,
                'sales': sales,
                'price': price,
                'dayofweek': date.dayofweek,
                'month': date.month,
                'is_weekend': 1 if date.dayofweek >= 5 else 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = "tests/data/test_sales.csv"
    df.to_csv(csv_path, index=False)
    
    # Create stats JSON
    stats = {
        'sales': {'mean': 50.0, 'std': 25.0, 'min': 10.0, 'max': 100.0},
        'price': {'mean': 30.0, 'std': 10.0, 'min': 10.0, 'max': 50.0}
    }
    
    stats_path = "tests/data/test_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    
    return {
        'csv_path': csv_path,
        'stats_path': stats_path,
        'products': products,
        'dates': dates
    }

def test_time_series_dataset(sample_data):
    """Test that TimeSeriesDataset loads and preprocesses data correctly."""
    # Create dataset
    dataset = TimeSeriesDataset(
        data_path=sample_data['csv_path'],
        feature_columns=['price', 'dayofweek', 'month', 'is_weekend'],
        lookback=7,
        stats_path=sample_data['stats_path']
    )
    
    # Check dataset has correct number of sequences
    assert len(dataset) > 0, "Dataset should contain sequences"
    
    # Check sequence shape
    x, y = dataset[0]
    assert x.shape == (7, 4), f"Expected sequence shape (7, 4), got {x.shape}"
    assert y.shape == (), f"Expected target shape (), got {y.shape}"
    
    # Check sequence type
    assert isinstance(x, torch.Tensor), "Sequence should be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "Target should be a torch.Tensor"
    
    # Check there are no NaNs
    assert not torch.isnan(x).any(), "Sequence contains NaN values"
    assert not torch.isnan(y).any(), "Target contains NaN values"

def test_dataset_with_product_filtering(sample_data):
    """Test that TimeSeriesDataset correctly filters by product IDs."""
    # Create dataset with only first product
    single_product = [sample_data['products'][0]]
    
    dataset = TimeSeriesDataset(
        data_path=sample_data['csv_path'],
        feature_columns=['price', 'dayofweek', 'month', 'is_weekend'],
        product_ids=single_product,
        lookback=7
    )
    
    # Check if all sequences are from the specified product
    data = pd.read_csv(sample_data['csv_path'])
    data['date'] = pd.to_datetime(data['date'])
    
    product_data = data[data['unique_id'].isin(single_product)]
    expected_sequences = max(0, len(product_data) - 7)
    
    # Account for grouping by product
    if expected_sequences > 0:
        assert len(dataset) > 0, "Dataset should contain sequences"
        assert len(dataset) <= expected_sequences, f"Expected at most {expected_sequences} sequences, got {len(dataset)}"

# Clean up after tests
def teardown_module(module):
    """Clean up test data after tests run."""
    import shutil
    if os.path.exists("tests/data"):
        shutil.rmtree("tests/data") 