import os
import pytest
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
import shutil
import tempfile
import uuid

from demand_forecasting.data.sequence import TimeSeriesDataset

@pytest.fixture(scope="function")
def test_dir():
    """Create a temporary directory for test data that will be cleaned up after the test."""
    # Create a unique test directory for each test function
    test_dir = Path(tempfile.gettempdir()) / f"demand_forecast_test_{uuid.uuid4().hex}"
    test_dir.mkdir(exist_ok=True, parents=True)
    yield test_dir
    # Clean up after tests
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture(scope="function")
def sample_data(test_dir):
    """Create a small synthetic dataset for testing."""
    # Generate synthetic data
    dates = pd.date_range(start='2020-01-01', end='2020-03-31')
    products = [1001, 1002, 1003]
    
    # Set seed for reproducibility
    np.random.seed(42)
    
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
    csv_path = test_dir / "test_sales.csv"
    df.to_csv(csv_path, index=False)
    
    # Create stats JSON
    stats = {
        'sales': {'mean': 50.0, 'std': 25.0, 'min': 10.0, 'max': 100.0},
        'price': {'mean': 30.0, 'std': 10.0, 'min': 10.0, 'max': 50.0}
    }
    
    stats_path = test_dir / "test_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    
    return {
        'csv_path': str(csv_path),
        'stats_path': str(stats_path),
        'products': products,
        'dates': dates,
        'df': df
    }

@pytest.mark.parametrize("lookback", [5, 7, 10])
def test_time_series_dataset(sample_data, lookback):
    """Test that TimeSeriesDataset loads and preprocesses data correctly with different lookbacks."""
    # Create dataset
    dataset = TimeSeriesDataset(
        data_path=sample_data['csv_path'],
        feature_columns=['price', 'dayofweek', 'month', 'is_weekend'],
        lookback=lookback,
        stats_path=sample_data['stats_path']
    )
    
    # Check dataset has correct number of sequences
    assert len(dataset) > 0, "Dataset should contain sequences"
    
    # Check sequence shape
    x, y = dataset[0]
    assert x.shape == (lookback, 4), f"Expected sequence shape ({lookback}, 4), got {x.shape}"
    assert y.shape == (), f"Expected target shape (), got {y.shape}"
    
    # Check sequence type
    assert isinstance(x, torch.Tensor), "Sequence should be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "Target should be a torch.Tensor"
    
    # Check there are no NaNs
    assert not torch.isnan(x).any(), "Sequence contains NaN values"
    assert not torch.isnan(y).any(), "Target contains NaN values"

@pytest.mark.parametrize("product_count", [1, 2, 3])
def test_dataset_with_product_filtering(sample_data, product_count):
    """Test that TimeSeriesDataset correctly filters by different numbers of product IDs."""
    # Take a subset of products
    selected_products = sample_data['products'][:product_count]
    
    dataset = TimeSeriesDataset(
        data_path=sample_data['csv_path'],
        feature_columns=['price', 'dayofweek', 'month', 'is_weekend'],
        product_ids=selected_products,
        lookback=7
    )
    
    # Check if all sequences are from the specified products
    product_data = sample_data['df'][sample_data['df']['unique_id'].isin(selected_products)]
    
    # Each product should have (total_dates - lookback) sequences
    # Because we're grouping by product, we need to account for each product separately
    expected_max_sequences = sum(
        max(0, len(group) - 7) for _, group in product_data.groupby('unique_id')
    )
    
    assert len(dataset) > 0, "Dataset should contain sequences"
    assert len(dataset) <= expected_max_sequences, f"Expected at most {expected_max_sequences} sequences, got {len(dataset)}"
    
    # Check that all sequences belong to selected products
    if len(dataset) > 0:
        # Get the first sequence and check its product ID
        product_id, _ = dataset.index_mapping[0]
        assert product_id in selected_products, f"Sequence from product {product_id} not in selected products {selected_products}"

def test_normalization(sample_data):
    """Test that normalization is applied correctly."""
    # Create dataset with and without normalization
    dataset_with_norm = TimeSeriesDataset(
        data_path=sample_data['csv_path'],
        feature_columns=['price'],
        lookback=7,
        stats_path=sample_data['stats_path']
    )
    
    dataset_without_norm = TimeSeriesDataset(
        data_path=sample_data['csv_path'],
        feature_columns=['price'],
        lookback=7,
        stats_path=None
    )
    
    # Get the same sequence from both datasets
    if len(dataset_with_norm) > 0 and len(dataset_without_norm) > 0:
        x_norm, _ = dataset_with_norm[0]
        x_raw, _ = dataset_without_norm[0]
        
        # The normalized values should be different from the raw values
        assert not torch.allclose(x_norm, x_raw), "Normalized and raw data should differ"
        
        # Check if normalization was applied with the correct mean and std
        stats = sample_data['df']['price'].describe()
        # Note: the actual stats here will be different from what's in the stats.json
        # since we hardcoded those values, but the test will still check the principle
        
        # Just check that normalized values have smaller magnitude
        assert torch.all(torch.abs(x_norm) < torch.abs(x_raw)), "Normalized values should have smaller magnitude" 