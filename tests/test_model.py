import pytest
import torch
import numpy as np

from demand_forecasting.modelling.lstm import LSTMModel, WaveNetModel

def test_lstm_model_forward():
    """Test that LSTM model can do a forward pass without errors."""
    # Setup model parameters
    batch_size = 8
    seq_length = 30
    input_features = 10
    hidden_size = 32
    num_layers = 2
    
    # Create model
    model = LSTMModel(
        input_size=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Create random input
    x = torch.randn(batch_size, seq_length, input_features)
    
    # Forward pass
    y = model(x)
    
    # Check output shape
    assert y.shape == (batch_size, 1), f"Expected output shape (batch_size, 1), got {y.shape}"
    
    # Check model has correct structure
    assert model.hidden_size == hidden_size
    assert model.num_layers == num_layers
    
    # Check that model produces non-NaN outputs
    assert not torch.isnan(y).any(), "Model output contains NaN values"

def test_wavenet_model_forward():
    """Test that WaveNet model can do a forward pass without errors."""
    # Setup model parameters
    batch_size = 8
    seq_length = 30
    input_features = 10
    residual_channels = 16
    
    # Create model
    model = WaveNetModel(
        input_channels=input_features,
        residual_channels=residual_channels,
        dilation_channels=residual_channels,
        skip_channels=residual_channels,
        layers=2,
        blocks=1
    )
    
    # Create random input
    x = torch.randn(batch_size, seq_length, input_features)
    
    # Forward pass
    y = model(x)
    
    # Check output shape
    assert y.shape == (batch_size, 1), f"Expected output shape (batch_size, 1), got {y.shape}"
    
    # Check that model produces non-NaN outputs
    assert not torch.isnan(y).any(), "Model output contains NaN values"

def test_model_parameter_count():
    """Test that model parameter count is as expected."""
    # Create small and large models
    small_model = LSTMModel(input_size=10, hidden_size=16, num_layers=1)
    large_model = LSTMModel(input_size=10, hidden_size=64, num_layers=3)
    
    # Count parameters
    small_param_count = sum(p.numel() for p in small_model.parameters())
    large_param_count = sum(p.numel() for p in large_model.parameters())
    
    # Check that large model has more parameters
    assert large_param_count > small_param_count, "Large model should have more parameters than small model"
    
    # Optional: Check specific parameter counts (adjust these values as needed)
    # assert 1000 < small_param_count < 5000, f"Small model has unexpected parameter count: {small_param_count}"
    # assert 20000 < large_param_count < 100000, f"Large model has unexpected parameter count: {large_param_count}"

def test_model_deterministic_output():
    """Test that model produces the same output for the same input with fixed seed."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model
    model = LSTMModel(input_size=5, hidden_size=16, num_layers=1)
    
    # Create input
    x = torch.randn(4, 10, 5)
    
    # Get first output
    y1 = model(x)
    
    # Reset seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Recreate model and input
    model = LSTMModel(input_size=5, hidden_size=16, num_layers=1)
    x = torch.randn(4, 10, 5)
    
    # Get second output
    y2 = model(x)
    
    # Check that outputs are identical
    assert torch.allclose(y1, y2), "Model does not produce deterministic outputs" 