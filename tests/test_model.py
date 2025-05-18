import pytest
import torch
import numpy as np

from demand_forecasting.modelling.lstm import LSTMModel
from demand_forecasting.modelling.wavenet import WaveNetModel

@pytest.fixture
def random_sequence():
    """Create a random input sequence for testing."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 8
    seq_length = 30
    input_features = 10
    
    return torch.randn(batch_size, seq_length, input_features)

@pytest.fixture
def lstm_model_params():
    """Return standard LSTM model parameters for testing."""
    return {
        'input_size': 10,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.2
    }

@pytest.fixture
def wavenet_model_params():
    """Return standard WaveNet model parameters for testing."""
    return {
        'input_channels': 10,
        'residual_channels': 16,
        'dilation_channels': 16,
        'skip_channels': 16,
        'layers': 1,  # Reduce complexity for testing
        'blocks': 1,
        'dropout': 0.2
    }

@pytest.fixture
def lstm_model(lstm_model_params):
    """Create an LSTM model for testing."""
    return LSTMModel(**lstm_model_params)

@pytest.fixture
def wavenet_model(wavenet_model_params):
    """Create a WaveNet model for testing."""
    return WaveNetModel(**wavenet_model_params)

def test_lstm_model_forward(lstm_model, random_sequence):
    """Test that LSTM model can do a forward pass without errors."""
    # Forward pass
    y = lstm_model(random_sequence)
    
    # Check output shape
    batch_size = random_sequence.shape[0]
    assert y.shape == (batch_size, 1), f"Expected output shape (batch_size, 1), got {y.shape}"
    
    # Check model has correct structure
    assert lstm_model.hidden_size == 32, f"Expected hidden_size=32, got {lstm_model.hidden_size}"
    assert lstm_model.num_layers == 2, f"Expected num_layers=2, got {lstm_model.num_layers}"
    
    # Check that model produces non-NaN outputs
    assert not torch.isnan(y).any(), "Model output contains NaN values"

def test_wavenet_model_forward(wavenet_model, random_sequence):
    """Test that WaveNet model can do a forward pass without errors."""
    # Forward pass
    y = wavenet_model(random_sequence)
    
    # Check output shape
    batch_size = random_sequence.shape[0]
    assert y.shape == (batch_size, 1), f"Expected output shape (batch_size, 1), got {y.shape}"
    
    # Check that model produces non-NaN outputs
    assert not torch.isnan(y).any(), "Model output contains NaN values"

@pytest.mark.parametrize("model_config", [
    {'input_size': 10, 'hidden_size': 16, 'num_layers': 1},
    {'input_size': 10, 'hidden_size': 64, 'num_layers': 3}
])
def test_lstm_parameter_count(model_config):
    """Test that model parameter count increases with size as expected."""
    small_model = LSTMModel(**model_config)
    
    # Count parameters
    param_count = sum(p.numel() for p in small_model.parameters())
    
    # Parameter count should be related to model size
    expected_min_params = model_config['input_size'] * model_config['hidden_size'] + model_config['hidden_size'] * model_config['hidden_size'] * model_config['num_layers']
    
    assert param_count > expected_min_params, f"Model has too few parameters. Expected more than {expected_min_params}, got {param_count}"

def test_model_parameter_count_comparison():
    """Test that larger models have more parameters than smaller ones."""
    small_model = LSTMModel(input_size=10, hidden_size=16, num_layers=1)
    large_model = LSTMModel(input_size=10, hidden_size=64, num_layers=3)
    
    # Count parameters
    small_param_count = sum(p.numel() for p in small_model.parameters())
    large_param_count = sum(p.numel() for p in large_model.parameters())
    
    # Check that large model has more parameters
    assert large_param_count > small_param_count, (
        f"Large model should have more parameters than small model. "
        f"Large: {large_param_count}, Small: {small_param_count}"
    )

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
    assert torch.allclose(y1, y2), (
        "Model does not produce deterministic outputs. "
        f"First output: {y1}, Second output: {y2}, Max difference: {torch.max(torch.abs(y1 - y2))}"
    ) 