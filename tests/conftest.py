import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility across all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    
@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon
    else:
        return torch.device("cpu")

@pytest.fixture(scope="session")
def test_output_dir():
    """Create a directory for test outputs that will persist after tests."""
    output_dir = Path("tests/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir 