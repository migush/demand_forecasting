import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting with normalization and regularization.
    This model takes a sequence of feature vectors and predicts a single target value.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of features in input
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization for LSTM output
        self.lstm_norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc_norm = nn.LayerNorm(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with better defaults to prevent exploding gradients"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:  # Only apply Xavier to 2D weights
                    nn.init.xavier_uniform_(param, gain=0.1)  # Reduced gain
                else:  # For 1D weights
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Forward propagate LSTM - no need to initialize h0/c0
        # PyTorch will use zeros by default
        out, _ = self.lstm(x)
        
        # Normalize LSTM output - take only the last time step
        out = self.lstm_norm(out[:, -1, :])
        
        # Fully connected layers with normalization
        out = self.fc1(out)
        out = self.fc_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out 