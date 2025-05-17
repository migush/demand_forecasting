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


class WaveNetModel(nn.Module):
    """
    WaveNet-inspired model for time series forecasting with dilated convolutions.
    This model is an alternative to LSTM that can capture long-range dependencies more efficiently.
    """
    def __init__(self, input_channels, residual_channels=32, dilation_channels=32, 
                 skip_channels=32, output_channels=1, layers=3, blocks=2, dropout=0.2):
        """
        Initialize the WaveNet model.
        
        Args:
            input_channels: Number of features in input
            residual_channels: Channels in residual connections
            dilation_channels: Channels in dilated convolutions
            skip_channels: Channels in skip connections
            output_channels: Number of output channels (1 for single-target regression)
            layers: Number of layers in each block
            blocks: Number of dilated convolution blocks
            dropout: Dropout rate
        """
        super(WaveNetModel, self).__init__()
        
        # Initial 1x1 convolution to adjust input channels
        self.start_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        # Dilated convolution blocks
        self.blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Current dilation factor
        dilation = 1
        
        for b in range(blocks):
            for l in range(layers):
                # Create dilated convolution layer
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(residual_channels, dilation_channels, 2, dilation=dilation, padding=dilation),
                    nn.LayerNorm([dilation_channels, 30]),  # Assume 30 is sequence length, adjust if needed
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(dilation_channels, residual_channels + skip_channels, 1)
                ))
                
                # Dilation grows exponentially (1, 2, 4, 8, ...)
                dilation *= 2
        
        # Final 1x1 convolutions to output
        self.end_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.LayerNorm([skip_channels, 30]),  # Adjust as needed
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(skip_channels, output_channels, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with better defaults"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Transpose to (batch_size, input_channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Initial 1x1 convolution
        x = self.start_conv(x)
        
        # Skip connections sum
        skip_sum = 0
        
        # Process through dilated convolution blocks
        for block in self.blocks:
            # Apply dilated convolution block
            out = block(x)
            
            # Split into residual and skip
            residual, skip = torch.split(out, [x.size(1), skip_sum.size(1) if isinstance(skip_sum, torch.Tensor) else 0], dim=1)
            
            # Residual connection
            x = x + residual
            
            # Skip connection
            if isinstance(skip_sum, int):
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip
        
        # Final 1x1 convolutions
        out = self.end_conv(skip_sum)
        
        # Take the mean across the sequence dimension
        out = torch.mean(out, dim=2)
        
        return out 