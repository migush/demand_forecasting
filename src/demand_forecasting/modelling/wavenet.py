import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Save configuration
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        
        # Initial 1x1 convolution to adjust input channels
        self.start_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        # Dilated convolution layers
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        # Current dilation factor
        dilation = 1
        
        for b in range(blocks):
            for l in range(layers):
                # Dilated convolution
                self.dilated_convs.append(
                    nn.Conv1d(residual_channels, dilation_channels, 
                             kernel_size=2, dilation=dilation,
                             padding='same')  # Use 'same' padding to maintain sequence length
                )
                
                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Conv1d(dilation_channels, residual_channels, 1)
                )
                
                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv1d(dilation_channels, skip_channels, 1)
                )
                
                # Dilation grows exponentially (1, 2, 4, 8, ...)
                dilation *= 2
        
        # Final 1x1 convolutions for output
        self.final_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.final_conv2 = nn.Conv1d(skip_channels, output_channels, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.relu = nn.ReLU()
        
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
        
        # Skip connections accumulator
        skip_connections = []
        
        # Process through dilated convolution layers
        for i in range(len(self.dilated_convs)):
            # Get current input
            residual = x
            
            # Dilated convolution
            x = self.relu(self.dilated_convs[i](x))
            
            # Residual and skip connections
            res = self.residual_convs[i](x)
            skip = self.skip_convs[i](x)
            
            # Add residual connection
            x = residual + res
            
            # Collect skip connection
            skip_connections.append(skip)
        
        # Sum all skip connections
        x = torch.stack(skip_connections).sum(dim=0)
        
        # Final 1x1 convolutions
        x = self.relu(x)
        x = self.final_conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final_conv2(x)
        
        # Take the mean across the sequence dimension to get final prediction
        x = torch.mean(x, dim=2)
        
        return x 