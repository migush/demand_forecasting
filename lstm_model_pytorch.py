import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from datetime import datetime
import os
import pandas as pd
from pathlib import Path

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class TimeSeriesDataset(Dataset):
    """Dataset for on-demand loading and processing of time series data for a product"""
    
    def __init__(self, data_path, product_ids=None, lookback=30, train=True, test_size=0.2, feature_columns=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the prepared sales data file (CSV or Parquet)
            product_ids: List of product IDs to include (None for all)
            lookback: Number of time steps to look back
            train: Whether this is for training or testing
            test_size: Proportion of data to use for testing
            feature_columns: List of feature column names (None to auto-detect)
        """
        self.data_path = Path(data_path)
        self.lookback = lookback
        self.train = train
        self.test_size = test_size
        
        # Check if data file is Parquet or CSV
        if self.data_path.suffix.lower() == '.parquet':
            # Sample a small amount to get metadata
            sample_df = pd.read_parquet(self.data_path, rows=1000)
        else:
            # Assume CSV
            sample_df = pd.read_csv(self.data_path, nrows=1000)
        
        # Get feature columns if not provided
        if feature_columns is None:
            self.feature_columns = [col for col in sample_df.columns 
                                   if col not in ['date', 'unique_id', 'sales']]
        else:
            self.feature_columns = feature_columns
            
        self.n_features = len(self.feature_columns)
        
        # Get all unique product IDs from the data if not provided
        if product_ids is None:
            if self.data_path.suffix.lower() == '.parquet':
                unique_ids_df = pd.read_parquet(self.data_path, columns=['unique_id'])
            else:
                unique_ids_df = pd.read_csv(self.data_path, usecols=['unique_id'])
            self.product_ids = unique_ids_df['unique_id'].unique()
        else:
            self.product_ids = product_ids
            
        # Create index of product sequences
        self._create_product_index()
        
        # Compute normalization statistics across the dataset and save them
        self._compute_normalization_stats()
        
    def _create_product_index(self):
        """Create an index of sequences for each product"""
        self.product_sequences = []
        
        for product_id in self.product_ids:
            # Read data for this product
            if self.data_path.suffix.lower() == '.parquet':
                product_df = pd.read_parquet(self.data_path, filters=[('unique_id', '==', product_id)])
            else:
                # For CSV, load the entire dataset and filter (less efficient)
                product_df = pd.read_csv(self.data_path)
                product_df = product_df[product_df['unique_id'] == product_id]
                
            # Sort by date to ensure correct sequence order
            product_df['date'] = pd.to_datetime(product_df['date'])
            product_df = product_df.sort_values('date')
            
            # Calculate number of sequences for this product
            n_sequences = max(0, len(product_df) - self.lookback)
            
            if n_sequences > 0:
                # Split into train/test indices
                split_point = int(n_sequences * (1 - self.test_size))
                
                if self.train:
                    # Training set: Use first part
                    for i in range(split_point):
                        self.product_sequences.append((product_id, i))
                else:
                    # Test set: Use second part
                    for i in range(split_point, n_sequences):
                        self.product_sequences.append((product_id, i))
        
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        # We'll use a sample of products to estimate the normalization stats
        sample_size = min(100, len(self.product_ids))
        sampled_products = np.random.choice(self.product_ids, sample_size, replace=False)
        
        # Collect feature values
        feature_values = []
        
        for product_id in sampled_products:
            # Load data for this product
            if self.data_path.suffix.lower() == '.parquet':
                product_df = pd.read_parquet(self.data_path, filters=[('unique_id', '==', product_id)])
            else:
                product_df = pd.read_csv(self.data_path)
                product_df = product_df[product_df['unique_id'] == product_id]
                
            if not product_df.empty:
                feature_values.append(product_df[self.feature_columns].values)
        
        # Combine all feature values and compute stats
        if feature_values:
            all_features = np.concatenate(feature_values, axis=0)
            self.feature_mean = np.mean(all_features, axis=0)
            self.feature_std = np.std(all_features, axis=0)
            # Prevent division by zero
            self.feature_std[self.feature_std == 0] = 1.0
            
            # Save normalization parameters
            os.makedirs('data/processed', exist_ok=True)
            np.save('data/processed/X_mean.npy', self.feature_mean)
            np.save('data/processed/X_std.npy', self.feature_std)
            
            logging.info(f"Normalization stats computed from {sample_size} products")
        else:
            # If no data, use zeros and ones
            self.feature_mean = np.zeros(self.n_features)
            self.feature_std = np.ones(self.n_features)
            logging.warning("No data found to compute normalization stats")
    
    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.product_sequences)
    
    def __getitem__(self, idx):
        """Get a sequence by index"""
        product_id, seq_start = self.product_sequences[idx]
        
        # Load data for this product
        if self.data_path.suffix.lower() == '.parquet':
            product_df = pd.read_parquet(self.data_path, filters=[('unique_id', '==', product_id)])
        else:
            product_df = pd.read_csv(self.data_path)
            product_df = product_df[product_df['unique_id'] == product_id]
            
        # Sort by date to ensure correct sequence order
        product_df['date'] = pd.to_datetime(product_df['date'])
        product_df = product_df.sort_values('date')
        
        # Extract the sequence
        sequence = product_df[self.feature_columns].iloc[seq_start:seq_start + self.lookback].values
        target = product_df['sales'].iloc[seq_start + self.lookback]
        
        # Apply normalization on-the-fly
        sequence = (sequence - self.feature_mean) / self.feature_std
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
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
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Normalize LSTM output
        out = self.lstm_norm(out[:, -1, :])
        
        # Fully connected layers with normalization
        out = self.fc1(out)
        out = self.fc_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_model(model, train_loader, val_loader, device, epochs=50, learning_rate=0.001):
    """
    Train the LSTM model with improved monitoring.
    
    Args:
        model: PyTorch LSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (CPU/GPU)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        Trained model and training history
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added L2 regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    logging.info(f"Starting training with {epochs} epochs")
    logging.info(f"Initial learning rate: {learning_rate}")
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_mae = 0
        train_predictions = []
        train_targets = []
        batch_count = 0
        
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        logging.info("Training phase:")
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            
            # Check for NaN in predictions
            if torch.isnan(y_pred).any():
                logging.error(f"NaN detected in predictions at batch {batch_count}")
                logging.error(f"Input shape: {X_batch.shape}, Output shape: {y_pred.shape}")
                logging.error(f"Input stats - Mean: {X_batch.mean():.4f}, Std: {X_batch.std():.4f}")
                logging.error(f"Output stats - Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}")
                raise ValueError("NaN detected in model predictions")
            
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping with smaller max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Reduced max_norm
            
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(y_pred - y_batch)).item()
            
            # Convert to numpy and check for NaN
            pred_np = y_pred.cpu().detach().numpy()
            target_np = y_batch.cpu().numpy()
            
            if np.isnan(pred_np).any() or np.isnan(target_np).any():
                logging.error(f"NaN detected in batch {batch_count} after conversion to numpy")
                raise ValueError("NaN detected in predictions or targets")
            
            train_predictions.extend(pred_np)
            train_targets.extend(target_np)
            batch_count += 1
            
            if batch_count % 10 == 0:  # Log every 10 batches
                logging.info(f"Batch {batch_count}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_mae = 0
        val_predictions = []
        val_targets = []
        
        logging.info("\nValidation phase:")
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                
                # Check for NaN in predictions
                if torch.isnan(y_pred).any():
                    logging.error("NaN detected in validation predictions")
                    raise ValueError("NaN detected in validation predictions")
                
                val_loss += criterion(y_pred, y_batch).item()
                val_mae += torch.mean(torch.abs(y_pred - y_batch)).item()
                
                # Convert to numpy and check for NaN
                pred_np = y_pred.cpu().numpy()
                target_np = y_batch.cpu().numpy()
                
                if np.isnan(pred_np).any() or np.isnan(target_np).any():
                    logging.error("NaN detected in validation data after conversion to numpy")
                    raise ValueError("NaN detected in validation predictions or targets")
                
                val_predictions.extend(pred_np)
                val_targets.extend(target_np)
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mae /= len(train_loader)
        val_mae /= len(val_loader)
        
        # Convert lists to numpy arrays and ensure they're 1D
        train_predictions = np.array(train_predictions).flatten()
        train_targets = np.array(train_targets).flatten()
        val_predictions = np.array(val_predictions).flatten()
        val_targets = np.array(val_targets).flatten()
        
        # Calculate additional metrics
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(current_lr)
        
        # Log progress
        logging.info(f"\nEpoch {epoch+1} Summary:")
        logging.info(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
        logging.info(f"Learning Rate: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'outputs/best_model.pth')
            logging.info("Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
    
    return model, history

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Define parameters
    lookback = 30
    hidden_size = 64
    num_layers = 2
    dropout = 0.2
    learning_rate = 0.001
    batch_size = 64
    epochs = 50
    test_size = 0.2
    num_workers = 4  # Number of parallel workers for data loading
    
    # Create datasets with streaming data loading
    data_path = 'data/prepared_sales_data.csv'  # or change to .parquet if converted
    
    logging.info(f"Creating training dataset...")
    train_dataset = TimeSeriesDataset(
        data_path=data_path,
        lookback=lookback,
        train=True,
        test_size=test_size
    )
    
    logging.info(f"Creating testing dataset...")
    test_dataset = TimeSeriesDataset(
        data_path=data_path,
        lookback=lookback,
        train=False,
        test_size=test_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Testing samples: {len(test_dataset)}")
    
    # Create model
    input_size = len(train_dataset.feature_columns)
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    logging.info(f"Created model with {input_size} input features")
    
    # Train the model
    model, history = train_model(
        model, 
        train_loader, 
        test_loader, 
        device, 
        epochs=epochs, 
        learning_rate=learning_rate
    )
    
    # Save the model
    torch.save(model.state_dict(), 'outputs/lstm_model.pth')
    logging.info("Model saved to outputs/lstm_model.pth")
    
    # Evaluate on test set
    test_loss, test_mae = evaluate_model(model, test_loader, device)
    logging.info(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main() 