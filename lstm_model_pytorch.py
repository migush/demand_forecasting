import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from datetime import datetime
import os

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

def load_data():
    """Load processed data from numpy files."""
    data_dir = 'data/processed'
    X_train = np.load(f'{data_dir}/X_train.npy')
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    # Normalize input features
    X_mean = X_train.mean(axis=(0, 1))  # Mean across batch and sequence
    X_std = X_train.std(axis=(0, 1))    # Std across batch and sequence
    X_std[X_std == 0] = 1  # Prevent division by zero
    
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    # Save normalization parameters
    np.save(f'{data_dir}/X_mean.npy', X_mean)
    np.save(f'{data_dir}/X_std.npy', X_std)
    
    logging.info(f"Input data normalized. Mean: {X_mean.mean():.4f}, Std: {X_std.mean():.4f}")
    
    return X_train, X_test, y_train, y_test

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
    device = torch.device("mps" if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load processed data
    logging.info("Loading processed data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Split training data into train and validation sets
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create and train model
    input_size = X_train.shape[2]  # Number of features
    model = LSTMModel(input_size=input_size).to(device)
    
    logging.info("Training LSTM model...")
    model, history = train_model(model, train_loader, val_loader, device)
    
    # Load best model for evaluation
    checkpoint = torch.load('outputs/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    logging.info("\nEvaluating model...")
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            test_predictions.extend(y_pred.cpu().numpy())
            test_targets.extend(y_batch.cpu().numpy())
    
    # Calculate final metrics
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    
    logging.info(f"Test MAE: {test_mae:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history['train_mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('outputs/training_history.png')
    plt.close()

if __name__ == "__main__":
    main() 