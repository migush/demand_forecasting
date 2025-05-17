import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from datetime import datetime
import os
import yaml
import argparse
from typing import Dict, Tuple, Any, Optional

# Local imports from package
from demand_forecasting.modelling.lstm import LSTMModel, WaveNetModel
from demand_forecasting.data.sequence import get_data_loaders

def setup_logging(log_dir: str) -> None:
    """
    Set up logging with file and console outputs.
    
    Args:
        log_dir: Directory to save log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(config: Dict[str, Any], input_size: int) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        input_size: Number of input features
        
    Returns:
        PyTorch model
    """
    model_type = config['model']['type']
    
    if model_type == 'lstm':
        lstm_config = config['model']['lstm']
        return LSTMModel(
            input_size=input_size,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout']
        )
    elif model_type == 'wavenet':
        wavenet_config = config['model']['wavenet']
        return WaveNetModel(
            input_channels=input_size,
            residual_channels=wavenet_config['residual_channels'],
            dilation_channels=wavenet_config['dilation_channels'],
            skip_channels=wavenet_config['skip_channels'],
            layers=wavenet_config['layers'],
            blocks=wavenet_config['blocks'],
            dropout=wavenet_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train the model with improved monitoring and using configuration parameters.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (CPU/GPU)
        config: Configuration dictionary
    
    Returns:
        Trained model and training history
    """
    # Training parameters
    train_config = config['training']
    output_config = config['output']
    
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    patience = train_config['patience']
    clip_grad_norm = train_config['clip_grad_norm']
    use_amp = train_config.get('use_amp', False)
    log_batch_interval = output_config['log_batch_interval']
    
    # Create output directories
    model_dir = output_config['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_config['lr_scheduler']['factor'],
        patience=train_config['lr_scheduler']['patience']
    )
    
    # Initialize gradient scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logging.info(f"Starting training with {epochs} epochs")
    logging.info(f"Initial learning rate: {learning_rate}")
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    logging.info(f"Device: {device}")
    logging.info(f"Model type: {config['model']['type']}")
    
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
            
            # Forward pass with optional AMP
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch.unsqueeze(1))
            else:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
            
            # Check for NaN in predictions
            if torch.isnan(y_pred).any():
                logging.error(f"NaN detected in predictions at batch {batch_count}")
                logging.error(f"Input shape: {X_batch.shape}, Output shape: {y_pred.shape}")
                logging.error(f"Input stats - Mean: {X_batch.mean():.4f}, Std: {X_batch.std():.4f}")
                logging.error(f"Output stats - Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}")
                raise ValueError("NaN detected in model predictions")
            
            # Backward pass and optimize with optional AMP
            optimizer.zero_grad()
            
            if use_amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(y_pred - y_batch.unsqueeze(1))).item()
            
            # Convert to numpy and check for NaN
            pred_np = y_pred.cpu().detach().numpy()
            target_np = y_batch.cpu().numpy()
            
            if np.isnan(pred_np).any() or np.isnan(target_np).any():
                logging.error(f"NaN detected in batch {batch_count} after conversion to numpy")
                raise ValueError("NaN detected in predictions or targets")
            
            train_predictions.extend(pred_np)
            train_targets.extend(target_np)
            batch_count += 1
            
            if batch_count % log_batch_interval == 0:
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
                
                if use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        y_pred = model(X_batch)
                        batch_loss = criterion(y_pred, y_batch.unsqueeze(1))
                else:
                    y_pred = model(X_batch)
                    batch_loss = criterion(y_pred, y_batch.unsqueeze(1))
                
                # Check for NaN in predictions
                if torch.isnan(y_pred).any():
                    logging.error("NaN detected in validation predictions")
                    raise ValueError("NaN detected in validation predictions")
                
                val_loss += batch_loss.item()
                val_mae += torch.mean(torch.abs(y_pred - y_batch.unsqueeze(1))).item()
                
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
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model checkpoint
            checkpoint_path = os.path.join(model_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            
            logging.info(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return model, history

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str
) -> Tuple[float, float]:
    """
    Evaluate the model on test data.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        output_dir: Directory to save evaluation results
        
    Returns:
        Tuple of (MAE, RMSE) metrics
    """
    model.eval()
    test_predictions = []
    test_targets = []
    
    logging.info("\nEvaluating model on test data...")
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            test_predictions.extend(y_pred.cpu().numpy())
            test_targets.extend(y_batch.cpu().numpy())
    
    # Calculate metrics
    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets).flatten()
    
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    
    logging.info(f"Test MAE: {test_mae:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    
    # Save predictions for further analysis
    np.save(os.path.join(output_dir, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(output_dir, 'test_targets.npy'), test_targets)
    
    return test_mae, test_rmse

def plot_history(history: Dict[str, list], output_dir: str) -> None:
    """
    Plot and save training history.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def main(config_path: str) -> None:
    """
    Main function to train and evaluate the model.
    
    Args:
        config_path: Path to configuration YAML
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config['output']['log_dir'])
    
    # Set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else 'cuda' if torch.cuda.is_available() 
        else 'cpu'
    )
    logging.info(f"Using device: {device}")
    
    # Get data loaders
    data_config = config['data']
    train_config = config['training']
    
    train_loader, val_loader, test_loader = get_data_loaders(
        data_path=data_config['processed_path'],
        stats_path=data_config['stats_path'],
        lookback=data_config['lookback'],
        batch_size=train_config['batch_size'],
        test_size=data_config['test_size'],
        val_size=data_config['val_size']
    )
    
    # Create model
    # Get a sample batch to determine input size
    sample_batch, _ = next(iter(train_loader))
    input_size = sample_batch.shape[2]  # (batch_size, seq_len, features)
    
    model = get_model(config, input_size)
    model.to(device)
    
    # Log model architecture
    logging.info(f"Model architecture:\n{model}")
    
    # Train model
    logging.info("Starting model training...")
    model, history = train_model(model, train_loader, val_loader, device, config)
    
    # Load best model for evaluation
    best_model_path = os.path.join(config['output']['model_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate model
    evaluate_model(model, test_loader, device, config['output']['model_dir'])
    
    # Plot training history
    plot_history(history, config['output']['plot_dir'])
    
    logging.info("Training and evaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand forecasting model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/demand_forecasting/config.yaml",
        help="Path to configuration YAML"
    )
    args = parser.parse_args()
    
    main(args.config) 