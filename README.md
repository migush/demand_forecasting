# Demand Forecasting with LSTM

This project implements a demand forecasting solution using Long Short-Term Memory (LSTM) neural networks. The goal is to predict future sales based on historical data, helping businesses optimize their inventory management and supply chain operations.

## Dataset

The project uses the Rohlík Sales Forecasting Challenge dataset from Kaggle. The dataset contains historical sales data from Rohlík, a Czech online grocery store. You can find the dataset and competition details at:
[Kaggle Competition Link](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/overview)

## Data Pipeline

This project uses a streaming data pipeline that loads and processes data on-demand:

1. **Data Preparation**: The raw data is transformed into a prepared dataset with feature engineering (saved as CSV)
2. **Streaming Dataset**: During training, data is loaded for each product ID only when needed
3. **On-the-fly Processing**: Window slicing and normalization happen in real-time
4. **Parallel Processing**: Multiple worker processes handle data loading while GPU trains

### Benefits of Streaming Pipeline:
- Reduced memory usage (only loads necessary data)
- Improved training speed (parallel processing)
- Better scalability for large datasets

## Project Structure

```
demand_forecasting/
├── src/                       # Source code directory
│   └── demand_forecasting/    # Python package
│       ├── data/              # Data processing modules
│       │   ├── etl.py         # Data extraction and transformation
│       │   └── sequence.py    # Sequence generation for LSTM
│       ├── modelling/         # Model definitions
│       │   └── lstm.py        # LSTM and WaveNet model architecture
│       ├── train.py           # Training loop and metrics
│       └── config.yaml        # Configuration parameters
├── tests/                     # Unit tests
├── data/                      # Data directory (gitignored except for metadata)
│   └── processed/             # Processed data files
├── logs/                      # Training logs
├── outputs/                   # Model checkpoints and visualizations
├── prepare_data.py            # Script for data preparation
├── convert_to_parquet.py      # Script to convert CSV to Parquet
└── lstm_model_pytorch.py      # LSTM model implementation with streaming dataset
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/migush/demand-forecasting.git
cd demand-forecasting
```

2. Create a virtual environment and install dependencies using `uv`:
```bash
# Install uv if you don't have it already
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies in one command
uv venv .venv
uv pip install -r requirements.txt

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
uv pip install -e .
```

## Usage

### Data Preparation

Before training, you need to prepare the data:

```bash
# Run the ETL module to prepare the data
python -m src.demand_forecasting.data.etl
```

This will:
1. Process the raw sales data and calendar data
2. Create necessary features
3. Save the prepared data to `data/processed/prepared_sales_data.csv`
4. Generate feature statistics in `data/processed/feature_stats.json`

### Running the Training Pipeline

After data preparation, you can train the model:

```bash
# Run with default configuration
python -m src.demand_forecasting.train

# Or specify a custom config file
python -m src.demand_forecasting.train --config path/to/your/config.yaml
```

The default configuration file is located at `src/demand_forecasting/config.yaml`.

> **Note**: If you encounter NaN errors during training, you may need to adjust the learning rate or other hyperparameters in the config file. Try reducing the learning rate to `0.0001` or adding gradient clipping by setting a lower value for `clip_grad_norm` in the config.

### Testing

Run the test suite using pytest:

```bash
# Install the package in development mode
uv pip install -e .

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run tests in parallel (faster)
pytest -n auto

# Run tests with verbose output
pytest -xvs
```

The test suite includes:
- Model tests to verify model architecture and forward passes
- Data processing tests to check dataset functionality
- Integration tests to ensure components work together

## Model Architecture

The project includes two neural network architectures:

1. **LSTM Model**: Captures long-term dependencies in time series data
2. **WaveNet Model**: Uses dilated convolutions for efficient long-range modeling

You can select the model type in the configuration file.

## Evaluation

The model's performance is evaluated using Weighted Mean Absolute Error (WMAE) between the predicted sales and the actual sales. The weights for the test evaluation are provided in `data/test_weights.csv`.

The WMAE is calculated as:
```
WMAE = Σ(weight_i * |actual_i - predicted_i|) / Σ(weight_i)
```

## Configuration

Example configuration (from config.yaml):

```yaml
# Data parameters
data:
  lookback: 30        # Sequence length for time series
  test_size: 0.2      # Proportion for test set

# Model parameters
model:
  type: "lstm"        # Model type (lstm or wavenet)
  lstm:
    hidden_size: 64
    num_layers: 2
    dropout: 0.2

# Training parameters
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  num_workers: 4      # Number of parallel data loading workers
```

## Future Improvements

- Dynamic lookback window size based on product characteristics
- Attention mechanisms for handling long sequences
- Transformer-based architectures
- Multi-task learning for related products
- Bayesian approaches for uncertainty estimation

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
