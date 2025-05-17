# Demand Forecasting with LSTM

This project implements a demand forecasting solution using Long Short-Term Memory (LSTM) neural networks. The goal is to predict future sales based on historical data, helping businesses optimize their inventory management and supply chain operations.

## Dataset

The project uses the Rohlík Sales Forecasting Challenge dataset from Kaggle. The dataset contains historical sales data from Rohlík, a Czech online grocery store. You can find the dataset and competition details at:
[Kaggle Competition Link](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/overview)

## Project Structure

```
demand_forecasting/
└── data/               # Data directory for storing the dataset
    └── test_weights.csv  # Weights for test evaluation
```

## Evaluation

The model's performance is evaluated using Weighted Mean Absolute Error (WMAE) between the predicted sales and the actual sales. The weights for the test evaluation are provided in `data/test_weights.csv`.

The WMAE is calculated as:
```
WMAE = Σ(weight_i * |actual_i - predicted_i|) / Σ(weight_i)
```

This metric ensures that errors in predictions are weighted according to their importance, providing a more meaningful evaluation of the model's performance.

## Getting Started

1. Clone the repository
2. Download the dataset from Kaggle and place it in the `data/` directory

## Technical Approach

The project will use an LSTM (Long Short-Term Memory) neural network implemented in PyTorch, which is particularly well-suited for time series forecasting tasks. LSTM networks can:
- Capture long-term dependencies in time series data
- Handle complex patterns and seasonality
- Learn from sequential data while maintaining memory of important historical information

## Model Architecture

The LSTM model is designed to:
- Process time series data with multiple features
- Capture seasonal patterns and trends
- Generate accurate sales forecasts for future time periods

## Future Improvements

- Implement additional feature engineering
- Experiment with different model architectures
- Add support for multiple product categories
- Implement ensemble methods
- Add real-time prediction capabilities

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
