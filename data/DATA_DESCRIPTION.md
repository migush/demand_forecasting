# Data Description

## Overview
The dataset contains historical sales data from Rohlík, a Czech online grocery store. The goal is to predict future sales for each product.

## Files

### sales_train.csv
- Contains historical sales data for training
- Columns:
  - `date`: Date of the sale
  - `product_id`: Unique identifier for each product
  - `sales`: Number of units sold

### sales_test.csv
- Contains the test set for making predictions
- Columns:
  - `date`: Date of the sale
  - `product_id`: Unique identifier for each product
  - `sales`: Target variable to predict

### calendar.csv
- Contains calendar information
- Columns:
  - `date`: Date
  - `dayofweek`: Day of the week (0-6, where 0 is Monday)
  - `month`: Month (1-12)
  - `year`: Year
  - `is_holiday`: Binary indicator for holidays
  - `holiday_name`: Name of the holiday (if applicable)

### inventory.csv
- Contains inventory information
- Columns:
  - `date`: Date
  - `product_id`: Unique identifier for each product
  - `inventory`: Number of units in stock

### test_weights.csv
- Contains weights for the test set evaluation
- Used to calculate the Weighted Mean Absolute Error (WMAE)

### solution.csv
- Contains the actual sales values for the test set
- Used for final evaluation

## Evaluation
Submissions are evaluated on Weighted Mean Absolute Error (WMAE) between the predicted sales and the actual sales. Weights for the test evaluation can be found in `test_weights.csv`.

## Important Notes
1. The test set contains future dates for which we need to predict sales
2. The evaluation weights are provided to ensure that errors in predictions are weighted according to their importance
3. The calendar data can be used to capture seasonal patterns and holiday effects
4. The inventory data can be used to understand stock availability and its impact on sales 