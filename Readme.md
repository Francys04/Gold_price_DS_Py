# Gold Price Prediction using Random Forest Regressor

This Python script analyzes historical gold price data and uses a Random Forest Regressor model to make price predictions.

## Getting Started

1. Clone this repository.
2. Install the required libraries using `pip install pandas numpy matplotlib sklearn`.
3. Run the script by executing `python Final_Result`.

## Code Overview

- `processing.py`: Main script for data analysis and machine learning.
- `config.py`: Configuration file with file paths and constants.
- `models/training_test.py`: Module containing the model training and testing code.
- `data/gold_price_data.csv`: Sample dataset with historical gold price data.

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Load the dataset, explore its basic information, and visualize data distributions.
2. Split the dataset into features (X) and the target variable (Y).
3. Train a Random Forest Regressor model on the training data.
4. Make predictions on the test data and evaluate the model's performance using R-squared error.
5. Visualize the actual vs. predicted gold prices.
