import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from src import config
import pandas as pd


data = pd.read_csv(config.file_path)
X = data.drop(['Date', 'GLD'], axis=1)
Y = data['GLD']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Creating and fitting the regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, Y_train)

'''prediction on test data'''
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

'''R squared error'''
error_square = metrics.r2_score(Y_test, test_data_prediction)
print("R square error : ", error_square)


