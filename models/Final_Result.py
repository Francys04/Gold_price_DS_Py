import matplotlib.pyplot as plt
from models.training_test import train_test_split, test_data_prediction, X, Y


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
'''Compare the actual values and predicted values in a plot'''
Y_test = list(Y_test)
plt.plot(Y_test, color='red', label='Actual label')
plt.plot(test_data_prediction, color='orange', label='Predicted value')
plt.title('Actual price vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
