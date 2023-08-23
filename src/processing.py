import matplotlib.pyplot as plt
import pandas as pd
from src import config
import seaborn as sns

# Read the CSV file using pandas
data = pd.read_csv(config.file_path)

'''print first 5 rows in the dataframe'''
print(data.head())

'''print las 5 rows of the dataframe'''
print(data.tail())

''' getting some basic info about the data'''
print(data.info())

'''checking the nr. of missing values'''
print(data.isnull().sum())

'''getting the statical measures of the data'''
print(data.describe())

'''Correlation:
      Positive correlation
      Negative correlation'''

correlation = data.corr(numeric_only=True)
'''constructing a heatmap for correlation'''
plt.figure(1)
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.legend()
# plt.show()

plt.figure(2)
print(data['GLD'].head())

sns.displot(data['GLD'], color='red')
plt.legend(fontsize=1)
plt.show()

'''splitting the features and target'''
X = data.drop(['Date', 'GLD'], axis=1)
Y = data['GLD']
print(X.head())
print(Y.head())
