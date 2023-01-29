"""
Python 3.10 linear regression program with pre-processing of kaggle competition data
File name: Linear_Regression.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2023-01-22
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Read in the training and test sets
train_df = pd.read_csv('../data/train.csv')
# print(len(train_df))
test_df = pd.read_csv('../data/test.csv')
# print(len(test_df))
result_df = pd.read_csv('../data/submission-Hs.csv')   # 100% result
# print(len(result_df))

###################################### Preprocess the data #############################################################
# Identify most relevant features
# You can use techniques like feature importance or correlation analysis to help you identify the most important features
relevant_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
train_df[relevant_features] = imputer.fit_transform(train_df[relevant_features])
test_df[relevant_features] = imputer.transform(test_df[relevant_features])

# Transform skewed or non-normal features
# Instead of normalizing all of the numeric features, you could try using techniques like log transformation or
# Box-Cox transformation to make the distribution of a feature more normal
scaler = StandardScaler()
train_df[relevant_features] = scaler.fit_transform(train_df[relevant_features])
test_df[relevant_features] = scaler.transform(test_df[relevant_features])

# Split the data into features (X) and labels (y)
X_train = train_df[relevant_features]
y_train = train_df['SalePrice']
X_test = test_df[relevant_features]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=33)


############################################## Train the model #########################################################
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Evaluate the logistic regression classifier
scores = cross_val_score(regr, X_val, y_val, cv=5)
print("Accuracy of linear regression classifier: ", scores.mean())

# Make predictions on the test set
y_pred = regr.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred})
output['SalePrice']= output['SalePrice'].astype(int)
output.to_csv('00.submission-linr-19633.84499.csv', index=False)

# print(output)
print('Correlation with ideal submission:', output['SalePrice'].corr(result_df['SalePrice']))
result_df['percent'] = result_df['SalePrice'] == output['SalePrice']
print('percent: \n', (result_df['percent'].value_counts('True')))
print('Real score on submission: 19633.84499')

'''
The predictions of this code did not give any correct
0% - True
'''


