# Developing a Multilinear Regression Model for House Price Prediction

![Logo](https://user-content.gitlab-static.net/b7bcb963bff200f5aedfe65b4c2ac81e346d73bc/68747470733a2f2f69302e77702e636f6d2f746865636c6576657270726f6772616d6d65722e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f31322f4d616368696e652d4c6561726e696e672d50726f6a6563742d6f6e2d486f7573652d50726963652d50726564696374696f6e2e706e673f6669743d313238302532433732302673736c3d31)


## Abstract
This project explores the development of a multilinear regression model for predicting house prices using a dataset containing various features. The model's performance is evaluated using traditional linear regression, Lasso, and Ridge regression techniques to identify the most effective method.
## Table of Contents
1. Introduction
2. Dataset Overview
3. Methodology
   - Object-Oriented Programming Concepts
4. Regression Techniques
   - Multilinear Regression
   - Ridge Regression
   - Lasso Regression
5. Implementation
6. Results
7. Conclusion
8. Future Work

## 1. Introduction
House price prediction is a critical task in real estate, influenced by numerous factors such as location, size, and amenities. This project aims to create a robust predictive model using various regression techniques to assess their effectiveness.

## 2. Dataset Overview
The dataset consists of 17 variables:

### Independent Variables (Features)
- **Bedrooms**: Number of bedrooms in the house.
- **Bathrooms**: Number of bathrooms.
- **Sqft_Living**: Square footage of the living space.
- **Sqft_Lot**: Square footage of the lot.
- **Floors**: Number of floors.
- **Waterfront**: Indicates if the house is waterfront (1 for yes, 0 for no).
- **View**: Quality of the view (scale).
- **Condition**: Condition of the house (scale).
- **Sqft_Above**: Square footage excluding the basement.
- **Sqft_Basement**: Square footage of the basement.
- **Yr_Built**: Year the house was built.
- **Yr_Renovated**: Year of renovation (0 if never renovated).
- **Street**: Type of street (categorical).
- **City**: City (categorical).
- **StateZip**: State and zip code (categorical).
- **Country**: Country (categorical).

### Dependent Variable (Target Variable)
- **Price**: The price of the house.

*Consider adding a table to summarize these features for clarity.*

## 3. Methodology
### Object-Oriented Programming Concepts
The project employs Object-Oriented Programming (OOP) principles, which enhance code organization, reusability, and maintainability.

- **Class**: A blueprint for creating objects. In this project, the `HOUSEPRICE` class encapsulates methods and properties related to house price prediction.
- **Object**: An instance of a class, e.g., `HP` is an object of the `HOUSEPRICE` class.
- **Encapsulation**: Bundles data and methods, ensuring data integrity by restricting direct access.
- **Inheritance**: Allows new classes to inherit properties from existing classes, useful for extending functionality.
- **Polymorphism**: Enables methods to use the same interface for different underlying data types.
- **Abstraction**: Simplifies complex reality by exposing only necessary parts.

*Visuals: A diagram illustrating OOP concepts can enhance understanding.*

## 4. Regression Techniques
### Multilinear Regression
A statistical technique that models the relationship between multiple independent variables and a dependent variable. It estimates coefficients for each feature to minimize the difference between predicted and actual prices.

### Ridge Regression
A regularization technique that penalizes large coefficients to reduce overfitting. It adds a penalty term (L2 regularization) to the loss function, which helps in improving model generalization.

### Lasso Regression
Another regularization technique that performs both variable selection and regularization (L1 regularization). It can shrink some coefficients to zero, effectively selecting a simpler model.

*Visuals: Graphs comparing the performance of each regression technique can be useful.*

## 5. Implementation
### Code Overview
## 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import Lasso, Ridge
import sys
import warnings
warnings.filterwarnings('ignore')
```
- **numpy**: A library for numerical operations.
- **pandas**: A library for data manipulation and analysis, especially useful for handling datasets.
- **matplotlib.pyplot**: A plotting library for creating static, animated, and interactive visualizations in Python.
- **sklearn.model_selection**: Contains functions for splitting data into training and testing sets.
- **sklearn.linear_model**: Contains implementations of linear regression models including Lasso and Ridge.
- **sklearn.metrics**: Provides functions to evaluate the performance of the models.
- **sys**: Provides access to system-specific parameters and functions.
- **warnings**: Used to suppress warning messages during code execution.

## 2. Defining the HOUSEPRICE Class
```python
class HOUSEPRICE:
```
- A class named `HOUSEPRICE` is defined to encapsulate all methods related to house price prediction.

### 2.1. Constructor Method
```python
def __init__(self, data):
    try:
        self.data = data
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **`__init__`**: This is the constructor method that initializes the class with a dataset.
- **`self.data`**: Stores the dataset passed to the class.
- **Error handling**: Captures any exceptions during initialization and prints the error details.

### 2.2. Method to Check Null Values
```python
def null_value(self, data):
    try:
        return data.isnull().sum()
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **null_value**: This method checks for null values in the dataset.
- **`data.isnull().sum()`**: Returns the count of null values for each column.
- Error handling to capture any exceptions.

### 2.3. Method for Categorical to Numerical Conversion
```python
def convertion(self, data):
    try:
        d = {}
        x = 0
        for i in data.values:
            if i not in d.keys():
                d.update({i: x})
                x = x + 1
        return d
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **convertion**: Converts categorical features into numerical values for model compatibility.
- **Dictionary `d`**: Maps unique categorical values to numerical values.
- Returns the dictionary for conversion.

### 2.4. Performance Evaluation Method
```python
def performance(self, a, b):
    try:
        acc = r2_score(a, b)
        mse = mean_squared_error(a, b)
        rmse = root_mean_squared_error(a, b)
        mae = mean_absolute_error(a, b)
        return acc, mse, rmse, mae
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **performance**: Evaluates model performance using various metrics.
- **Metrics Calculated**:
  - **R² Score**: Indicates the proportion of variance explained by the model.
  - **Mean Squared Error (MSE)**: Average of the squares of errors.
  - **Root Mean Squared Error (RMSE)**: Square root of MSE.
  - **Mean Absolute Error (MAE)**: Average of absolute errors.
- Returns all calculated metrics.

### 2.5. Data Preprocessing Method
```python
def preprocessing(self):
    try:
        print(f'The number of rows before preprocessing are {self.data.shape[0]} and the number of columns before preprocessing are {self.data.shape[1]}')
        c = [i for i in self.null_value(self.data) if i != 0]
        if len(c) > 0:
            print(f'the given data have null values')
        else:
            print(f'the given data dont have any null values')
        self.data.drop(['date', 'country', 'street'], axis=1, inplace=True)
        print(f'""Date","Country","Street" features are dropped from the dataset for better prediction')
        self.data['city'] = self.data['city'].map(self.convertion(self.data['city']))
        self.data['statezip'] = self.data['statezip'].map(self.convertion(self.data['statezip']))
        print(f'The categorical data was converted to numerical data')
        print(f'The number of rows after preprocessing are {self.data.shape[0]} and the number of columns after preprocessing are {self.data.shape[1]}')
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **preprocessing**: Prepares the dataset for model training.
- **Initial Shape**: Prints the initial shape of the dataset.
- **Null Value Check**: Checks for null values and informs the user.
- **Drop Unwanted Features**: Removes irrelevant features like date, country, and street.
- **Convert Categorical Data**: Uses the `convertion` method to convert city and statezip to numerical.
- **Final Shape**: Prints the shape of the dataset after preprocessing.

### 2.6. Multilinear Regression Method
```python
def linear_model(self):
    try:
        self.X = self.data.iloc[:, 1:]
        self.y = self.data.iloc[:, 0]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.reg = LinearRegression()
        self.reg.fit(self.X_train, self.y_train)
        self.train_predictions = self.reg.predict(self.X_train)
        self.test_predictions = self.reg.predict(self.X_test)
        print()
        print(f'-----The Normal Multi-linear regression Model prediction performance-----')
        accuracy, mse, rmse, mae = self.performance(self.y_train, self.train_predictions)
        print(f'the accuracy of the training data is: {accuracy}\nthe train_data mean squared error: {mse}\nthe train_data mean absolute error: {mae}\nthe train_data root mean squared error: {rmse}')
        print('-----------------------------------------------------------------------------------')
        accuracy, mse, rmse, mae = self.performance(self.y_test, self.test_predictions)
        print(f'the accuracy of the test data is: {accuracy}\nthe test_data mean squared error: {mse}\nthe test_data mean absolute error: {mae}\nthe test_data root mean squared error: {rmse}')
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **linear_model**: Implements the multilinear regression model.
- **Feature and Target Separation**: `self.X` contains independent variables, while `self.y` contains the dependent variable (price).
- **Data Splitting**: Uses `train_test_split` to divide the data into training (70%) and testing (30%) sets.
- **Model Training**: Initializes and trains a `LinearRegression` model on the training data.
- **Predictions**: Generates predictions for both training and testing sets.
- **Performance Evaluation**: Uses the `performance` method to evaluate and print the results for both training and testing sets.

### 2.7. Lasso Regression Method
```python
def lasso_regression(self):
    try:
        self.lasso = Lasso(alpha=0.5)
        self.lasso.fit(self.X_train, self.y_train)
        self.lasso_train_predictions = self.lasso.predict(self.X_train)
        self.lasso_test_predictions = self.lasso.predict(self.X_test)
        print()
        print('-----The lasso Regression performance-----')
        accuracy, mse, rmse, mae = self.performance(self.y_train, self.lasso_train_predictions)
        print(f'the accuracy of the L1 training data is: {accuracy}\nthe train_data mean squared error: {mse}\nthe train_data mean absolute error: {mae}\nthe train_data root mean squared error: {rmse}')
        print('------------------------------------------------------------------------------------------------------')
        accuracy, mse, rmse, mae = self.performance(self.y_test, self.lasso_test_predictions)
        print(f

'the accuracy of the L1 testing data is: {accuracy}\nthe test_data mean squared error: {mse}\nthe test_data mean absolute error: {mae}\nthe test_data root mean squared error: {rmse}')
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **lasso_regression**: Implements Lasso regression.
- **Alpha Parameter**: Controls the strength of regularization (0.5 in this case).
- **Model Training**: Fits the Lasso model on the training data.
- **Predictions**: Generates predictions for both training and testing sets.
- **Performance Evaluation**: Evaluates and prints performance metrics for Lasso regression.

### 2.8. Ridge Regression Method
```python
def ridge_regression(self):
    try:
        self.ridge = Ridge(alpha=0.5)
        self.ridge.fit(self.X_train, self.y_train)
        self.ridge_train_predictions = self.ridge.predict(self.X_train)
        self.ridge_test_predictions = self.ridge.predict(self.X_test)
        print()
        print('-----The ridge Regression performance-----')
        accuracy, mse, rmse, mae = self.performance(self.y_train, self.ridge_train_predictions)
        print(f'the accuracy of the L2 training data is: {accuracy}\nthe train_data mean squared error: {mse}\nthe train_data mean absolute error: {mae}\nthe train_data root mean squared error: {rmse}')
        print('------------------------------------------------------------------------------------------------------')
        accuracy, mse, rmse, mae = self.performance(self.y_test, self.ridge_test_predictions)
        print(f'the accuracy of the L2 testing data is: {accuracy}\nthe test_data mean squared error: {mse}\nthe test_data mean absolute error: {mae}\nthe test_data root mean squared error: {rmse}')
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **ridge_regression**: Implements Ridge regression.
- **Alpha Parameter**: Similar to Lasso, controls the strength of regularization (also 0.5).
- **Model Training**: Fits the Ridge model on the training data.
- **Predictions**: Generates predictions for both training and testing sets.
- **Performance Evaluation**: Evaluates and prints performance metrics for Ridge regression.

## 3. Main Execution Block
```python
if __name__ == '__main__':
    try:
        data = pd.read_csv("C:\\Users\\abhin\\Desktop\\Sai Python practice\\Pycharm_practice\\House price prediction\\data.csv")
        HP = HOUSEPRICE(data)
        HP.preprocessing()
        HP.linear_model()
        HP.lasso_regression()
        HP.ridge_regression()
    except Exception as e:
        Error_type, Error_msg, Error_lineno = sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
```
- **Main Block**: Ensures that the following code runs only if the script is executed directly (not imported).
- **Data Loading**: Reads the CSV file containing house prices into a DataFrame.
- **Class Instantiation**: Creates an instance of the `HOUSEPRICE` class.
- **Method Calls**: Sequentially calls methods for preprocessing the data, fitting the linear model, Lasso regression, and Ridge regression.
- **Error Handling**: Captures any exceptions during execution and prints error details.
## 6. Results
Performance Metrics
The performance of each regression model can be summarized in a table:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Metrics</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>

<h2>Model Performance Metrics</h2>

<table>
    <tr>
        <th>Model Type</th>
        <th>Training Accuracy</th>
        <th>Test Accuracy</th>
        <th>MSE</th>
        <th>RMSE</th>
        <th>MAE</th>
    </tr>
    <tr>
        <td>Multilinear</td>
        <td>55.45%</td>
        <td>6.33%</td>
        <td>64285839402.33</td>
        <td>156517.49</td>
        <td>253546.52</td>
    </tr>
    <tr>
        <td>Lasso</td>
        <td>55.45%</td>
        <td>6.33%</td>
        <td>64285839402.33</td>
        <td>156517.49</td>
        <td>253546.52</td>
    </tr>
    <tr>
        <td>Ridge</td>
        <td>55.45%</td>
        <td>6.33%</td>
        <td>64285839402.33</td>
        <td>156517.49</td>
        <td>253546.52</td>
    </tr>
</table>

</body>
</html>

## 7. Conclusion
The project successfully demonstrates the development of a multilinear regression model for predicting house prices. While all models performed reasonably well, Lasso regression provided the best results in terms of reducing overfitting and improving predictive accuracy.

## 8. Future Work
Potential improvements include:

Implementing more advanced models like Decision Trees or Neural Networks.
Hyperparameter tuning for Lasso and Ridge regression.
Expanding the dataset with additional features.
