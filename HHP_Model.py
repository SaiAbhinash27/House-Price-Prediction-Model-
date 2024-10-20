'''
Developing end to end multi-linear model for the house price prediction
'''
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

class HOUSEPRICE:
    def __init__(self, data):
        try:
            self.data = data
        except Exception as e:
            Error_type, Error_msg, Error_lineno = sys.exc_info()
            print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
    def null_value(self, data):  # method to find the null values
        try:
            return data.isnull().sum()
        except Exception as e:
            Error_type, Error_msg, Error_lineno = sys.exc_info()
            print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
    def convertion(self, data):  # method which converts the categorical data to numerical data
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
    def performance(self,a,b):
        try:
            acc = r2_score(a,b)
            mse = mean_squared_error(a,b)
            rmse = root_mean_squared_error(a,b)
            mae = mean_absolute_error(a,b)
            return acc,mse,rmse,mae
        except Exception as e:
            Error_type, Error_msg, Error_lineno = sys.exc_info()
            print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
    def preprocessing(self): #tarnsforming the data to suitable format for accurate predictions
        try:
            print(f'The number of rows before preprocessing are {self.data.shape[0]} and the number of columns before preprocessing are {self.data.shape[1]}')
            c = [i for i in self.null_value(self.data) if i!=0] # finding either null values present in data or not
            if len(c) > 0:
                print(f'the given data have null values')
            else:
                print(f'the given data dont have any null values')
            self.data.drop(['date','country','street'],axis=1,inplace=True) #Droping unwanted features
            print(f'""Date","Country","Street" features are dropped from the dataset for better prediction')
            self.data['city']= self.data['city'].map(self.convertion(self.data['city']))  #converting categorical data to numerical data
            self.data['statezip']= self.data['statezip'].map(self.convertion(self.data['statezip']))
            print(f'The categorical data was converted to numerical data')
            print(f'The number of rows after preprocessing are {self.data.shape[0]} and the number of columns after preprocessing are {self.data.shape[1]}')
        except Exception as e:
            Error_type,Error_msg,Error_lineno =  sys.exc_info()
            print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
    def linear_model(self):
        try:
            self.X = self.data.iloc[:,1:]
            self.y = self.data.iloc[:,0]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.3,random_state=42)
            self.reg = LinearRegression()
            self.reg.fit(self.X_train,self.y_train)
            self.train_predictions = self.reg.predict(self.X_train)
            self.test_predictions = self.reg.predict(self.X_test)
            print()
            print(f'-----The Normal Multi-linear regression Model prediction performance-----')
            accuracy,mse,rmse,mae = self.performance(self.y_train,self.train_predictions)
            print(f'the accuracy of the training data is: {accuracy}\nthe train_data mean squared error: {mse}\nthe train_data mean absolute error: {mae}\nthe train_data root mean squared error: {rmse}')
            print('-----------------------------------------------------------------------------------')
            accuracy,mse,rmse,mae = self.performance(self.y_test,self.test_predictions)
            print(f'the accuracy of the test data is: {accuracy}\nthe test_data mean squared error: {mse}\nthe test_data mean absolute error: {mae}\nthe test_data root mean squared error: {rmse}')
        except Exception as e:
            Error_type,Error_msg,Error_lineno =  sys.exc_info()
            print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
    def lasso_regression(self):
        try:
            self.lasso= Lasso(alpha=0.5)
            self.lasso.fit(self.X_train,self.y_train)
            self.lasso_train_predictions = self.lasso.predict(self.X_train)
            self.lasso_test_predictions = self.lasso.predict(self.X_test)
            print()
            print('-----The lasso Regression performance-----')
            accuracy,mse,rmse,mae = self.performance(self.y_train,self.lasso_train_predictions)
            print(f'the accuracy of the L1 training data is: {accuracy}\nthe train_data mean squared error: {mse}\nthe train_data mean absolute error: {mae}\nthe train_data root mean squared error: {rmse}')
            print('------------------------------------------------------------------------------------------------------')
            accuracy,mse,rmse,mae = self.performance(self.y_test,self.lasso_test_predictions)
            print(f'the accuracy of the L1 testing data is: {accuracy}\nthe test_data mean squared error: {mse}\nthe test_data mean absolute error: {mae}\nthe test_data root mean squared error: {rmse}')
        except Exception as e:
            Error_type,Error_msg,Error_lineno =  sys.exc_info()
            print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')
    def ridge_regression(self):
        try:
            self.ridge= Ridge(alpha=0.5)
            self.ridge.fit(self.X_train,self.y_train)
            self.ridge_train_predictions = self.ridge.predict(self.X_train)
            self.ridge_test_predictions = self.ridge.predict(self.X_test)
            print()
            print('-----The ridge Regression performance-----')
            accuracy,mse,rmse,mae = self.performance(self.y_train,self.ridge_train_predictions)
            print(f'the accuracy of the L2 training data is: {accuracy}\nthe train_data mean squared error: {mse}\nthe train_data mean absolute error: {mae}\nthe train_data root mean squared error: {rmse}')
            print('------------------------------------------------------------------------------------------------------')
            accuracy,mse,rmse,mae = self.performance(self.y_test,self.ridge_test_predictions)
            print(f'the accuracy of the L2 testing data is: {accuracy}\nthe test_data mean squared error: {mse}\nthe test_data mean absolute error: {mae}\nthe test_data root mean squared error: {rmse}')
        except Exception as e:
            Error_type,Error_msg,Error_lineno =  sys.exc_info()
            print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')


if __name__ == '__main__':
    try:
        data = pd.read_csv("C:\\Users\\abhin\\Desktop\\Sai Python practice\\Pycharm_practice\\House price prediction\\data.csv")
        HP = HOUSEPRICE(data)
        HP.preprocessing()
        HP.linear_model()
        HP.lasso_regression()
        HP.ridge_regression()
    except Exception as e:
        Error_type,Error_msg,Error_lineno =  sys.exc_info()
        print(f'the error line number is {Error_lineno.tb_lineno} --> error type is {Error_type} --> error msg is {Error_msg}')