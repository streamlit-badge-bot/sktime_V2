# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 05:40:28 2020

@author: uy308417
"""

import numpy as np
import pandas as pd
from sktime.forecasting.compose import ReducedRegressionForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import six
import sys
sys.modules['sklearn.externals.six'] = six
#from sktime.forecasting.all import plot_ys
import time
#st.beta_set_page_config(layout="wide")
st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title("Upload Your Sales History")
uploaded_file = st.sidebar.file_uploader("Upload a file in csv format", type=("csv"))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({"date" : ["1/1/2016","1/2/2016","1/3/2016","1/4/2016","1/5/2016","1/6/2016","1/7/2016","1/8/2016",
                                 "1/9/2016","1/10/2016","1/11/2016","1/12/2016","1/1/2017","1/2/2017","1/3/2017","1/4/2017",
                                 "1/5/2017","1/6/2017","1/7/2017","1/8/2017","1/9/2017","1/10/2017","1/11/2017","1/12/2017"], 
                     "sales" : [25,30,35,40,45,20,50,80,70,60,20,10,70,70,40,30,20,60,55,67,34,56,90,40]})
    
st.title("Forecaster")
#col1, col2 = st.beta_columns(2)

st.subheader('Example csv file format to upload')
example_df=pd.DataFrame(np.array([['2016-01-01', 3000], ['2016-02-01' , 4200], ['2016-03-01', 1500]]),columns=['date', 'sales'])
example_df.head()
st.write(example_df)
if st.checkbox('Show uploaded dataframe'):
    st.write(df)


#df=pd.read_csv(r'C:\Users\uy308417\OneDrive - GSK\Desktop\Python Projects\my-notebook\3.2. Sktime\AUG1GUAETRADE.csv')
#uploaded_file.seek(0)


def select_regressor(selection):
    regressors = {
    'Linear Regression': LinearRegression(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(verbosity = 0),
    'Support Vector Machines': LinearSVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=0),
     }
    
    return regressors[selection]

def generate_forecast(df_, regressor, forecast_horizon, window_length):
    df = df_.copy()
    #Replacing NaN values with the forward fill method
    #df.fillna(method = 'ffill', inplace = True)
    
    #Resetting the index of the time series,
    #because sktime doesn't support DatetimeIndex for now
    y_train = df.iloc[:,-1].reset_index(drop=True)

    fh = np.arange(forecast_horizon) + 1
    regressor = select_regressor(regressor)
    forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=window_length,
                                             strategy='recursive')
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict(fh)
      
    date = '1/1/2016' #df.index[0]
    periods = df.shape[0] + forecast_horizon
    #Creating a new DatetimeIndex that goes
    #as far in the future as the forecast horizon
    date_index = pd.date_range(date, periods=periods, freq='M')
    
    col_name = ' Forecast' 
    df_pred = pd.DataFrame({col_name: y_pred}) 
    #Appending the forecast as a new column to the dataframe
    df = df.append(df_pred, ignore_index=True)
    #Setting the DatetimeIndex we created
    #as the new index of the dataframe
    df.set_index(date_index, inplace=True)
    
    return df

def calculate_smape(df_, regressor, forecast_horizon, window_length):
    df = df_.copy()
    df.fillna(method = 'ffill', inplace = True)
    y = df.iloc[:,-1].reset_index(drop=True)
    y_train, y_test = temporal_train_test_split(y, test_size = 12)
    fh = np.arange(y_test.shape[0]) + 1
    regressor = select_regressor(regressor)
    forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=window_length,
                                             strategy='recursive')
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict(fh)
    
    return smape_loss(y_pred, y_test)



#df=pd.read_csv(r'C:\Users\uy308417\OneDrive - GSK\Desktop\Python Projects\my-notebook\3.2. Sktime\AUG1GUAETRADE.csv')


st.subheader('Forecasting Sales with Machine Learning')
st.markdown(" Use different Machine Learning Algortihms to generate time series forecasting for your product.  \n"
        "Forecast Length: How many future months you want to generate forecast  \n"
        "Sliding Window length: Rolling past months to use in future forecasting .  \n"
        "SMAPE is used for accuarcy performance of the model [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)  \n"
        "Choose different ML algorithms and Window Length to see best performing model (minimized SMAPE)  \n"
        )
regressor = st.sidebar.selectbox("Select a Machine Learning Algorithm",   
                                 ['Linear Regression','Support Vector Machines',
                                  'K-Nearest Neighbors','XGBoost',
                                  'Random Forest', 'Gradient Boosting',
                                  'Decision Tree'
                                  ])
st.subheader('Total Monthly Sales - Actual and Future Predictions')
forecast_horizon = st.sidebar.slider(label = 'Forecast Length (months)',
                                     min_value = 3, max_value = 36, value = 12)
window_length = st.sidebar.slider(label = 'Sliding Window Length ',
                                  min_value = 1, value = 8)



smape = calculate_smape(df[['sales']], regressor, forecast_horizon, window_length)*100
st.subheader('SMAPE - Accuracy Performance: %.2f' % smape+'%')
#Generating and plotting a forecast for each renewable energy type
df_forecast = generate_forecast(df[['sales']], regressor, forecast_horizon, window_length)
st.line_chart(df_forecast, use_container_width = False, width = 800)
if uploaded_file is not None:
    uploaded_file.seek(0)

