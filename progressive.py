# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:47:41 2022

@author: mackt

Here I would like to look at stock price data for Progressive Insurance, ticket PGR

A simple look at the stock price, returns, and a forecast using ARIMA, following documentation or
an article that I can figure out.

Let's get started

**Some ideas and code will be taken from this Kaggle article:
    https://www.kaggle.com/nageshsingh/stock-market-forecasting-arima
"""

#Section I: Loading and Inspecting the Data

#Necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


#Time series analysis and modeling tools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima

#Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Download the data
pgr = yf.download('PGR', start = '2018-01-1', ends = '2022-01-1')
print(type(pgr))

#Basic data inspection
print(pgr.head())
print(pgr.shape)
print(pgr.info())
print(pgr.describe())

#Create adjusted close df and returns df

pgr_close = pgr['Adj Close']

pgr_returns = pgr_close.pct_change().dropna()
print(pgr_returns.describe())


#Section II: Visualizing the Data


#Let's take a look at how PGR has changed over time
plt.plot(pgr.index, pgr_close)
plt.title('Progressive Ins Stock Price')
plt.show()

#And how its returns have looked over time
plt.plot(pgr_returns.index, pgr_returns)
plt.title('Progressive Ins Stock Returns')
plt.show()

#20 day Rolling Mean for both pgr['Adj Close'] and pgr_returns

#Adj Close
plt.plot(pgr.index, pgr_close.rolling(window = 20).mean())
plt.title('20 Day Rolling Mean Price')
plt.show()

#Returns
plt.plot(pgr_returns.index, pgr_returns.rolling(window = 20).mean())
plt.title('20 Day Rolling Mean Return')
plt.show()

#Histogram of returns
plt.hist(pgr_returns, bins = 50)
plt.title("Distribution of Returns")
plt.show()

#Distribution of PGR Price
pgr_close.plot(kind = 'kde')
plt.title('Distribution of Closing Prices')
plt.show()

#Now it's time for a stationarity test
#Eyeballing it, the closing prices are probably not stationary, while the returns might be

#ADF Test and a function for it are defined in the statsmodels documentation
#But here, we will use the function that Kaggle defined
#Kaggle uses a 12 period moving average, we will use a parameter 'size' to choose

def test_stationarity(timeseries, size):
    #Calculating and plotting rolling statistics
    rolmean = timeseries.rolling(window = size).mean()
    rolstd = timeseries.rolling(window = size).std()
    
    
    plt.plot(timeseries, color = 'blue', label = 'Original')
    plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    plt.plot(rolstd, color = 'black', label = 'Rolling STD')
    plt.legend(loc = 'best')
    plt.title(f'Rolling {size} Day Mean and Std')
    plt.show(block = False)
    
    #ADF test
    print("Results of ADF Test: ")
    
    adft = adfuller(timeseries, autolag = 'AIC')
    #The adfuller() function does not format the output, that's on us
    #The documentation offers a way to do this, here we will follow Kaggle
    output = pd.Series(adft[0:4], index = ['Test Statistics', 'p-value', 'No. Lags Used', 'No. Observations Used'])
    for key, values in adft[4].items():
        output['Critical Values (%s)'%key] = values
    print(output)
    
    #Returns nothing

test_stationarity(pgr_close, 20)

#The p-value is far higher than the significance level 0.05
#The closing price series is *not* stationary (We cannot reject the null hypothesis)

#Use this code to see if the pgr_returns are stationary or not!
#test_stationarity(pgr_returns, size = 20)

#To get the closing prices stationary, we need to decompose the series

decomp = seasonal_decompose(pgr_close, model = 'multiplicative', freq = 30)
#And plot
fig = plt.figure()
fig = decomp.plot()
fig.set_size_inches(16, 9)

#One way to get a good model is to reduce the variance which can be done by 
#taking the log of the series

pgr_log = np.log(pgr_close)

moving_avg = pgr_log.rolling(window = 20).mean()
std_dev = pgr_log.rolling(window = 20).std()

plt.plot(moving_avg, color = 'red', label = 'Moving Avg')
plt.plot(std_dev, color = 'black', label = 'Std Dev')
plt.title("Moving Average of Logged Prices")
plt.legend(loc = 'best')
plt.show()

#Now we will examine acf and pacf plots
#Let's do so with a differenced series
sm.graphics.tsa.plot_acf(pgr_log.diff(1).dropna(), lags = 20)
sm.graphics.tsa.plot_pacf(pgr_log.diff(1).dropna(), lags = 20)

#From this we can see that (partial) autocorrelations start to fall within range at around lag = 3
#We may use this with our ARIMA


#Time for ARIMA, first things first: train test split
train, test = pgr_log[3:int(len(pgr_log)*0.9)], pgr_log[int(len(pgr_log)*0.9):]
print("Train shape: ", train.shape)
print("Test shape: ", test.shape)

#Take a look at the split:
plt.figure(figsize = (10, 6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Adj Closing Price')
plt.plot(train, color = 'green', label = 'Training Data')
plt.plot(test, color = 'blue', label = 'Testing Data')

#We will start with auto.arima

pgr_autoarima = auto_arima(pgr_log, start_p = 0, start_q = 0, test = 'adf', 
                             max_p = 3, max_q = 3,
                             m = 5,
                             d = None,
                             seasonal = False,
                             start_P = 0,
                             D = 0,
                             trace = True,
                             error_action = 'ignore',
                             suppress_warnings = True,
                             stepwise = True)

print(pgr_autoarima.summary())
pgr_autoarima.plot_diagnostics(figsize = (16,9))

#From the summary, we see that (p,d,q) = (1,0,2) is optimal, interesting that d = 0
#Now let's build a model that we can use for forecasting

pgr_arima = ARIMA(train, order = (1,0,2))
fit = pgr_arima.fit(disp = -1)
print(fit.summary())

#Time to forecast

fc, se, conf_int = fit.forecast(test.shape[0], alpha = 0.05)

#We meed to do a little massaging in order to plot this
fc_series = pd.Series(fc, index = test.index)
lower_series = pd.Series(conf_int[:, 0], index = test.index)
upper_series = pd.Series(conf_int[:, 1], index = test.index)

plt.figure(figsize = (10, 5), dpi = 100)
plt.plot(train, color = 'black', label = 'Training Data')
plt.plot(test, color = 'blue', label = 'Actual Price')
plt.plot(fc_series, color = 'red', label = 'Forecast Price')
plt.fill_between(lower_series.index, lower_series, upper_series, color = 'k', alpha = 0.10)
plt.title('Progressive Stock Price Forecast')
plt.xlabel('Time')
plt.ylabel('Log Price')
plt.legend(loc = 'best')
plt.show()

#From the graphic, this is a pretty terrible forecast

#Model Performance Metrics
mse = mean_squared_error(test, fc)
print('MSE: ' + str(mse))

mae = mean_absolute_error(test, fc)
print('MAE: ' + str(mae))

rmse = np.sqrt(mean_squared_error(test, fc))
print('RMSE: ' + str(rmse))

mape = np.mean(np.abs(fc - test)/np.abs(test))
print('MAPE: ' + str(mape))


#MAPE is a good metric for this kind of model
#And with a mape of 0.025, our model is 97.5% accurate over the next 105 days
#If you placed long-trades or investments in PGR based off this model, you would be glad
#If you shorted PGR based off this model, you might be disappointed
