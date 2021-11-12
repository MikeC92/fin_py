# -*- coding: utf-8 -*-

"""Created 08/07 at 5:25pm

The goal is to write a few functions that will handle more generally the enph_sedg exercise

I will combine specifying and getting the tickers, and looking at the data frame into one function
'return stock_df' will be important

Next I can put the visualization into a function, maybe I can make it so that I can use it later too
We will see

A returns function to log, dropna, and plot, including scatter

Simple OLS regression, plot

Correlation + 20 day rolling correlation plot, overlay OLS line

Tip: for plotting, x, y must be the position index of each stock.
So like, stock_returns.iloc[0] for stock_1 and stock_returns.iloc[1] for stock_2

But that will be figured out"""

import yfinance as yf
import numpy as np
import pandas as pd


def get_tickers(tickers, start_date, end_date):
    stock_df = yf.download(tickers, start = start_date, end = end_date)
    stock_df.head()
    names = stock_df.columns
    return stock_df, names

def stock_plot(stock_df, names):
    stock_df['Adj Close'].plot(secondary_y = names[1], figsize = (18,12))


def returns(stock_df):
    returns_df = np.log(stock_df['Adj Close'] / stock_df['Adj Close'].shift(1))
    returns_df.dropna(inplace = True)
    returns_df.plot(subplots = True, figsize = (14,6), title = 'Log Returns Plot')
    returns_names = returns_df.columns
    return returns_df, returns_names

def returns_hist(returns_df):
    pd.plotting.scatter_matrix(returns_df, alpha = 0.2,
                               diagonal = 'hist',
                               hist_kwds = {'bins': 50}, figsize = (18,12))
    
def lin_reg(returns_df, returns_names):
    lin_reg = np.polyfit(returns_df[returns_names[0]],returns_df[returns_names[1]], deg = 1)
    ax = returns_df.plot(kind = 'scatter', x = returns_names[0], y = returns_names[1], figsize = (18,12))
    ax.plot(returns_df[returns_names[0]], np.polyval(lin_reg, returns_df[returns_names[0]]), linewidth = 2, color = 'r')
    return lin_reg

def returns_corr(returns_df, returns_names, window):
    ax = returns_df[returns_names[0]].rolling(window = window).corr(returns_df[returns_names[1]]).plot(figsize = (18,12))
    ax.axhline(returns_df.corr().iloc[0,1], c = 'r')
    print(returns_df.corr())
    return returns_df.corr()


"""Finished the session last night, but I did not give a time. This code does exactly
what the prior code does, but for any two stocks."""
