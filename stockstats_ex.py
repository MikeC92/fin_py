# -*- coding: utf-8 -*-

"""08/19 7:58pm

There is a package I discovered browsing the internet called stockstats

It is a wrapper around pandas which grants the ability to return many different
technical indicators and financial metrics on any pandas data frame.

Here are the links: 
    
Article: https://towardsdatascience.com/stockstats-a-handy-stocks-oriented-pandas-dataframe-wrapper-a879a06d9c3f

Github: https://github.com/jealous/stockstats

Let's get to it!"""


import pandas as pd
import numpy as np
import yfinance as yf
from stockstats import StockDataFrame as sts

enph = yf.download("ENPH", start = '2015-01-01', end = '2021-08-13')

enph_sts = sts.retype(enph)

#So up to this point all we have done is typecasted a pandas df to a stockstats stockdf
#I encourage you to try out head, shape, info, describe to convince yourself that this is okay

#This library is background methods and 'columns' in a sense.
#To see certain values, we can do sts['the_value'] and it will produce it
#despite us not having created the column, take a look:
    
enph_sts[['change', 'rate', 'close_-1_d', 'log-ret']]

# change and rate produce identical output while close_-1_d and log-ret produce their outputs
# Note the NaNs because of first-differencing

#In regards to 'close_-1_d' what this means is: one day change in closing prices between
#time t and t - 1. In this case, t = days. 

#Note now that these 4 columns are appended to the sts
#Do this with info() and shape and head if you want

#Now for something which may render the datacamp tutorial obsolete (but not really)

enph_sts[['close_10_sma']] #closing price 10 day simple moving average [the intuitive one]

#We can now see a pattern in how to retrive information, from the article:
# columnName_window_statistics

#So, I will make up an example: 
enph_sts[['open_20_ema']] #Here I produced a 20 day exponential MA using open prices

#Now let's get fancy!!

enph_sts[['close', 'close_20_sma', 'close_50_sma']].plot(title = 'SMA Example')

#Now we're talkin!!


#And just so we are clear: this is still a pandas dataframe

enph_sts.loc["2020-06-01":, ["close", "close_10_sma", "close_50_sma"]].plot(title="SMA example")
#This plot starts at june last year

"""WHAT'S SO NUTS AND I AM JUST NOTICING NOW IS THAT THIS ARTICLE IS FROM JUNE 20 THIS YEAR!!"""

#Bollinger band time!!

enph_sts[['close', 'boll', 'boll_ub', 'boll_lb']].plot(grid = True, title = 'Bollinger Bands!!')

#This lower plot looks better
enph_sts.loc["2020-06-01":, ["close", "boll", "boll_ub", 'boll_lb']].plot(title="SMA example")


#Now if we wanted to do something more automatically in regards to moving averages
#we could try something like the following to tell us when a short term MA crosses the long term

enph_sts[['close_xu_close_20_sma']]
#When does the closing price cross over the 20 day sma close price?

#We can also do more complicated crossovers, such as a 20d sma cross over 50d sma

enph_sts[['close_20_sma_xd_close_50_sma']]

#note that in with crossovers, we use xu or xd to say cross up (over) or crossover below

""" I will have to play with this and plotting, I think I might be able to use it to my advantage.

Creativity.

Completed at 8:33pm"""
