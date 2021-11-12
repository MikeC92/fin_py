# -*- coding: utf-8 -*-

"""This file is created on 09/02 Thursday at 4:39pm

It will perform basic web scraping and manipulation of financial statements from yahoo finance, 
following this article:
    https://towardsdatascience.com/web-scraping-for-accouwaynting-analysis-using-python-part-1-b5fc016a1c9a
    
Enjoy!"""

#Imports

import pandas as pd
from bs4 import BeautifulSoup
import urllib.request as ur


#Income statement, balance sheet, cash flow
#We will use these with ur and beso

# Enter a stock symbol

index= 'MSFT'

# URL link 
url = 'https://finance.yahoo.com/quote/ENPH?p=ENPH&.tsrc=fin-srch'
url_is = 'https://finance.yahoo.com/quote/ENPH/financials?p=ENPH'

#url_bs = 'https://finance.yahoo.com/quote/ENPH?p=ENPH&.tsrc=fin-srch'
#url_cf = 'https://finance.yahoo.com/quote/' + index + '/cash-flow?p='+ index

#Grabbing the data and parsing using lxml for readability

read_data = ur.urlopen(url_is).read()
soup_is = BeautifulSoup(read_data, 'lxml')


"""10/4/21 this has been paused indefinitely because the code is not working for me.
I am not sure if it is my fault or if packages have changed, or something else enirely.

Thought I believe the problem has to do with how the yahoo finance webpage details
have changed since the article was written - no fault of the author or me perhaps."""
