# -*- coding: utf-8 -*-

"""08/23 Monday at 4:15pm on my lunch break. Took it at 3:45 due to a training.

Here I will conduct a brief walkthrough of how to do webscraping with yahoo finance
and some packages. I will follow this link's uh...code:

https://towardsdatascience.com/web-scraping-yahoo-finance-477fe3daa852

Let's get started!"""


#So, let's go ahead and import what we need

#beso, selenium and chromedriver to help with getting the stuff from the web and transforming it
#These are important because yahoofinance uses JavaScript and t
#re and string to help with identifying text patterns and getting them into consistent formatting
#pandas for pandas!! 

import pandas as pd
from bs4 import BeautifulSoup as beso
import re
from selenium import webdriver
import chromedriver_binary
import string

pd.options.display.float_format = '{:.0f}'.format


#And let us begin! 
#The article uses AAPL, as per normal we will use ENPH
#And then at some point maybe we add AAPL NVDA AMZN etc


#income statement = is_link

is_link = 'https://finance.yahoo.com/quote/ENPH/financials?p=ENPH' 
driver = webdriver.Chrome(executable_path = r'C:\bin\chromedriver')
driver.get(is_link)

"""So I had a ton of issues configuring chromedriver_binary and my version of Google Chrome
which resulted in some frustration. I asked around and googled and yada yada and found that the
path, including the little r you see above, solved the issues.

The issues involved version compatability betweeen chromedriver and chrome (93 vs 92)
and then also issues with Windows PATH environment variables (which were easier to fix)

You see the directory C:\bin is one of my own creation at the advice of others on the internet.

Interesting really. I have no idea what I am doing. But this is part of the process, and I shall
learn and master it too.

I have no choice.""" 


#But moving right along! =)

html = driver.execute_script('return document.body.innerHTML;')
soup = beso(html, 'lxml')

#So these 5 lines of code pop up a dummy browser to yahoo finance ENPH page
#then the html and soup grab things for you and format to lxml
#A quote from the article: 
"""Since Yahoo Finance operates on JavaScript, running the code through this method pulls 
all of the data and saves it as if it were a static website. 
This is important for pulling the stock price, as those are dynamic items on the webpage and 
can refresh/update at regular intervals."""

close_price = [entry.text for entry in soup.find_all('span', {'class':'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)'})]

#So what I am noting here is that you need to manually inspect
#elements of a website.
#There has got to be a way to automate that even further.
#Maybe though it will always require having that knowledge, so maybe collecting it once
#and automating from there. Idk yet lol

#So the author of the article clicked around on the webpage
#and found that financial statement data can be found by inspecting elements
#and looking for div class attributes of D(tbr) form
#So now we are ready to start saving them

features = soup.find_all('div', class_ = 'D(tbr)')

#So the author now creates a few lists to store different things from the data
#i do not know how to see what he saw though, when i print features i get a null
#that may be proper but idk

headers = []
temp_list = []
label_list = []
final = []
index = 0

#And now we create headers, then proceed to a while loop to get things done

for item in features[0].find_all('div', class_ = 'D(ib)'):
    headers.append(item.text)
    
#content of financial statements

while index <= len(features) - 1: 
    #filter each line of statement
    temp = features[index].find_all('div', class_ = 'D(tbc)')
    for line in temp:
        temp_list.append(line.text)
    final.append(temp_list) #so we grab things from the line, add to temp, to final, and empty it
    temp_list = []
    index += 1

df = pd.DataFrame(final[1:])
df.columns = headers

#Neat
#An interesting dataframe, not used to the way the columns are.
#For example: the string '12/31/2017' is a column containing income statement entry items
#all numeric or dashes lol

#Remember that in pandas, strings are called 'object' 
#So we should transform them into numbers

#To do this, our author writes a function which takes in column from a data frame
#and does somethings to it, let's take a look:
def convert_to_numeric(column):
    first_col = [i.replace(',', '') for i in column] #removing the commas from values
    #second_col = [i.replace('-', '') for i in first_col] #takes out the dashes for blank values
    second_col = [i.replace('-', '') if i == '-' else i for i in first_col]
    final_col = pd.to_numeric(second_col)
    return final_col

#Noticed how this function's goal is to prepare the data to be valid inputs
#for the pd.to_numeric function. We first have to remove the pieces which will cause errors
#Do you see why such a function is useful?

#Now to apply it over everyone

for column in headers[1:]:
    df[column] = convert_to_numeric(df[column])
final_df = df.fillna('-') #NaNs are a dash, though I might prefer to keep them NaNs
#in my own analysis
print(final_df)

#Also, by replacing dashes, we get rid of negative numbers which is an issue
#There's a comment to the article about that
 
#Using this code second_col = [i.replace('-', '') if i == '-' else i for i in first_col]
#We add the condition that if i is only a dash, replace it, otherwise leave it be
#this will preseve negative numbers which have a dash AND numerics

"""This is the end of this tutorial. For funsies, I might try to look at the balance sheet
and cash flow later today. Today by the way is Tues 08/24 at 11:45 am.

I have a Dr's appointment at 4, so I took the day off. """
