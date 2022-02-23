#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://github.com/hennypurwadi/Web_Scraping

import requests 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup 
import datetime
from datetime import datetime, timedelta
import time
import schedule
import csv
import h5py
import pytrends
from pytrends.request import TrendReq

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Write header in empty csv file

filedf = "dailybtcprice.csv"
with open(filedf, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'BTC'])
        print("file written")


# In[2]:


#Write url_options.csv

url_options = "url_options.csv"
with open(url_options, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['list_url'])
        writer.writerow(['https://www.google.com/search?&q=bitcoin price in US'])
        writer.writerow(['https://finance.yahoo.com/quote/BTC-USD?p=BTC-USD'])
        writer.writerow(['https://coinmarketcap.com/currencies/bitcoin/'])
        writer.writerow(['https://trends.google.com/trends/explore?date=today%201-m&geo=US&q=bitcoin'])
        print("file written")


# In[3]:


today = datetime.now().date()
today = str(today)

Some_url = pd.read_csv('url_options.csv')
url = Some_url['list_url'].iloc[0]

req = requests.get(url)
scrap = BeautifulSoup(req.text, 'lxml')
btc_price = scrap.find("div", class_ = "BNeawe iBp4i AP7Wnd").text

filedf = "dailybtcprice.csv"

# write new data into csv
with open(filedf, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([today, btc_price])
    print("file written")
    
def job(time, price):
    
    url = 'https://www.google.com/search?&q=bitcoin price in US'
    req = requests.get(url)
    scrap = BeautifulSoup(req.text, 'lxml')
    btc_price = scrap.find("div", class_ = "BNeawe iBp4i AP7Wnd").text
        
    filedf = "dailybtcprice.csv"
    
    # append new data into csv new row
    with open(filedf, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([today, btc_price])
        print("file written")        


# In[4]:


#read url_options.csv
Some_url = pd.read_csv('url_options.csv')
Some_url['list_url'].iloc[0]


# In[5]:


btc_price


# In[6]:


get_ipython().run_cell_magic('writefile', 'web_scrape.py', '\nimport requests \nimport pandas as pd \nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom bs4 import BeautifulSoup \nimport datetime\nfrom datetime import datetime, timedelta\nimport time\nimport schedule\nimport csv\nimport h5py\nimport pytrends\nfrom pytrends.request import TrendReq\n    \ntoday = datetime.now().date()\ntoday = str(today)\n\nSome_url = pd.read_csv(\'url_options.csv\')\n#url = \'https://www.google.com/search?&q=bitcoin price in US\'\nurl = Some_url[\'list_url\'].iloc[0]\n\nreq = requests.get(url)\nscrap = BeautifulSoup(req.text, \'lxml\')\nbtc_price = scrap.find("div", class_ = "BNeawe iBp4i AP7Wnd").text\n        \ndef job(time, price):\n        \n    filedf = "dailybtcprice.csv"\n    \n    # write new data into csv\n    with open(filedf, \'a\', newline=\'\') as f:\n        writer = csv.writer(f)\n        writer.writerow([today, btc_price])\n        print("new row written")     \n        \nschedule.every().day.at("09:00").do(job)\n\nif __name__ == "__main__":\n    job(today, btc_price)')


# In[7]:


#read dataset csv
df = pd.read_csv('dailybtcprice.csv')
df.head(3)


# In[8]:


#Create empty HDF file
file = h5py.File('dailybtcprice2.h5','w')

#Create an empty dataset in HDF5 file
filefh = file.create_dataset('dailybtcprice2.h5',(29,6))
filefh


# In[9]:


#read dataset HDF5
hf5 = df.to_hdf('dailybtcprice2.h5', 'w')
#print('hf5 file saved')

file = h5py.File('dailybtcprice2.h5','r+')
list(file.keys())


# In[10]:


#Load the datasets
hf = pd.read_hdf('dailybtcprice2.h5')
hf.head(3)


# In[11]:


#Slice datasets HDF5
hf['BTC_Price'] = hf['BTC'].str.slice(0, 6)
hf['BTC_Currency'] = hf['BTC'].str.slice(10, 31)
hf.head(3)


# In[12]:


#drop unused Columns in HDF5 file
hf=hf.drop(['BTC','BTC_Currency'], axis = 1)
hf.head(3)


# In[ ]:





# In[14]:


#Slice datasets csv
df['BTC_Price'] = df['BTC'].str.slice(0, 6)
df['BTC_Currency'] = df['BTC'].str.slice(10, 31)
df.head(3)


# In[15]:


#drop unused Columns
df=df.drop(['BTC','BTC_Currency'], axis = 1)
df


# In[25]:


#Plot BTC Time Series

plt.style.use("fivethirtyeight")
plt.figure(figsize=(12, 4))

plt.xlabel("Date")
plt.ylabel("Values")
plt.title("BTC-Price in US Time Series")
 
# plotting the columns
plt.plot(df["BTC_Price"], label = 'BTC_Price', color = 'r')
plt.legend()
plt.savefig("BTC_US_Time Series_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# ## Google Trends

# In[17]:


#Change dtypes from str become float

df['BTC_Price'] = df[('BTC_Price')].astype(float)
df.dtypes


# In[18]:


#https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
#https://trends.google.com/trends/explore

pytrend = TrendReq()
keys=['Bitcoin'] 
keys_codes=[pytrend.suggestions(keyword=i)[0] for i in keys] 
df_codes= pd.DataFrame(keys_codes)
df_codes

country=["US"] #US
date_interval='2022-02-01 2022-02-19'
real_keys=df_codes['mid'].to_list()

#categories
category = 0 
search_t = ""

specific_real_key = list(zip(*[iter(real_keys)]*1))
specific_real_key = [list(x) for x in specific_real_key]

collect = {}
i = 1
for country in country:
    for key in specific_real_key:
        pytrend.build_payload(kw_list=key, timeframe = date_interval, 
            geo = country, cat=category, gprop=search_t) 
        collect[i] = pytrend.interest_over_time()
        i+=1
df_trends = pd.concat(collect, axis=1)

df_trends.columns = df_trends.columns.droplevel(0) 
df_trends = df_trends.drop('isPartial', axis = 1) 
df_trends.reset_index(level=0,inplace=True) 

#change column names
df_trends.columns=['date','BTC_Trends_US']  
df_trends


# In[23]:


#Save as HDF5 dataset

today = datetime.now().date()
today = str(today)
df_trends.to_csv(today + '_trends.csv','w')
print("csv file trends saved")

h5File_df_trends = (today + '_trends.h5')
df_trends.to_hdf(h5File_df_trends, 'w')
print('h5File_df_trends file saved')


# In[24]:


#Load the datasets
hf_trends = pd.read_hdf(h5File_df_trends)
hf_trends.head(3)


# In[21]:


plt.style.use("fivethirtyeight")
plt.figure(figsize=(12, 4))

plt.xlabel("date")
plt.ylabel("Index Trends")
plt.title("BTC Google Trends FEBRUARY 2022")
 
# plotting the columns
plt.plot(hf_trends["BTC_Trends_US"], label = 'BTC_Trends')
plt.legend()
plt.savefig("BTC_Trends_FEB2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[22]:


sns.set(color_codes=True, palette='deep')
ht = hf_trends.plot(figsize = (12,4),x="date", y=['BTC_Trends_US'], title = "BTC Google Trends FEBRUARY 2022", kind="line")
ht.set_ylabel('Index Trends')
ht.tick_params( which='both', axis='both', labelsize=12)
plt.savefig("BTC_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[27]:


#Plot BTC Time Series and Google Trends before standardized

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 12))

plt.xlabel("Date")
plt.ylabel("Values")
plt.title("BTC-Price in US Time Series")
 
# plotting the columns
plt.plot(df["BTC_Price"], label = 'BTC_Price', color = 'r')
plt.plot(hf_trends["BTC_Trends_US"], label = 'BTC_Trends')
plt.legend()
plt.savefig("BTC_vs Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# ### Standarize BTC Price and BTC Trends 

# In[28]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

for df in [df]:
    df['BTC_Price'] = scaler.fit_transform(df['BTC_Price'].to_numpy().reshape(-1, 1))    
df.head(4)    


# In[29]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

for hf_trends in [hf_trends]:
    hf_trends["BTC_Trends_US"] = scaler.fit_transform(hf_trends["BTC_Trends_US"].to_numpy().reshape(-1, 1))    
hf_trends.head(4)   


# In[30]:


#Plot STANDARDIZED BTC Time Series and Google Trends

plt.style.use("fivethirtyeight")
plt.figure(figsize=(18, 4))

plt.xlabel("Date")
plt.ylabel("Values")
plt.title("BTC-Price in US Time Series After Standardized")
 
# plotting the columns
plt.plot(df["BTC_Price"], label = 'BTC_Price standardized', color = 'r')
plt.plot(hf_trends["BTC_Trends_US"], label = 'BTC_Trends standardized')
plt.legend()
plt.savefig("BTC_vs Trends_FEBRUARY2022_Standardized.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[31]:


hf_trends["date"].dtypes


# In[32]:


#Change datatype
hf_trends["date"] = hf_trends["date"].astype(str)
hf_trends["date"].dtypes


# In[33]:


hf_trends["Date"] = hf_trends["date"].str.slice(0, 10)
hf_trends.head(3)


# In[34]:


hf_trends = hf_trends.drop(columns=['date'])
hf_trends.head(3)


# In[35]:


#Combine 2 dataframes
dfhf = pd.merge(df,hf_trends, on=["Date", "Date"])
dfhf


# ### Correlation Heatmap

# In[36]:


# generate heatmap of correlation coefficients 
plt.figure(figsize=(5, 3)) # create figure  

sns.heatmap(dfhf.corr(),linewidths = .4,  cmap="YlGnBu", annot = True)  
plt.title("Heatmap Correlation Coefficient between BTC Price VS BTC Trends", size = 14,)
plt.savefig("heatmap_BTC_Price_Trends.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)
plt.show() # show figure  


# In[ ]:


#There are positive weak correlation between BTC Price vs BTC Trends

