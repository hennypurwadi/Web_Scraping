#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup 
from sklearn.preprocessing import LabelEncoder
import datetime
from datetime import datetime, timedelta
import time
import schedule
import h5py
import pytrends
from pytrends.request import TrendReq

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Create directory
import os

path = './data'
# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist: 
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("new directory is created!")


# In[3]:


today = datetime.now().date()
time_delta_D = timedelta(days=7)
lastweek = today - time_delta_D

today = str(today)
lastweek = str(lastweek)

today


# In[4]:


#Show list of url.csv

file_url = pd.read_csv('url.csv')
file_url


# In[5]:


def web_scrape(url):
    
    Bsoup = BeautifulSoup(requests.get(url).text)
    headers = [header.text for listing in Bsoup.find_all('thead') for header in listing.find_all('th')]
    r_data = {header:[] for header in headers}

    for rows in Bsoup.find_all('tbody'):
      for row in rows.find_all('tr'):
        
        if len(row) != len(headers): continue
        for idx, cell in enumerate(row.find_all('td')):
              r_data[headers[idx]].append(cell.text)
    return pd.DataFrame(r_data)


def job():

    df0 = web_scrape(file_url['url_list'].iloc[0])
    df00 = df0.iloc[:, 0:3]
    df1 = web_scrape(file_url['url_list'].iloc[1])
    df01 = df1.iloc[:, 0:3]
    df2 = web_scrape(file_url['url_list'].iloc[2])
    df02 = df2.iloc[:, 0:5]
    df3 = web_scrape(file_url['url_list'].iloc[3])
    df03 = df3.iloc[:, 0:3]
    
    #chosen url
    df = df03 
    
    h5File = (today + '_web_scrape.h5')
    df.to_hdf(h5File, 'w')
    print("wrote hdf5 file done")   
    
#schedule.every(1).minutes.do(job)
#schedule.every(1).hour.do(job)
schedule.every(1).day.at("10:30").do(job)


# In[6]:


#https://finance.yahoo.com/gainers
df0 = web_scrape(file_url['url_list'].iloc[0]) #first row
df00 = df0.iloc[:, 0:3]#first until 4th columns

#https://finance.yahoo.com/currencies
df1 = web_scrape(file_url['url_list'].iloc[1])
df01 = df1.iloc[:, 0:3]

#https://coinmarketcap.com/nft/collections/
df2 = web_scrape(file_url['url_list'].iloc[2])
df02 = df2.iloc[:, 0:5]

#https://finance.yahoo.com/cryptocurrencies/
df3 = web_scrape(file_url['url_list'].iloc[3])
df03 = df3.iloc[:, 0:3]


# In[7]:


#chose url 00
df = df00 

#Save to csv file
df.to_csv(today + '_web_scrape.csv')

#Save to HDF5 file
h5File = (today + '_web_scrape.h5')
df.to_hdf(h5File, 'w')
print("wrote hdf5 file done")

hf = pd.read_hdf(h5File)
list(hf.keys())
hf.shape

hf.head(2)


# In[8]:


#chose url 1
df = df01 

#Save to csv file
df.to_csv(today + '_web_scrape.csv')

#Save to HDF5 file
h5File = (today + '_web_scrape.h5')
df.to_hdf(h5File, 'w')
print("wrote hdf5 file done")

hf = pd.read_hdf(h5File)
#list(hf.keys())
#hf.shape

hf.head(2)


# In[9]:


#chose url 2
df = df02 

#Save to csv file
df.to_csv(today + '_web_scrape.csv')

#Save to HDF5 file
h5File = (today + '_web_scrape.h5')
df.to_hdf(h5File, 'w')
print("wrote hdf5 file done")

hf = pd.read_hdf(h5File)
#list(hf.keys())
#hf.shape

hf.head(2)


# In[10]:


#chose url 3
df = df03 

#Save to csv file
df.to_csv(today + '_web_scrape.csv')

#Save to HDF5 file
h5File = (today + '_web_scrape.h5')
df.to_hdf(h5File, 'w')
print("wrote hdf5 file done")

hf = pd.read_hdf(h5File)
hf.head(2)


# # Merge datasets collected from several days

# In[11]:


#day01
hf220219 = pd.read_hdf('2022-02-19_web_scrape.h5')
hf220219.rename(columns = {'Price (Intraday)':'2022-02-19'},inplace = True)
hf220219 = hf220219.drop(columns=['Name'])
hf220219.head(2)


# In[12]:


#day02
hf220220 = pd.read_hdf('2022-02-20_web_scrape.h5')
hf220220.rename(columns = {'Price (Intraday)':'2022-02-20'},inplace = True)
hf220220 = hf220220.drop(columns=['Name'])
hf220220.head(2)


# In[13]:


#day03
hf220221 = pd.read_hdf('2022-02-21_web_scrape.h5')
hf220221.rename(columns = {'Price (Intraday)':'2022-02-21'},inplace = True)
hf220221 = hf220221.drop(columns=['Name'])
hf220221.head(2)


# In[68]:


#Merge datasets become one
hfmerged = pd.merge(hf220219, hf220220, on=["Symbol", "Symbol"])
hfmerged = pd.merge(hfmerged, hf220221, on=["Symbol", "Symbol"])

#We will focus on BTC and ETH 
hfmerged = hfmerged.iloc[0:2]
hfmerged


# In[69]:


print(hfmerged.isnull().sum())


# In[70]:


hfmerged.dtypes


# In[62]:


#Split decimals then removed
hfmerged[['20220219', '2022-02-19B']] = hfmerged['2022-02-19'].str.split('.', 1, expand=True)
hfmerged[['20220220', '2022-02-20B']] = hfmerged['2022-02-20'].str.split('.', 1, expand=True)
hfmerged[['20220221', '2022-02-21B']] = hfmerged['2022-02-21'].str.split('.', 1, expand=True)
hfmerged = hfmerged.drop(columns=['2022-02-19', '2022-02-19B','2022-02-20', '2022-02-20B','2022-02-21', '2022-02-21B'])

hfmerged.head(2)


# In[63]:


#Remove comma inside values. Otherwise can't convert them into float

hfmerged['20220219'] = hfmerged[('20220219')].replace(',','', regex=True)
hfmerged['20220220'] = hfmerged[('20220220')].replace(',','', regex=True)
hfmerged['20220221'] = hfmerged[('20220221')].replace(',','', regex=True)
hfmerged.head(2)


# In[64]:


#Change dtypes from str become float

hfmerged['20220219'] = hfmerged[('20220219')].astype(float)
hfmerged['20220220'] = hfmerged[('20220220')].astype(float)
hfmerged['20220221'] = hfmerged[('20220221')].astype(float)
hfmerged.dtypes


# In[65]:


hfmerged = hfmerged.iloc[0:2]
hfmerged


# In[66]:


#Transpose
hfmerged_T = hfmerged.set_index('Symbol').T
hfmerged_T.head(5)


# # writefile to .py files

# In[67]:


get_ipython().run_cell_magic('writefile', 'web_scraping.py', '\nimport requests \nimport pandas as pd \nfrom bs4 import BeautifulSoup \nimport datetime\nfrom datetime import datetime, timedelta\nimport time\nimport schedule\nimport h5py\n\nimport warnings\nwarnings.filterwarnings(\'ignore\')\n\ntoday = datetime.now().date()\ntoday = str(today)\n\nfile_url = pd.read_csv(\'url.csv\')\n\ndef web_scrape(url):\n    \n    Bsoup = BeautifulSoup(requests.get(url).text)\n    headers = [header.text for listing in Bsoup.find_all(\'thead\') for header in listing.find_all(\'th\')]\n    r_data = {header:[] for header in headers}\n\n    for rows in Bsoup.find_all(\'tbody\'):\n          for row in rows.find_all(\'tr\'):\n        \n            if len(row) != len(headers): continue\n            for idx, cell in enumerate(row.find_all(\'td\')):\n                  r_data[headers[idx]].append(cell.text)\n    return pd.DataFrame(r_data)\n\ndef job():\n    \n    #https://finance.yahoo.com/gainers\n    df0 = web_scrape(file_url[\'url_list\'].iloc[0])\n    df00 = df0.iloc[:, 0:3]\n\n    #https://finance.yahoo.com/currencies\n    df1 = web_scrape(file_url[\'url_list\'].iloc[1])\n    df01 = df1.iloc[:, 0:3]\n\n    #https://coinmarketcap.com/nft/collections/\n    df2 = web_scrape(file_url[\'url_list\'].iloc[2])\n    df02 = df2.iloc[:, 0:5]\n\n    #https://finance.yahoo.com/cryptocurrencies/\n    df3 = web_scrape(file_url[\'url_list\'].iloc[3])\n    df03 = df3.iloc[:, 0:3]\n    \n    #chosen url\n    df = df03 \n    \n    h5File = (today + \'_web_scrape.h5\')\n    df.to_hdf(h5File, \'w\')\n    print("wrote hdf5 file done")   \n    \nschedule.every().day.at("10:30").do(job)\n\nif __name__ == "__main__":\n    job()')


# #### Automated task using Task Scheduler on Windows (Run Task Scheduler)
# 1.Open Start
# 
# 2.Search for Task Scheduler , click the top result to open the experience.
# 
# 3.Expand the Task Scheduler Library branch.
# 
# 4.Select the folder with your tasks.
# 
# 5.To run a task on demand, right-click it and select the Run option.
# 
# 6.To edit a task, right-click it and select the Properties options.
# 
# 7.To delete a task, right-click it and select the Delete option.

# In[47]:


from datetime import datetime
with open('timestamps.txt','a+') as file:
    file.write(str(datetime.now()))    


# ### Collect BTH and ETH from Google Trends using pytrends

# In[48]:


#https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
#https://trends.google.com/trends/explore

pytrend = TrendReq()
keys=['Bitcoin','Ethereum'] 
keys_codes=[pytrend.suggestions(keyword=i)[0] for i in keys] 
df_codes= pd.DataFrame(keys_codes)
df_codes

country=["US","RU","UA","AF","SG"] #US, Russia, Ukraine, Afghanistan, Singapore
date_interval='2022-01-02 2022-02-28'
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
        pytrend.build_payload(kw_list=key, timeframe = date_interval, geo = country, 
                              cat=category, gprop=search_t) 
        collect[i] = pytrend.interest_over_time()
        i+=1
df_trends = pd.concat(collect, axis=1)

df_trends.columns = df_trends.columns.droplevel(0) 
df_trends = df_trends.drop('isPartial', axis = 1) 
df_trends.reset_index(level=0,inplace=True) 

#change column names
df_trends.columns=['date','BTC-US','ETH-US','BTC-Russia','ETH-Russia','BTC-Ukraine','ETH-Ukraine',
                   'BTC-Afghanistan','ETH-Afghanistan','BTC-Singapore', 'ETH-Singapore']  
#df_trends.tail(5)


# In[49]:


#Save as HDF5 dataset

today = datetime.now().date()
today = str(today)
df_trends.to_csv(today + '_trends.csv','w')
print("csv file trends saved")

h5File_df_trends = (today + '_trends.h5')
df_trends.to_hdf(h5File_df_trends, 'w')
print('h5File_df_trends file saved')


# In[50]:


#Load the datasets
hf_trends = pd.read_hdf(h5File_df_trends)
hf_trends.head(5)


# In[51]:


sns.set(color_codes=True, palette='deep')
ht = hf_trends.plot(figsize = (12,5),x="date", y=['BTC-US','BTC-Russia','BTC-Ukraine','BTC-Afghanistan', 'BTC-Singapore'], 
                    title = "BTC Google Trends FEBRUARY 2022", kind="line")
ht.set_ylabel('Idx Trends')
ht.tick_params( which='both', axis='both', labelsize=12)
plt.savefig("BTC_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[52]:


sns.set(color_codes=True, palette='deep')
ht = hf_trends.plot(figsize = (12,5),x="date", y=['ETH-US','ETH-Russia','ETH-Ukraine','ETH-Afghanistan','ETH-Singapore'], 
                    kind="line", title = "ETH Google Trends FEBRUARY 2022")
ht.set_ylabel('Idx Trends')
ht.tick_params( which='both', axis='both', labelsize=12)
plt.savefig("ETH_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# Cryptocurrency use—especially Bitcoin use—has spiked in Afghanistan since the U.S. and its allies chaotically exited the country after a 20-year. Afghanistan has one of the fastest rates of cryptocurrency adoption, according to blockchain analytics firm Chainalysis.A number of high-profile Bitcoin advocates used the situation in Afghanistan last week to make crude jokes, while others made wide-eyed suggestions about the way cryptocurrencies could be used to escape the economic control of repressive regimes. https://decrypt.co/79980/cryptocurrency-save-afghanistan-cardano-ethereum-cofounder

# In[53]:


#BTC vs ETH trends in Afghanistan
sns.set(color_codes=True, palette='dark')
ht = hf_trends.plot(figsize = (5,3),x="date", y=['BTC-US','ETH-US'], kind="line", 
                    title = "BTC vs ETH in US Google Trends FEBRUARY 2022")
ht.tick_params( which='both', axis='both', labelsize=11)
ht.set_ylabel('Idx Trends')
plt.savefig("BTCvsETH_US_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[54]:


#BTC vs ETH trends in Russia
sns.set(color_codes=True, palette='dark')
ht = hf_trends.plot(figsize = (5,3),x="date", y=['BTC-Russia','ETH-Russia'], 
                    title = "BTC vs ETH in Russia Google Trends FEBRUARY 2022",kind="line")
ht.set_ylabel('Idx Trends')
ht.tick_params( which='both', axis='both', labelsize=11)
plt.savefig("BTCvsETH_Russia_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[55]:


#BTC vs ETH trends in Ukraine
sns.set(color_codes=True, palette='dark')
ht = hf_trends.plot(figsize = (5,3),x="date", y=['BTC-Ukraine','ETH-Ukraine'], 
                    title = "BTC vs ETH in Ukraine Google Trends FEBRUARY 2022", kind="line")
ht.set_ylabel('Idx Trends')
ht.tick_params( which='both', axis='both', labelsize=11)
plt.savefig("BTCvsETH_Ukraine_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[56]:


#BTC vs ETH trends in Singapore
sns.set(color_codes=True, palette='dark')
ht = hf_trends.plot(figsize = (5,3),x="date", y=['BTC-Singapore','ETH-Singapore'], 
                    title = "BTC vs ETH in Singapore Google Trends FEBRUARY 2022", kind="line")
ht.set_ylabel('Idx Trends')
ht.tick_params( which='both', axis='both', labelsize=11)
plt.savefig("BTCvsETH_Singapore_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[58]:


#BTC vs ETH trends in Afghanistan
sns.set(color_codes=True, palette='dark')
ht = hf_trends.plot(figsize = (5,3),x="date", y=['BTC-Afghanistan','ETH-Afghanistan'], 
                    title = "BTC vs ETH in Afghanistan Google Trends FEBRUARY 2022", kind="line")
ht.set_ylabel('Idx Trends')
ht.tick_params( which='both', axis='both', labelsize=11)
plt.savefig("BTCvsETH_Afghanistan_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)

