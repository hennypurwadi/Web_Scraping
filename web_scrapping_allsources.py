#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:


#Create directory
import os

path = './data'
# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist: 
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("new directory is created!")


# In[24]:


today = datetime.now().date()
time_delta_D = timedelta(days=7)
lastweek = today - time_delta_D

today = str(today)
lastweek = str(lastweek)

today


# In[4]:


#Show list of url
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
df0 = web_scrape(file_url['url_list'].iloc[0])
df00 = df0.iloc[:, 0:3]

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
hf220219.rename(columns = {'Price (Intraday)':'day01'},inplace = True)
hf220219.head(2)


# In[12]:


#day02
hf220220 = pd.read_hdf('2022-02-20_web_scrape.h5')
hf220220.rename(columns = {'Price (Intraday)':'day02'},inplace = True)
hf220220.head(2)


# In[13]:


hf220220 = hf220220.drop(columns=['Name'])
hf220220.head(2)


# In[14]:


#Merge datasets become one

hfmerged = pd.merge(hf220219, hf220220, on=["Symbol", "Symbol"])
hfmerged.head(5)


# In[15]:


print(hfmerged.isnull().sum())


# In[16]:


hfmerged.dtypes


# In[17]:


#Split decimals then remove

hfmerged[['day_01', 'day_01B']] = hfmerged['day01'].str.split('.', 1, expand=True)
hfmerged[['day_02', 'day_02B']] = hfmerged['day02'].str.split('.', 1, expand=True)
hfmerged = hfmerged.drop(columns=['day01', 'day_01B','day02', 'day_02B'])

hfmerged.head(5)


# In[19]:


hfmerged['day_01'] = hfmerged[('day_01')].replace(',','', regex=True)
hfmerged['day_02'] = hfmerged[('day_02')].replace(',','', regex=True)
hfmerged.head(5)


# In[21]:


#Change dtypes from str become float
hfmerged['day_01'] = hfmerged[('day_01')].astype(float)
hfmerged['day_02'] = hfmerged[('day_02')].astype(float)
hfmerged.dtypes


# In[ ]:


#Plot
# %% plot the data in a simple scatter plot
plt.figure(figsize=(5, 5))
sns.set_theme(style="whitegrid")
sns.scatterplot(data=df, x="FULLY_VACCINATED_PER100", y="Cases_cumulative_total_per_100000_pop",
               color = 'blue')
plt.savefig("FullyVaccinated_Cases",transparent=False, bbox_inches='tight',pad_inches=0.1)
plt.show()


# In[ ]:





# # writefile to .py files

# In[66]:


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

# In[ ]:


from datetime import datetime
with open('timestamps.txt','a+') as file:
    file.write(str(datetime.now()))    


# ### Collect BTH and ETH using pytrends

# In[35]:


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


# In[33]:


#Save as HDF5 dataset

today = datetime.now().date()
today = str(today)

#df_trends.to_csv(today + '_trends.csv','w')
#print("csv file trends saved")

h5File_df_trends = (today + '_trends.h5')
df_trends.to_hdf(h5File_df_trends, 'w')
print('h5File_df_trends file saved')


# In[34]:


#Load dataset

hf_trends = pd.read_hdf(h5File_df_trends)
hf_trends.head(5)


# In[64]:


sns.set(color_codes=True, palette='deep')

ht = hf_trends.plot(figsize = (12,5),x="date", y=['BTC-US','BTC-Russia','BTC-Ukraine','BTC-Afghanistan', 'BTC-Singapore'], 
                    title = "BTC Google Trends FEBRUARY 2022", kind="line")
ht.set_xlabel('Date')
ht.set_ylabel('Trends Index')
ht.tick_params(axis='both', which='both', labelsize=10)

plt.savefig("BTC_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[62]:


sns.set(color_codes=True, palette='deep')

ht = hf_trends.plot(figsize = (12,5),x="date", y=['ETH-US','ETH-Russia','ETH-Ukraine','ETH-Afghanistan','ETH-Singapore'], 
                    kind="line", title = "ETH Google Trends FEBRUARY 2022")
ht.set_xlabel('Date')
ht.set_ylabel('Trends Index')
ht.tick_params(axis='both', which='both', labelsize=10)

plt.savefig("ETH_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[56]:


#BTC vs ETH trends in Afghanistan
sns.set(color_codes=True, palette='dark')

ht = hf_trends.plot(figsize = (5,4),x="date", y=['BTC-US','ETH-US'], kind="line", 
                    title = "BTC vs ETH in US Google Trends FEBRUARY 2022")
ht.set_xlabel('Date')
ht.set_ylabel('Trends Index')
ht.tick_params(axis='both', which='both', labelsize=10)

plt.savefig("BTCvsETH_US_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[57]:


#BTC vs ETH trends in Russia
sns.set(color_codes=True, palette='dark')

ht = hf_trends.plot(figsize = (5,4),x="date", y=['BTC-Russia','ETH-Russia'], 
                    title = "BTC vs ETH in Russia Google Trends FEBRUARY 2022",kind="line")

ht.set_xlabel('Date')
ht.set_ylabel('Trends Index')
ht.tick_params(axis='both', which='both', labelsize=10)

plt.savefig("BTCvsETH_Russia_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[58]:


#BTC vs ETH trends in Ukraine
sns.set(color_codes=True, palette='dark')

ht = hf_trends.plot(figsize = (5,4),x="date", y=['BTC-Ukraine','ETH-Ukraine'], 
                    title = "BTC vs ETH in Ukraine Google Trends FEBRUARY 2022", kind="line")

ht.set_xlabel('Date')
ht.set_ylabel('Trends Index')
ht.tick_params(axis='both', which='both', labelsize=10)

plt.savefig("BTCvsETH_Ukraine_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[59]:


#BTC vs ETH trends in Afghanistan
sns.set(color_codes=True, palette='dark')

ht = hf_trends.plot(figsize = (5,4),x="date", y=['BTC-Afghanistan','ETH-Afghanistan'], 
                    title = "BTC vs ETH in Afghanistan Google Trends FEBRUARY 2022", kind="line")

ht.set_xlabel('Date')
ht.set_ylabel('Trends Index')
ht.tick_params(axis='both', which='both', labelsize=10)

plt.savefig("BTCvsETH_Afghanistan_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)


# In[60]:


#BTC vs ETH trends in Singapore
sns.set(color_codes=True, palette='dark')

ht = hf_trends.plot(figsize = (5,4),x="date", y=['BTC-Singapore','ETH-Singapore'], 
                    title = "BTC vs ETH in Singapore Google Trends FEBRUARY 2022", kind="line")

ht.set_xlabel('Date')
ht.set_ylabel('Trends Index')
ht.tick_params(axis='both', which='both', labelsize=10)

plt.savefig("BTCvsETH_Singapore_Google_Trends_FEBRUARY2022.jpg",transparent=False, bbox_inches='tight',pad_inches=0.1)

