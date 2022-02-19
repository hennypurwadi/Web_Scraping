
import requests 
import pandas as pd 
from bs4 import BeautifulSoup 
import datetime
from datetime import datetime, timedelta
import time
import schedule
from pytrends import dailydata
import numpy as np
import h5py

import warnings
warnings.filterwarnings('ignore')

today = datetime.now().date()
today = str(today)

file_url = pd.read_csv('url.csv')

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

    df1 = web_scrape(file_url['url_list'].iloc[0])
    df = df1.iloc[:, 0:3]
    
    h5File = (today + '_d_web_scrape.h5')
    df.to_hdf(h5File, 'w')
    print("wrote hdf5 file done")   
    
#schedule.every(1).minutes.do(job) 

if __name__ == "__main__":
    job()
