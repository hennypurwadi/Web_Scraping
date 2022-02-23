
import requests 
import pandas as pd 
import matplotlib.pyplot as plt
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
    
today = datetime.now().date()
today = str(today)

Some_url = pd.read_csv('url_options.csv')
#url = 'https://www.google.com/search?&q=bitcoin price in US'
url = Some_url['list_url'].iloc[0]

req = requests.get(url)
scrap = BeautifulSoup(req.text, 'lxml')
btc_price = scrap.find("div", class_ = "BNeawe iBp4i AP7Wnd").text
        
def job(time, price):
        
    filedf = "dailybtcprice.csv"
    
    # write new data into csv
    with open(filedf, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([today, btc_price])
        print("new row written")     
        
schedule.every().day.at("09:00").do(job)

if __name__ == "__main__":
    job(today, btc_price)
