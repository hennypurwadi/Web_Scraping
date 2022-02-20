
# Get daily data from Google Trends

import datetime
import requests
import pandas as pd
from pytrends import dailydata

def get_google_trends_data(keyword, fr_date, to_date):  
    fr_year, fr_month = datetime.date.fromisoformat(fr_date).year, datetime.date.fromisoformat(fr_date).month
    to_year, to_month = datetime.date.fromisoformat(to_date).year, datetime.date.fromisoformat(to_date).month
    data = dailydata.get_daily_data(keyword, fr_year, fr_month, to_year, to_month)
    return data[keyword]

today = datetime.datetime.now().date()
time_delta_D = timedelta(days=7)
lastweek = today - time_delta_D

today = str(today)
lastweek = str(lastweek)

#Fill keywords to web scrape
keyword_chosen1 = "BTC"
keyword_chosen2 = "ETH"

# Get Google Trends data and save

#google = get_google_trends_data(keyword_chosen, lastweek, today)
google1 = get_google_trends_data(keyword_chosen1, '2022-01-01', '2022-02-28')
google2 = get_google_trends_data(keyword_chosen2, '2022-01-01', '2022-02-28')

h5File_google1 = (today + '_google1.h5')
google1.to_hdf(h5File_google1, 'w')
print('h5File_google1 saved')

h5File_google2 = (today + '_google2.h5')
google2.to_hdf(h5File_google2, 'w')
print('h5File_google2 saved')
