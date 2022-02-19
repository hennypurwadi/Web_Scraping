To help decision maker to make strategic decision, we demonstrate web scrapping skills to automate collection of information from some web pages.

df1 = web_scrape(file_url['url_list'].iloc[0])

df = df1.iloc[:, 0:3]

url_list:

https://finance.yahoo.com/cryptocurrencies/

https://finance.yahoo.com/gainers

https://finance.yahoo.com/currencies

https://coinmarketcap.com/nft/collections/
