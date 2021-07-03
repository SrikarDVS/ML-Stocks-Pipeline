from datetime import datetime
import pandas_datareader as web
from pandas import DataFrame
from util import n_day_sma
from util import n_day_ema
from util import MinMaxScaler

start = datetime(2020,1,1)
end = datetime.now()

stocks = web.DataReader('BSE:HINDUNILVR', 'av-daily', start, end, api_key='YXSMCBBE8H56GXU2')

stocks.drop(stocks.columns[[4]],axis = 1, inplace = True)
columns_titles = ['close','high','low','open']
stocks = stocks.reindex(columns = columns_titles)

print(stocks.head())
stocks.to_csv('./ML StockMarket Production/file1.csv')



