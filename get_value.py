from util import parse
from util import MinMaxScaler
from util import get_macd
import pandas as pd


dataset = pd.read_csv('./file1.csv',index_col=0, date_parser=parse)
values = dataset.values

values = values.astype('int')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled_macd = get_macd(values)