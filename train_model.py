import pandas as pd

from util import train_the_model
from util import reframe_values
from get_value import scaled_macd
from reframe import reframed_close,reframed_high,reframed_low,reframed_open




train_X,train_y,test_X,test_y = reframe_values(reframed_open)
open_model = train_the_model("open_train_loss",train_X,train_y,test_X,test_y,scaled_macd)
train_X,train_y,test_X,test_y = reframe_values(reframed_close)
close_model = train_the_model("close_train_loss",train_X,train_y,test_X,test_y,scaled_macd)
train_X,train_y,test_X,test_y = reframe_values(reframed_high)
high_model = train_the_model("high_train_loss",train_X,train_y,test_X,test_y,scaled_macd)
train_X,train_y,test_X,test_y = reframe_values(reframed_low)
low_model = train_the_model("low_train_loss",train_X,train_y,test_X,test_y,scaled_macd)

open_model.save("./ML StockMarket Production/open_model")
close_model.save("./ML StockMarket Production/close_model")
high_model.save("./ML StockMarket Production/high_model")
low_model.save("./ML StockMarket Production/low_model")
