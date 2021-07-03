from util import prediction_for_days,reframe_values
from get_value import scaled_macd,scaler,values
from reframe import reframed_open
from matplotlib import pyplot
import keras.models as load
import os

open_model = load.load_model("./open_model")
close_model = load.load_model("./close_model")
high_model = load.load_model("./high_model")
low_model = load.load_model("./low_model")
train_X,train_y,test_X,test_y = reframe_values(reframed_open)
x,append_ds = prediction_for_days(5,test_X[-1:,:],scaled_macd[-1:,:],values,scaler,open_model,close_model,high_model,low_model)
x = scaler.inverse_transform(x)


pyplot.plot(x, label = '100 day prediction')
base_dir = './Prediction/'
name = "N_day_prediction"
pyplot.savefig(os.path.join(base_dir,name))