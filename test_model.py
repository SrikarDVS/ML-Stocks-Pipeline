from util import validation_test
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import keras.models as load
from reframe import reframed_close,reframed_high,reframed_low,reframed_open
from util import reframe_values
from get_value import scaled_macd,scaler
import os

open_model = load.load_model("./open_model")
close_model = load.load_model("./close_model")
high_model = load.load_model("./high_model")
low_model = load.load_model("./low_model")
train_X,train_y,test_X,test_y = reframe_values(reframed_open)

combined_prediction = validation_test(open_model,close_model,high_model,low_model,test_X,scaled_macd)
combined_prediction = scaler.inverse_transform(combined_prediction)
# print(test_X.shape)
real = scaler.inverse_transform(test_X.reshape(test_X.shape[0],test_X.shape[2]))

names = ['close','high','low','open']
for i in range(len(names)):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plot_no = i
    rmse = sqrt(mean_squared_error(combined_prediction[:,plot_no],real[:,plot_no]))
    plt.plot(combined_prediction[:,plot_no], label=names[i]+" prediction")
    plt.plot(real[:,plot_no], label=names[i]+" real")
#    base_dir = "./Test Prediction Graphs/"
    plt.legend()
    plt.text(1,0,rmse,transform=ax.transAxes)
    plt.savefig(names[i]+" Test Predictions")

    
    print(rmse)
