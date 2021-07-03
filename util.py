from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import concatenate as conc
from keras import optimizers
import numpy as np
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from keras.utils.vis_utils import plot_model
import pandas_datareader as web


def parse(x):
	return datetime.strptime(x, '%Y-%m-%d')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def n_day_sma(n,val_col):
    out = np.zeros(shape = val_col.shape)
    # print(out.shape)
    for i in range(n-1):
        out[i] = 0
    for i in range(n-1,len(val_col)):
        out[i] = np.mean(val_col[i:i+n])
    return out.reshape(val_col.shape[0],1)

# Calculating the EMA of the closing price value column
def n_day_ema(n_ema,n_sma,val_col):
    y = n_day_sma(n_sma,val_col)
    k = float(2/(n_ema+1))
    ema = np.zeros(shape = val_col.shape)
    previous_ema = y[n_sma-1]
    for i in range(n_sma, len(val_col)):
        ema[i] = k*(val_col[i]-previous_ema) + previous_ema
        previous_ema = ema[i]
    # print(ema.shape)
    ema = ema.reshape(ema.shape[0],1)
    return ema


def train_the_model(name,train_X,train_y,test_X,test_y,tech_ind):
    tech_ind_train = tech_ind[:320]
    tech_ind_test  = tech_ind[320:]
    print(tech_ind_train.shape)

    # Input Declarations
    lstm_input = Input(batch_shape = (32,train_X.shape[1], train_X.shape[2]))
    dense_input = Input(shape = (tech_ind.shape[1],1),name = 'tech_input')
    
    # LSTM Branch AddOnns
    x = LSTM(32,stateful = True, return_sequences = True,name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x - LSTM(32,stateful = True,name = 'lstm_1')(x)
    output = Activation('relu', name='relu_output')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=output)
    # Tech Ind Branch AddOnns
    y = Dense(32, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    
    
    # Combining the 2 branches
    combined = conc([lstm_branch.output, technical_indicators_branch.output], name = 'concat')
    # Puting the output through 2 dense layers to get a single value output
    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)
    # Declaring the model
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    # getting the optimizer
    model.compile( loss = 'mse',optimizer = 'adam')
    model.summary()
    history = model.fit([train_X,tech_ind_train],train_y, epochs = 100, batch_size = 32, validation_data=([test_X[:32,:],tech_ind_test[:32,:]],test_y[:32,:]), verbose = 0, shuffle = False)

    lstm_input = Input(batch_shape = (1,train_X.shape[1], train_X.shape[2]))
    dense_input = Input(shape = (tech_ind.shape[1],1),name = 'tech_input')
    

    x = LSTM(32,stateful = True, return_sequences = True,name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x - LSTM(32,stateful = True,name = 'lstm_1')(x)
    output = Activation('relu', name='relu_output')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=output)

    y = Dense(32, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)
    combined = conc([lstm_branch.output, technical_indicators_branch.output], name = 'concat')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    new_model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    new_model.compile( loss = 'mse',optimizer = 'adam')

    evaluation = model.evaluate([test_X[:32,:],tech_ind_test[:32,:]],test_y[:32,:])
    # print(evaluation)
    
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig(name)

    return new_model


def reframe_values(reframed):
    values = reframed.values
    n_train_hours = 320
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:,-1:]
    test_X, test_y = test[:, :-1], test[:, -1:]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X,train_y,test_X,test_y 

def validation_test(open_model,close_model,high_model,low_model,test_X,scaled_macd):
    y = []
    for i in range(len(test_X)):
        y.append(future_pred_model(open_model,close_model,high_model,low_model,test_X[i,:].reshape(1,1,4),scaled_macd[i,:]))
    
    y = np.array(y).reshape(test_X.shape[0],test_X.shape[2])
    print(y.shape)
    return y

def get_macd(values):
    z1 = n_day_ema(12,9,values[:,3])
    z2 = n_day_ema(26,9,values[:,3])
    macd = z1-z2
    macd_scaler = MinMaxScaler(feature_range=(0,1))
    scaled_macd = macd_scaler.fit_transform(macd)
    return scaled_macd


def future_pred_model(open_model,close_model,high_model,low_model,test_X,tech_ind):
    yhat_open = open_model.predict([test_X,tech_ind])
    yhat_close = close_model.predict([test_X,tech_ind])
    yhat_high = high_model.predict([test_X,tech_ind])
    yhat_low = low_model.predict([test_X,tech_ind])

    # print(yhat_close.shape,yhat_open,yhat_high,yhat_low)
    # inv_yhat_open = yhat_open
    # inv_yhat_close = yhat_close
    # inv_yhat_high = yhat_high
    # inv_yhat_low = yhat_low

    inv_yhat_open = yhat_open.reshape(yhat_open.shape)
    inv_yhat_close = yhat_close.reshape(yhat_close.shape)
    inv_yhat_high = yhat_high.reshape(yhat_high.shape)
    inv_yhat_low = yhat_low.reshape(yhat_low.shape)
    combined = [inv_yhat_close, inv_yhat_high, inv_yhat_low, inv_yhat_open]
    combined = np.array(combined)
    # print(combined)
    return combined

def get_new_tech(x):
    z1 = n_day_ema(12,3,x[:,3])
    z2 = n_day_ema(26,3,x[:,3])
    macd = z1-z2
    macd_scaler = MinMaxScaler(feature_range=(0,1))
    scaled_macd = macd_scaler.fit_transform(macd)
    return scaled_macd[-1,:]

def prediction_for_days(days,todays_values,tech_ind,values,scaler,open_model,close_model,high_model,low_model):
    output_arr = np.zeros(shape = (days,4))
    append_ds = values
    for i in range(days):
        todays_values = todays_values.reshape(1,1,4)
        new_row = future_pred_model(open_model,close_model,high_model,low_model,todays_values,tech_ind)
        # print(new_row[1,:,:])
        # print(todays_values)
        out_row = scaler.inverse_transform(new_row.reshape(1,4))
        append_ds = np.vstack((append_ds,out_row))
        append_ds = append_ds.astype('int')
        tech_ind = get_new_tech(append_ds)
        output_arr[i] = new_row.reshape(4,)
        todays_values = new_row
    return output_arr,append_ds


