#Recurrent Neural Network

#Part 1 - Data PreProcessing

#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from bokeh.plotting import figure, reset_output
from bokeh.io import show

#importing training dataset
dataset_train = pd.read_csv('dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature Scaling - Normalization
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN


# Initialising the RNN
rnn_regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
rnn_regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
rnn_regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
rnn_regressor.add(LSTM(units = 50, return_sequences = True))
rnn_regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
rnn_regressor.add(LSTM(units = 50, return_sequences = True))
rnn_regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
rnn_regressor.add(LSTM(units = 50))
rnn_regressor.add(Dropout(0.2))

# Adding the output layer
rnn_regressor.add(Dense(units = 1))

# Compiling the RNN
rnn_regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
rnn_regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)

# Part 3 - Performing predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('dataset/GOOG.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 543):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = rnn_regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualizing the results using Bokeh

days = []
for i in range(483):
    days.append(i)
    
real_stock= []
real_stock = real_stock_price.ravel()
    
predict_stock= []
predict_stock = predicted_stock_price.ravel()


#Figure 1
reset_output()
p2 = figure( x_axis_label ='Total Number of Days', y_axis_label ='US Dollars')
p2.line(days, real_stock, color='red', legend='real_stock')
p2.line(days, predict_stock, color='navy', legend='predicted_stock')
p2.title.text = "Predicting Google stocks from January 2017 - November 2018"
p2.legend.location = "top_left"
p2.grid.grid_line_alpha = 0
p2.ygrid.band_fill_color = "olive"
p2.ygrid.band_fill_alpha = 0.1
show(p2)
reset_output()


