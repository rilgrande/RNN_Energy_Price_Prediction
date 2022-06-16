#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Created By: Roger IL Grande
# ---------------------------------------------------------------------------
"""EM-623 Final Project"""

import numpy as np
import pandas as pd
import datetime
from pylab import rcParams
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

# DATA PRE-PROCESSING, CONVERTING THE DATE FORMAT

#dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y') # Did not end up needing this

# Read in the dataset (comment out each one, there are two separate datasets to run in the model)
df = pd.read_csv('BrentOilPrices.csv', parse_dates=['Date'])
#df = pd.read_csv('Henry_Hub_Natural_Gas_Spot_Price.csv', parse_dates=['Date'])
print(df)

# Sorting the dataset by the Date column
df = df.sort_values('Date')
df = df.groupby('Date')['Price'].sum().reset_index()
df.set_index('Date', inplace=True)

df = df.loc[datetime.date(year=1989, month=12, day=1):] # Brent Oil Dataset Dates
#df = df.loc[datetime.date(year=1997, month=1, day=1):] # Henry Hub Dataset Dates

df.head()


# This function provides basic information about the dataset
def df_check_nulls(df_initial):
    tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values (nb)'}))
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum() / df_initial.shape[0] * 100).T.
                               rename(index={0: 'null values (%)'}))
    return tab_info

df_check_nulls(df)

df.index

y = df['Price'].resample('MS').mean()


# Plot shows the full input dataset
y.plot(figsize=(15, 6))
plt.show()

# Plot shows the trendline, seasonal spikes/dips, and residual values
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# Normalize the data using MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
df = sc.fit_transform(df)

# Split the data into test and train sets
train_size = int(len(df) * 0.70)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]


# Create matrix definition from the array of values
def matrix_definition(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return np.array(data_x), np.array(data_y)


# Reshape the datasets into X=t and Y=t+1
look_back = 90
# Lookback defines how many previous timesteps are used in order to predict the subsequent timestep
X_train, Y_train, X_test, Y_test = [], [], [], []
X_train, Y_train = matrix_definition(train, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test, Y_test = matrix_definition(test, look_back)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# BUILDING THE MODEL

# Using the Sequential() model from Keras
# The correct model for a plain stack of layers where each layer has exactly one input tensor and one output tensor
# Create a Sequential model by passing a list of layers to the Sequential constructor
model = Sequential()

model.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.1))

model.add(LSTM(units = 60, return_sequences = True))
model.add(Dropout(0.1))

model.add(LSTM(units = 60))
model.add(Dropout(0.1))

model.add(Dense(units = 1))

from keras import backend as K
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

# Instantiate optimizer before passing it to model.compile()
# The default parameters for the Adam optimizer will be used
# Compile configures the model for training
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[auc])
# Loss function computes the quantity that a model should seek to minimize during training
# mean_squared_error computes the mean of squares of errors between labels and predictions

reduce_LR = ReduceLROnPlateau(monitor='val_loss',patience=5) # Reduce learning rate when a metric has stopped improving

# Train the model
history = model.fit(X_train, Y_train, epochs = 20, batch_size = 15, validation_data=(X_test, Y_test),
                        callbacks=[reduce_LR], shuffle=False)

# Use the model to do prediction with the train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert the predictions
train_predict = sc.inverse_transform(train_predict)
Y_train = sc.inverse_transform([Y_train])
test_predict = sc.inverse_transform(test_predict)
Y_test = sc.inverse_transform([Y_test])

# Output these error metrics
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()


# Compare the actual prices and the predicted prices
aa=[x for x in range(180)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[0][:180], marker='.', label="Actual")
plt.plot(aa, test_predict[:,0][:180], 'r', label="Prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time Step', size=15)
plt.legend(fontsize=15)
plt.show()


# AUC Plot
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()
