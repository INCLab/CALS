import matplotlib.pyplot as plt
import numpy as np
import keras
import joblib
import pandas as pd

import os
import tensorflow as tf
import joblib
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataloader import DataLoader
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_plot import model_train_plot, corr_heatmap

TIMESTEMP = 50
MAX_EPOCHS = 100

# Load pretrained model
print('======> Load model')
model = keras.models.load_model('./cnn1d_model')
model.summary()
print('======> Success')

# Load scaler
print('======> Load scaler')
scaler = joblib.load('./std_scaler.pkl')
print('======> Success')

dataPath = '../data/0608_labeled'

pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, TIMESTEMP, scaler=scaler)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))

# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]

# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Change to ndarray
X_train = np.array(X_train)
X_test = np.array(X_test)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
y_train = np.array(y_train)
y_test = np.array(y_test)

print('Train: X shape: {}'.format(X_train.shape))
print('Train: y shape: {}'.format(y_train.shape))
print('Valid: X shape: {}'.format(X_valid.shape))
print('Valid: y shape: {}'.format(y_valid.shape))
print('Test: X shape: {}'.format(X_test.shape))
print('Test: y shape: {}'.format(y_test.shape))

inp = (-1, X_train.shape[1], 1)

X_train = X_train.reshape(inp)  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
X_valid = X_valid.reshape(inp)
X_test = X_test.reshape(inp)

print('X reshape: {}'.format(X_train.shape))

learning_rate = 1e-3
decay = learning_rate / MAX_EPOCHS

optimizer = Adam(
    learning_rate=learning_rate,
    decay=decay
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                        tf.keras.metrics.Precision(name='precision'),
                                                                        tf.keras.metrics.Recall(name='recall')])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid), callbacks=[es])

acc = model.evaluate(X_test, y_test)[1]
print("\n 테스트 정확도: %.4f" % (acc))
