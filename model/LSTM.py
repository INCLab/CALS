import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, MaxPooling1D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataloader import DataLoader
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_plot import model_train_plot

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TIMESTEMP = 70
MAX_EPOCHS = 100

dataPath = '../data/pe'

# Load Person Exist dataset
# pe_df, npe_df = DataLoader().loadPEdata(dataPath, ['_30', '_31', '_33', '_34'])
# pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)
pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, TIMESTEMP)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

# from data_analysis import dataAnalysisPE
#
# dataAnalysisPE(csi_df)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))

# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]

# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)

# # Scaling
# standardizer = StandardScaler()
# X_train = standardizer.fit_transform(X_train)
# X_valid = standardizer.transform(X_valid)
# X_test = standardizer.transform(X_test)

# Change to ndarray
X_train = np.array(X_train)
X_test = np.array(X_test)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
y_train = np.array(y_train)
y_test = np.array(y_test)

# # Sampling
# SAMPLE_NUM = 8000
# X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
# X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]

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


model = Sequential()
model.add(LSTM(64, input_shape=(TIMESTEMP, 1), return_sequences=False))
# model.add(Dropout(0.3))
# model.add(LSTM(128, input_shape=(TIMESTEMP, 1), return_sequences=False))
# model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, batch_size=32, validation_data=(X_valid, y_valid), callbacks=[es])

acc = model.evaluate(X_test, y_test)[1]
print("\n 테스트 정확도: %.4f" % (acc))

# plot train process
model_train_plot(history)
