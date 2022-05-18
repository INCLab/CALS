import numpy as np
import pandas as pd
from dataloader import DataLoader
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataPath = '../data/pe'

# Load Person Exist dataset
pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))


# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]

# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42)

# Change to ndarray
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# # Sampling
# SAMPLE_NUM = 1000
# X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
# X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]


print('X shape: {}'.format(X_train.shape))
print('y shape: {}'.format(y_train.shape))

TIMESTEMP = 10

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
print('X reshape: {}'.format(X_train.shape))

model = Sequential()
model.add(LSTM(10, activation='sigmoid', input_shape=(TIMESTEMP, 1), return_sequences=True))
model.add(LSTM(10, activation='sigmoid', return_sequences=True))  # (None, TIMESTEMP, 10)을 받는다
model.add(LSTM(3))  # 마지막은 return_sequence X
# return_sequence를 쓰면 dimension이 한개 추가 되므로 다음 Dense Layer의 인풋에 3 dim이 들어가게 되므로 안씀
# LSTM 두개를 엮을 때
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics='accuracy')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2, callbacks=[early_stopping])

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

yhat = model.evaluate(X_test, y_test)
print(yhat)
