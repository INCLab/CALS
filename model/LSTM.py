import numpy as np
import pandas as pd
from dataloader import DataLoader
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataPath = '../data/pe'

# null_pilot_col_list = ['_' + str(x + 32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]
# total_sub = list(np.arange(0, 63, 1))
# sub_list = []
# for sub in total_sub:
#     sub = '_' + str(sub)
#     if sub not in null_pilot_col_list:
#         sub_list.append(sub)

# max_acc = 0
# max_sub_idx = None

# Load Person Exist dataset
# pe_df, npe_df = DataLoader().loadPEdata(dataPath, ['_30', '_31', '_33', '_34'])
pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)
# pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, ['_30', '_31', '_33', '_34'])

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

# from data_analysis import dataAnalysisPE
#
# dataAnalysisPE(csi_df)

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
# SAMPLE_NUM = 8000
# X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
# X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]


print('X shape: {}'.format(X_train.shape))
print('y shape: {}'.format(y_train.shape))

TIMESTEMP = 50

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
print('X reshape: {}'.format(X_train.shape))

model = Sequential()
model.add(LSTM(128, input_shape=(TIMESTEMP, 1)))  #원래는 256
#model.add(LSTM(50, return_sequences=True))
#model.add(LSTM(50))  # (None, TIMESTEMP, 10)을 받는다
#model.add(LSTM(3, activation='relu'))  # 마지막은 return_sequence X
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)
model.fit(X_train, y_train, epochs=50, batch_size=32,  validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping])

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

result = model.evaluate(X_test, y_test)
print(result)

# if result[-1] > max_acc:
#     max_acc = result[-1]
#     max_sub_idx = sub

# print('max acc: {}, sub_idx: {}'.format(max_acc, max_sub_idx))
# print(acc_list)
