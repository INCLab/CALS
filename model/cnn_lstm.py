import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataloader import DataLoader
from numpy import array
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataPath = '../data/pe'

# null_pilot_col_list = ['_' + str(x + 32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]
# total_sub = list(np.arange(0, 63, 1))
# sub_list = []
# for sub in total_sub:
#     sub = '_' + str(sub)
#     if sub not in null_pilot_col_list:
#         sub_list.append(sub)
#
# max_acc = 0
# max_sub_idx = None
# acc_list = []


# Load Person Exist dataset
# pe_df, npe_df = DataLoader().loadPEdata(dataPath)
# pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, ['_30', '_31', '_33', '_34'])
pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))


# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]

# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42, shuffle=False)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)

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
# SAMPLE_NUM = 8000=
# X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
# X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]

print('Train: X shape: {}'.format(X_train.shape))
print('Train: y shape: {}'.format(y_train.shape))
print('Valid: X shape: {}'.format(X_valid.shape))
print('Valid: y shape: {}'.format(y_valid.shape))
print('Test: X shape: {}'.format(X_test.shape))
print('Test: y shape: {}'.format(y_test.shape))

TIMESTEMP = 50
inp = (-1, X_train.shape[1], 1)

X_train = X_train.reshape(inp)  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
X_valid = X_valid.reshape(inp)
X_test = X_test.reshape(inp)

print('X reshape: {}'.format(X_train.shape))

inp = keras.layers.Input(shape=(TIMESTEMP, 1))

lstm = keras.layers.LSTM(128, return_sequences=False, stateful=False)(inp)


cnn_1 =  keras.layers.Conv1D(128, 3, padding='valid', activation='relu')(inp)
maxpool = keras.layers.MaxPooling1D()(cnn_1)
cnn_2 = keras.layers.Conv1D(128, 1, padding='valid', activation='relu')(maxpool)
gmaxpool = keras.layers.GlobalMaxPooling1D()(cnn_2)

merged = keras.layers.concatenate([lstm, gmaxpool], axis=1)

hl_2 = keras.layers.Dense(128, activation='relu')(merged)
hl_3 = keras.layers.Dense(64, activation='relu')(hl_2)
out = keras.layers.Dense(1, activation='sigmoid')(hl_3)

model = keras.Model(inp, out)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_valid, y_valid), callbacks=[es])

acc = model.evaluate(X_test, y_test)[1]
print("\n 테스트 정확도: %.4f" % (acc))

# model save
#model.save("cnn1d_model")

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

loss_ax.plot(train_loss, 'y', label='train loss')
loss_ax.plot(val_loss, 'r', label='val loss')

acc_ax.plot(train_acc, 'b', label='train acc')
acc_ax.plot(val_acc, 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()