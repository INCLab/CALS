from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
from dataloader import DataLoader
from numpy import array
from sklearn.model_selection import train_test_split

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
#
# max_acc = 0
# max_sub_idx = None
# acc_list = []


# Load Person Exist dataset
pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, ['_30', '_31', '_33', '_34'])

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
# SAMPLE_NUM = 8000=
# X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
# X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]


print('X shape: {}'.format(X_train.shape))
print('y shape: {}'.format(y_train.shape))

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
print('X reshape: {}'.format(X_train.shape))

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

TIMESTEMP = 10

dropout_ratio = 0.3 # 드롭아웃 비율
num_filters = 256 # 커널의 수
kernel_size = 10 # 커널의 크기
hidden_units = 128 # 뉴런의 수

model = Sequential()
#model.add(Dropout(dropout_ratio))
model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_units, activation='relu'))
#model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[es, mc])

loaded_model = load_model('best_model.h5')

acc = loaded_model.evaluate(X_test, y_test)[1]
print("\n 테스트 정확도: %.4f" % (acc))

# acc_list.append([sub, acc])
# if acc > max_acc:
#     max_acc = acc
#     max_sub_idx = sub
#
# print('max acc: {}, sub_idx: {}'.format(max_acc, max_sub_idx))
# print(acc_list)
