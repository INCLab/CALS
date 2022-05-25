import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
pe_df, npe_df = DataLoader().loadPEdata(dataPath, ['_30', '_31', '_33', '_34'])
# pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, ['_30', '_31', '_33', '_34'])
# pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)

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

TIMESTEMP = 4

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
print('X reshape: {}'.format(X_train.shape))

model = Sequential()
model.add(Conv1D(128, 3, padding='valid', activation='relu', input_shape=(TIMESTEMP, 1)))
model.add(MaxPooling1D())
model.add(Conv1D(128, 1, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[es])

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

acc = model.evaluate(X_test, y_test)[1]
print("\n 테스트 정확도: %.4f" % (acc))

import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')  # ‘bo’는 파란색 점을 의미합니다.
plt.plot(epochs, val_loss, 'b', label='Validation loss') # ‘b’는 파란색 실선을 의미합니다.
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf() # 그래프를 초기화합니다.
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()