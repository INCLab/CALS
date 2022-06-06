import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataloader import DataLoader
from numpy import array
from sklearn.model_selection import train_test_split
from model_plot import model_train_plot, corr_heatmap

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
# pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, ['_30', '_31'], filter=True)
pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))


# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]

# Display correlation
# corr = csi_data.corr()
# corr_heatmap(corr)


# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

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

TIMESTEMP = 50
MAX_EPOCHS = 100

inp = (-1, X_train.shape[1], 1)

X_train = X_train.reshape(inp)  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
X_valid = X_valid.reshape(inp)
X_test = X_test.reshape(inp)

print('X reshape: {}'.format(X_train.shape))

model = tf.keras.Sequential([
    tf.keras.layers.GRU(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
    tf.keras.layers.SimpleRNN(52, 'relu', kernel_regularizer='l1'),
    tf.keras.layers.Dense(1, 'sigmoid')
])

model.summary()

learning_rate = 1e-3
decay = learning_rate / MAX_EPOCHS

optimizer = Adam(
    learning_rate=learning_rate,
    decay=decay
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid), callbacks=[es])

acc = model.evaluate(X_test, y_test)[1]
print("\n 테스트 정확도: %.4f" % (acc))

# model save
#model.save("cnn1d_model")

# plot train process
model_train_plot(history)

