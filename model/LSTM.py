import numpy as np
import pandas as pd
from dataloader import DataLoader

from sklearn.model_selection import train_test_split

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

# Sampling
SAMPLE_NUM = 10
X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]


print('X shape: {}'.format(X_train.shape))
print('y shape: {}'.format(y_train.shape))

# sentences = ["i like dog", "i love coffee", "i hate milk", "you like cat", "you love milk", "you hate coffee"]
# dtype = torch.float
#
# """
# Word Processing
# """
# word_list = list(set(" ".join(sentences).split()))
# word_dict = {w: i for i, w in enumerate(word_list)}
# number_dict = {i: w for i, w in enumerate(word_list)}
# n_class = len(word_dict)
#
# """
# TextRNN Parameter
# """
# batch_size = len(sentences)
# n_step = 2  # 학습 하려고 하는 문장의 길이 - 1
# n_hidden = 5  # 은닉층 사이즈
#
#
# def make_batch(sentences):
#     input_batch = []
#     target_batch = []
#
#     for sen in sentences:
#         word = sen.split()
#         input = [word_dict[n] for n in word[:-1]]
#         target = word_dict[word[-1]]
#
#         input_batch.append(np.eye(n_class)[input])  # One-Hot Encoding
#         target_batch.append(target)
#
#     return input_batch, target_batch
#
#
# input_batch, target_batch = make_batch(sentences)
# input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True)
# target_batch = torch.tensor(target_batch, dtype=torch.int64)
#
# """
# TextLSTM
# """
#
#
# class TextLSTM(nn.Module):
#     def __init__(self):
#         super(TextLSTM, self).__init__()
#
#         self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, dropout=0.3)
#         self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
#         self.b = nn.Parameter(torch.randn([n_class]).type(dtype))
#         self.Softmax = nn.Softmax(dim=1)
#
#     def forward(self, hidden_and_cell, X):
#         X = X.transpose(0, 1)
#         outputs, hidden = self.lstm(X, hidden_and_cell)
#         outputs = outputs[-1]  # 최종 예측 Hidden Layer
#         model = torch.mm(outputs, self.W) + self.b  # 최종 예측 최종 출력 층
#         return model
#
#
# """
# Training
# """
# model = TextLSTM()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# for epoch in range(500):
#     hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
#     cell = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
#     output = model((hidden, cell), input_batch)
#     loss = criterion(output, target_batch)
#
#     if (epoch + 1) % 100 == 0:
#         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# input = [sen.split()[:2] for sen in sentences]
#
# hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
# cell = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
# predict = model((hidden, cell), input_batch).data.max(1, keepdim=True)[1]
# print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
TIMESTEMP = 10

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # LSTM은 input으로 3차원 (datasize, timestamp, feature)
print('X reshape: {}'.format(X_train.shape))

model = Sequential()
model.add(LSTM(10, activation='sigmoid', input_shape=(TIMESTEMP, 1), return_sequences=True))
model.add(LSTM(10, activation='sigmoid', return_sequences=True))  # (None, TIMESTEMP, 10)을 받는다
model.add(LSTM(10, activation='sigmoid', return_sequences=True))
model.add(LSTM(10, activation='sigmoid', return_sequences=True))
model.add(LSTM(10, activation='sigmoid', return_sequences=True))
model.add(LSTM(10, activation='sigmoid', return_sequences=True))
model.add(LSTM(10, activation='sigmoid', return_sequences=True))
model.add(LSTM(10, activation='sigmoid', return_sequences=True))
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
