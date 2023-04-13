from database.csi_db import csi_db
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pywt

# =========  CSI labeling  =========
db = csi_db()
csi_label_list = []

csi_datas = db.get_csi()
mot_datas = dict(db.get_mot())


# Make time list
# timeList: [[Start_t, End_t, Label], [Start_t, End_t, Label], ...]
def make_time_list(mot_dict):
    motTimeList = list(mot_dict.keys())
    timeList = []

    # initialize with 1st mot data
    label = mot_dict[motTimeList[0]]
    start_time = motTimeList[0]

    for idx, time in enumerate(motTimeList):
        current_label = mot_dict[time]

        if current_label == label:
            if idx == len(motTimeList) - 1:
                timeList.append([start_time, time, label])
            continue
        else:
            timeList.append([start_time, time, label])
            start_time = time
            label = current_label

    return timeList


# Make time list
timeList = make_time_list(mot_datas)

# Labeling CSI data
for csi_data in csi_datas:
    csi_time = csi_data[0]

    # CSI Time이 MOT 시작 시간보다 빠를경우 -1로 labeling
    if csi_time < timeList[0][0]:
        csi_label_list.append(-1)
        continue
    # CSI Time이 MOT 끝 시간을 넘어간 경우 -1로 labeling
    elif csi_time >= timeList[-1][1]:
        csi_label_list.append(-1)
        continue

    # tset: [start_time, end_time, label]
    for tset in timeList:
        start_time, end_time, label = tset

        # 현재 iteration에서 csi time이 end time과 같거나 클경우 다음 iter로
        if csi_time >= end_time:
            continue
        # 시간 범위안에 들어오는경우 해당 label append
        elif start_time <= csi_time < end_time:
            csi_label_list.append(label)

# # Datalist에서 Value와 가장 유사한 값을 찾음
# def matching_values(datalist, value):
#     # Threshold 범위 내의 값만 Numpy Array에 담음
#     array = datalist[np.where(abs(datalist - value) <= threshold)]
#
#     # 해당하는 값이 하나 이상이면
#     if array.size > 0:
#         # 그 중 가장 작은 값(minimum error)의 Index를 리턴
#         minvalue = np.argmin(abs(array - value))
#
#         return array[minvalue]
#     else:
#         # 하나도 없다면 -1
#         return -1
#
#
# # MOT Timestamp List (nparray)
# mot_times = np.asarray(list(map((lambda x: x), mot_datas.keys())))
#
# for csi_data in csi_datas:
#     # 두 값을 비교해 현재 CSI Time과 가장 유사한 MOT Time을 구함
#     compare = matching_values(mot_times, csi_data[0])
#
#     # -1이면 매칭 실패
#     if compare == -1:
#         label[csi_data[0]] = -1
#         label_list.append(-1)
#     # 그 외에는 해당하는 MOT Timestamp가 Return 되므로 그 값을 Label에 담음
#     else:
#         target_num = mot_datas[compare]  # total target number in the frame
#         exist_label = int()  # (target_num == 0, 0) ,(target_num > 0, 1)
#
#         if target_num == 0:
#             exist_label = 0
#         elif target_num > 0:
#             exist_label = 1
#
#         label[csi_data[0]] = exist_label
#         label_list.append(exist_label)
#
#         # 이미 뽑은건 제거
#         mot_times = np.delete(mot_times, np.argwhere(mot_times == compare))


csi_df = db.get_csi_df()

'''
    <csi_df>
    Feature: 64 Subcarriers(-32~32), Elements: Amplitude 
'''
csi_df['label'] = label_list

# Drop unnecessary rows (label '-1')
csi_df.drop(csi_df[csi_df['label'] == -1].index, inplace=True)
csi_df.reset_index(drop=True, inplace=True)


# ============= Preprocessing ================

'''
    0. Set time window
    1. Drop null subcarriers
    2. Denoising with DWT(Discrete Wavelet Transform)
    3. Normalization
    4. extract dynamic moving feature with AST(Amplitude Shape Trend)
'''
# Drop timestamp
csi_df.drop([csi_df.columns[0]], axis=1, inplace=True)

# 0. Set time window (n second)
packets_ps = 100  # packets per second
n_second = 3
time_window =  packets_ps * n_second
tw_list = []  # time window list

# Todo: time window 내에서 label이 바뀌는경우에 대한 전처리를 어떻게 할것인지,
# solution: FairMOT result와 csi sync 과정에서 FairMOT label에 따라서
# sync 데이터 따로 만들기. 이때 time window에 포함되는 csi data가 label이 통일 되지 않는경우 drop


# 1. Drop null subcarriers
# Indexes of Null and Pilot OFDM subcarriers
# {-32, -31, -30, -29, 31, 30, 29, 0}
null_idx = [-32, -31, -30, -29, 31, 30, 29, 0]
null_idx = ['_' + str(idx + 32) for idx in null_idx]

for idx in null_idx:
    csi_df.drop(idx, axis=1, inplace=True)

# 2. Denoising with DWT
# pywt.dwt()
# https://hyongdoc.tistory.com/429wanton


def lowpassfilter(signal, thresh=0.63, wavelet="db4"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per", level=8)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal


for sub_idx in csi_df.columns[:-1]:
    csi_df[sub_idx] = lowpassfilter(np.abs(csi_df[sub_idx]), 0.3)

# 3. Normalization
scaler = MinMaxScaler()
scaler.fit(csi_df.iloc[:, 0:-1])
scaled_df = scaler.transform(csi_df.iloc[:, 0:-1])
csi_df.iloc[:, 0:-1] = scaled_df


# # 4. AST
#
# # ==========================================
# from csi_ML_train import train_rf
# from csi_DL_train import deep_model
# #train_rf(csi_df)
# deep_model(csi_df)



