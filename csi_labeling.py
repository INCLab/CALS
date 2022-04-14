from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import pywt
from datetime import datetime
import time

TIME_THRESHOLD = 1

# ========= Read Data =========
test_name = 'test3_3rasp'
data_path = 'data'
csi_path = os.path.join(data_path, test_name, 'csi')
mot_path = os.path.join(data_path, test_name, 'mot')

# ========= Output data path ========
out_path = os.path.join(data_path, test_name, 'labeled')
os.makedirs(out_path, exist_ok=True)

# =========  Create file list  =========
csi_flist = os.listdir(csi_path)
mot_flist = os.listdir(mot_path)

# ========= video data: start_time, end_time ########
date, start, end = mot_flist[0].split('-')[1:-1]
mot_start_time = date + start
mot_end_time = date + end

# Unix time
mot_start_ut = time.mktime(datetime.strptime(mot_start_time, '%Y%m%d%H%M%S').timetuple())
mot_end_ut = time.mktime(datetime.strptime(mot_end_time, '%Y%m%d%H%M%S').timetuple())


'''
    Read all mot output files (e.g., ch01, ch02, ..., ch04)
    1. Count the number of detected person and make { time: total_person} dict by each files
    2. Then concatenate all list and remove duplicate: choose the highest total_person value
    3. return mot_dict
    mot_dict = {time1:total_person1, time2:total_person2, ...}
'''
def create_mot_dict(mot_file_list):
    # === 1 ===
    mot_dict_by_files = {}
    for mot_fname in mot_file_list:
        with open(os.path.join(mot_path, mot_fname), 'r') as f:
            data = f.read()
            row_list = data.split('\n')

            if row_list[-1] == '':
                row_list.remove('')


            tmp_dict = {}
            before_time = 0.0
            for row in row_list:
                t = float(row.split(' ')[0])

                if t != before_time:
                    tmp_dict[t] = 1
                    before_time = t
                else:
                    tmp_dict[t] += 1
            mot_dict_by_files[mot_fname] = tmp_dict

    # === 2 ===
    base_file = list(mot_dict_by_files.keys())[0]
    base_dict = mot_dict_by_files[base_file]

    if len(mot_dict_by_files.keys()) > 1:
        for file in mot_dict_by_files.keys():
            if file == base_file:
                continue

            for current_time in mot_dict_by_files[file].keys():
                if current_time in base_dict:
                    if mot_dict_by_files[file][current_time] > base_dict[current_time]:
                        base_dict[current_time] = mot_dict_by_files[file][current_time]
                else:
                    base_dict[current_time] = mot_dict_by_files[file][current_time]

        return dict(sorted(base_dict.items()))

    else:
        return base_dict


# Make time list
# timeList: [[Start_t, End_t, Label], [Start_t, End_t, Label], ...]
def make_time_list(mot_dict):
    motTimeList = list(mot_dict.keys())
    timeList = []

    # 비디오 시작 시간에 타겟이 없고, Time threshold 보다 긴 시간동안
    # 타겟이 등장하지 않는 경우 해당 시간은 label 0으로 처리
    if motTimeList[0] - mot_start_ut >= TIME_THRESHOLD:
        timeList.append([mot_start_ut, motTimeList[0], 0])
    # 비디오 시작 시간과 mot 시작시간에 차이가 있지만, Time threshold 보다 작을 경우
    # 해당 시간은 label 1으로 처리
    elif 0 < motTimeList[0] - mot_start_ut < TIME_THRESHOLD:
        timeList.append([mot_start_ut, motTimeList[0], 1])

    # initialize with 1st mot data
    label = mot_dict[motTimeList[0]]
    start_time = motTimeList[0]
    before_time = motTimeList[0]

    for idx, time in enumerate(motTimeList):
        current_label = mot_dict[time]

        if current_label == label and (time - before_time) < TIME_THRESHOLD:
            if idx == len(motTimeList) - 1:
                timeList.append([start_time, time, label])
            before_time = time
            continue
        # time threshold보다 크다는것은 중간에 타겟이 없었던 시간이 존재한다는 뜻
        elif current_label == label and (time - before_time) >= TIME_THRESHOLD:
            # 타겟이 존재했던 시간까지 timeList에 삽입
            timeList.append([start_time, before_time, label])
            # 타겟이 존재하지 않는 시간에 대해 timeList에 삽입
            timeList.append([before_time, time, 0])
            start_time = time
        # label이 바뀌고 threshold를 넘지 않는 경우
        elif current_label != label and (time - before_time) < TIME_THRESHOLD:
            # label이 바뀌기 이전시간까지 List에 삽입
            timeList.append([start_time, time, label])
            start_time = time
            label = current_label
        # label이 바뀌고 threshold를 넘는 경우
        elif current_label != label and (time - before_time) >= TIME_THRESHOLD:
            # 타겟이 존재했던 시간까지 timeList에 삽입
            timeList.append([start_time, before_time, label])
            # 타겟이 존재하지 않는 시간에 대해 timeList에 삽입
            timeList.append([before_time, time, 0])
            start_time = time
            label = current_label

        before_time = time

    # MOT output txt의 마지막 타임과 비디오 끝 타임의 간격이 Time threshold보다 클 경우
    # 해당 구간은 타겟이 없는것으로 labeling
    if mot_end_ut - before_time >= TIME_THRESHOLD:
        timeList.append([before_time, mot_end_ut, 0])
    # MOT output txt의 마지막 타임과 비디오 끝 타임의 간격이 존재하고, Time threshold보다 작을 경우
    # 해당 구간은 mot 마지막 label로 labeling
    elif 0 < mot_end_ut - before_time <  TIME_THRESHOLD:
        timeList.append([before_time, mot_end_ut, label])

    return timeList


# Make mot dict
mot_datas = create_mot_dict(mot_flist)

# Make time list
timeList = make_time_list(mot_datas)

for csi_file in csi_flist:
    csi_label_list = []
    mac = csi_file[9:-4]

    csi_df = pd.read_csv(os.path.join(csi_path, csi_file))

    # mac address column 제외
    df = csi_df.iloc[:, 1:]

    # Labeling CSI data
    for csi_data in df.values.tolist():
        csi_time = csi_data[0]

        # CSI Time이 MOT 시작 시간보다 빠를경우 -1로 labeling
        if csi_time < timeList[0][0]:
            csi_label_list.append(-1)
            continue
        # CSI Time이 MOT 끝 시간을 넘어간 경우 -1로 labeling
        elif csi_time >= timeList[-1][1]:
            csi_label_list.append(-1)
            continue

        # tset: [start_time, end_time, plabel]
        for tset in timeList:
            start_time, end_time, plabel = tset

            # Person exist label로 변경
            if plabel > 0:
                plabel = 1

            # 현재 iteration에서 csi time이 end time과 같거나 클경우 다음 iter로
            if csi_time >= end_time:
                continue
            # 시간 범위안에 들어오는경우 해당 label append
            elif start_time <= csi_time < end_time:
                csi_label_list.append(plabel)
    csi_df['label'] = csi_label_list

    csi_df.to_csv(os.path.join(out_path, 'labeled_csi_data_{}.csv'.format(mac)), index=False)


# csi_df = db.get_csi_df()
#
# '''
#     <csi_df>
#     Feature: 64 Subcarriers(-32~32), Elements: Amplitude
# '''
# csi_df['label'] = label_list
#
# # Drop unnecessary rows (label '-1')
# csi_df.drop(csi_df[csi_df['label'] == -1].index, inplace=True)
# csi_df.reset_index(drop=True, inplace=True)
#
#
# # ============= Preprocessing ================
#
# '''
#     0. Set time window
#     1. Drop null subcarriers
#     2. Denoising with DWT(Discrete Wavelet Transform)
#     3. Normalization
#     4. extract dynamic moving feature with AST(Amplitude Shape Trend)
# '''
# # Drop timestamp
# csi_df.drop([csi_df.columns[0]], axis=1, inplace=True)
#
# # 0. Set time window (n second)
# packets_ps = 100  # packets per second
# n_second = 3
# time_window =  packets_ps * n_second
# tw_list = []  # time window list
#
# # Todo: time window 내에서 label이 바뀌는경우에 대한 전처리를 어떻게 할것인지,
# # solution: FairMOT result와 csi sync 과정에서 FairMOT label에 따라서
# # sync 데이터 따로 만들기. 이때 time window에 포함되는 csi data가 label이 통일 되지 않는경우 drop
#
#
# # 1. Drop null subcarriers
# # Indexes of Null and Pilot OFDM subcarriers
# # {-32, -31, -30, -29, 31, 30, 29, 0}
# null_idx = [-32, -31, -30, -29, 31, 30, 29, 0]
# null_idx = ['_' + str(idx + 32) for idx in null_idx]
#
# for idx in null_idx:
#     csi_df.drop(idx, axis=1, inplace=True)
#
# # 2. Denoising with DWT
# # pywt.dwt()
# # https://hyongdoc.tistory.com/429wanton
#
#
# def lowpassfilter(signal, thresh=0.63, wavelet="db4"):
#     thresh = thresh * np.nanmax(signal)
#     coeff = pywt.wavedec(signal, wavelet, mode="per", level=8)
#     coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
#     reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
#     return reconstructed_signal
#
#
# for sub_idx in csi_df.columns[:-1]:
#     csi_df[sub_idx] = lowpassfilter(np.abs(csi_df[sub_idx]), 0.3)
#
# # 3. Normalization
# scaler = MinMaxScaler()
# scaler.fit(csi_df.iloc[:, 0:-1])
# scaled_df = scaler.transform(csi_df.iloc[:, 0:-1])
# csi_df.iloc[:, 0:-1] = scaled_df




