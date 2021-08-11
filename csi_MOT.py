from database.tracking_db import tracking_db
import numpy as np
import pandas as pd

# =========  CSI labeling  =========
label = {}
label_list = []
threshold = 0.9  # error threshold
db = tracking_db()
csi_datas = db.get_csi()
mot_datas = dict(db.get_mot())


# Datalist에서 Value와 가장 유사한 값을 찾음
def matching_values(datalist, value):
    # Threshold 범위 내의 값만 Numpy Array에 담음
    array = datalist[np.where(abs(datalist - value) <= threshold)]

    # 해당하는 값이 하나 이상이면
    if array.size > 0:
        # 그 중 가장 작은 값(minimum error)의 Index를 리턴
        minvalue = np.argmin(abs(array - value))

        return array[minvalue]
    else:
        # 하나도 없다면 -1
        return -1


# MOT Timestamp List (nparray)
mot_times = np.asarray(list(map((lambda x: x), mot_datas.keys())))

for csi_data in csi_datas:
    # 두 값을 비교해 현재 CSI Time과 가장 유사한 MOT Time을 구함
    compare = matching_values(mot_times, csi_data[0])

    # -1이면 매칭 실패
    if compare == -1:
        label[csi_data[0]] = -1
        label_list.append(-1)
    # 그 외에는 해당하는 MOT Timestamp가 Return 되므로 그 값을 Label에 담음
    else:
        target_num = mot_datas[compare]  # total target number in the frame
        exist_label = int()  # (target_num == 0, 0) ,(target_num > 0, 1)

        if target_num == 0:
            exist_label = 0
        elif target_num > 0:
            exist_label = 1

        label[csi_data[0]] = exist_label
        label_list.append(exist_label)

        # 이미 뽑은건 제거
        mot_times = np.delete(mot_times, np.argwhere(mot_times == compare))


csi_df = db.get_csi_df()

'''
    <csi_df>
    Feature: 64 Subcarriers, Elements: Amplitude 
'''
csi_df['label'] = label_list

# Drop unnecessary rows (label '-1')
csi_df.drop(csi_df[csi_df['label'] == -1].index, inplace=True)
csi_df.reset_index(drop=True, inplace=True)


# ============= Preprocessing ================

'''
    1. Normalization
    2. Butterworth filter
'''

# ============= Using all subcarrier Ver. ==============
from csi_ML_train import train_rf
from csi_DL_train import deep_model
#train_rf(csi_df)
deep_model(csi_df)




