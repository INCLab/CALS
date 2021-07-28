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

print(label_list)
csi_df = db.get_csi_df()

csi_df['label'] = label_list
print(csi_df)

# Define number of features
feature_num = 100

# Define subcarrier index
'''
    Null carrier index: 0, 1, 2, 3, 32 , 61, 62, 63
'''
sub_list = [7, 8, 14, 15, 20, 21, 40, 41, 54, 55]

'''
    Create label '1(one person)' data 
'''
for sub_idx in range(0, len(csi_df.columns)):
    packet_num = 0
    subcarrier_array = []

    while (packet_num <= len(csi_df) - feature_num):
        if packet_num == 0:
            subcarrier = csi_df[sub_idx].iloc[packet_num:packet_num + feature_num]
            sub_array = np.array(subcarrier)
            subcarrier_array = sub_array.reshape(1, -1)
        else:
            subcarrier = csi_df[sub_idx].iloc[packet_num:packet_num + feature_num]
            sub_array = np.array(subcarrier)
            sub_array = sub_array.reshape(1, -1)

            subcarrier_array = np.concatenate((subcarrier_array, sub_array), axis=0)

        packet_num += 100

    new_df = pd.DataFrame(subcarrier_array)

    # Add label '1(one person)' column in dataframe
    one_label = [1 for i in range(0, len(new_df))]
    new_df['label'] = one_label
    one_data_list.append(new_df)

