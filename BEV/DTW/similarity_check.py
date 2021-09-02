import pandas as pd
import math
import numpy as np
import dtw

txt_name = ['BEV_result0', 'BEV_result1', 'BEV_result2']


def make_df_list(txt_name):
    result = pd.read_csv('../temp/' + txt_name + '.txt', delimiter=' ', header=None)
    result.columns = ['frame', 'id', 'x', 'y']

    ##### 임시로 수동 전처리 ##############
    '''
        result0.txt
        1. id 16, 23을 10으로 변경
        2. id 40, 42 제거
        
        result1.txt
        1. id 24, 33을 9로 변경
        2. id 13, 31, 36을 14로 변경
        3. id 32, 43을 17로 변경
        4. id 41제거
        
        result2.txt
        1. id 26을 12로 변경
        2. id 34를 20으로 변경
        3. id 5 제거
    '''

    if txt_name == 'BEV_result0':
        result['id'][(result['id'] == 16)] = 10
        result['id'][(result['id'] == 23)] = 10
        result.drop(result[result['id'] == 40].index, inplace=True)
        result.drop(result[result['id'] == 42].index, inplace=True)
    elif txt_name == 'BEV_result1':
        result['id'][(result['id'] == 24)] = 9
        result['id'][(result['id'] == 33)] = 9
        result['id'][(result['id'] == 13)] = 14
        result['id'][(result['id'] == 31)] = 14
        result['id'][(result['id'] == 36)] = 14
        result['id'][(result['id'] == 32)] = 17
        result['id'][(result['id'] == 43)] = 17
        result.drop(result[result['id'] == 41].index, inplace=True)
    elif txt_name == 'BEV_result2':
        result['id'][(result['id'] == 26)] = 12
        result['id'][(result['id'] == 34)] = 20
        result.drop(result[result['id'] == 5].index, inplace=True)
    ###################################

    id_df = result.drop_duplicates(['id'])
    id_list = id_df['id'].tolist()

    df_list = []

    for id in id_list:
        df = result[result['id'] == id]
        df_list.append(df)

    return df_list


def create_dist_list(df):
    frame_list = df['frame'].to_list()
    id = df['id'].iloc[0]
    x_list = df['x'].to_list()
    y_list = df['y'].to_list()

    # [[fist_frame, last_frame], id, [dist_list]]
    info_list = [[frame_list[0], frame_list[-1]], id]
    unit_vec_list = []

    # calculate unit vector
    for i in range(0, len(x_list) - 1):
        vec = np.array([x_list[i + 1] - x_list[i], y_list[i + 1] - y_list[i]])

        # For divide by 0
        if np.linalg.norm(vec) == 0:
            unit_vec = vec
        else:
            unit_vec = vec / np.linalg.norm(vec)
        unit_vec_list.append(unit_vec)

    unit_vec_list = np.array(unit_vec_list)
    info_list.append(unit_vec_list)
    return info_list


result0_df_list = make_df_list(txt_name[0])
result1_df_list = make_df_list(txt_name[1])
result2_df_list = make_df_list(txt_name[2])

result0_info_list = []
result1_info_list = []
result2_info_list = []

for df in result0_df_list:
    result0_info_list.append(create_dist_list(df))

for df in result1_df_list:
    result1_info_list.append(create_dist_list(df))

for df in result2_df_list:
    result2_info_list.append(create_dist_list(df))


def check_similarity(info, info_list1, info_list2):

    min1 = 999999
    idx1 = -1

    min2 = 999999
    idx2 = -1

    for k in info_list1:
        if info[0][0] < 150 and k[0][0] < 150:
            dist = dtw.dtw(k[2], info[2], keep_internals=True).distance
            print(dist)
            if min1 > dist:
                min1 = dist
                idx1 = k[1]
        elif info[0][0] < 150 and k[0][0] > 150:
            break
        elif info[0][0] > 150 and k[0][0] < 150:
            continue
        else:
            dist = dtw.dtw(k[2], info[2], keep_internals=True).distance
            if min1 > dist:
                min1 = dist
                idx1 = k[1]

    for k in info_list2:
        if info[0][0] < 150 and k[0][0] < 150:
            dist = dtw.dtw(k[0][2], info[2], keep_internals=True).distance
            if min2 > dist:
                min2 = dist
                idx2 = k[1]
        elif info[0][0] < 150 and k[0][0] > 150:
            break
        elif info[0][0] > 150 and k[0][0] < 150:
            continue
        else:
            dist = dtw.dtw(k[2], info[2], keep_internals=True).distance
            if min2 > dist:
                min2 = dist
                idx2 = k[1]

    return idx1, idx2

id_map_list = []
for info in result0_info_list:
    id_map = [info[1]]
    idx1, idx2 = check_similarity(info, result1_info_list, result2_info_list)
    if idx1 != -1:
        id_map.append(idx1)
    if idx2 != -1:
        id_map.append(idx2)
    id_map_list.append(id_map)

print(id_map_list)




#global_idx = 1000  # start