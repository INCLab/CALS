import pandas as pd
import math
import numpy as np
import dtw

txt_name = ['result0', 'result1', 'result2']


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

    if txt_name == 'result0':
        result['id'][(result['id'] == 16)] = 10
        result['id'][(result['id'] == 23)] = 10
        result.drop(result[result['id'] == 40].index, inplace=True)
        result.drop(result[result['id'] == 42].index, inplace=True)
    elif txt_name == 'result1':
        result['id'][(result['id'] == 24)] = 9
        result['id'][(result['id'] == 33)] = 9
        result['id'][(result['id'] == 13)] = 14
        result['id'][(result['id'] == 31)] = 14
        result['id'][(result['id'] == 36)] = 14
        result['id'][(result['id'] == 32)] = 17
        result['id'][(result['id'] == 43)] = 17
        result.drop(result[result['id'] == 41].index, inplace=True)
    elif txt_name == 'result2':
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
    info_list = [[frame_list[0], frame_list[-1]], id, []]

    # calculate Euclidean distance
    for i in range(0, len(x_list) - 1):
        dist = math.sqrt((x_list[i + 1] - x_list[i]) ** 2 + (y_list[i + 1] - y_list[i]) ** 2)
        info_list[2].append(dist)

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

print(dtw.dtw(result0_info_list[0][2], result1_info_list[0][2], keep_internals=True).distance)


# def check_similarity(info, info_list1, info_list2):
#     threshold = 5  # 5 frame
#
#     info1 = []
#     dist = 999999
#     for k in info_list1:
#         # compare start frame
#         if abs(info[0][0] - k[0][0]) < threshold:




