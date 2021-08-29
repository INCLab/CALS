import pandas as pd
import math
import numpy as np
import dtw

txt_name = ['result0', 'result1', 'result2']


def make_df_list(txt_name):
    result = pd.read_csv('../temp/' + txt_name + '.txt', delimiter=' ', header=None)
    result.columns = ['frame', 'id', 'x', 'y']

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


def check_similarity(info, info_list1, info_list2):
    threshold = 5  # 5 frame

    info1 = []
    dist = 999999
    for k in info_list1:
        # compare start frame
        if abs(info[0][0] - k[0][0]) < threshold:




