import math
import numpy as np
import dtw
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

FRAME_THRESHOLD = 20

def make_df_list(filename):
    result = pd.read_csv('../temp/' + filename + '.txt', delimiter=' ', header=None)
    result.columns = ['frame', 'id', 'x', 'y']

    ##### 임시로 수동 전처리 ##############
    '''
        result0.txt
        1. id 5를 7으로 변경
        2. id 2 제거

        result1.txt
        1. id 1을 8로변경
        4. id 9제거

        result2.txt
        1. id 4 제거
    '''

    if filename == 'BEV_result0':
        result['id'][(result['id'] == 5)] = 7
        result.drop(result[result['id'] == 2].index, inplace=True)
        result.drop(result[result['id'] == 1].index, inplace=True)
    elif filename == 'BEV_result1':
        result['id'][(result['id'] == 1)] = 8
        result.drop(result[result['id'] == 9].index, inplace=True)

        result['id'][(result['id'] == 3)] = 13
        result['id'][(result['id'] == 5)] = 15
        result['id'][(result['id'] == 8)] = 18
    elif filename == 'BEV_result2':
        result['id'][(result['id'] == 1)] = 21
        result['id'][(result['id'] == 2)] = 22
        result['id'][(result['id'] == 3)] = 23
        result.drop(result[result['id'] == 4].index, inplace=True)
    ###################################

    id_df = result.drop_duplicates(['id'])
    id_list = id_df['id'].tolist()

    df_list = []

    for id in id_list:
        df = result[result['id'] == id]
        df_list.append(df)

    return df_list, id_list


# ########## Create Feature for DTW #################
def create_unit_vec(df, threshold):
    frame_list = df['frame'].to_list()
    id = df['id'].iloc[0]
    x_list = df['x'].to_list()
    y_list = df['y'].to_list()

    # return form : [frame_list[:-1], id, [dist_list]]
    info_list = [frame_list[:-1], id]
    unit_vec_list = []

    # calculate unit vector
    for i in range(0, len(x_list) - 1):
        if frame_list[i+1] - frame_list[i] > threshold:
            continue
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


def create_scalar(df, threshold):
    # Min-Max normalization
    scaler = MinMaxScaler()
    scaler.fit(df.iloc[:, 2:])
    scaled_df = scaler.transform(df.iloc[:, 2:])
    df.iloc[:, 2:] = scaled_df

    frame_list = df['frame'].to_list()
    id = df['id'].iloc[0]
    x_list = df['x'].to_list()
    y_list = df['y'].to_list()

    # return form : [frame_list[:-1], id, [dist_list]]
    info_list = [frame_list[:-1], id]
    scalar_list = []

    # calculate distance
    for i in range(0, len(x_list) - 1):
        if frame_list[i+1] - frame_list[i] > threshold:
            continue
        dist = math.sqrt((x_list[i + 1] - x_list[i]) ** 2 + (y_list[i + 1] - y_list[i]) ** 2)
        scalar_list.append(dist)

    info_list.append(scalar_list)

    return info_list


def create_vec(df, threshold):
    frame_list = df['frame'].to_list()
    id = df['id'].iloc[0]
    x_list = df['x'].to_list()
    y_list = df['y'].to_list()

    # return form : [frame_list[:-1], id, [dist_list]]
    info_list = [frame_list[:-1], id]
    vec_list = []

    # calculate unit vector
    for i in range(0, len(x_list) - 1):
        if frame_list[i+1] - frame_list[i] > threshold:
            continue
        vec = np.array([x_list[i + 1] - x_list[i], y_list[i + 1] - y_list[i]])
        vec_list.append(vec)

    vec_list = np.array(vec_list)
    info_list.append(vec_list)

    return info_list
#########################################################################


# Select 1.unit vector or 2. normalized scalar or 3. vector
# Default: unit vector
def select_feature(result_df_list, info_list, feature='unit'):
    if feature == 'unit':
        for df_list in result_df_list:
            info = []
            for df in df_list:
                info.append(create_unit_vec(df, FRAME_THRESHOLD))
            info_list.append(info)
    elif feature == 'scalar':
        for df_list in result_df_list:
            info = []
            for df in df_list:
                info.append(create_scalar(df, FRAME_THRESHOLD))
            info_list.append(info)
    elif feature == 'vector':
        for df_list in result_df_list:
            info = []
            for df in df_list:
                info.append(create_vec(df, FRAME_THRESHOLD))
            info_list.append(info)

    return


def check_similarity(info_list, compare_list):
    '''
        기준 아이디에 대해 다른 result 파일에서 나온 모든 아이디들과 케이스별로 유사도 측정(혹은 제외) 후,
        DTW distance를 모두 저장
    '''

    # list of total dtw distance info
    result_list = []
    for _ in range(0, len(compare_list)):
        result_list.append([])

    # Loop for each result file
    for info in info_list:
        for i in range(0, len(compare_list)):
            for k in compare_list[i]:

                # *** 겹치지 않는경우: 일단 제외한다
                if info[0][0] > k[0][-1] or info[0][-1] < k[0][0]:
                    continue

                # *** 포함하는 경우 : DTW로 유사도 측정
                # case 1
                elif info[0][0] <= k[0][0] and info[0][-1] >= k[0][-1]:
                    dist = dtw_overlap_frames(info, k, 1)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compared_id, DTW_dist]
                # case 2
                elif info[0][0] >= k[0][0] and info[0][-1] <= k[0][-1]:
                    dist = dtw_overlap_frames(info, k, 2)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compared_id, DTW_dist]

                # *** 절반이상 겹치는 경우 : DTW로 유사도 측정
                # case 3
                elif info[0][0] >= k[0][0] and info[0][int(len(info[0]) / 2)] <= k[0][-1] <= info[0][-1]:
                    dist = dtw_overlap_frames(info, k, 3)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compared_id, DTW_dist]
                # case 4
                elif info[0][0] <= k[0][0] and k[0][int(len(k[0]) / 2)] <= info[0][-1] <= k[0][-1]:
                    dist = dtw_overlap_frames(info, k, 4)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compared_id, DTW_dist]

                # *** 절반이하로 겹치는 경우: 제외?(포함하려면 위 코드와 합치기)
                elif k[0][0] <= info[0][0] < k[0][-1] < info[0][int(len(info[0]) / 2)]:
                    dist = dtw_overlap_frames(info, k, 3)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compared_id, DTW_dist]
                elif info[0][0] <= k[0][0] < info[0][-1] < k[0][int(len(k[0]) / 2)]:
                    dist = dtw_overlap_frames(info, k, 4)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compared_id, DTW_dist]
                else:
                    print('Not matching case!!!!')

    return result_list


'''
    이동경로를 비교할때 overlap 되는 frame에 해당하는 feature들만 골라서 DTW 적용 
'''


def dtw_overlap_frames(x_id_info, y_id_info, case):
    dist = -1
    x_frame_list = x_id_info[0]
    y_frame_list = y_id_info[0]

    x_vec_list = x_id_info[2]
    y_vec_list = y_id_info[2]

    start_idx = 0
    end_idx = 0

    # Case 1,2: 포함하는 경우
    if case == 1:
        try:
            start_idx = x_frame_list.index(y_frame_list[0])
        except:
            min = 99999
            for i in range(0, len(x_frame_list)):
                if abs(x_frame_list[i] - y_frame_list[0]) < min:
                    min = abs(x_frame_list[i] - y_frame_list[0])
                    start_idx = i
        try:
            end_idx = x_frame_list.index(y_frame_list[-1])
        except:
            min = 99999
            for i in range(0, len(x_frame_list)):
                if abs(x_frame_list[i] - y_frame_list[-1]) < min:
                    min = abs(x_frame_list[i] - y_frame_list[-1])
                    end_idx = i
        dist = dtw.dtw(x_vec_list[start_idx:end_idx + 1], y_vec_list, keep_internals=True).distance

    elif case == 2:
        try:
            start_idx = y_frame_list.index(x_frame_list[0])
        except:
            min = 99999
            for i in range(0, len(y_frame_list)):
                if abs(y_frame_list[i] - x_frame_list[0]) < min:
                    min = abs(y_frame_list[i] - x_frame_list[0])
                    start_idx = i
        try:
            end_idx = y_frame_list.index(x_frame_list[-1])
        except:
            min = 99999
            for i in range(0, len(y_frame_list)):
                if abs(y_frame_list[i] - x_frame_list[-1]) < min:
                    min = abs(y_frame_list[i] - x_frame_list[-1])
                    end_idx = i
        dist = dtw.dtw(x_vec_list, y_vec_list[start_idx:end_idx + 1], keep_internals=True).distance
    # Case 3,4: 겹치는 경우
    elif case == 3:
        try:
            start_idx = y_frame_list.index(x_frame_list[0])
        except:
            min = 99999
            for i in range(0, len(y_frame_list)):
                if abs(y_frame_list[i] - x_frame_list[0]) < min:
                    min = abs(y_frame_list[i] - x_frame_list[0])
                    start_idx = i
        try:
            end_idx = x_frame_list.index(y_frame_list[-1])
        except:
            min = 99999
            for i in range(0, len(x_frame_list)):
                if abs(x_frame_list[i] - y_frame_list[-1]) < min:
                    min = abs(x_frame_list[i] - y_frame_list[-1])
                    end_idx = i
        dist = dtw.dtw(x_vec_list[:end_idx], y_vec_list[start_idx:], keep_internals=True).distance

    elif case == 4:
        try:
            start_idx = x_frame_list.index(y_frame_list[0])
        except:
            min = 99999
            for i in range(0, len(x_frame_list)):
                if abs(x_frame_list[i] - y_frame_list[0]) < min:
                    min = abs(x_frame_list[i] - y_frame_list[0])
                    start_idx = i
        try:
            end_idx = y_frame_list.index(x_frame_list[-1])
        except:
            min = 99999
            for i in range(0, len(y_frame_list)):
                if abs(y_frame_list[i] - x_frame_list[-1]) < min:
                    min = abs(y_frame_list[i] - x_frame_list[-1])
                    end_idx = i
        dist = dtw.dtw(x_vec_list[start_idx:], y_vec_list[:end_idx], keep_internals=True).distance

    return dist


# Todo: result1, 2 의 similarity도 반영해서 id mapping을 진행해야함
def id_mapping(distance_list, mapping_list):
    for dist_list in distance_list:
        sorted_list = sorted(dist_list, key=lambda x: (x[2], x[0]))

        while sorted_list:
            compare_id = sorted_list[0][0]
            compared_id = sorted_list[0][1]

            check = 0
            for ids in mapping_list:
                if compare_id in ids:
                    if compared_id not in ids:
                        ids.append(compared_id)
                    check = 1
                    break
                if compared_id in ids:
                    if compare_id not in ids:
                        ids.append(compare_id)
                    check = 1
                    break

            if check == 0:
                mapping_list.append([compare_id, compared_id])

            # Delete duplicate id list
            sorted_list = [i for i in sorted_list if not compare_id in i]
            sorted_list = [i for i in sorted_list if not compared_id in i]

    return


# Input: dataframe list by camera & id
# Output: Same as input but id == global_id
def change_to_global(T_set, id_set, gid_set):
    for T in T_set:
        for id_info in T:
            for i in range(0, len(id_set)):
                if id_info['id'][0] in id_set[i]:
                    id_info['id'] = gid_set[i]
                    break
    return


# Input: total_info = [[frame, id, x, y], ...]
def generate_global_info(total_info):
    I_G = list()

    accum_x = [0, 0]  # [accumulative x, number of target]
    accum_y = [0, 0]

    for i in range(0, len(total_info)):
        if i != len(total_info) - 1 and total_info[i][0] == total_info[i + 1][0] and total_info[i][1] == \
                total_info[i + 1][1]:
            accum_x[0] += total_info[i][2]
            accum_y[0] += total_info[i][3]

            accum_x[1] += 1
            accum_y[1] += 1
        else:
            if accum_x[1] != 0:
                accum_x[0] += total_info[i][2]
                accum_y[0] += total_info[i][3]

                accum_x[1] += 1
                accum_y[1] += 1

                avg_x = int(accum_x[0] / accum_x[1])
                avg_y = int(accum_y[0] / accum_y[1])

                I_G.append([total_info[i][0], total_info[i][1], avg_x, avg_y])

                # init
                accum_x = [0, 0]
                accum_y = [0, 0]
            else:
                I_G.append(total_info[i])

    return I_G
