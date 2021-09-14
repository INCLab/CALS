import pandas as pd
import math
import numpy as np
import dtw

txt_name = ['BEV_result0', 'BEV_result1', 'BEV_result2']
FRAME_THRESHOLD = 40

# 정익:2,6,9,10,12  민재:1,8,11,14,18  찬영:3,4,17,19,20
GT = [[1, 8], [11, 14, 18], [2, 6], [9, 10, 12], [3, 4], [17, 19, 20]]  # Ground Truth


def make_df_list(filename):
    result = pd.read_csv('../temp/' + filename + '.txt', delimiter=' ', header=None)
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

    if filename == 'BEV_result0':
        result['id'][(result['id'] == 16)] = 10
        result['id'][(result['id'] == 23)] = 10
        result.drop(result[result['id'] == 40].index, inplace=True)
        result.drop(result[result['id'] == 42].index, inplace=True)
    elif filename == 'BEV_result1':
        result['id'][(result['id'] == 24)] = 9
        result['id'][(result['id'] == 33)] = 9
        result['id'][(result['id'] == 13)] = 14
        result['id'][(result['id'] == 31)] = 14
        result['id'][(result['id'] == 36)] = 14
        result['id'][(result['id'] == 32)] = 17
        result['id'][(result['id'] == 43)] = 17
        result.drop(result[result['id'] == 41].index, inplace=True)
    elif filename == 'BEV_result2':
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

    result_list = []

    for df in df_list:
        result_list += divide_df(df)

    return result_list, id_list


# If dataframe is spaced more than threshold, divide it
def divide_df(dataframe, frame_threshold=FRAME_THRESHOLD):
    list_by_row = []
    div_idx_list = []

    for i in range(len(dataframe)):
        list_by_row.append(dataframe.iloc[i].to_list())

    # Check frame interval
    for j in range(1, len(list_by_row)):
        if frame_threshold < list_by_row[j][0] - list_by_row[j - 1][0]:  # frame interval
            div_idx_list.append(j)

    # If elements in div_idx_list are consecutive, it means front div point consist dataframe ifself.
    # So, discard front point
    if len(div_idx_list) > 1:
        for k in range(len(div_idx_list) - 1):
            if div_idx_list[k] + 1 == div_idx_list[k + 1]:
                div_idx_list[k] = -1

    remove_idx = {-1}
    div_idx = [i for i in div_idx_list if i not in remove_idx]

    # Divide dataframe
    df_list = []
    if div_idx:
        for i in range(len(div_idx)):
            if i == 0:
                df_list.append(list_by_row[:div_idx[i]])
            else:
                df_list.append(list_by_row[div_idx[i - 1]:div_idx[i]])
        df_list.append(list_by_row[div_idx[-1]:])
    else:
        df_list.append(list_by_row)

    result_df = []
    for rows in df_list:
        result_df.append(pd.DataFrame(rows, columns=['frame', 'id', 'x', 'y']))

    return result_df


# Create Feature for DTW
def create_unit_vec(df):
    frame_list = df['frame'].to_list()
    id = df['id'].iloc[0]
    x_list = df['x'].to_list()
    y_list = df['y'].to_list()

    # return form : [frame_list[:-1], id, [dist_list]]
    info_list = [frame_list[:-1], id]
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
                elif info[0][0] < k[0][0] and info[0][-1] > k[0][-1]:
                    dist = dtw_overlap_frames(info, k, 1)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]
                # case 2
                elif info[0][0] > k[0][0] and info[0][-1] < k[0][-1]:
                    dist = dtw_overlap_frames(info, k, 2)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]

                # *** 절반이상 겹치는 경우 : DTW로 유사도 측정
                # case 3
                elif info[0][0] > k[0][0] and info[0][int(len(info[0]) / 2)] <= k[0][-1] <= info[0][-1]:
                    dist = dtw_overlap_frames(info, k, 3)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]
                # case 4
                elif info[0][0] < k[0][0] and k[0][int(len(k[0]) / 2)] <= info[0][-1] <= k[0][-1]:
                    dist = dtw_overlap_frames(info, k, 4)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]

                # *** 절반이하로 겹치는 경우: 제외?(포함하려면 위 코드와 합치기)
                elif k[0][0] < info[0][0] < k[0][-1] < info[0][int(len(info[0]) / 2)]:
                    dist = dtw_overlap_frames(info, k, 3)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]
                elif info[0][0] < k[0][0] < info[0][-1] < k[0][int(len(k[0]) / 2)]:
                    dist = dtw_overlap_frames(info, k, 4)
                    result_list[i].append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]

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


# Create Dataframes by id
result_df_list = []
total_id_list = []
for name in txt_name:
    df_list, id_list = make_df_list(name)
    result_df_list.append(df_list)
    total_id_list.append(id_list)

# Create id info list
result_info_list = []
for df_list in result_df_list:
    info = []
    for df in df_list:
        info.append(create_unit_vec(df))
    result_info_list.append(info)

# Create high similarity ID list
for i in range(0, len(result_info_list)-1):
    id_map_list = []
    result_dist_list = check_similarity(result_info_list[i], result_info_list[i+1:])
    id_mapping(result_dist_list, id_map_list)
    print(id_map_list)

# Accurate

# Assign global id
