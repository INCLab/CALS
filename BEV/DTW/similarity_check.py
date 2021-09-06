import pandas as pd
import numpy as np
import dtw

txt_name = ['BEV_result0', 'BEV_result1', 'BEV_result2']
FRAME_THRESHOLD = 40

# 정익:2,6,9,10,12  민재:1,8,11,14,18  찬영:3,4,17,19,20
GT = [[1, 8, 11, 14, 18], [2, 6, 9, 10, 12], [3, 4, 17, 19, 20]]  # Ground Truth


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

    result_list = []

    for df in df_list:
        result_list += divide_df(df)

    return result_list


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


# ToDO: 유사도 비교에 대한 모든 경우의 수 리스트를 return 해야함, arg를 단일 df로 받으면 안됨
def check_similarity(info, info_list):
    '''
        기준 아이디에 대해 다른 result 파일에서 나온 모든 아이디들과 케이스별로 유사도 측정(혹은 제외) 후,
        DTW distance를 모두 저장한 뒤에 마지막에 가장 낮은 distance 값을 갖는 id쌍 부터 차례대로 mapping하며
        나머지는 지움
    '''

    # result_list: [result0_dist_list: [[compare_id, compared_id, DTW_dist], ...], result1_dist_list: , ...]
    # length of result_list: camera result files - 1
    result_list = []

    # Loop for each result file
    for i in range(len(info_list)):
        dist_list = []
        for k in info_list[i]:

            # 1. 겹치지 않는경우: 일단 제외한다
            if info[0][0] > k[0][-1] or info[0][-1] < k[0][0]:
                continue

            # 2. 포함하는 경우 : DTW로 유사도 측정
            elif (info[0][0] < k[0][0] and info[0][-1] > k[0][-1]) or (info[0][0] > k[0][0] and info[0][-1] < k[0][-1]):
                dist = dtw.dtw(k[2], info[2], keep_internals=True).distance
                dist_list.append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]

            # 3. 절반이상 겹치는 경우 : DTW로 유사도 측정
            elif (info[0][0] > k[0][0] and info[0][int(len(info[0]) / 2)] <= k[0][-1] <= info[0][-1]) or (
                    info[0][0] < k[0][0] and k[0][int(len(k[0]) / 2)] <= info[0][-1] <= k[0][-1]):
                dist = dtw.dtw(k[2], info[2], keep_internals=True).distance
                dist_list.append([info[1], k[1], dist])  # [compare_id, compare_id, DTW_dist]

            # 4. 절반이하로 겹치는 경우: 제외
            elif (info[0][0] > k[0][0] and info[0][int(len(info[0]) / 2)] > k[0][-1]) or (
                    info[0][0] < k[0][0] and k[0][int(len(k[0]) / 2)] > info[0][-1]):
                continue

        result_list.append(dist_list)

    return result_list


def id_mapping(dist_dictionary, mapping_list):
    keys = list(dist_dictionary.keys())

    for key in keys:
        sorted_list = sorted(dist_dictionary[key], key=lambda x: (x[2], x[0]))

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
for name in txt_name:
    result_df_list.append(make_df_list(name))

# Create id info list
result_info_list = []
for df_list in result_df_list:
    info = []
    for df in df_list:
        info.append(create_unit_vec(df))
    result_info_list.append(info)

# Create high similarity ID list
id_map_list = []

# dist_dict.keys : result.txt file number
dist_dict = dict()

# initialize
for i in range(0, len(result_info_list) - 1):
    dist_dict[i] = []


for info in result_info_list[0]:
    result_dist_list = check_similarity(info, result_info_list[1:])

    for num in range(0, len(result_dist_list)):
        if result_dist_list[num]:
            dist_dict[num] += result_dist_list[num]

id_mapping(dist_dict, id_map_list)
print(id_map_list)