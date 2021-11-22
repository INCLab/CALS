'''
    Set the distance(threshold) and two targets.
    Then, find number of frames(times) that two targets are closer than the distance.
'''
import sys
import pandas as pd
import math

# Parameters
threshold = 1000
target_id1 = 1001
target_id2 = 1003

# # ###################### Function ##########################
#
#
# # Check the distance and count all frames that meet the condition
# # 앞에서 Target1 Target2 겹치는 frame있는지 먼저 체크해야함
# def check_distance(thres, tar1_df, tar2_df):
#     tar1_list = []
#     tar2_list = []
#     total_frame = 0
#
#     for i in range(0, len(tar1_df)):
#         tar1_list.append(tar1_df.iloc[i].tolist())
#     for i in range(0, len(tar2_df)):
#         tar2_list.append(tar2_df.iloc[i].tolist())
#
#     for tar1_info in tar1_list:
#         for tar2_info in tar2_list:
#             # Check frame number
#             if tar1_info[0] == tar2_info[0]:
#                 dist = math.sqrt((tar1_info[2] - tar2_info[2])**2 + (tar1_info[3] - tar2_info[3])**2)
#                 if dist <= thres:
#                     total_frame += 1
#             elif tar1_info[0] < tar2_info[0]:
#                 break
#
#     return total_frame
#
#
# def check_overlap(tar1_df, tar2_df):
#     tar1_first_frame = tar1_df.iloc[0].tolist()[0]
#     tar1_last_frame = tar1_df.iloc[-1].tolist()[0]
#
#     tar2_first_frame = tar2_df.iloc[0].tolist()[0]
#     tar2_last_frame = tar2_df.iloc[-1].tolist()[0]
#
#     result = True
#
#     if tar1_last_frame < tar2_first_frame or tar1_first_frame > tar2_last_frame:
#         result = False
#
#     return result
# # ###########################################################
#
#
# Read global_result.txt and create dataframe
glob_info_df = pd.read_csv('../temp/global_result.txt', delimiter=' ', header=None)
glob_info_df.columns = ['frame', 'id', 'x', 'y']

target_id1_df = glob_info_df[(glob_info_df['id'] == target_id1)]
target_id2_df = glob_info_df[(glob_info_df['id'] == target_id2)]

# Check empty dataframe
if target_id1_df.empty:
    print("Target 1 doesn't exist!")
    exit()
elif target_id2_df.empty:
    print("Target 2 doesn't exist!")
    exit()
#
# is_overlap = check_overlap(target_id1_df, target_id2_df)
#
# if is_overlap:
#     total = check_distance(threshold, target_id1_df, target_id2_df)
#     print("{0} frames".format(total))
# else:
#     print("There's no overlap frames!")
#
#


# ################## target - space #######################
# Check target space data file
def is_empty(txt_dir):
    try:
        with open(txt_dir, 'rt', encoding='UTF8') as f:
            info_list = f.readlines()
            if not info_list:
                return True
            else:
                return False
    except Exception as e:
        return True


def make_info_dict(txt_dir):
    space_info_dict = dict()
    with open(txt_dir, 'rt', encoding='UTF8') as f:
        info_list = f.readlines()

        for info in info_list:
            split_info = info.split(" ")

            if split_info[0] not in space_info_dict.keys():
                space_info_dict[split_info[0]] = [[int(split_info[1]), int(split_info[2].rstrip('\n'))]]
            else:
                space_info_dict[split_info[0]].append([int(split_info[1]), int(split_info[2].rstrip('\n'))])

    return space_info_dict


# Point in polygon algorithm
def is_inside(p_info_dict, p_space_tar_name, p_point):
    inside = False
    space_point = p_info_dict[p_space_tar_name]

    i = 0
    j = len(space_point) - 1

    while i <= len(space_point) - 1:
        # intersection point
        inter_point = ((space_point[j][0] - space_point[i][0]) * (p_point[1] - space_point[i][1]) / (
                    space_point[j][1] - space_point[i][1])) + space_point[i][0]

        if (((space_point[i][1] > p_point[1]) != (space_point[j][1] > p_point[1])) and
                (p_point[0] < inter_point)):
            inside = not inside
        j = i
        i = i + 1

    return inside



point_txt_dir = "../temp/target_space.txt"


if is_empty(point_txt_dir):
    print("타겟 데이터가 없습니다")
else:
    info_dict = make_info_dict(point_txt_dir)

    tar_id_list = []
    total_frame = 0

    for i in range(0, len(target_id1_df)):
        tar_id_list.append(target_id1_df.iloc[i].tolist())

    for tar_id_info in tar_id_list:
        point = [tar_id_info[1], tar_id_info[2]]

        if is_inside(info_dict, '놀이터', point):
            total_frame += 1

    print(total_frame)



