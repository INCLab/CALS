import pandas as pd
import dtwfunction as dfunc

result_save_dir = '../tmp/'
txt_name = ['BEV_result0', 'BEV_result1', 'BEV_result2']


# 정익:6 / 13 / 21 민재:3 / 15 / 22 찬영:7 / 18 / 23


# Create Dataframes by id
result_df_list = []
total_id_list = []
for name in txt_name:
    df_list, id_list = dfunc.make_df_list(name)
    result_df_list.append(df_list)
    total_id_list.append(id_list)

# Create id info list
result_info_list = []

# Select feature 1.unit(unit vector) 2.scalar(normalized scalar) 3.vector  (default: unit)
# and generate result_info_list
dfunc.select_feature(result_df_list, result_info_list, feature='vector')

# Create high similarity ID list
# ToDo: 현재는 result0를 기준으로 result1,2를 비교한 결과만 사용, 후에 result1을 기준으로 구한 값도 고려해야함
id_map_list = [[], []]
for i in range(0, len(result_info_list)-1):
    result_dist_list = dfunc.check_similarity(result_info_list[i], result_info_list[i+1:])
    dfunc.id_mapping(result_dist_list, id_map_list[i])  # id_mapping에서 todo 처리

print(id_map_list[0])

# # Assign global id
# global_id_num = 1000
# global_id_set = []
#
# for i in range(1, len(id_map_list[0]) + 1):
#     global_id_set.append(global_id_num + i)
#
# dfunc.change_to_global(result_df_list, id_map_list[0], global_id_set)
#
# total_list = list()
# for T in result_df_list:
#     for id_info in T:
#         total_list += id_info.values.tolist()
#
# total_list.sort()

# global_I = dfunc.generate_global_info(total_list)
# global_df = pd.DataFrame(global_I)
# global_df.columns = ['frame', 'id', 'x', 'y']
#
# print(global_df)
#
# # Create Global information txt file
# global_df.to_csv('global_result.txt', sep=' ', header=None, index=None)



