import pandas as pd
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


result0_df_list = make_df_list(txt_name[0])
result1_df_list = make_df_list(txt_name[1])
result2_df_list = make_df_list(txt_name[2])


def create_dist_list(df):
    frame_list = df['frame'].to_list()
    id = df['id'][0]
    x_list = df['x'].to_list()
    y_list = df['y'].to_list()


