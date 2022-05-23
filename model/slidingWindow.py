import pandas as pd
import numpy as np
PACKETPERSECOND = 50

def makeSlidingWindow(df, subcarrier):
    s_df = df[[subcarrier, 'label']]

    label_list = s_df['label'].drop_duplicates().to_list()

    sliding_list = []
    for label in label_list:
        tmp_df = s_df[s_df['label'] == label]

        subc_list = tmp_df[subcarrier].to_list()

        total_second = int(len(subc_list) / PACKETPERSECOND)
        start_num = 0
        end_num = PACKETPERSECOND

        while True:
            end_num = start_num + PACKETPERSECOND
            if end_num > total_second * PACKETPERSECOND:
                break
            current_csi = subc_list[start_num:end_num]
            current_csi.append(label)
            sliding_list.append(current_csi)
            start_num += int(PACKETPERSECOND/2)

    column_list = [str(i) for i in range(1, PACKETPERSECOND + 1)] + ['label']
    sliding_df = pd.DataFrame(sliding_list, columns=column_list)

    return sliding_df
