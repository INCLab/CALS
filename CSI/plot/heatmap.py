import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot.dataPreprocess import data_preprocess

'''
Heatmap
---------------------------

Plot 
'''


def heatmap(csi_df, sample_start, sample_end, preprocess=False):

    df = csi_df[sample_start:sample_end]

    if preprocess is True:
        df = data_preprocess(df)

    packet_idx = [i for i in range(1, len(df) + 1)]

    x_list = []
    for idx in packet_idx:
        # x_list.append(idx / 150)
        x_list.append(idx)

    y_list = []
    for col in df.columns:
        y_list.append(col)

    plt.pcolor(x_list, y_list, df.transpose(), cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (dBm)')

    xtic = np.arange(0, x_list[-1] + 1, 100)
    xtic[0] = 1
    ytic = np.arange(0, 52, 13)

    plt.xticks(xtic)
    plt.yticks(ytic, [y_list[idx] for idx in [0, int(len(y_list)/4), int(len(y_list)/4*2), int(len(y_list)/4*3)]])
    #plt.xlabel('Time (s)')
    plt.xlabel('Packet Index')
    plt.ylabel('Subcarrier Index')

    plt.show()


