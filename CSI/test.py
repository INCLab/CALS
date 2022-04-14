# Todo: real Time plotter 완성하기
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.animation import FuncAnimation

import time
from datetime import datetime


def AmpRTPlotter(csi_df, spf_sub=None):
    subcarrier_list = []
    for col in csi_df.columns:
        subcarrier_list.append(csi_df[col].to_list())


    x = np.arange(0, 100, 1)
    y_list = []

    for i in range(0, len(subcarrier_list)):
        y_list.append([subcarrier_list[i][j] for j in range(0, 100)])

    plt.ion()

    fig, ax = plt.subplots(figsize=(12, 8))

    line_list = []

    for y in y_list:
        line, = ax.plot(x, y, alpha=0.5)
        line_list.append(line)

    plt.ylabel('Signal Amplitude', fontsize=16)
    plt.xlabel('Packet', fontsize=16)

    idx = 99
    for l in range(0, 100):
        idx += 1
        for i, y in enumerate(y_list):
            del y[0]
            y.append(subcarrier_list[i][idx])

        for i, line in enumerate(line_list):
            line.set_xdata(x)
            line.set_ydata(y_list[i])

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)


if __name__ == '__main__':
    # Path
    test_name = 'test3_3rasp'
    data_path = '../data'
    data_path = os.path.join(data_path, test_name, 'csi/csi_data_b827ebba9e7b.csv')
    #csi_list = os.listdir(data_path)

    null_pilot_col_list = ['_' + str(x + 32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]

    # Read csi.csv
    df = pd.read_csv(data_path)

    # Remove MAC address, timestamp
    csi_df = df.iloc[:, 2:]

    # Create timestamp list
    time_list = df['time'].tolist()

    # Remove null & pilot subcarriers
    csi_df.drop(null_pilot_col_list, axis=1, inplace=True)

    AmpRTPlotter(csi_df)


