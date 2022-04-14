# Todo: real Time plotter 완성하기
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.animation import FuncAnimation

import time
from datetime import datetime

global xval
xval = 0


def AmpRTPlotter(csi_df, spf_sub=None):

    if spf_sub is not None:
        subcarrier = csi_df[spf_sub].to_list()

        # ============ Denoising with DWT ==================
        signal = subcarrier

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Amp-SampleIndex plot')
        ax.plot(signal, color="b", alpha=0.5, label=spf_sub)
        rec = lowpassfilter(signal, 0.2)
        ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
        ax.legend()
        ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
        ax.set_ylabel('Signal Amplitude', fontsize=16)
        ax.set_xlabel('Sample Index', fontsize=16)
        plt.show()
    else:
        subcarrier_list = []
        for col in csi_df.columns:
            subcarrier_list.append(csi_df[col].to_list())
        x = []
        y_list = [[] for i in range(0, len(subcarrier_list))]

        def animate(i):
            # 100packet 이상 넘어가면 맨 앞 packet 제거
            if len(x) > 100:
                del x[0]
                for y in y_list:
                    del y[0]

            global xval
            x.append(xval)
            for j in range(0, len(subcarrier_list)):
                y_list[j].append(subcarrier_list[j][xval])
            plt.cla()
            for idx, y in enumerate(y_list):
                plt.plot(x, y, alpha=0.5, label=csi_df.columns[idx])

            plt.ylabel('Signal Amplitude', fontsize=16)
            plt.xlabel('Time', fontsize=16)
            plt.tight_layout()
            xval += 1


        ani = FuncAnimation(plt.gcf(), animate, interval=10)
        # ani.save(r'csi.gif', fps=10)

        plt.tight_layout()
        plt.show()




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


