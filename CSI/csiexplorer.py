import pandas as pd
import os
from plot.ampPlotter import AmpPlotter
from plot.heatmap import heatmap


def string_is_int(s):
    '''
    Check if a string is an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False

# Path
data_path = '../data'
data_fname = 'csi_data.csv'

data_path = os.path.join(data_path, data_fname)

# Read csi.csv
df = pd.read_csv(data_path)

csi_df = df.iloc[:, 3:-1]

s_start = 0
s_end = 190


if __name__ == "__main__":

    while True:
        plot_mode = input('1. Amplitude plot, 2.Heatmap, 3.Exit: ')

        if plot_mode == '1':
            #sub = input('subcarrier index(-32 ~ 32):  ')
            AmpPlotter(csi_df, s_start, s_end)

        elif plot_mode == '2':
            pre = input('Data Preprocessing (True or False):  ')

            if pre == 'True':
                heatmap(csi_df, s_start, s_end, preprocess=True)
            elif pre == 'False':
                heatmap(csi_df, s_start, s_end)
            else:
                print("Wrong input!")

        elif plot_mode == '3':
            break
        else:
            print('Unknown command.')


