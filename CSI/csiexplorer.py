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
data_path = '../data/csi/'
csi_list = os.listdir(data_path)

s_start = 0
s_end = -1


if __name__ == "__main__":
    print('== Choose csv file == ')

    for idx, csi_fname in enumerate(csi_list):
        print('{}: {}'.format(idx+1, csi_fname))

    is_true = True

    while is_true:
        try:
            select_num = int(input("Select: "))
            if select_num > len(csi_list) or select_num <= 0:
                print("Error!")
                continue
            else:
                is_true = False
        except:
            print("Error!")
            continue

    data_fname = csi_list[select_num-1]

    data_path = os.path.join(data_path, data_fname)

    # Read csi.csv
    df = pd.read_csv(data_path)

    csi_df = df.iloc[:, 3:-1]

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


