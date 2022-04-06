import pandas as pd
import os
from plot.ampPlotter import AmpPlotter
from plot.heatmap import heatmap

'''
   <<  Null & Pilot subcarriers on 2.4MHz >>
    
    Null: [x+32 for x in [-32, -31, -30, -29, 31,  30,  29,  0]]
    Pilot: [x+32 for x in [-21, -7, 21,  7]]
    
    Available number of subcarriers: 64-12 = 52
'''

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

null_pilot_col_list = ['_' + str(x+32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]

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
    csi_df = df.iloc[:, 3:]

    # Remove null & pilot subcarriers
    csi_df.drop(null_pilot_col_list, axis=1, inplace=True)

    while True:
        plot_mode = input('1. Amplitude plot, 2.Heatmap, 3.Exit: ')

        if plot_mode == '1':
            # select specific subcarrier
            spf_subc = input('Select specific subcarrier(True or False):  ')

            if spf_subc == 'True':
                spf_sub_idx = input('Select one subcarrier {}:  '.format(csi_df.columns))

                while spf_sub_idx not in csi_df.columns:
                    print("Wrong input!")
                    spf_sub_idx = input('Select one subcarrier {}:  '.format(csi_df.columns))

                AmpPlotter(csi_df, s_start, s_end, spf_sub_idx)
            elif spf_subc == 'False':
                AmpPlotter(csi_df, s_start, s_end)
            else:
                print("Wrong input!")

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


