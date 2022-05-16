import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def complexToAmp(comp_df):

    comp_df = comp_df.astype('complex')
    amp_df = comp_df.apply(np.abs, axis=1)

    return amp_df


peList = ['leftbottom', 'lefttop', 'middle', 'rightbottom', 'righttop', 'all_loc']  # person exist
npeList = ['no_person', 'no_person2', 'no_person3']  # none person

dataPath = 'data/0502_csi_mot'
outPath = 'data/pe'

# Make output directory
os.makedirs(outPath, exist_ok=True)

# Create merged Person Exist CSI data
pe_dict = dict()
for pe_case in peList:
    csiPath = os.path.join(dataPath, pe_case, 'labeled/PE')
    file_list = os.listdir(csiPath)

    for file in file_list:
        if file[:2] == 'pe':
            mac = file.split('_')[2][:-4]
            csi_df = pd.read_csv(os.path.join(csiPath, file))

            # Delete null, pilot subcarriers
            null_pilot_col_list = ['_' + str(x + 32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]
            csi_df.drop(null_pilot_col_list, axis=1, inplace=True)

            df_plt = csi_df.iloc[:, 2:-1]
            df_plt = complexToAmp(df_plt)

            columns = df_plt.columns

            scaler = StandardScaler()
            df_plt = pd.DataFrame(scaler.fit_transform(df_plt), columns=columns)

            # # Show plot
            # subcarrier_list = []
            # for col in df_plt.columns:
            #     subcarrier_list.append(df_plt[col].to_list())
            #
            # fig, ax = plt.subplots(figsize=(12, 8))
            # fig.suptitle('Amp-SampleIndex plot')
            #
            # for idx, sub in enumerate(subcarrier_list):
            #     ax.plot(sub, alpha=0.5, label=df_plt.columns[idx])
            #
            # ax.set_ylabel('Signal Amplitude', fontsize=16)
            # ax.set_xlabel('Sample Index', fontsize=16)
            # plt.show()

            df_plt.insert(0, 'mac', csi_df['mac'].to_list())
            df_plt.insert(1, 'time', csi_df['time'].to_list())
            df_plt['label'] = csi_df['label'].to_list()

            if mac in pe_dict.keys():
                pe_dict[mac] = pd.concat([pe_dict[mac], df_plt], ignore_index=True)
            else:
                pe_dict[mac] = df_plt

for mac in pe_dict.keys():
    df = pe_dict[mac]
    df.drop(df[df['label'] == -1].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.to_csv(os.path.join(outPath, 'pe_' + mac + '.csv'), index=False)

# Create merged None Person CSI data
npe_dict = dict()
for npe_case in npeList:
    csiPath = os.path.join(dataPath, npe_case, 'labeled/PE')
    file_list = os.listdir(csiPath)

    for file in file_list:
        if file[:2] == 'pe':
            mac = file.split('_')[2][:-4]
            csi_df = pd.read_csv(os.path.join(csiPath, file))

            # Delete null, pilot subcarriers
            null_pilot_col_list = ['_' + str(x + 32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]
            csi_df.drop(null_pilot_col_list, axis=1, inplace=True)

            df_plt = csi_df.iloc[:, 2:-1]
            df_plt = complexToAmp(df_plt)

            columns = df_plt.columns

            scaler = StandardScaler()
            df_plt = pd.DataFrame(scaler.fit_transform(df_plt), columns=columns)

            # # Show plot
            # subcarrier_list = []
            # for col in df_plt.columns:
            #     subcarrier_list.append(df_plt[col].to_list())
            #
            # fig, ax = plt.subplots(figsize=(12, 8))
            # fig.suptitle('Amp-SampleIndex plot')
            #
            # for idx, sub in enumerate(subcarrier_list):
            #     ax.plot(sub, alpha=0.5, label=df_plt.columns[idx])
            #
            # ax.set_ylabel('Signal Amplitude', fontsize=16)
            # ax.set_xlabel('Sample Index', fontsize=16)
            # plt.show()

            df_plt.insert(0, 'mac', csi_df['mac'].to_list())
            df_plt.insert(1, 'time', csi_df['time'].to_list())
            df_plt['label'] = csi_df['label'].to_list()

            if mac in npe_dict.keys():
                npe_dict[mac] = pd.concat([npe_dict[mac], df_plt], ignore_index=True)
            else:
                npe_dict[mac] = df_plt

for mac in npe_dict.keys():
    df = npe_dict[mac]
    df.drop(df[df['label'] == -1].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.to_csv(os.path.join(outPath, 'npe_' + mac + '.csv'), index=False)