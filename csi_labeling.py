import os
import argparse
import numpy as np
import pandas as pd
from database.csi_db import csi_db
from util import *
import cfg

# Arg_parser
parser = argparse.ArgumentParser(description='Argument parser for processing CSI labeling')
parser.add_argument('--dbmode', type=str2bool, default=False, help='use database')
args = parser.parse_args()

# ========= Output data path ========
os.makedirs(cfg.out_path, exist_ok=True)

'''
    Use csv (csi, vision)data
       Read all mot output files (e.g., ch01, ch02, ..., ch04)
       1. Count the number of detected person and make { time: total_person} dict by each files
       2. Then concatenate all list and remove duplicate: choose the highest total_person value
       3. return mot_dict
       mot_dict = {time1:total_person1, time2:total_person2, ...}
   '''
if args.dbmode is False:
    TIME_THRESHOLD = 1

    # Create file list
    try:
        csi_flist = [val for val in os.listdir(cfg.csi_path) if os.path.splitext(val)[1]=='.csv']
        vis_file = [val for val in os.listdir(cfg.vision_path) if os.path.splitext(val)[1]=='.txt'][0]
        print(f"Vision file:'{vis_file}' is selected.")
        print(f"CSI file list:'{csi_flist}'")
    except:
        print('Data file is not exist!')
        exit()

    # Make mot dict
    mot_datas = create_vis_dict(vis_file)

    # Make time list
    timeList = make_time_list(mot_datas)

    for csi_file in csi_flist:
        csi_label_list = []
        mac = csi_file[9:-4]

        csi_df = pcap_to_df(os.path.join(cfg.csi_path, csi_file))

        # mac address column 제외
        df = csi_df.iloc[:, 1:]

        # Labeling CSI data
        for csi_data in df.values.tolist():
            csi_time = csi_data[0]

            # CSI Time이 MOT 시작 시간보다 빠를경우 -1로 labeling
            if csi_time < timeList[0][0]:
                csi_label_list.append(-1)
                continue
            # CSI Time이 MOT 끝 시간을 넘어간 경우 -1로 labeling
            elif csi_time >= timeList[-1][1]:
                csi_label_list.append(-1)
                continue

            # tset: [start_time, end_time, plabel]
            for tset in timeList:
                start_time, end_time, plabel = tset

                # Person exist label로 변경
                if plabel > 0:
                    plabel = 1

                # 현재 iteration에서 csi time이 end time과 같거나 클경우 다음 iter로
                if csi_time >= end_time:
                    continue
                # 시간 범위안에 들어오는경우 해당 label append
                elif start_time <= csi_time < end_time:
                    csi_label_list.append(plabel)
        csi_df['label'] = csi_label_list

        csi_df.to_csv(os.path.join(out_path, 'labeled_csi_data_{}.csv'.format(mac)), index=False)
else:
    db = csi_db()
    csi_df = db.get_csi_df()

    '''
        <csi_df>
        Feature: 64 Subcarriers(-32~32), Elements: Amplitude
    '''
    csi_df['label'] = label_list

    # Drop unnecessary rows (label '-1')
    csi_df.drop(csi_df[csi_df['label'] == -1].index, inplace=True)
    csi_df.reset_index(drop=True, inplace=True)


    # ============= Preprocessing ================

    '''
        0. Set time window
        1. Drop null subcarriers
        2. Denoising with DWT(Discrete Wavelet Transform)
        3. Normalization
        4. extract dynamic moving feature with AST(Amplitude Shape Trend)
    '''
    # Drop timestamp
    csi_df.drop([csi_df.columns[0]], axis=1, inplace=True)

    # 0. Set time window (n second)
    packets_ps = 100  # packets per second
    n_second = 3
    time_window =  packets_ps * n_second
    tw_list = []  # time window list

    # Todo: time window 내에서 label이 바뀌는경우에 대한 전처리를 어떻게 할것인지,
    # solution: FairMOT result와 csi sync 과정에서 FairMOT label에 따라서
    # sync 데이터 따로 만들기. 이때 time window에 포함되는 csi data가 label이 통일 되지 않는경우 drop


    # 1. Drop null subcarriers
    # Indexes of Null and Pilot OFDM subcarriers
    # {-32, -31, -30, -29, 31, 30, 29, 0}
    null_idx = [-32, -31, -30, -29, 31, 30, 29, 0]
    null_idx = ['_' + str(idx + 32) for idx in null_idx]

    for idx in null_idx:
        csi_df.drop(idx, axis=1, inplace=True)





