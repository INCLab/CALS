import os
import argparse
import importlib
import pandas as pd
import numpy as np
import csi.config as config
import time
import cfg
from datetime import datetime


decoder = importlib.import_module(f'csi.decoders.{config.decoder}') # This is also an import

def pcap_to_df(filename, amp=False, del_null=False, add_MAC=True, add_time=True):
    nulls = {
        20: [x + 32 for x in [
            -32, -31, -30, -29,
            31, 30, 29, 0
        ]],

        40: [x + 64 for x in [
            -64, -63, -62, -61, -60, -59, -1,
            63, 62, 61, 60, 59, 1, 0
        ]],

        80: [x + 128 for x in [
            -128, -127, -126, -125, -124, -123, -1,
            127, 126, 125, 124, 123, 1, 0
        ]],

        160: [x + 256 for x in [
            -256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
            255, 254, 253, 252, 251, 129, 128, 127, 5, 4, 3, 3, 1, 0
        ]]
    }

    # Read pcap file and create dataframe
    try:
        csi_samples = decoder.read_pcap(filename)
    except FileNotFoundError:
        print(f'File {filename} not found.')
        exit(-1)

    bw = csi_samples.bandwidth

    num_20MHz_sc = 64
    num_sc = num_20MHz_sc * bw//20  # number of subcarriers

    # Create csi data frame
    colums = ['_' + str(i) for i in range(0, num_sc)]

    if amp is True:
        csi_df = pd.DataFrame(np.abs(csi_samples.get_all_csi()), columns=colums)  # Get csi amplitude dataframe
    else:
        csi_df = pd.DataFrame(csi_samples.get_all_csi(), columns=colums)  # Get I/Q complex num dataframe

    if del_null is True:
        csi_df = csi_df[csi_df.columns.difference(nulls[bw])]

    if add_time is True:
        pkttimes = csi_samples.get_times()
        csi_df.insert(0, 'time', pkttimes)

    if add_MAC is True:
        mac_list = [csi_samples.get_mac(i).hex() for i in range(0, csi_samples.nsamples)]
        csi_df.insert(0, 'mac', mac_list)

    return csi_df


# Todo
def create_vis_dict(mot_file):
    mot_start_ut = time.mktime(datetime.strptime(cfg.vis_start_time, cfg.time_format).timetuple())
    mot_end_ut = time.mktime(datetime.strptime(cfg.vis_end_time, cfg.time_format).timetuple())

    # === 1 ===
    mot_dict_by_files = {}

    with open(os.path.join(mot_path, mot_file), 'r') as f:
        data = f.read()
        row_list = data.split('\n')

        if row_list[-1] == '':
            row_list.remove('')


        tmp_dict = {}
        before_time = 0.0
        for row in row_list:
            t = float(row.split(' ')[0])

            if t != before_time:
                tmp_dict[t] = 1
                before_time = t
            else:
                tmp_dict[t] += 1
        mot_dict_by_files[mot_fname] = tmp_dict

    # === 2 ===
    base_file = list(mot_dict_by_files.keys())[0]
    base_dict = mot_dict_by_files[base_file]

    if len(mot_dict_by_files.keys()) > 1:
        for file in mot_dict_by_files.keys():
            if file == base_file:
                continue

            for current_time in mot_dict_by_files[file].keys():
                if current_time in base_dict:
                    if mot_dict_by_files[file][current_time] > base_dict[current_time]:
                        base_dict[current_time] = mot_dict_by_files[file][current_time]
                else:
                    base_dict[current_time] = mot_dict_by_files[file][current_time]

        return dict(sorted(base_dict.items()))

    else:
        return base_dict


# Make time list
# timeList: [[Start_t, End_t, Label], [Start_t, End_t, Label], ...]
def make_time_list(mot_dict):
    motTimeList = list(mot_dict.keys())
    timeList = []

    # 비디오 시작 시간에 타겟이 없고, Time threshold 보다 긴 시간동안
    # 타겟이 등장하지 않는 경우 해당 시간은 label 0으로 처리
    if motTimeList[0] - mot_start_ut >= TIME_THRESHOLD:
        timeList.append([mot_start_ut, motTimeList[0], 0])
    # 비디오 시작 시간과 mot 시작시간에 차이가 있지만, Time threshold 보다 작을 경우
    # 해당 시간은 label 1으로 처리
    elif 0 < motTimeList[0] - mot_start_ut < TIME_THRESHOLD:
        timeList.append([mot_start_ut, motTimeList[0], 1])

    # initialize with 1st mot data
    label = mot_dict[motTimeList[0]]
    start_time = motTimeList[0]
    before_time = motTimeList[0]

    for idx, time in enumerate(motTimeList):
        current_label = mot_dict[time]

        if current_label == label and (time - before_time) < TIME_THRESHOLD:
            if idx == len(motTimeList) - 1:
                timeList.append([start_time, time, label])
            before_time = time
            continue
        # time threshold보다 크다는것은 중간에 타겟이 없었던 시간이 존재한다는 뜻
        elif current_label == label and (time - before_time) >= TIME_THRESHOLD:
            # 타겟이 존재했던 시간까지 timeList에 삽입
            timeList.append([start_time, before_time, label])
            # 타겟이 존재하지 않는 시간에 대해 timeList에 삽입
            timeList.append([before_time, time, 0])
            start_time = time
        # label이 바뀌고 threshold를 넘지 않는 경우
        elif current_label != label and (time - before_time) < TIME_THRESHOLD:
            # label이 바뀌기 이전시간까지 List에 삽입
            timeList.append([start_time, time, label])
            start_time = time
            label = current_label
        # label이 바뀌고 threshold를 넘는 경우
        elif current_label != label and (time - before_time) >= TIME_THRESHOLD:
            # 타겟이 존재했던 시간까지 timeList에 삽입
            timeList.append([start_time, before_time, label])
            # 타겟이 존재하지 않는 시간에 대해 timeList에 삽입
            timeList.append([before_time, time, 0])
            start_time = time
            label = current_label

        before_time = time

    # MOT output txt의 마지막 타임과 비디오 끝 타임의 간격이 Time threshold보다 클 경우
    # 해당 구간은 타겟이 없는것으로 labeling
    if mot_end_ut - before_time >= TIME_THRESHOLD:
        timeList.append([before_time, mot_end_ut, 0])
    # MOT output txt의 마지막 타임과 비디오 끝 타임의 간격이 존재하고, Time threshold보다 작을 경우
    # 해당 구간은 mot 마지막 label로 labeling
    elif 0 < mot_end_ut - before_time <  TIME_THRESHOLD:
        timeList.append([before_time, mot_end_ut, label])

    return timeList

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 'TRUE', 'T', 'Y', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'FALSE', 'F', 'N', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
