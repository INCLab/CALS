import os
import numpy as np
import pandas as pd
import pywt
from datetime import datetime
import time

TIME_THRESHOLD = 1  # Second

'''
    Read all mot output files (e.g., ch01, ch02, ..., ch04)
    1. Count the number of detected person and make { time: total_person} dict by each files
    2. Then concatenate all list and remove duplicate: choose the highest total_person value
    3. return mot_dict
    mot_dict = {time1:total_person1, time2:total_person2, ...}
'''
def create_mot_dict(mot_file_list, mot_path):
    # === 1 ===
    mot_dict_by_files = {}
    for mot_fname in mot_file_list:
        with open(os.path.join(mot_path, mot_fname), 'r') as f:
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
def make_time_list(mot_dict, mot_flist):
    # ========= video data: start_time, end_time ########
    # mot file name format: channelNum-date-startTime-endTime-number.txt
    date, start, end = mot_flist[0].split('-')[1:-1]
    mot_start_time = date + start
    mot_end_time = date + end

    # Unix time
    mot_start_ut = time.mktime(datetime.strptime(mot_start_time, '%Y%m%d%H%M%S').timetuple())
    mot_end_ut = time.mktime(datetime.strptime(mot_end_time, '%Y%m%d%H%M%S').timetuple())

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

    for idx, t in enumerate(motTimeList):
        current_label = mot_dict[t]

        if current_label == label and (t - before_time) < TIME_THRESHOLD:
            if idx == len(motTimeList) - 1:
                timeList.append([start_time, t, label])
            before_time = t
            continue
        # time threshold보다 크다는것은 중간에 타겟이 없었던 시간이 존재한다는 뜻
        elif current_label == label and (t - before_time) >= TIME_THRESHOLD:
            # 타겟이 존재했던 시간까지 timeList에 삽입
            timeList.append([start_time, before_time, label])
            # 타겟이 존재하지 않는 시간에 대해 timeList에 삽입
            timeList.append([before_time, t, 0])
            start_time = t
        # label이 바뀌고 threshold를 넘지 않는 경우
        elif current_label != label and (t - before_time) < TIME_THRESHOLD:
            # label이 바뀌기 이전시간까지 List에 삽입
            timeList.append([start_time, t, label])
            start_time = t
            label = current_label
        # label이 바뀌고 threshold를 넘는 경우
        elif current_label != label and (t - before_time) >= TIME_THRESHOLD:
            # 타겟이 존재했던 시간까지 timeList에 삽입
            timeList.append([start_time, before_time, label])
            # 타겟이 존재하지 않는 시간에 대해 timeList에 삽입
            timeList.append([before_time, t, 0])
            start_time = t
            label = current_label

        before_time = t

    # MOT output txt의 마지막 타임과 비디오 끝 타임의 간격이 Time threshold보다 클 경우
    # 해당 구간은 타겟이 없는것으로 labeling
    if mot_end_ut - before_time >= TIME_THRESHOLD:
        timeList.append([before_time, mot_end_ut, 0])
    # MOT output txt의 마지막 타임과 비디오 끝 타임의 간격이 존재하고, Time threshold보다 작을 경우
    # 해당 구간은 mot 마지막 label로 labeling
    elif 0 < mot_end_ut - before_time <  TIME_THRESHOLD:
        timeList.append([before_time, mot_end_ut, label])

    return timeList


def personExistLabeling(mot_flist, csi_flist, mot_path, csi_path, out_path):
    # Make mot dict
    mot_datas = create_mot_dict(mot_flist, mot_path)

    # Make time list
    timeList = make_time_list(mot_datas, mot_flist)

    for csi_file in csi_flist:
        csi_label_list = []
        mac = csi_file[9:-4]

        csi_df = pd.read_csv(os.path.join(csi_path, csi_file))

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

        csi_df.to_csv(os.path.join(out_path, 'pe_csi_data_{}.csv'.format(mac)), index=False)


def gridAllocateLabeling(mot_flist, csi_flist, mot_path, csi_path, out_path):
    '''
        1920 x 1080을 120 x 120 정사각형으로 나누면 16 x 9
        양 끝 2 x 9  제거 후, 남은 12 x 9를 3 x 3으로 나누면 4 x 3 = 총 12개의 class
        좌상단 부터 class 1~12 labeling

        1440 x 1080를 360 x 360 정사각형으로 나눈꼴
    '''
    BLOCK_SIZE = 360

    x_start = 240

    '''
          grid space dict elements
                class : 
                    left_top: left_top_coord, 
                    right_bottom: right_bottom_coord
                    
        total 12 keys in grid space -> number of class
    '''
    grid_space_dict = {}

    for i in range(0, 12):
        x_top = x_start + ((i % 4) * BLOCK_SIZE)
        y_top = int(i / 4) * BLOCK_SIZE
        x_bottom = x_top + BLOCK_SIZE
        y_bottom = y_top + BLOCK_SIZE

        grid_space_dict[i+1] = {
            'left_top': [x_top, y_top],
            'right_bottom': [x_bottom, y_bottom]
        }


    # Make mot dict
    mot_datas = create_mot_dict(mot_flist, mot_path)

    # Make time list
    timeList = make_time_list(mot_datas, mot_flist)

    for csi_file in csi_flist:
        csi_label_list = []
        mac = csi_file[9:-4]

        csi_df = pd.read_csv(os.path.join(csi_path, csi_file))

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

        csi_df.to_csv(os.path.join(out_path, 'pe_csi_data_{}.csv'.format(mac)), index=False)
