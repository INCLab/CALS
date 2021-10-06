import numpy as np
from function import save_lonlat_frame
from function import save_dict
from function import getcolor
import cv2
import os
import sys
import shutil
##############################################################################

'''
pixel : 실제 공간
lonloat : 도면 공간
실제 mapping 되는 곳에 좌표를 입력 @@@.py 사용
오른쪽 위, 왼쪽 위, 왼쪽 아래, 오른쪽 아래 순서
'''
input_path = sys.argv[1]
output_path = sys.argv[2]
global_output_path = os.path.join(output_path, 'global_map_frame')
map_path = sys.argv[3]
temp_path = "./temp"

if not os.path.exists(global_output_path):
    os.makedirs(global_output_path)
else:
    shutil.rmtree(global_output_path)
    os.makedirs(global_output_path)

# ==============  Global ID BEV result  ===================

file = open(os.path.join(temp_path, "global_result.txt"), 'r')
globals()['g_frame'], globals()['g_point'] = save_dict(file)

map = cv2.imread(map_path, -1)

for frames in range(1, int(globals()['g_frame'])):
    if globals()['g_point'].get(str(frames)) is not None:
        for label in globals()['g_point'].get(str(frames)):
            lonlat = [label[1], label[2]]
            color = getcolor(abs(label[0]))
            cv2.circle(map, (lonlat[0], lonlat[1]), 3, color, -1)

    src = os.path.join(global_output_path, str(frames) + '.jpg')
    cv2.imwrite(src, map)
