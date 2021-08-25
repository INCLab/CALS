import numpy as np
from function import PixelMapper
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
heatmap_path = os.path.join(output_path, 'heatmap.png')
output_path = os.path.join(output_path, 'map_frame')
map_path = sys.argv[3]
temp_path = "./temp"

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)

num = len(os.listdir(input_path))
f = open(os.path.join(temp_path, 'points.txt'), 'r')
data = f.read()
data = data.split('\n')

point = []
frame_point = []
map_point = []

for i in data:
    point.append(i.split(' '))

for i in point:
    if i[0] == 'map':
        map_point.append(i)
    elif i[0] == 'frame':
        frame_point.append(i)

if not(len(map_point) % 4 == 0 and len(frame_point) % 4 == 0):
    print('Point Error')
    sys.exit()

quad_coords_list = []

for i in range(num):
    quad_coords = {
        "pixel": np.array([
            [frame_point[i*4][1]  ,   frame_point[i*4][2]],  # Third lampost top right
            [frame_point[i*4+1][1], frame_point[i*4+1][2]],  # Corner of white rumble strip top left
            [frame_point[i*4+2][1], frame_point[i*4+2][2]],  # Corner of rectangular road marking bottom left
            [frame_point[i*4+3][1], frame_point[i*4+3][2]]  # Corner of dashed line bottom right
        ]),
        "lonlat": np.array([
            [map_point[i*4][1]  ,   map_point[i*4][2]],  # Third lampost top right
            [map_point[i*4+1][1], map_point[i*4+1][2]],  # Corner of white rumble strip top left
            [map_point[i*4+2][1], map_point[i*4+2][2]],  # Corner of rectangular road marking bottom left
            [map_point[i*4+3][1], map_point[i*4+3][2]]  # Corner of dashed line bottom right
        ])
    }
    quad_coords_list.append(quad_coords)

print(quad_coords_list)
#PixelMapper로 값 전달

for i in range(num):
##############변경해야하는 부분#######################
# 좌표값을 받아야함(하나씩)
    file = open(os.path.join(temp_path,"result"+str(i)+".txt"), 'r')
    globals()['frame{}'.format(i)], globals()['point{}'.format(i)] = save_dict(file)


map = cv2.imread(map_path, -1)
for i in range(num):
    globals()['BEV_Point{}'.format(i)] = dict()

for frames in range(1, int(globals()['frame{}'.format(0)])):
    for i in range(num):
        pm = PixelMapper(quad_coords_list[i]["pixel"], quad_coords_list[i]["lonlat"])
        if globals()['point{}'.format(i)].get(str(frames)) != None:
            for label in globals()['point{}'.format(i)].get(str(frames)) :
                uv = (label[1], label[2])
                lonlat = list(pm.pixel_to_lonlat(uv))
                li = [label[0], int(lonlat[0][0]), int(lonlat[0][1])]
                if frames in globals()['BEV_Point{}'.format(i)]:
                    line = globals()['BEV_Point{}'.format(i)].get(frames)
                    line.append(li)
                else:
                    globals()['BEV_Point{}'.format(i)][frames] = [li]

                color = getcolor(abs(label[0]))
                cv2.circle(map, (int(lonlat[0][0]), int(lonlat[0][1])), 3, color, -1)

        src = os.path.join(output_path, str(frames) + '.jpg')
        cv2.imwrite(src, map)

# heatmap

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np


# df = pd.DataFrame(index=range(0, 10), columns=range(0, 13))
df = [[0 for col in range(13)] for row in range(10)]

# df = df.fillna(0)

print(type(map.shape))
print(map.shape[1])

for frames in range(1, int(globals()['frame{}'.format(0)])):

    for i in range(num):

        if globals()['BEV_Point{}'.format(i)].get(frames) is not None:

            for label in globals()['BEV_Point{}'.format(i)].get(frames):
                if label[2] < 0 or label[1] < 0 or label[1] > map.shape[1] or label[2] > map.shape[0] :

                    continue

                x = round(int(label[2]) / map.shape[0] * 9)
                y = round(int(label[1]) / map.shape[1] * 12)
                df[x][y] += 1

print(df)

sns.heatmap(df, linewidths=0.1, linecolor="black")

plt.savefig(heatmap_path)