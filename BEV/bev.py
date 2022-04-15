from copy import copy

import cv2
import os
import sys
import shutil
import time

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mimetypes

LOCAL_INIT_ID = 10000
##############################################################################

'''
pixel : 실제 공간
lonloat : 도면 공간
실제 mapping 되는 곳에 좌표를 입력 @@@.py 사용
오른쪽 위, 왼쪽 위, 왼쪽 아래, 오른쪽 아래 순서
'''

def start(output_path, map_path, temp_path='./temp'):

    heatmap_path = os.path.join(output_path, 'heatmap.png')
    original_output_path = output_path
    output_path = os.path.join(output_path, 'map_frame')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    filelist = []
    for file in os.listdir(original_output_path):
        if file.endswith(".txt"):
            filelist.append(file.rstrip('.txt'))

    # Sort files
    # Among the file names, you must specify a location indicating the order
    # e.g., 'ch01-....' -> [2:4]
    filelist = strange_sort(filelist, 2, 4)

    f = open(os.path.join(temp_path, 'points_map1.txt'), 'r')
    data = f.read()
    data = data.split('\n')

    point = []
    frame_point = {}
    map_point = {}

    for i in data:
        point.append(i.split(' '))

    crtfile = None
    for i in point:
        if crtfile is not None and (i[0].startswith("map") or i[0].startswith("frame")):
            if i[0] == 'map':
                map_point[crtfile].append([i[1], i[2]])
            elif i[0] == 'frame':
                frame_point[crtfile].append([i[1], i[2]])
        else:
            if not i[0] == '' and not i[0].isspace():
                crtfile = i[0]
                map_point[crtfile] = []
                frame_point[crtfile] = []

    # if not(len(map_point) % 4 == 0 and len(frame_point) % 4 == 0):
    #     print('Point Error')
    #     sys.exit()

    quad_coords_list = {}

    for i in list(map_point.keys()):
        quad_coords = {
            "pixel": np.array([
                [frame_point[i][0][0],   frame_point[i][0][1]],  # Third lampost top right
                [frame_point[i][1][0],   frame_point[i][1][1]],  # Corner of white rumble strip top left
                [frame_point[i][2][0],   frame_point[i][2][1]],  # Corner of rectangular road marking bottom left
                [frame_point[i][3][0],   frame_point[i][3][1]]  # Corner of dashed line bottom right
            ]),
            "lonlat": np.array([
                [map_point[i][0][0],   map_point[i][0][1]],  # Third lampost top right
                [map_point[i][1][0],   map_point[i][1][1]],  # Corner of white rumble strip top left
                [map_point[i][2][0],   map_point[i][2][1]],  # Corner of rectangular road marking bottom left
                [map_point[i][3][0],   map_point[i][3][1]]  # Corner of dashed line bottom right
            ])
        }
        quad_coords_list[i] = quad_coords

    max_frame = 0
    file_num = 1
    for filename in filelist:
        file = open(os.path.join(original_output_path, filename + '.txt'), 'r')
        globals()['frame{}'.format(filename)], globals()['point{}'.format(filename)] = save_dict(file, LOCAL_INIT_ID, file_num)
        globals()['BEV_Point{}'.format(filename)] = dict()

        if int(globals()['frame{}'.format(filename)]) > max_frame:
            max_frame = int(globals()['frame{}'.format(filename)])
        file_num += 1

    map = cv2.imread(str(map_path), -1)
    pointset = set()

    print("Create BEV map_frame...")
    for frames in range(1, max_frame + 1):
        for filename in filelist:
            pm = PixelMapper(quad_coords_list[filename]["pixel"], quad_coords_list[filename]["lonlat"])
            if globals()['point{}'.format(filename)].get(frames) is not None:
                for label in globals()['point{}'.format(filename)].get(frames):
                    uv = (label[1], label[2])
                    lonlat = list(pm.pixel_to_lonlat(uv))
                    li = [frames, label[0], int(lonlat[0][0]), int(lonlat[0][1])]
                    if frames in globals()['BEV_Point{}'.format(filename)]:
                        line = globals()['BEV_Point{}'.format(filename)].get(frames)
                        line.append(li)
                    else:
                        globals()['BEV_Point{}'.format(filename)][frames] = [li]

                    tlabel = tuple(label)
                    if tlabel not in pointset:
                        color = getcolor(abs(label[0]))
                        cv2.circle(map, (int(lonlat[0][0]), int(lonlat[0][1])), 10, color, -1)
                        pointset.add(tlabel)

                    #cv2.imshow('Video', map)
                    #cv2.waitKey(1)

            src = os.path.join(output_path, str(frames) + '.jpg')

            cv2.imwrite(src, map)
    print("Done")

    # #### Create BEV_Result txt files
    # Check the directory already exist
    if not os.path.isdir(os.path.join(original_output_path, 'bev_result')):
        os.mkdir(os.path.join(original_output_path, 'bev_result'))

    is_success = False
    for filename in filelist:
        with open(os.path.join(original_output_path, 'bev_result', 'BEV_{}.txt'.format(filename)), 'w') as f:
            for key in globals()['BEV_Point{}'.format(filename)]:
                for info in globals()['BEV_Point{}'.format(filename)][key]:
                    temp = ''
                    for e in info:
                        temp += str(e) + ' '
                    temp.rstrip()
                    f.write(temp.rstrip() + '\n')
            is_success = True

    return is_success


'''
id 라벨값에 맞춰 색깔을 지정하는 function
'''
def getcolor(idx):
    idx = idx * 3
    return (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255

'''
실제공간과 도면을 mapping해주는 class
'''
class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """

    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape == (4, 2), "Need (4,2) input array"
        assert lonlat_array.shape == (4, 2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array), np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array), np.float32(pixel_array))

    #실제 공간을 도면으로 바꿈
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1, 2)
        assert pixel.shape[1] == 2, "Need (N,2) input array"
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0], 1))], axis=1)
        lonlat = np.dot(self.M, pixel.T)

        return (lonlat[:2, :] / lonlat[2, :]).T

    #도면 공간을 실제 공간으로 바꿈
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1, 2)
        assert lonlat.shape[1] == 2, "Need (N,2) input array"
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0], 1))], axis=1)
        pixel = np.dot(self.invM, lonlat.T)

        return (pixel[:2, :] / pixel[2, :]).T

"""
lonlat에 frame을 한번에 저장하는 function
"""
def save_lonlat_frame(point, pm,frame_num ,input_dir, output_dir):
    map = cv2.imread(input_dir, -1)

    #1541
    for frames in range(1, frame_num): #object ID마다 색깔바꿔서 점찍기
        if point.get(str(frames)) != None:
            for label in point.get(str(frames)) :
                uv = (label[1], label[2])
                lonlat = list(pm.pixel_to_lonlat(uv))
                color = getcolor(abs(label[0]))
                cv2.circle(map, (int(lonlat[0][0]), int(lonlat[0][1])), 3, color, -1)

        src = os.path.join(output_dir, str(frames)+'.jpg')
        cv2.imwrite(src, map)


def save_dict(file, local_init_id, file_num):
    frame = 0
    point = dict()

    # All mot result files start with same ID '1'
    # So we should change the start ID number in each files
    init_id = local_init_id * file_num

    while True:
        line = file.readline()

        if not line:
            break

        info = list(map(int, line[:-1].split(" ")))

        frame = info[0]

        if info[0] in point:
            data = point.get(info[0])
            data.append(list(map(int, info[1:])))
            # Change ID with init_id
            data[-1][0] = init_id + data[-1][0]
        else:
            point[info[0]] = [list(map(int, info[1:]))]
            # Change ID with init_id
            point[info[0]][0][0] = init_id + point[info[0]][0][0]

    file.close()

    return frame, point


def strange_sort(strings, n, m):
    return sorted(strings, key=lambda element: element[n:m])

if __name__ == "__main__":
    start('../output/paper_17person_byte/no_skip/1', '../input/edu_map.png', temp_path="../temp")