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

'''
pixel : 실제 공간
lonloat : 도면 공간
오른쪽 위, 왼쪽 위, 왼쪽 아래, 오른쪽 아래 순서
'''


def start(output_path, map_path, temp_path='../temp'):

    original_output_path = output_path
    output_path = os.path.join(output_path, 'map_frame')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    # MOT result file list
    filelist = []
    for file in os.listdir(original_output_path):
        if file.endswith(".txt"):
            filelist.append(file.rstrip('.txt'))

    f = open(os.path.join(temp_path, 'points.txt'), 'r')
    data_lines = f.read()
    data_lines= data_lines.split('\n')

    point = []
    frame_point = []
    map_point = []

    for line in data_lines:
        point.append(line.split(' '))

    for i in point:
        if i[0] == 'map':
            map_point.append([i[1], i[2]])
        elif i[0] == 'frame':
            frame_point.append([i[1], i[2]])

    quad_coords = {
        "pixel": np.array([
            [frame_point[0][0],   frame_point[0][1]],  # Third lampost top right
            [frame_point[1][0],   frame_point[1][1]],  # Corner of white rumble strip top left
            [frame_point[2][0],   frame_point[2][1]],  # Corner of rectangular road marking bottom left
            [frame_point[3][0],   frame_point[3][1]]  # Corner of dashed line bottom right
        ]),
        "lonlat": np.array([
            [map_point[0][0],   map_point[0][1]],  # Third lampost top right
            [map_point[1][0],   map_point[1][1]],  # Corner of white rumble strip top left
            [map_point[2][0],   map_point[2][1]],  # Corner of rectangular road marking bottom left
            [map_point[3][0],   map_point[3][1]]  # Corner of dashed line bottom right
        ])
    }

    # Read MOT result txt file
    file = open(os.path.join(original_output_path, filelist[0] + '.txt'), 'r')
    time_list, frame_list, point_dict = save_dict(file)
    bev_point = dict()

    map = cv2.imread(str(map_path), -1)
    pointset = set()

    print("Create BEV map_frame...")
    filename = filelist[0]
    for frames in range(1, frame_list[-1] + 1):
        pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])
        if point_dict.get(frames) is not None:
            for coord in point_dict.get(frames):
                uv = (coord[1], coord[2])
                lonlat = list(pm.pixel_to_lonlat(uv))
                li = [frames, coord[0], int(lonlat[0][0]), int(lonlat[0][1])]
                if frames in bev_point:
                    line = bev_point.get(frames)
                    line.append(li)
                else:
                   bev_point[frames] = [li]

                tcoord = tuple(coord)
                if tcoord not in pointset:
                    color = getcolor(abs(coord[0]))
                    cv2.circle(map, (int(lonlat[0][0]), int(lonlat[0][1])), 10, color, -1)
                    pointset.add(tcoord)

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

    time_idx = 0
    with open(os.path.join(original_output_path, 'bev_result', 'BEV_{}.txt'.format(filename)), 'w') as f:
        for key in bev_point:
            for info in bev_point[key]:
                temp = ''
                for e in info:
                    temp += str(e) + ' '
                temp.rstrip()
                temp = time_list[time_idx] + ' ' + temp
                f.write(temp.rstrip() + '\n')
                time_idx += 1
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


def save_dict(file):
    frame_list = []
    time_list = []
    point = dict()

    while True:
        line = file.readline()

        if not line:
            break

        current_time = line.split(" ")[0]
        time_list.append(current_time)
        # If Time is added, line[1:-1] or line[:-1]
        info = list(map(int, line.split(" ")[1:]))

        current_frame = info[0]
        frame_list.append(current_frame)

        if current_frame in point:
            data = point.get(current_frame)
            data.append(list(map(int, info[1:])))
        else:
            point[current_frame] = [list(map(int, info[1:]))]

    file.close()

    return time_list, frame_list, point


if __name__ == "__main__":
    test_list = ['lefttop', 'middle', 'rightbottom', 'righttop', 'all_loc', 'testset']

    for case in test_list:
        start('../data/0502_csi_mot/' + case + '/mot', '../temp/csi_grid_map.png')