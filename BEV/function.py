import numpy as np
import cv2
import os


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

def save_dict(file):
    ##################################################
    frame = 0
    point = dict()
    while True:
        line = file.readline()

        if not line:
            break

        info = line[:-1].split(" ")

        frame = info[0]

        if info[0] in point:
            line = point.get(info[0])
            line.append(list(map(int, info[1:])))
        else:
            point[info[0]] = [list(map(int, info[1:]))]

    file.close()

    return frame, point
    ###########################################################################
