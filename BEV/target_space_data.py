import numpy as np
import os
import cv2
import sys
import shutil

des_x, des_y = -1, -1
point_txt_dir = "./temp/target_space.txt"
global f

f = open(point_txt_dir, 'w')


def select_points_des(event, x, y, flags, param):
    global des_x, des_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        des_x, des_y = x, y
        print("map coordinate:", des_x, des_y)
        f.write('space_name' + str(des_x) + ' ' + str(des_y)+'\n')
        cv2.circle(map, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


map_name = './input/map.png'

map = cv2.imread(map_name, -1)
map_copy = map.copy()
cv2.namedWindow('map')
cv2.moveWindow('map', 780, 80)
cv2.setMouseCallback('map', select_points_des)
print(map.shape)

while True:
    cv2.imshow('map', map)

    k = cv2.waitKey(1)
    if k == ord('s'):
        break

f.close()