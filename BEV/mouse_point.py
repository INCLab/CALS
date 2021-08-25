import numpy as np
import os
import cv2
import sys
import shutil

src_x, src_y = -1,-1
des_x, des_y = -1,-1
point_txt_dir = "./temp/points.txt"
global f

if not os.path.exists(point_txt_dir):
    f = open(point_txt_dir, 'w')
else:
    os.remove(point_txt_dir)
    f = open(point_txt_dir, 'w')

def select_points_src(event, x, y, flags, param):
    global src_x, src_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x, y
        print("frame coordinate:", src_x, src_y)
        f.write('frame '+str(src_x) + ' ' + str(src_y)+'\n')
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def select_points_des(event, x, y, flags, param):
    global des_x, des_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        des_x, des_y = x, y
        print("map coordinate:", des_x, des_y)
        f.write('map '+str(des_x) + ' ' + str(des_y)+'\n')
        cv2.circle(map, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

pixel_list = []
map_list = []



map_name = sys.argv[1]
frame_dir = "./output/frame"

for i in os.listdir(frame_dir):
    print(i)
    dir = os.path.join(frame_dir,i)
    print(os.listdir(dir)[0])
    frame = cv2.imread(os.path.join(dir, os.listdir(dir)[0]),-1)

    frame_copy = frame.copy()
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 80, 80)
    cv2.setMouseCallback('frame', select_points_src)
    print(frame.shape)

    map = cv2.imread(map_name, -1)
    map_copy = map.copy()
    cv2.namedWindow('map')
    cv2.moveWindow('map', 780, 80)
    cv2.setMouseCallback('map', select_points_des)
    print(map.shape)



    while(1):
        cv2.imshow('map', map)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == ord('s'):
            break;

f.close()