# -*- coding:utf-8 -*-
import numpy as np
import os
import cv2
import sys
import shutil

des_x, des_y = -1, -1
point_txt_dir = "../temp/target_space.txt"
map_dir = '../input/map.png'
global f
global space_name


def delete_data(name, txt_dir):
    with open(txt_dir, 'rt', encoding='UTF8') as data:
        lines = data.readlines()

        modified_data = []
        for tar_info in lines:
            if name + ' ' not in tar_info:
                modified_data.append(tar_info)

    with open(txt_dir, 'w', encoding='UTF8') as data:
        for modified_tar in modified_data:
            data.write(modified_tar)


def get_space_tar_list(txt_dir):
    try:
        with open(txt_dir, 'rt', encoding='UTF8') as data:
            lines = data.readlines()
            tar_list = []

            for line in lines:
                info_list = line.split(' ')
                tar_list.append(info_list[0])

            tar_list = list(set(tar_list))

            print("#### 타겟 리스트 ####")

            if not tar_list:
                print("데이터가 없습니다")

            for i in range(0, len(tar_list)):
                print("{0}. {1}".format(i+1, tar_list[i]))

            print("#################")
            return tar_list
    except Exception as e:
        print("타겟 리스트가 비어있습니다")


def write_data(txt_dir, map_name, target_list):

    def select_points_des(event, x, y, flags, param):
        global des_x, des_y, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            des_x, des_y = x, y
            print("map coordinate:", des_x, des_y)
            f.write(space_name + ' ' + str(des_x) + ' ' + str(des_y) + '\n')
            cv2.circle(map, (x, y), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    with open(txt_dir, 'a', encoding='UTF8') as f:
        print("Enter the space name: ")
        space_name = input()

        if target_list:
            while space_name in target_list:
                print("이미 존재하는 타겟입니다. \n")
                print("Enter the space name: ")
                space_name = input()

        map = cv2.imread(map_name, -1)
        cv2.namedWindow('map')
        cv2.moveWindow('map', 780, 80)
        cv2.setMouseCallback('map', select_points_des)
        print("Map shape{0}".format(map.shape))

        while True:
            cv2.imshow('map', map)

            k = cv2.waitKey(1)
            if k == ord('s'):
                break


# Display target list
tar_list = get_space_tar_list(point_txt_dir)

# Write Data
write_data(point_txt_dir, map_dir, tar_list)

# Delete data
tar_list = get_space_tar_list(point_txt_dir)
if not tar_list:
    print("타겟 리스트가 없습니다.")
else:
    print('제거할 타겟이름을 입력하세요: ')
    del_tar = input()

    while del_tar not in tar_list:
        print("\n해당하는 타겟이 없습니다 ")
        print('제거할 타겟이름을 입력하세요: ')
        del_tar = input()

    delete_data(del_tar, point_txt_dir)

tar_list = get_space_tar_list(point_txt_dir)