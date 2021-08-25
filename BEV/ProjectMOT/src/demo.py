from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pymongo import MongoClient

import _init_paths

import shutil
import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
from variable import variableClass
import cv2


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    result_root = "./output"
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    ####video make
    video_file_dir = opt.input_video
    videos_list = []
    result_filename_list = []
    temp_dir = "./temp"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

    for i, j in enumerate(os.listdir(video_file_dir)):
        videos_list.append(os.path.join(video_file_dir, j))
        result_filename_list.append(os.path.join(temp_dir, "result"+str(i)+".txt"))

    print(videos_list)

    # dataloader = datasets.LoadVideo('../videos/our.mp4', opt.img_size)
    image_size_list = []
    for i in range(len(videos_list)):
        image_size_list.append(opt.img_size)


    dataloader = datasets.MultiLoadVideo(videos_list, image_size_list, result_filename_list)
    frame_rate = dataloader.frame_rate_list

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

    #'''
    # eval_seq(track) -> plot_tracking(visualization)
    # eval_seq(num, opt, variable_list, 'mot', collection, show_image=False)
    eval_seq(opt, dataloader, 'mot',
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate)




    for i in range(len(videos_list)):
        if opt.output_format == 'video':
            output_video_path = osp.join(result_root, 'result'+ str(i)+'.mp4')
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame', str(i)),
                                                                                      output_video_path)
            os.system(cmd_str)
    #'''


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
