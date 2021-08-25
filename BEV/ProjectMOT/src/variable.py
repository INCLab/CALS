

import datasets.dataset.jde as datasets
import os
import os.path as osp



class variableClass:  # for inference
    def __init__(self, opt, video, result_filename, result_root, i):

        self.dataloader = datasets.LoadVideo(video, opt.img_size)
        self.result_filename = os.path.join(result_filename)
        self.frame_rate = self.dataloader.frame_rate
        self.frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame' + str(i))
