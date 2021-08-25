import re
import os
import numpy as np
import cv2
import sys

output_path = sys.argv[1]
path = os.path.join(output_path, 'map_frame')
paths = [os.path.join(path, i) for i in os.listdir(path) if re.search(".jpg$", i)]

## 정렬 작업
store1, store2, store3, store4 = [], [], [], []
for i in paths:
    if len(i.split('/')[-1]) == 9:
        store4.append(i)
    elif len(i.split('/')[-1]) == 8:
        store3.append(i)
    elif len(i.split('/')[-1]) == 7:
        store2.append(i)
    elif len(i.split('/')[-1]) == 6:
        store1.append(i)

paths = list(np.sort(store1)) + list(np.sort(store2)) + list(np.sort(store3)) + list(np.sort(store4))
# len('ims/2/a/2a.2710.png')

pathOut = os.path.join(output_path, 'output.mov')
fps = 25
frame_array = []
for idx, path in enumerate(paths):
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(pathOut, fourcc, fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()