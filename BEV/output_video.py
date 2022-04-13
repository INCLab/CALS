import re
import os
import numpy as np
import cv2
import subprocess


def start(output_path):
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

    fps = 25
    frame_array = []
    size = None
    output_idx = 1

    for idx, path in enumerate(paths):
        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

        if len(frame_array) >= 4500 or idx == len(path) - 1:
            writer = cv2.VideoWriter(_gst_write_pipeline(os.path.join(output_path, str(output_idx) + 'output.mp4')),
                                     cv2.CAP_GSTREAMER,
                                     fps,
                                     (1280, 720))
            for i in range(len(frame_array)):
                frame_array[i] = cv2.resize(frame_array[i], dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)
                writer.write(frame_array[i])

            writer.release()
            frame_array.clear()
            output_idx += 1

    print(size)


def _gst_write_pipeline(output_uri):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    # use hardware encoder if found
    if 'omxh264enc' in gst_elements:
        h264_encoder = 'omxh264enc'
    elif 'x264enc' in gst_elements:
        h264_encoder = 'x264enc'
    else:
        raise RuntimeError('GStreamer H.264 encoder not found')
    pipeline = (
            'appsrc ! autovideoconvert ! %s ! mp4mux ! filesink location=%s '
            % (
                h264_encoder,
                output_uri
            )
    )
    return pipeline
