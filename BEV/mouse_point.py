import os
import cv2

src_x, src_y = -1, -1
des_x, des_y = -1, -1
drawing = False
f = None

def start(tempfile, ref_video_dir, map_name):
    global f

    if not os.path.exists(tempfile):
        f = open(tempfile, 'w+')
    else:
        os.remove(tempfile)
        f = open(tempfile, 'w+')

    if not os.listdir(ref_video_dir):
        print("RefVideoFile NotFound!")
        exit()

    for i in os.listdir(ref_video_dir):
        camera_channel = i[:4]
        print(i)

        # Capture video frame
        vidcap = cv2.VideoCapture(os.path.join(ref_video_dir, i))

        ret, img = vidcap.read()

        # Get 30th frame image
        while vidcap.isOpened():
            ret, img = vidcap.read()

            if int(vidcap.get(1)) == 5:
                break

        f.write(camera_channel + '\n')

        frame = img
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.moveWindow('frame', 80, 80)
        cv2.setMouseCallback('frame', select_points_src, frame)
        print(frame.shape)

        map = cv2.imread(map_name, -1)
        cv2.namedWindow('map')
        cv2.moveWindow('map', 80, 80)
        cv2.setMouseCallback('map', select_points_des, map)
        print(map.shape)

        while True:
            cv2.imshow('map', map)
            cv2.imshow('frame', frame)

            k = cv2.waitKey(1)
            if k == ord('s'):
                break

    cv2.destroyAllWindows()
    f.close()


def select_points_src(event, x, y, flags, param):
    global f, src_x, src_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x, y
        print("frame coordinate:", src_x, src_y)
        f.write('frame ' + str(src_x) + ' ' + str(src_y) + '\n')
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def select_points_des(event, x, y, flags, param):
    global f, des_x, des_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        des_x, des_y = x, y
        print("map coordinate:", des_x, des_y)
        f.write('map ' + str(des_x) + ' ' + str(des_y) + '\n')
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

if __name__ == '__main__':
    start('test/testpoint.txt', 'test/ref_point_video/map1/', 'test/map_img/ref0807_1.png')