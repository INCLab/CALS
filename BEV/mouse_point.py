import os
import cv2
#import DB.database as Database

src_x, src_y = -1, -1
des_x, des_y = -1, -1
drawing = False
f = None

def start(tempfile, frame_dir, map_name):
    global f

    if not os.path.exists(tempfile):
        f = open(tempfile, 'w+')
    else:
        os.remove(tempfile)
        f = open(tempfile, 'w+')

    for i in os.listdir(frame_dir):
        print(i)
        dir = os.path.join(frame_dir, i)
        print(os.listdir(dir)[0])

        f.write(i + '\n')

        frame = cv2.imread(os.path.join(dir, os.listdir(dir)[0]), -1)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.moveWindow('frame', 80, 80)
        cv2.setMouseCallback('frame', select_points_src, frame)
        print(frame.shape)

        map = cv2.imread(map_name, -1)
        cv2.namedWindow('map')
        cv2.moveWindow('map', 780, 80)
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
        #f.write('frame ' + str(src_x) + ' ' + str(src_y) + '\n')
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
    start('../temp/points_ref.txt', '../output/ref_0408/frame/', '../input/edu_map2.png')