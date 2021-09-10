# Video에서 Frame 추출하기 (신규 Framework인 FastMOT와의 호환용)

import cv2
import sys
import os

print('Video Frame Extracting.....')

if len(sys.argv) != 2:
    print("Invalid Parameter")
    sys.exit()

videos_dir = sys.argv[1]

# 폴더 내 모든 파일에 대해 Loop
for f in os.listdir(videos_dir):
    # OpenCV로 Video File Open 
    videofile = cv2.VideoCapture(videos_dir + '/' + f)
    
    # Video File이 열렸으면
    while videofile.isOpened():
        # Frame Load 
        ret, frame = videofile.read()
        
        if not ret:
            print('Invaild Frame')
            sys.exit()

        # Frame 저장
        imgfilename = 'output/frame/' + os.path.splitext(f)[0] + '_frame.jpg'

        cv2.imwrite(imgfilename, frame)
        print('Saved to ' + imgfilename)

        # Break
        break
    
    # Release
    videofile.release()
