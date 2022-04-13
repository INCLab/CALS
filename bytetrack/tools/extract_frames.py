import cv2
import os

video_path = "video/paper_5person_video"
video_fname_list = os.listdir(video_path)
save_dir = "output/paper_5person_video"
os.makedirs(save_dir, exist_ok=True)

FPS = 20


def extract_frames(mot_videoPath, outputPath, fps):
    for video_fname in video_fname_list:
        vpath = os.path.join(video_path, video_fname)

        vidcap = cv2.VideoCapture(vpath)

        total_frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_video = int(DIVIDE_SECOND * FPS)

        buffer_list = []
        frame_buffer = []

        count = 1

        while vidcap.isOpened() and total_frame_num > int(vidcap.get(cv2.CAP_PROP_POS_FRAMES)):
            ret, image = vidcap.read()
            if ret is False:
                print("{} frame can't read".format(int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))))
                continue
            height, width, layers = image.shape
            size = (width, height)

            frame_buffer.append(image)

            # 비디오 할당되는 frame 수를 넘어갈때마다 현재 buffer는 저장하고, 새로운 buffer 생성
            if int(vidcap.get(cv2.CAP_PROP_POS_FRAMES)) % frames_per_video == 0 \
                    and int(vidcap.get(cv2.CAP_PROP_POS_FRAMES)) != 0:
                print('video {} buffer is added!'.format(count))
                buffer_list.append(frame_buffer)
                frame_buffer = []
                count += 1

        vidcap.release()

        # 마지막 iter에서 buffer에 남은 image가 있는경우 buffer_list에 추가
        if len(frame_buffer) != 0:
            buffer_list.append(frame_buffer)


        for idx, buffer in enumerate(buffer_list):
            v_save_dir = os.path.join(save_dir, str(idx+1))
            os.makedirs(v_save_dir, exist_ok=True)
            video_save_path = os.path.join(v_save_dir, '{}.mp4'.format(video_fname[:4]))
            out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'X264'), FPS, size)

            for img in buffer:
                out.write(img)

            out.release()

        print('{} process is done'.format(video_fname))