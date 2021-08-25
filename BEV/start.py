import sys
import os

input_path = sys.argv[1]
output_path = sys.argv[2]
map_path = sys.argv[3]

if len(sys.argv) != 4:
    print("Argument Error")
    sys.exit()

print(input_path)
print(output_path)


#python start.py ./input/video ./output ./input/map.png


os.system("python ./ProjectMOT/src/demo.py mot --load_model ./ProjectMOT/models/fairmot_dla34.pth --conf_thres 0.4 --input-video "+input_path)
os.system("python mouse_point.py "+map_path)
os.system("python BEV.py "+input_path + " " + output_path + " "+map_path)
os.system("python output_video.py "+ output_path)
# os.system("rm -r temp")