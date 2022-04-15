import os
import seaborn as sns
from matplotlib import pyplot as plt
'''
    1920 x 1080을 120 x 120 정사각형으로 나누면 16 x 9
'''
BLOCK_SIZE = 120


def start(file_path, save_path):
    ht_map = [[0 for col in range(16)] for row in range(9)]

    for case in range(1, 51):
        f_path = os.path.join(file_path, str(case), 'bev_result/')
        file_list = os.listdir(f_path)
        print(f_path)

        for file_name in file_list:
            print(file_name)
            f = open(os.path.join(f_path, file_name), 'r')
            data = f.read()
            data = data.split('\n')
            data.remove('')

            coord_list = []

            for line in data:
                coord_list.append(list(map(int, line.split(' ')[2:])))

            for coord in coord_list:
                x = coord[0]
                y = coord[1]

                loc_x = int(x / BLOCK_SIZE)
                loc_y = int(y / BLOCK_SIZE)

                try:
                    ht_map[loc_y][loc_x] += 1
                except:
                    continue

    sns.heatmap(ht_map, linewidths=0.1, linecolor="black", annot=True)
    #plt.title('BEV Heatmap')
    plt.show()
    # plt.savefig(save_path)




if __name__ == '__main__':
    start('../output/paper_eval_data/no_skip/', '../output')
