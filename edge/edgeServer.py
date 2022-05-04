# Todo: 젯슨나노에 matplotlib 설치
import socketserver
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

HOST = '221.148.100.53'
PORT = 9010

selected_mac = 'dca6328e1dcb'
show_packet_length = 100
GAP_PACKET_NUM = 20


class MyTcpHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print('{0} is connected'.format(self.client_address[0]))
        buffer = self.request.recv(2048)  # receive data
        buffer = buffer.decode()

        if not buffer:
            print("Fail to receive!")
            return
        else:
            csi_data = list(map(float, buffer.split(' ')))

            # real time plot
            idx += 1
            for i, y in enumerate(y_list):
                del y[0]
                new_y = csi_data[i]
                y.append(new_y)
                line_list[i].set_xdata(x)
                line_list[i].set_ydata(y)

                # Min-Max Gap
                if gap_count == 0:
                    minmax.append([new_y, new_y])
                else:
                    # Update min
                    if minmax[i][0] > new_y:
                        minmax[i][0] = new_y
                    # Update max
                    if minmax[i][1] < new_y:
                        minmax[i][1] = new_y

            gap_list = []
            for mm in minmax:
                gap_list.append(mm[1] - mm[0])

            gap = max(gap_list)

            Artist.remove(txt)
            txt = ax.text(40, 1600, 'Amp Min-Max Gap: {}'.format(gap), fontsize=14)
            gap_count += 1
            if gap_count == 20:
                gap_count = 0
                minmax = []

            fig.canvas.draw()
            fig.canvas.flush_events()


def runServer(HOST, PORT):
    print('==== Start Edge Server ====')
    print('==== Exit with Ctrl + C ====')

    try:
        # ####### RealTime plot ###############
        x = np.arange(0, show_packet_length, 1)
        y_list = []

        for i in range(0, 64):
            y_list.append([0 for j in range(0, show_packet_length)])

        plt.ion()

        fig, ax = plt.subplots(figsize=(12, 8))

        line_list = []

        for y in y_list:
            line, = ax.plot(x, y, alpha=0.5)
            line_list.append(line)

        plt.title('{}'.format(selected_mac), fontsize=18)
        plt.ylabel('Signal Amplitude', fontsize=16)
        plt.xlabel('Packet', fontsize=16)
        plt.ylim(0, 1500)

        # Amp Min-Max gap text on plot figure
        txt = ax.text(40, 1600, 'Amp Min-Max Gap: None', fontsize=14)
        gap_count = 0
        minmax = []

        idx = show_packet_length - 1

        server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
        server.serve_forever()  # server_forever()메소드를 호출하면 클라이언트의 접속 요청을 받을 수 있음
    except KeyboardInterrupt:
        print('==== Exit Edge server ====')


runServer(HOST, PORT)
