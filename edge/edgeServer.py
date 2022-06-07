# Todo: 젯슨나노에 matplotlib 설치
import socketserver
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

HOST = '192.9.203.146'
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
            print(csi_data)


def runServer(HOST, PORT):
    print('==== Start Edge Server ====')
    print('==== Exit with Ctrl + C ====')

    try:
        server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
        server.serve_forever()  # server_forever()메소드를 호출하면 클라이언트의 접속 요청을 받을 수 있음

    except KeyboardInterrupt:
        print('==== Exit Edge server ====')


runServer(HOST, PORT)
