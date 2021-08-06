# Server
import socketserver
from os.path import exists

HOST = '221.148.100.53'
PORT = 9009


class MyTcpHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data_transferred = 0
        print('{0} is connected'.format(self.client_address[0]))
        filename = self.request.recv(1024)
        filename = filename.decode()

        if not exists(filename):
            return

        print('Start transport {0}'.format(filename))
        with open(filename, 'rb') as f:
            try:
                data = f.read(1024)
                while data:
                    data_transferred += self.request.send(data)
                    data = f.read(1024)
            except Exception as e:
                print(e)

        print('Transport complete {0}, Transferred data: {1}'.format(filename, data_transferred))


def runServer(HOST, PORT):
    print('==== Start File Server ====')
    print('==== Exit with Ctrl + C ====')

    try:
        server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print('==== Exit File server ====')


runServer(HOST, PORT)
