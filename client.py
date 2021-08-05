# Client
import socket

HOST = 'localhost'
PORT = 9009


def getFileFromServer(filename):
    data_transferred = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        sock.sendall(filename.encode())

        data = sock.recv(1024)
        if not data:
            print('File: {0}: Not exist in Server or transport error'.format(filename))
            return

        with open(filename, 'wb') as f:
            try:
                while data:
                    f.write(data)
                    data_transferred += len(data)
                    data = sock.recv(1024)
            except Exception as e:
                print(e)

    print('Transport {0} done. Transferred data: {1}'.format(filename, data_transferred))


filename = 'lite_model.tf'  # lite 파일 이름
getFileFromServer(filename)