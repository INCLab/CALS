import socket

HOST = '192.9.201.242'
PORT = 9009


def sendCSIPacketToServer(csi):

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((HOST, PORT))
            sock.sendall(csi.encode())
        finally:
            sock.close()


for i in range(0, 100):
    list = [i*2 for j in range(0, 64)]

    csi = ""
    for l in list:
        csi += str(l) + ' '
    print(csi)
    sendCSIPacketToServer(csi)
