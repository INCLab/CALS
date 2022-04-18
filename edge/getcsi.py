import socket

# AF_INET: IPv4, SOCK_STREAM: TCP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 포트 사용중이라 연결할 수 없다는
# WinError 10048 에러 해결를 위해 필요합니다
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_address = ('127.0.0.1', 9999)
print('Starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# 연결을 기다림
sock.listen()

while True:
    #연결을 기다림
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('connection from', client_address)

        #작은 데이터를 받고 다시 전송
        while True:
            data = connection.recv(16)
            print('received {!r}'.format(data))
            if data:
                print('sending data back to the client')
                connection.sendall(data)
            else:
                print('no data from', client_address)
            break
    finally:
        # 연결 모두 지움
        print("closing current connection")
        connection.close()