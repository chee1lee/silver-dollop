import socket

Host = 'localhost'
Port = 8080

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((Host, Port))

while True:
    rcv = client.recv(1024)
    # receive msg : state reward done
    print(rcv.decode())
    action = input('action:')
    if action == 'exit':
        break
    else:
        client.send(action.encode())

client.close()
