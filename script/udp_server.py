import socket
import json

ip = "192.168.1.20"
port = 29703

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((ip, int(port)))
print(f"UDP {ip} bound on port {port}...")

while True:
    data, addr = s.recvfrom(512)
    print(data)
