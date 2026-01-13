import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import netifaces
import os,json
import argparse

def get_wired_ip():
    ip_list = []
    wired_prefixes = ('en', 'eth')
    for interface in netifaces.interfaces():
        if interface == 'lo':
            continue
        if interface.startswith(wired_prefixes):
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                ip_info = addrs[netifaces.AF_INET][0]
                ip = ip_info['addr']
                ip_list.append(ip)
    return ip_list

def check_server_thread(ip):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((ip, 29701))
        sock.close()
        if result == 0:
            print(f"find WebSocket server: {ip}:29701")
            return ip
    except:
        pass
    return None

def get_server_ip(net):

    ip_list = [f"192.168.{net}.{i}" for i in range(1, 255)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(check_server_thread, ip) for ip in ip_list]
        for future in as_completed(futures):
            ip = future.result()
            if ip:
                return ip

def auto_update_server_ip(args):
    for ip in get_wired_ip():
        server_ip = get_server_ip(ip.split('.')[2])
        if server_ip:
            client_config_path = args.file
            with open(client_config_path, 'r') as f:
                client_config = json.load(f)
            client_config['client']['ip'] = server_ip
            with open(client_config_path, 'w') as f:
                json.dump(client_config, f, indent=4, sort_keys=False)
            break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='', help='store config path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    auto_update_server_ip(args)