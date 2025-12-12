import asyncio
import websockets
import logging,json
import argparse

logging.basicConfig(level=logging.INFO)

connected_clients = set()

async def handle_client(websocket):
    connected_clients.add(websocket)
    client_addr = websocket.remote_address
    logging.info(f"新客户端连接: {client_addr}")

    try:
        async for message in websocket:
            logging.info(f"收到来自 {client_addr} 的消息: {json.loads(message)['data']}")
            #logging.info(f"收到来自 {client_addr} 的消息")

    except websockets.exceptions.ConnectionClosed:
        logging.info(f"客户端断开: {client_addr}")
    finally:
        # 客户端断开时从集合中移除
        connected_clients.discard(websocket)

async def main(ip,port):
    server = await websockets.serve(handle_client, ip, port)
    logging.info(f"WebSocket 服务已启动，监听 ws://{ip}:{port}")

    # 保持服务运行
    await server.wait_closed()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",default="192.168.1.20",type=str)
    parser.add_argument("--port",default=29701,type=int)
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.ip,args.port))
    except KeyboardInterrupt:
        logging.info("服务已停止")

