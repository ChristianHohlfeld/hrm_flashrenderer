import socket
import asyncio
import json
import os
from aiohttp import web

UDP_IP = "0.0.0.0"
UDP_PORT = 9999
STOP_FLAG = "../stop_engine.flag"

async def udp_receiver(app):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    print(f"UDP Receiver started on port {UDP_PORT}")
    
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            if len(data) >= 28:
                import struct
                # step (I), loss (f), cos (f), euclid (f), x (f), y (f), z (f)
                vals = struct.unpack("Iffffff", data[:28])
                packet = {
                    "step": vals[0], "loss": vals[1], "cos_sim": vals[2],
                    "euclid_dist": vals[3], "proj_x": vals[4], "proj_y": vals[5], "proj_z": vals[6]
                }
                msg = json.dumps(packet)
                for ws in app['sockets']:
                    await ws.send_str(msg)
        except BlockingIOError:
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['sockets'].add(ws)
    try:
        async for msg in ws:
            pass
    finally:
        request.app['sockets'].remove(ws)
    return ws

async def handle_index(request):
    return web.FileResponse('index.html')

async def handle_stop(request):
    with open(STOP_FLAG, "w") as f:
        f.write("STOP")
    return web.Response(text="Stop signal sent")

async def on_startup(app):
    if os.path.exists(STOP_FLAG):
        os.remove(STOP_FLAG)
    asyncio.create_task(udp_receiver(app))

app = web.Application()
app['sockets'] = set()
app.router.add_get('/', handle_index)
app.router.add_get('/ws', handle_ws)
app.router.add_post('/stop', handle_stop)
app.on_startup.append(on_startup)

if __name__ == '__main__':
    web.run_app(app, port=9998)
