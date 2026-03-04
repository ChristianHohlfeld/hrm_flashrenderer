import socket
import asyncio
import json
import os
import struct
import psutil
import pynvml
from transformers import AutoTokenizer
from aiohttp import web

# Initialize Global Tokenizer
try:
    pynvml.nvmlInit()
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
    print(f"[*] NVML Initialized: {GPU_COUNT} GPUs detected.")
except Exception as e:
    print(f"[!] NVML Init failed: {e}")
    GPU_COUNT = 0

try:
    TOKENIZER = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
except Exception as e:
    print(f"[!] Tokenizer load failed: {e}")
    TOKENIZER = None

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
            data_len = len(data)
            packet = None
            
            if data_len == 116:
                # Legacy / run_llm_telemetry.sh style (1 GPU)
                vals = struct.unpack("I" + "f"*12 + "64s", data[:116])
                packet = {
                    "step": vals[0], "loss": vals[1], "cos_sim": vals[2],
                    "euclid_dist": vals[3], "proj_x": vals[4], "proj_y": vals[5], "proj_z": vals[6],
                    "gpu_util": vals[7:10], "gpu_mem": vals[10:13],
                    "prompt": vals[13].decode('utf-8', errors='ignore').strip('\x00'),
                    "model_id": "Unknown"
                }
            elif data_len >= 180:
                # Extended / run.sh style with model info
                vals = struct.unpack("I" + "f"*12 + "64s64s", data[:180])
                packet = {
                    "step": vals[0], "loss": vals[1], "cos_sim": vals[2],
                    "euclid_dist": vals[3], "proj_x": vals[4], "proj_y": vals[5], "proj_z": vals[6],
                    "gpu_util": vals[7:10], "gpu_mem": vals[10:13],
                    "model_id": vals[13].decode('utf-8', errors='ignore').strip('\x00'),
                    "prompt": vals[14].decode('utf-8', errors='ignore').strip('\x00')
                }
            elif data_len == 348:
                # 3 x 116 bytes or special multi-GPU packet
                # We'll take the first one or combine them
                vals = struct.unpack("I" + "f"*12 + "64s", data[:116])
                packet = {
                    "step": vals[0], "loss": vals[1], "cos_sim": vals[2],
                    "euclid_dist": vals[3], "proj_x": vals[4], "proj_y": vals[5], "proj_z": vals[6],
                    "gpu_util": vals[7:10], "gpu_mem": vals[10:13],
                    "prompt": vals[13].decode('utf-8', errors='ignore').strip('\x00'),
                    "model_id": "Multi-GPU Stream"
                }

            if packet:
                # Enrich with system stats
                packet["cpu_util"] = psutil.cpu_percent()
                packet["ram_util"] = psutil.virtual_memory().percent
                
                # Enrich with GPU stats from NVML (Real Load Spread)
                g_utils = []
                g_mems = []
                for i in range(min(3, GPU_COUNT)):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        g_utils.append(float(util.gpu))
                        g_mems.append(float(mem.used) / 1024 / 1024) # MB
                    except:
                        g_utils.append(0.0)
                        g_mems.append(0.0)
                
                # Pad to 3
                while len(g_utils) < 3: g_utils.append(0.0)
                while len(g_mems) < 3: g_mems.append(0.0)
                
                packet["gpu_util"] = g_utils
                packet["gpu_mem"] = g_mems
                
                # Token Decoding and SPAM filtering
                prompt_text = packet.get("prompt", "")
                token_id = packet.get("step", 0) # Often overloaded or 0
                is_system = prompt_text in ["SYSTEM_IDLE", "AWAITING_PROMPT", "llm_engine"] or prompt_text.startswith("TokenID:")
                
                token_text = ""
                if not is_system and TOKENIZER and token_id > 0:
                    try:
                        token_text = TOKENIZER.decode([token_id])
                    except:
                        token_text = f"[{token_id}]"
                
                packet["token_text"] = token_text
                packet["is_system"] = is_system
                
                msg = json.dumps(packet)
                for ws in app['sockets']:
                    await ws.send_str(msg)
        except BlockingIOError:
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[!] Relay parse error: {e} (len={len(data)})")
            await asyncio.sleep(0.1)

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, 'index.html')
    return web.FileResponse(index_path)

async def handle_stop(request):
    with open(STOP_FLAG, "w") as f:
        f.write("STOP")
    return web.Response(text="Stop signal sent")

async def handle_send_prompt(request):
    data = await request.json()
    prompt = data.get("prompt", "")
    if prompt:
        # Crucial: the engine polls /tmp/deepseek_prompt.txt
        # We ensure it exists and is writable.
        try:
            with open("/tmp/deepseek_prompt.txt", "w") as f:
                f.write(prompt)
            print(f"[*] Prompt relayed to engine: {prompt[:30]}...")
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)
    return web.json_response({"status": "error", "message": "empty prompt"}, status=400)

async def heartbeat(app):
    while True:
        try:
            if not app['sockets']:
                await asyncio.sleep(1)
                continue
                
            g_utils = []
            g_mems = []
            for i in range(min(3, GPU_COUNT)):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    g_utils.append(float(util.gpu))
                    g_mems.append(float(mem.used) / 1024 / 1024)
                except:
                    g_utils.append(0.0)
                    g_mems.append(0.0)
            
            while len(g_utils) < 3: g_utils.append(0.0)
            while len(g_mems) < 3: g_mems.append(0.0)
            
            status_msg = "ONLINE (WS_ACT)"
            if os.path.exists("/tmp/deepseek_status.txt"):
                try:
                    with open("/tmp/deepseek_status.txt", "r") as f:
                        status_msg = f.read().strip()
                except:
                    pass

            packet = {
                "step": 0, "loss": 0, "cos_sim": 0, "euclid_dist": 0,
                "proj_x": 0, "proj_y": 0, "proj_z": 0,
                "gpu_util": g_utils, "gpu_mem": g_mems,
                "cpu_util": psutil.cpu_percent(),
                "ram_util": psutil.virtual_memory().percent,
                "model_id": "System Monitor",
                "prompt": "SYSTEM_IDLE",
                "is_system": True,
                "sys_state": status_msg
            }
            msg = json.dumps(packet)
            for ws in list(app['sockets']):
                try:
                    await ws.send_str(msg)
                except:
                    pass
        except Exception as e:
            print(f"Heartbeat error: {e}")
        await asyncio.sleep(1)

async def on_startup(app):
    if os.path.exists(STOP_FLAG):
        os.remove(STOP_FLAG)
    asyncio.create_task(udp_receiver(app))
    asyncio.create_task(heartbeat(app))

app = web.Application()
app['sockets'] = set()
app.router.add_get('/', handle_index)
app.router.add_get('/ws', handle_ws)
app.router.add_post('/stop', handle_stop)
app.router.add_post('/send_prompt', handle_send_prompt)
app.on_startup.append(on_startup)

if __name__ == '__main__':
    print("[*] Killing any existing processes on ports 9998/9999...")
    os.system("fuser -k -9 9998/tcp 9999/udp >/dev/null 2>&1")
    print("[*] Starting Hardware Telemetry Relay...")
    web.run_app(app, host='0.0.0.0', port=9998)
