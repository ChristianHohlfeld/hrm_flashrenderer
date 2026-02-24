from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from multiprocessing.connection import Client
from typing import Any, Dict


@dataclass(frozen=True)
class FlashDaemonAddr:
    host: str
    port: int
    authkey: bytes


def parse_daemon_addr() -> FlashDaemonAddr | None:
    s = os.environ.get("HRM_FLASH_DAEMON")
    if not s:
        return None
    # format: host:port
    if ":" not in s:
        return None
    host, port_s = s.rsplit(":", 1)
    try:
        port = int(port_s)
    except ValueError:
        return None
    auth = os.environ.get("HRM_FLASH_AUTHKEY", "hrmflash").encode("utf-8")
    return FlashDaemonAddr(host=host, port=port, authkey=auth)


def ping(addr: FlashDaemonAddr, timeout_s: float = 0.5) -> bool:
    try:
        c = Client((addr.host, addr.port), authkey=addr.authkey)
        c.send({"cmd": "ping"})
        r = c.recv()
        c.close()
        return bool(r.get("ok"))
    except Exception:
        return False


def generate(addr: FlashDaemonAddr, prompt: str, max_new_tokens: int, prefill_chunk_size: int) -> str:
    c = Client((addr.host, addr.port), authkey=addr.authkey)
    c.send({
        "cmd": "generate",
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "prefill_chunk_size": int(prefill_chunk_size),
    })
    r: Dict[str, Any] = c.recv()
    c.close()
    if not r.get("ok"):
        raise RuntimeError(str(r.get("error", "unknown daemon error")))
    return str(r.get("text", ""))


def shutdown(addr: FlashDaemonAddr) -> None:
    c = Client((addr.host, addr.port), authkey=addr.authkey)
    c.send({"cmd": "shutdown"})
    _ = c.recv()
    c.close()
