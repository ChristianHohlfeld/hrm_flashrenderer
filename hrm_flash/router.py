# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _require_fastapi():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        return FastAPI, HTTPException, JSONResponse, BaseModel
    except Exception as e:
        raise SystemExit(
            "ERR: Missing server dependencies. Install with:\n"
            "  python -m pip install fastapi uvicorn pydantic\n"
            f"Details: {e}"
        )


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str


class _RouterState:
    def __init__(self):
        self.backends: Dict[str, Backend] = {}
        self.short_prompt_tokens: int = 256
        self.medium_prompt_tokens: int = 1200
        self.short_max_new_tokens: int = 192
        self.medium_max_new_tokens: int = 384
        self.long_max_new_tokens: int = 768
        self.request_timeout_s: float = 180.0
        self.health_timeout_s: float = 1.5
        self.chars_per_token: float = 4.0
        self.max_concurrent: int = 8
        self.sem: asyncio.Semaphore | None = None
        self.tokenizer: Any = None
        self.tokenizer_name: str | None = None


STATE = _RouterState()


def _normalize_base_url(url: str) -> str:
    return str(url).strip().rstrip("/")


def _http_json_get(url: str, timeout_s: float) -> tuple[Dict[str, Any], int]:
    req = urllib.request.Request(url, method="GET")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        raw = r.read()
    latency_ms = int((time.perf_counter() - t0) * 1000.0)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"invalid JSON from {url}: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid JSON shape from {url}: expected object")
    return payload, latency_ms


def _http_json_post(url: str, payload: Dict[str, Any], timeout_s: float) -> tuple[Dict[str, Any], int]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"HTTP {e.code} from backend {url}: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"request to backend failed ({url}): {e}") from e
    latency_ms = int((time.perf_counter() - t0) * 1000.0)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"invalid JSON from backend {url}: {e}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"invalid JSON shape from backend {url}: expected object")
    return obj, latency_ms


def _estimate_prompt_tokens(prompt: str) -> tuple[int, str]:
    if STATE.tokenizer is not None:
        try:
            ids = STATE.tokenizer(
                prompt,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            ).get("input_ids", [])
            return max(1, int(len(ids))), "tokenizer"
        except Exception:
            pass
    cpt = max(1e-6, float(STATE.chars_per_token))
    est = int((len(prompt) / cpt) + 0.999)
    return max(1, est), "chars"


def _normalize_backend_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = value.strip().lower()
    aliases = {
        "22gb": "solo_22gb",
        "solo22": "solo_22gb",
        "solo_22": "solo_22gb",
        "solo_22gb": "solo_22gb",
        "quality": "solo_22gb",
        "best": "solo_22gb",
        "pair": "nvlink_pair",
        "nvlink": "nvlink_pair",
        "nvlink_pair": "nvlink_pair",
        "balanced": "nvlink_pair",
        "default": "nvlink_pair",
        "3080": "solo_3080",
        "solo_3080": "solo_3080",
        "fast": "solo_3080",
        "low_latency": "solo_3080",
        "latency": "solo_3080",
    }
    return aliases.get(s, s)


def _collapsed_max_model_mode() -> bool:
    solo = STATE.backends.get("solo_22gb")
    pair = STATE.backends.get("nvlink_pair")
    fast = STATE.backends.get("solo_3080")
    if solo is None or pair is None or fast is None:
        return False
    return bool(
        solo.base_url == pair.base_url
        and solo.base_url != fast.base_url
    )


def _primary_backend(prompt_tokens: int, requested_max_new_tokens: int, route_hint: Optional[str], prefer_backend: Optional[str]) -> str:
    preferred = _normalize_backend_name(prefer_backend)
    if preferred in STATE.backends:
        return preferred

    hint = _normalize_backend_name(route_hint)
    if hint in STATE.backends:
        return hint

    # In collapsed max-model mode (quality + balanced share same endpoint),
    # default to that max-model lane unless caller explicitly requested "fast".
    if _collapsed_max_model_mode():
        return "nvlink_pair"

    if prompt_tokens <= STATE.short_prompt_tokens and requested_max_new_tokens <= STATE.short_max_new_tokens:
        return "solo_3080"
    if prompt_tokens <= STATE.medium_prompt_tokens and requested_max_new_tokens <= STATE.medium_max_new_tokens:
        return "nvlink_pair"
    return "solo_22gb"


def _max_new_limit_for_backend(backend_name: str) -> int:
    if backend_name == "solo_3080":
        return int(STATE.short_max_new_tokens)
    if backend_name == "nvlink_pair":
        return int(STATE.medium_max_new_tokens)
    return int(STATE.long_max_new_tokens)


def _candidate_order(primary: str, allow_failover: bool) -> list[str]:
    if not allow_failover:
        return [primary]
    order_map = {
        "solo_3080": ["solo_3080", "nvlink_pair", "solo_22gb"],
        "nvlink_pair": ["nvlink_pair", "solo_22gb", "solo_3080"],
        "solo_22gb": ["solo_22gb", "nvlink_pair", "solo_3080"],
    }
    order = order_map.get(primary, [primary, "nvlink_pair", "solo_22gb", "solo_3080"])
    out: list[str] = []
    for name in order:
        if name in STATE.backends and name not in out:
            out.append(name)
    return out


def _check_backend_health(backend: Backend, timeout_s: float) -> Dict[str, Any]:
    url = backend.base_url + "/v1/health"
    try:
        payload, latency_ms = _http_json_get(url, timeout_s=timeout_s)
        return {"ok": bool(payload.get("ok", False)), "latency_ms": latency_ms, "url": backend.base_url}
    except Exception as e:
        return {"ok": False, "error": str(e), "url": backend.base_url}


def main():
    FastAPI, HTTPException, JSONResponse, BaseModel = _require_fastapi()

    ap = argparse.ArgumentParser(prog="hrm-flash-router", description="Heterogeneous router for hrm-flash native services.")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--endpoint_solo_22gb", type=str, default="http://127.0.0.1:8081")
    ap.add_argument("--endpoint_nvlink_pair", type=str, default="http://127.0.0.1:8082")
    ap.add_argument("--endpoint_solo_3080", type=str, default="http://127.0.0.1:8083")
    ap.add_argument("--short_prompt_tokens", type=int, default=256)
    ap.add_argument("--medium_prompt_tokens", type=int, default=1200)
    ap.add_argument("--short_max_new_tokens", type=int, default=192)
    ap.add_argument("--medium_max_new_tokens", type=int, default=384)
    ap.add_argument("--long_max_new_tokens", type=int, default=768)
    ap.add_argument("--request_timeout_s", type=float, default=180.0)
    ap.add_argument("--health_timeout_s", type=float, default=1.5)
    ap.add_argument("--max_concurrent", type=int, default=8)
    ap.add_argument("--tokenizer_model", type=str, default=None)
    ap.add_argument("--chars_per_token", type=float, default=4.0)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--disable_tokenizer", action="store_true")
    args = ap.parse_args()

    STATE.backends = {
        "solo_22gb": Backend(name="solo_22gb", base_url=_normalize_base_url(args.endpoint_solo_22gb)),
        "nvlink_pair": Backend(name="nvlink_pair", base_url=_normalize_base_url(args.endpoint_nvlink_pair)),
        "solo_3080": Backend(name="solo_3080", base_url=_normalize_base_url(args.endpoint_solo_3080)),
    }
    STATE.short_prompt_tokens = max(16, int(args.short_prompt_tokens))
    STATE.medium_prompt_tokens = max(STATE.short_prompt_tokens + 1, int(args.medium_prompt_tokens))
    STATE.short_max_new_tokens = max(16, int(args.short_max_new_tokens))
    STATE.medium_max_new_tokens = max(STATE.short_max_new_tokens, int(args.medium_max_new_tokens))
    STATE.long_max_new_tokens = max(STATE.medium_max_new_tokens, int(args.long_max_new_tokens))
    STATE.request_timeout_s = max(1.0, float(args.request_timeout_s))
    STATE.health_timeout_s = max(0.2, float(args.health_timeout_s))
    STATE.chars_per_token = max(1.0, float(args.chars_per_token))
    STATE.max_concurrent = max(1, int(args.max_concurrent))
    STATE.sem = asyncio.Semaphore(STATE.max_concurrent)

    if not bool(args.disable_tokenizer) and args.tokenizer_model:
        try:
            from transformers import AutoTokenizer

            STATE.tokenizer = AutoTokenizer.from_pretrained(
                str(args.tokenizer_model),
                local_files_only=bool(args.local_files_only),
            )
            STATE.tokenizer_name = str(args.tokenizer_model)
            print(f"[router] tokenizer enabled: {STATE.tokenizer_name}", file=sys.stderr, flush=True)
        except Exception as e:
            STATE.tokenizer = None
            STATE.tokenizer_name = None
            print(f"[router] tokenizer unavailable, using char-estimate: {e}", file=sys.stderr, flush=True)
    else:
        print("[router] tokenizer disabled; using char-estimate token routing", file=sys.stderr, flush=True)

    app = FastAPI(title="hrm-flash-router", version="5.1.0")

    class GenerateReq(BaseModel):
        prompt: str
        max_new_tokens: Optional[int] = None
        mode: Optional[str] = None
        show_sources: Optional[bool] = False
        route_hint: Optional[str] = None
        prefer_backend: Optional[str] = None
        allow_failover: bool = True

    @app.get("/v1/health")
    async def health():
        checks = await asyncio.gather(
            asyncio.to_thread(_check_backend_health, STATE.backends["solo_22gb"], STATE.health_timeout_s),
            asyncio.to_thread(_check_backend_health, STATE.backends["nvlink_pair"], STATE.health_timeout_s),
            asyncio.to_thread(_check_backend_health, STATE.backends["solo_3080"], STATE.health_timeout_s),
        )
        backend_health = {
            "solo_22gb": checks[0],
            "nvlink_pair": checks[1],
            "solo_3080": checks[2],
        }
        all_ok = bool(backend_health["solo_22gb"]["ok"] or backend_health["nvlink_pair"]["ok"] or backend_health["solo_3080"]["ok"])
        return {
            "ok": all_ok,
            "router": {
                "tokenizer": STATE.tokenizer_name,
                "collapsed_max_model_mode": _collapsed_max_model_mode(),
                "request_timeout_s": STATE.request_timeout_s,
                "max_concurrent": STATE.max_concurrent,
                "thresholds": {
                    "short_prompt_tokens": STATE.short_prompt_tokens,
                    "medium_prompt_tokens": STATE.medium_prompt_tokens,
                    "short_max_new_tokens": STATE.short_max_new_tokens,
                    "medium_max_new_tokens": STATE.medium_max_new_tokens,
                    "long_max_new_tokens": STATE.long_max_new_tokens,
                },
            },
            "backends": backend_health,
        }

    @app.post("/v1/generate")
    async def generate(req: GenerateReq):
        if not req.prompt or not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt required")
        assert STATE.sem is not None
        async with STATE.sem:
            requested_max_new_tokens = int(req.max_new_tokens) if req.max_new_tokens is not None else 256
            requested_max_new_tokens = max(1, min(4096, requested_max_new_tokens))
            prompt_tokens, estimator = _estimate_prompt_tokens(req.prompt)

            primary = _primary_backend(
                prompt_tokens=prompt_tokens,
                requested_max_new_tokens=requested_max_new_tokens,
                route_hint=req.route_hint,
                prefer_backend=req.prefer_backend,
            )
            candidates = _candidate_order(primary=primary, allow_failover=bool(req.allow_failover))

            attempted: list[str] = []
            errors: list[str] = []
            for backend_name in candidates:
                backend = STATE.backends[backend_name]
                backend_max_new_tokens = min(requested_max_new_tokens, _max_new_limit_for_backend(backend_name))
                payload = {
                    "prompt": req.prompt,
                    "max_new_tokens": int(backend_max_new_tokens),
                    "mode": req.mode,
                    "show_sources": bool(req.show_sources),
                }
                attempted.append(backend_name)
                try:
                    body, latency_ms = await asyncio.to_thread(
                        _http_json_post,
                        backend.base_url + "/v1/generate",
                        payload,
                        STATE.request_timeout_s,
                    )
                    if not bool(body.get("ok", True)):
                        raise RuntimeError(str(body.get("detail") or body.get("error") or "backend returned ok=false"))
                    out = dict(body)
                    out["route"] = {
                        "primary": primary,
                        "selected": backend_name,
                        "attempted": attempted,
                        "estimator": estimator,
                        "prompt_tokens": prompt_tokens,
                        "requested_max_new_tokens": requested_max_new_tokens,
                        "backend_max_new_tokens": backend_max_new_tokens,
                        "latency_ms": latency_ms,
                    }
                    return JSONResponse(out)
                except Exception as e:
                    errors.append(f"{backend_name}: {e}")

            raise HTTPException(
                status_code=503,
                detail={
                    "error": "all backends failed",
                    "primary": primary,
                    "attempted": attempted,
                    "backend_errors": errors,
                },
            )

    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
