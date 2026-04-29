# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hrm_flash.deepseek_export import export_dsi_v4


def _safe_name(s: str) -> str:
    return s.replace("\\", "--").replace("/", "--").replace(":", "_")


def resolve_deepseek_engine_bin(repo_root: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise RuntimeError(f"DeepSeek engine binary not found: {p}")
        return p

    candidates = [
        repo_root / ".run" / "bin" / "deepseek_engine",
        repo_root / "deepseek_engine",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise RuntimeError(
        "DeepSeek engine binary not found. Expected one of:\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}"
    )


def ensure_deepseek_model(
    model_source: str,
    model_bin: str | None = None,
    model_quant: str = "q8",
    local_files_only: bool = False,
    project_root: Path | None = None,
) -> tuple[Path, str]:
    model_quant = str(model_quant).strip().lower()
    if model_quant != "q8":
        raise RuntimeError(
            f"unsupported model_quant='{model_quant}' (supported: q8 only; native q4 is not enabled in production mainline)"
        )

    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    if model_bin:
        mb = Path(model_bin).expanduser().resolve()
        if not mb.is_file():
            raise RuntimeError(f"--model_bin does not exist: {mb}")
        return mb, model_source

    ms = Path(model_source).expanduser()
    if ms.is_file() and ms.suffix.lower() == ".bin":
        return ms.resolve(), model_source
    if ms.is_dir():
        existing = ms / f"model_{model_quant}.bin"
        if existing.is_file():
            return existing.resolve(), str(ms.resolve())
        out = ms / f"model_{model_quant}.bin"
        export_dsi_v4(str(ms.resolve()), out, local_files_only=local_files_only, quant=model_quant)
        return out.resolve(), str(ms.resolve())

    target_dir = root / "llm_models" / _safe_name(model_source)
    out = target_dir / f"model_{model_quant}.bin"
    if not out.is_file():
        target_dir.mkdir(parents=True, exist_ok=True)
        export_dsi_v4(model_source, out, local_files_only=local_files_only, quant=model_quant)
    return out.resolve(), model_source


def ensure_deepseek_q8_model(
    model_source: str,
    model_bin: str | None = None,
    local_files_only: bool = False,
    project_root: Path | None = None,
) -> tuple[Path, str]:
    # Backward-compatible wrapper used by existing call sites.
    return ensure_deepseek_model(
        model_source=model_source,
        model_bin=model_bin,
        model_quant="q8",
        local_files_only=local_files_only,
        project_root=project_root,
    )


def build_deepseek_engine_if_needed(
    repo_root: Path,
    model_source: str,
    model_bin: Path,
    model_quant: str = "q8",
    force_rebuild: bool = False,
) -> None:
    build_sh = repo_root / "scripts" / "build_deepseek_native.sh"
    run_candidates = [
        repo_root / "scripts" / "deepseek_native_engine.sh",
        repo_root / "experiments" / "deepseek_int" / "run.sh",  # legacy fallback
    ]
    run_sh = next((c for c in run_candidates if c.is_file()), None)
    if not build_sh.is_file() and run_sh is None:
        return
    engine_bin = repo_root / ".run" / "bin" / "deepseek_engine"
    if engine_bin.is_file() and not force_rebuild:
        return

    env = os.environ.copy()
    env["MODEL_QUANT"] = str(model_quant).strip().lower()
    env["ENGINE_BIN"] = str(engine_bin)
    env["FORCE_REBUILD"] = "1" if force_rebuild else "0"
    if build_sh.is_file():
        # Preferred path: enforces mandatory HW selection and rebuild signature checks.
        subprocess.run(
            ["bash", str(build_sh), model_source, str(model_bin)],
            cwd=str(repo_root),
            env=env,
            check=True,
        )
        return

    # Legacy fallback (kept for compatibility with older layouts).
    env["WORKDIR"] = str(repo_root)
    env["MODEL_REPO"] = model_source
    env["MODEL_BIN"] = str(model_bin)
    env["SKIP_RUN"] = "1"
    subprocess.run(
        ["bash", str(run_sh)],
        cwd=str(repo_root),
        env=env,
        check=True,
    )


@dataclass(frozen=True)
class DeepSeekRuntimePaths:
    prompt_file: Path
    prompt_tokens_file: Path
    tokens_file: Path
    done_file: Path
    log_file: Path


class DeepSeekNativeEngine:
    def __init__(
        self,
        repo_root: Path,
        model_bin: Path,
        tokenizer_source: str,
        engine_bin: Path,
        runtime_name: str,
        local_files_only: bool = False,
        max_new_tokens: int = 256,
        max_prompt_tokens: int | None = None,
        startup_timeout_s: float = 120.0,
        request_timeout_s: float = 180.0,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.model_bin = Path(model_bin).resolve()
        self.tokenizer_source = str(tokenizer_source)
        self.engine_bin = Path(engine_bin).resolve()
        self.local_files_only = bool(local_files_only)
        self.max_new_tokens = max(1, int(max_new_tokens))
        env_max_prompt = os.environ.get("DSI8_MAX_PROMPT_TOKENS")
        if max_prompt_tokens is None and env_max_prompt:
            try:
                max_prompt_tokens = int(env_max_prompt)
            except Exception:
                max_prompt_tokens = None
        self.max_prompt_tokens = max(128, int(max_prompt_tokens or 2048))
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.request_timeout_s = max(1.0, float(request_timeout_s))

        base = self.repo_root / ".run" / "deepseek_native" / runtime_name
        base.mkdir(parents=True, exist_ok=True)
        self.paths = DeepSeekRuntimePaths(
            prompt_file=base / "prompt.txt",
            prompt_tokens_file=base / "prompt_tokens.txt",
            tokens_file=base / "tokens.txt",
            done_file=base / "done.flag",
            log_file=base / "engine.log",
        )
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._lock = threading.Lock()
        self._tokenizer = None

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError(f"transformers is required for native DeepSeek tokenization: {e}") from e
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_source,
            local_files_only=self.local_files_only,
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def is_running(self) -> bool:
        return self._is_running()

    def _tail_log(self, n_lines: int = 40) -> str:
        try:
            txt = self.paths.log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        lines = txt.splitlines()
        return "\n".join(lines[-n_lines:])

    def start(self) -> None:
        with self._lock:
            if self._is_running():
                return
            self._ensure_tokenizer()
            if not self.engine_bin.is_file():
                raise RuntimeError(f"DeepSeek engine binary not found: {self.engine_bin}")
            if not self.model_bin.is_file():
                raise RuntimeError(f"DeepSeek model bin not found: {self.model_bin}")

            self.paths.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.paths.log_file.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["DSI8_PROMPT_FILE"] = str(self.paths.prompt_file)
            env["DSI8_PROMPT_TOKENS_FILE"] = str(self.paths.prompt_tokens_file)
            env["DSI8_TOKENS_FILE"] = str(self.paths.tokens_file)
            env["DSI8_DONE_FILE"] = str(self.paths.done_file)
            env["DSI8_TGEN"] = str(self.max_new_tokens)
            env.setdefault("TELEM_ENABLE", "0")

            self._proc = subprocess.Popen(
                [str(self.engine_bin), str(self.model_bin), self.tokenizer_source],
                cwd=str(self.repo_root),
                env=env,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )

            t0 = time.time()
            ready_marker = "Ready. Entering interactive loop"
            while time.time() - t0 < self.startup_timeout_s:
                if self._proc.poll() is not None:
                    tail = self._tail_log()
                    raise RuntimeError(f"DeepSeek engine exited during startup.\n{tail}")
                if self.paths.log_file.exists():
                    tail = self._tail_log(200)
                    if ready_marker in tail:
                        return
                time.sleep(0.1)

            tail = self._tail_log()
            raise RuntimeError(f"DeepSeek engine startup timeout after {self.startup_timeout_s:.1f}s.\n{tail}")

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _write_lines_atomic(self, path: Path, lines: Iterable[str]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln)
                if not ln.endswith("\n"):
                    f.write("\n")
        tmp.replace(path)

    def _read_token_ids(self) -> list[int]:
        if not self.paths.tokens_file.is_file():
            return []
        out: list[int] = []
        with self.paths.tokens_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(int(s))
                except Exception:
                    continue
        return out

    def generate(self, prompt: str, max_new_tokens: int | None = None, timeout_s: float | None = None) -> str:
        if not self._is_running():
            self.start()
        assert self._tokenizer is not None
        requested = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        requested = max(1, min(requested, self.max_new_tokens))
        timeout = self.request_timeout_s if timeout_s is None else max(1.0, float(timeout_s))

        with self._lock:
            if not self._is_running():
                raise RuntimeError("DeepSeek engine is not running.")

            for p in (self.paths.tokens_file, self.paths.done_file, self.paths.prompt_file, self.paths.prompt_tokens_file):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

            enc = self._tokenizer(
                prompt,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids = enc.get("input_ids", [])
            if not ids:
                eos = self._tokenizer.eos_token_id
                bos = self._tokenizer.bos_token_id
                fallback = bos if bos is not None else eos
                ids = [int(fallback)] if fallback is not None else [0]
            if len(ids) > self.max_prompt_tokens:
                ids = ids[-self.max_prompt_tokens:]

            self._write_lines_atomic(self.paths.prompt_tokens_file, [str(int(i)) for i in ids])
            self._write_lines_atomic(self.paths.prompt_file, [prompt])

            t0 = time.time()
            while time.time() - t0 < timeout:
                if not self._is_running():
                    tail = self._tail_log()
                    raise RuntimeError(f"DeepSeek engine exited while waiting for result.\n{tail}")
                if self.paths.done_file.is_file():
                    break
                time.sleep(0.05)
            else:
                tail = self._tail_log()
                self._stop_unlocked()
                raise RuntimeError(f"DeepSeek generation timeout after {timeout:.1f}s.\n{tail}")

            out_ids = self._read_token_ids()
            if requested < len(out_ids):
                out_ids = out_ids[:requested]
            if not out_ids:
                return ""
            return self._tokenizer.decode(out_ids, skip_special_tokens=True)
