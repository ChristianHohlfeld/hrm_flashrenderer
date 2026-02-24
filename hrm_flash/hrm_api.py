import ctypes
import os
from pathlib import Path
from typing import Any, Dict


class HRMApiError(RuntimeError):
    pass


def _candidate_paths(repo_root: Path | None = None) -> list[Path]:
    cands: list[Path] = []

    # explicit env
    env = os.environ.get("HRM_API_LIB")
    if env:
        cands.append(Path(env))

    # repo build
    if repo_root is not None:
        cands.append(repo_root / "hrm_core" / "build" / "libhrm_api.so")

    # cwd build
    cands.append(Path.cwd() / "hrm_core" / "build" / "libhrm_api.so")

    # system locations (optional)
    for p in [
        Path("/usr/local/lib/libhrm_api.so"),
        Path("/usr/lib/libhrm_api.so"),
    ]:
        cands.append(p)

    return cands


def load_hrm_api(repo_root: Path | None = None) -> ctypes.CDLL:
    for p in _candidate_paths(repo_root=repo_root):
        try:
            if p.is_file():
                lib = ctypes.CDLL(str(p))
                return lib
        except OSError:
            continue
    raise HRMApiError(
        "HRM API shared library not found. Build it with:\n"
        "  cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build hrm_core/build -j\n"
        "Expected libhrm_api.so in hrm_core/build, or set HRM_API_LIB."
    )


class HRMHandle:
    def __init__(self, model_dir: Path, repo_root: Path | None = None):
        self.model_dir = Path(model_dir).resolve()
        self.repo_root = repo_root
        self.lib = load_hrm_api(repo_root=repo_root)

        self.lib.hrm_open.argtypes = [ctypes.c_char_p]
        self.lib.hrm_open.restype = ctypes.c_void_p

        self.lib.hrm_close.argtypes = [ctypes.c_void_p]
        self.lib.hrm_close.restype = None

        self.lib.hrm_query_json.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.hrm_query_json.restype = ctypes.c_void_p

        self.lib.hrm_free.argtypes = [ctypes.c_void_p]
        self.lib.hrm_free.restype = None

        self.lib.hrm_last_error.argtypes = []
        self.lib.hrm_last_error.restype = ctypes.c_char_p

        h = self.lib.hrm_open(str(self.model_dir).encode("utf-8"))
        if not h:
            err = self.lib.hrm_last_error()
            raise HRMApiError(f"hrm_open failed: {err.decode('utf-8', errors='replace') if err else 'unknown'}")
        self._h = ctypes.c_void_p(h)

    def close(self) -> None:
        if getattr(self, "_h", None):
            self.lib.hrm_close(self._h)
            self._h = None

    def query_json(self, prompt: str, top_k: int = 8, top_m: int = 400, k: int = 8, lam_num: int = 7, lam_den: int = 10) -> str:
        if not self._h:
            raise HRMApiError("handle closed")
        p = self.lib.hrm_query_json(
            self._h,
            prompt.encode("utf-8"),
            int(top_k),
            int(top_m),
            int(k),
            int(lam_num),
            int(lam_den),
        )
        if not p:
            err = self.lib.hrm_last_error()
            raise HRMApiError(f"hrm_query_json failed: {err.decode('utf-8', errors='replace') if err else 'unknown'}")
        try:
            s = ctypes.cast(p, ctypes.c_char_p).value
            return s.decode("utf-8", errors="replace") if s else "{}"
        finally:
            self.lib.hrm_free(p)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
