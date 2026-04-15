# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import json
import os
import subprocess
import unittest
from pathlib import Path


def _has_hrm_api_library(repo: Path) -> bool:
    if os.environ.get("HRM_API_LIB") and Path(os.environ["HRM_API_LIB"]).is_file():
        return True
    if (repo / "hrm_core" / "build" / "libhrm_api.so").is_file():
        return True
    if (Path.cwd() / "hrm_core" / "build" / "libhrm_api.so").is_file():
        return True
    if Path("/usr/local/lib/libhrm_api.so").is_file():
        return True
    if Path("/usr/lib/libhrm_api.so").is_file():
        return True
    return False


def _run(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        out = (exc.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}") from exc


def _ensure_hrm_prereqs(repo: Path, model_dir: Path) -> None:
    need_build = not _has_hrm_api_library(repo)
    need_model = not (model_dir / "router_index.bin").is_file() or not (model_dir / "index.sqlite").is_file()
    if not need_build and not need_model:
        return

    # Build C++ core + shared API library in-repo.
    _run(["bash", "scripts/build.sh"], cwd=repo)

    # Ensure minimal model artifacts for integration query.
    seed_file = repo / "tests" / "fixtures" / "hrm_seed.txt"
    if not seed_file.is_file():
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(
            "HRM integration seed corpus.\n"
            "This file is used to generate router_index.bin and index.sqlite for tests.\n"
            "DeepSeek-only, mixed and retrieval modes are supported.\n",
            encoding="utf-8",
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    _run(["bash", "scripts/make_model.sh", str(seed_file), str(model_dir), "8"], cwd=repo)


class TestHrmApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        model_dir_override = os.environ.get("HRM_TEST_MODEL_DIR", "").strip()
        if model_dir_override:
            cls.model_dir = Path(model_dir_override).resolve()
        else:
            cls.model_dir = cls.repo / ".run" / "test_artifacts" / "model_min"

        _ensure_hrm_prereqs(cls.repo, cls.model_dir)

        missing_after_bootstrap = []
        if not (cls.model_dir / "router_index.bin").is_file():
            missing_after_bootstrap.append(f"{cls.model_dir}/router_index.bin")
        if not (cls.model_dir / "index.sqlite").is_file():
            missing_after_bootstrap.append(f"{cls.model_dir}/index.sqlite")
        if not _has_hrm_api_library(cls.repo):
            missing_after_bootstrap.append("libhrm_api.so (set HRM_API_LIB or build hrm_core)")
        if missing_after_bootstrap:
            raise RuntimeError("HRM integration prerequisites not available: " + ", ".join(missing_after_bootstrap))

    def test_query_json_contains_required_fields(self):
        from hrm_flash.hrm_api import HRMHandle

        with HRMHandle(model_dir=self.model_dir, repo_root=self.repo) as handle:
            response = handle.query_json("hello world", top_k=3, top_m=50, k=4)

        obj = json.loads(response)
        self.assertIn("prompt", obj)
        self.assertIn("cids", obj)
        self.assertIn("chosen", obj)


if __name__ == "__main__":
    unittest.main()
