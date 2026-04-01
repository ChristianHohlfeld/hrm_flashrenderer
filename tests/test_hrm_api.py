# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import json
import os
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


class TestHrmApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.model_dir = cls.repo / "examples" / "model_min"

        missing = []
        if not (cls.model_dir / "router_index.bin").is_file():
            missing.append("examples/model_min/router_index.bin")
        if not (cls.model_dir / "index.sqlite").is_file():
            missing.append("examples/model_min/index.sqlite")
        if not _has_hrm_api_library(cls.repo):
            missing.append("libhrm_api.so (set HRM_API_LIB or build hrm_core)")

        if missing:
            raise unittest.SkipTest("Skipping HRM integration test; missing prerequisites: " + ", ".join(missing))

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

