# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import json
from pathlib import Path


def main():
    # Requires HRM core build output (libhrm_api.so) and example model.
    repo = Path(__file__).resolve().parents[1]
    model_dir = repo / "examples" / "model_min"
    assert (model_dir / "router_index.bin").is_file()
    assert (model_dir / "index.sqlite").is_file()

    from hrm_flash.hrm_api import HRMHandle

    h = HRMHandle(model_dir=model_dir, repo_root=repo)
    s = h.query_json("hello world", top_k=3, top_m=50, k=4)
    h.close()

    obj = json.loads(s)
    assert "prompt" in obj
    assert "cids" in obj
    assert "chosen" in obj
    print("OK")


if __name__ == "__main__":
    main()

