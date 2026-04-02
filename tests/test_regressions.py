# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    # Accept UTF-8 files with optional BOM (seen in some Windows-edited files).
    return path.read_text(encoding="utf-8-sig")


def _backend_arg_defaults(py_file: Path) -> list[str]:
    tree = ast.parse(_read_text(py_file))
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        arg_values: list[str] = []
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                arg_values.append(a.value)
        if "--backend" not in arg_values:
            continue
        default_val = None
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                default_val = kw.value.value
                break
        if default_val is not None:
            out.append(default_val)
    return out


def _backend_arg_choices(py_file: Path) -> list[list[str]]:
    tree = ast.parse(_read_text(py_file))
    out: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        arg_values: list[str] = []
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                arg_values.append(a.value)
        if "--backend" not in arg_values:
            continue
        for kw in node.keywords:
            if kw.arg != "choices":
                continue
            if isinstance(kw.value, (ast.List, ast.Tuple)):
                vals: list[str] = []
                for e in kw.value.elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        vals.append(e.value)
                if vals:
                    out.append(vals)
    return out


def _state_backend_default(py_file: Path) -> str | None:
    tree = ast.parse(_read_text(py_file))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "_State":
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or child.name != "__init__":
                continue
            for stmt in ast.walk(child):
                if not isinstance(stmt, ast.Assign):
                    if not isinstance(stmt, ast.AnnAssign):
                        continue
                    if not isinstance(stmt.target, ast.Attribute) or not isinstance(stmt.target.value, ast.Name):
                        continue
                    if stmt.target.value.id != "self" or stmt.target.attr != "backend":
                        continue
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        return stmt.value.value
                    continue
                if not isinstance(stmt.value, ast.Constant) or not isinstance(stmt.value.value, str):
                    continue
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name):
                        if tgt.value.id == "self" and tgt.attr == "backend":
                            return stmt.value.value
    return None


class TestNativeProdRegressions(unittest.TestCase):
    def test_cli_backend_defaults_to_deepseek(self):
        cli_path = REPO_ROOT / "hrm_flash" / "cli.py"
        defaults = _backend_arg_defaults(cli_path)
        self.assertGreaterEqual(len(defaults), 2, "expected backend args for serve and generate")
        self.assertTrue(
            all(v == "deepseek_int8" for v in defaults),
            f"unexpected backend default(s): {defaults}",
        )
        choices = _backend_arg_choices(cli_path)
        self.assertGreaterEqual(len(choices), 2, "expected backend choices for serve and generate")
        self.assertTrue(
            all(c == ["deepseek_int8"] for c in choices),
            f"unexpected backend choice set(s): {choices}",
        )

    def test_serve_backend_defaults_to_deepseek(self):
        serve_path = REPO_ROOT / "hrm_flash" / "serve.py"
        arg_defaults = _backend_arg_defaults(serve_path)
        self.assertEqual(arg_defaults, ["deepseek_int8"], f"unexpected serve backend arg defaults: {arg_defaults}")
        arg_choices = _backend_arg_choices(serve_path)
        self.assertEqual(arg_choices, [["deepseek_int8"]], f"unexpected serve backend arg choices: {arg_choices}")
        state_default = _state_backend_default(serve_path)
        self.assertEqual(state_default, "deepseek_int8", f"unexpected STATE.backend default: {state_default}")

    def test_no_deepseek_placeholder_alias_remains(self):
        for rel in ("scripts/start_native_stack.sh", "scripts/start_native_topology.sh"):
            p = REPO_ROOT / rel
            txt = _read_text(p)
            self.assertNotIn("<dein-deepseek-distill>", txt, f"placeholder alias still present in {rel}")

    def test_systemd_template_starts_native_stack(self):
        p = REPO_ROOT / "scripts" / "systemd" / "hrm-flash.service"
        txt = _read_text(p)
        self.assertIn("Type=oneshot", txt)
        self.assertIn("Environment=BACKEND=deepseek_int8", txt)
        self.assertIn("start_native_stack.sh", txt)
        self.assertIn("stop_native_stack.sh", txt)

    def test_repo_root_safe_helpers(self):
        checks = {
            "scripts/build.sh": ['ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"', 'cmake -S "$ROOT_DIR/hrm_core"'],
            "scripts/make_model.sh": ['ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"', 'HRM_BIN="$ROOT_DIR/hrm_core/build/hrm"'],
            "scripts/query_json.sh": ['ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"', 'HRM_BIN="$ROOT_DIR/hrm_core/build/hrm"'],
            "scripts/render.sh": ['ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"', '"$ROOT_DIR/renderer/hrm_render.py"'],
        }
        for rel, needles in checks.items():
            txt = _read_text(REPO_ROOT / rel)
            for needle in needles:
                self.assertIn(needle, txt, f"missing '{needle}' in {rel}")

    def test_native_lock_guards_in_scripts(self):
        checks = {
            "scripts/bootstrap.sh": "PROFILE=$PROFILE is not supported in production mainline",
            "scripts/start_native_stack.sh": "BACKEND=$BACKEND is not supported in production mainline",
            "scripts/start_native_topology.sh": "BACKEND=$BACKEND is not supported in production mainline",
            "scripts/serve.sh": "BACKEND=$BACKEND is not supported in production mainline",
            "scripts/hrm_flash_generate.sh": "BACKEND=$BACKEND is not supported in production mainline",
        }
        for rel, needle in checks.items():
            txt = _read_text(REPO_ROOT / rel)
            self.assertIn(needle, txt, f"missing native lock guard in {rel}")

    def test_max_model_fast_defaults_are_hardened(self):
        stack_txt = _read_text(REPO_ROOT / "scripts" / "start_native_stack.sh")
        topo_txt = _read_text(REPO_ROOT / "scripts" / "start_native_topology.sh")
        e2e_txt = _read_text(REPO_ROOT / "scripts" / "prod_live_e2e.sh")

        self.assertIn('TOPOLOGY_MODE="${TOPOLOGY_MODE:-max_model_fast}"', stack_txt)
        self.assertIn('RECO_MODEL_TRIPLE_MAX_Q8="${RECO_MODEL_TRIPLE_MAX_Q8:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"', stack_txt)
        self.assertIn('RECO_MODEL_TRIPLE_MAX_Q4="${RECO_MODEL_TRIPLE_MAX_Q4:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"', stack_txt)
        self.assertIn('PORT_NVLINK_PAIR="$PORT_SOLO_22GB"', stack_txt)
        self.assertIn('TOPOLOGY_MODE="${TOPOLOGY_MODE:-max_model_fast}"', topo_txt)
        self.assertIn('MAX_SEQ_TRIPLE_MAX="${MAX_SEQ_TRIPLE_MAX:-3072}"', topo_txt)
        self.assertIn('PREFILL_TRIPLE_MAX="${PREFILL_TRIPLE_MAX:-512}"', topo_txt)
        self.assertIn('TRIPLE_DEVICES="$GPU_22GB,$GPU_NVLINK_PAIR"', topo_txt)
        self.assertIn('ALLOW_PCIE_PAIR_FALLBACK="${ALLOW_PCIE_PAIR_FALLBACK:-1}"', topo_txt)
        self.assertIn('REQUIRE_NVLINK="${REQUIRE_NVLINK:-0}"', topo_txt)
        self.assertIn('TOPOLOGY_MODE="${TOPOLOGY_MODE:-max_model_fast}"', e2e_txt)

    def test_silent_mode_defaults_are_enabled(self):
        cli_txt = _read_text(REPO_ROOT / "hrm_flash" / "cli.py")
        serve_txt = _read_text(REPO_ROOT / "hrm_flash" / "serve.py")
        prompt_txt = _read_text(REPO_ROOT / "hrm_flash" / "prompt_builder.py")

        self.assertIn('g.add_argument("--top_k", type=int, default=16)', cli_txt)
        self.assertIn('g.add_argument("--k", type=int, default=16)', cli_txt)
        self.assertIn('g.add_argument("--max_sources", type=int, default=16)', cli_txt)
        self.assertIn('s.add_argument("--max_sources", type=int, default=16)', cli_txt)
        self.assertIn("top_k=16,", serve_txt)
        self.assertIn("k=16,", serve_txt)
        self.assertIn("timeout_s=1.8,", serve_txt)
        self.assertIn('ap.add_argument("--max_sources", type=int, default=16)', serve_txt)
        self.assertIn("SILENT_SYSTEM_PROMPT", prompt_txt)
        self.assertIn("Never mention retrieval, sources", prompt_txt)
        self.assertIn("max_sources: int = 16", prompt_txt)

    def test_three_modes_are_wired_in_mainline(self):
        cli_txt = _read_text(REPO_ROOT / "hrm_flash" / "cli.py")
        serve_txt = _read_text(REPO_ROOT / "hrm_flash" / "serve.py")
        router_txt = _read_text(REPO_ROOT / "hrm_flash" / "router.py")
        prompt_txt = _read_text(REPO_ROOT / "hrm_flash" / "prompt_builder.py")
        stack_txt = _read_text(REPO_ROOT / "scripts" / "start_native_stack.sh")

        self.assertIn('g.add_argument("--mode", type=str, choices=["retrieval", "mixed", "deepseek_only"], default="mixed")', cli_txt)
        self.assertIn('rt.add_argument("--default_mode", type=str, choices=["retrieval", "mixed", "deepseek_only"], default="mixed")', cli_txt)
        self.assertIn('mode: Optional[str] = None', serve_txt)
        self.assertIn('show_sources: Optional[bool] = None', serve_txt)
        self.assertIn('payload = {"ok": True, "text": text, "source_count": len(sources), "mode": mode}', serve_txt)
        self.assertIn('self.default_mode: str = "mixed"', router_txt)
        self.assertIn('def _resolve_mode(mode: Optional[str], default_mode: str) -> str:', router_txt)
        self.assertIn('"mode": mode,', router_txt)
        self.assertIn('"show_sources": show_sources,', router_txt)
        self.assertIn('ROUTER_DEFAULT_MODE="${ROUTER_DEFAULT_MODE:-mixed}"', stack_txt)
        self.assertIn('--default_mode "$ROUTER_DEFAULT_MODE"', stack_txt)
        self.assertIn('DEFAULT_MODE = "mixed"', prompt_txt)
        self.assertIn('SUPPORTED_MODES = {"retrieval", "mixed", "deepseek_only"}', prompt_txt)


if __name__ == "__main__":
    unittest.main()
