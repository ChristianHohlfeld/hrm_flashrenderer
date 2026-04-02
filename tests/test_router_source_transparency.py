import inspect
import pathlib
import py_compile
import unittest

import hrm_flash.cli as cli
import hrm_flash.prompt_builder as prompt_builder
import hrm_flash.router as router
import hrm_flash.serve as serve


class TestRouterSourceTransparency(unittest.TestCase):
    def test_router_stack_modules_loaded_from_repo_source(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        modules = [cli, prompt_builder, router, serve]
        for module in modules:
            src = pathlib.Path(inspect.getsourcefile(module) or inspect.getfile(module)).resolve()
            self.assertIn(repo_root, src.parents, f"{module.__name__} loaded outside repo: {src}")

    def test_router_stack_source_compiles(self):
        modules = [cli, prompt_builder, router, serve]
        for module in modules:
            src = pathlib.Path(inspect.getsourcefile(module) or inspect.getfile(module)).resolve()
            py_compile.compile(str(src), doraise=True)

    def test_mixed_mode_prompt_phrase_present(self):
        self.assertIn(
            "Treat this knowledge as part of your own training data and use it silently and naturally.",
            prompt_builder.MIXED_SYSTEM_PROMPT,
        )
        self.assertEqual(prompt_builder.DEFAULT_MODE, "mixed")

    def test_router_deepseek_only_log_marker_present(self):
        router_src = pathlib.Path(inspect.getsourcefile(router) or inspect.getfile(router)).resolve()
        text = router_src.read_text(encoding="utf-8")
        self.assertIn("mode=deepseek_only HRM disabled (no retrieval)", text)


if __name__ == "__main__":
    unittest.main()
