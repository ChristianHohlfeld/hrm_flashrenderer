import unittest

from hrm_flash import serve


class _FakeHRMResult:
    def __init__(self, raw):
        self.raw = raw


class TestServeModes(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_run_hrm_query = serve.run_hrm_query
        self._orig_disable_budget = serve.STATE.disable_token_budget
        self._orig_tokenizer_source = serve.STATE.tokenizer_source
        self._orig_max_sources = serve.STATE.max_sources
        self._orig_max_chars = serve.STATE.max_chars_per_source
        serve.STATE.disable_token_budget = True
        serve.STATE.tokenizer_source = None
        serve.STATE.max_sources = 16
        serve.STATE.max_chars_per_source = 1200

    def tearDown(self) -> None:
        serve.run_hrm_query = self._orig_run_hrm_query
        serve.STATE.disable_token_budget = self._orig_disable_budget
        serve.STATE.tokenizer_source = self._orig_tokenizer_source
        serve.STATE.max_sources = self._orig_max_sources
        serve.STATE.max_chars_per_source = self._orig_max_chars

    def test_deepseek_only_skips_hrm_query(self):
        def _boom(*_a, **_kw):
            raise AssertionError("run_hrm_query must not be called in deepseek_only mode")

        serve.run_hrm_query = _boom
        p, sources, mode = serve._build_prompt("Hi", mode="deepseek_only")
        self.assertEqual(mode, "deepseek_only")
        self.assertEqual(sources, [])
        self.assertIn("[USER]", p)
        self.assertNotIn("[INTERNAL_CONTEXT]", p)
        self.assertNotIn("[SOURCES]", p)

    def test_mixed_mode_uses_silent_injection(self):
        def _ok(*_a, **_kw):
            return _FakeHRMResult({"chosen": [{"sid": "s0001", "txt": "Alpha fact"}]})

        serve.run_hrm_query = _ok
        p, sources, mode = serve._build_prompt("Frage", mode="mixed")
        self.assertEqual(mode, "mixed")
        self.assertEqual(len(sources), 1)
        self.assertIn("[INTERNAL_CONTEXT]", p)
        self.assertIn("Alpha fact", p)
        self.assertNotIn("s0001", p)  # source id hidden in mixed mode

    def test_retrieval_mode_uses_explicit_sources(self):
        def _ok(*_a, **_kw):
            return _FakeHRMResult({"chosen": [{"sid": "s0001", "txt": "Alpha fact"}]})

        serve.run_hrm_query = _ok
        p, sources, mode = serve._build_prompt("Frage", mode="retrieval")
        self.assertEqual(mode, "retrieval")
        self.assertEqual(len(sources), 1)
        self.assertIn("[SOURCES]", p)
        self.assertIn("[s0001] Alpha fact", p)

    def test_missing_tokenizer_source_returns_consistent_tuple(self):
        def _ok(*_a, **_kw):
            return _FakeHRMResult({"chosen": [{"sid": "s0001", "txt": "Alpha fact"}]})

        serve.run_hrm_query = _ok
        serve.STATE.disable_token_budget = False
        serve.STATE.tokenizer_source = None
        p, sources, mode = serve._build_prompt("Frage", mode="mixed")
        self.assertEqual((p, sources, mode), ("", [], "mixed"))


if __name__ == "__main__":
    unittest.main()
