import unittest

from hrm_flash import serve


class _FakeHRMResult:
    def __init__(self, raw):
        self.raw = raw


class TestModeAuditRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_run_hrm_query = serve.run_hrm_query
        self._orig_disable_budget = serve.STATE.disable_token_budget
        self._orig_tokenizer_source = serve.STATE.tokenizer_source
        self._orig_total = serve.STATE.hrm_query_calls_total
        self._orig_by_mode = dict(serve.STATE.hrm_query_calls_by_mode)
        serve.STATE.disable_token_budget = True
        serve.STATE.tokenizer_source = None
        serve.STATE.hrm_query_calls_total = 0
        serve.STATE.hrm_query_calls_by_mode = {"mixed": 0, "retrieval": 0, "deepseek_only": 0}

    def tearDown(self) -> None:
        serve.run_hrm_query = self._orig_run_hrm_query
        serve.STATE.disable_token_budget = self._orig_disable_budget
        serve.STATE.tokenizer_source = self._orig_tokenizer_source
        serve.STATE.hrm_query_calls_total = self._orig_total
        serve.STATE.hrm_query_calls_by_mode = self._orig_by_mode

    def test_hrm_call_counters_reflect_mode_behavior(self):
        def _ok(*_a, **_kw):
            return _FakeHRMResult({"chosen": [{"sid": "s0001", "txt": "Alpha"}]})

        serve.run_hrm_query = _ok

        # mixed -> HRM call
        _, _, m1, active1 = serve._build_prompt("Frage 1", mode="mixed")
        self.assertEqual(m1, "mixed")
        self.assertTrue(active1)

        # retrieval -> HRM call
        _, _, m2, active2 = serve._build_prompt("Frage 2", mode="retrieval")
        self.assertEqual(m2, "retrieval")
        self.assertTrue(active2)

        # deepseek_only -> no HRM call
        _, _, m3, active3 = serve._build_prompt("Frage 3", mode="deepseek_only")
        self.assertEqual(m3, "deepseek_only")
        self.assertFalse(active3)

        self.assertEqual(serve.STATE.hrm_query_calls_total, 2)
        self.assertEqual(serve.STATE.hrm_query_calls_by_mode.get("mixed"), 1)
        self.assertEqual(serve.STATE.hrm_query_calls_by_mode.get("retrieval"), 1)
        self.assertEqual(serve.STATE.hrm_query_calls_by_mode.get("deepseek_only"), 0)


if __name__ == "__main__":
    unittest.main()

