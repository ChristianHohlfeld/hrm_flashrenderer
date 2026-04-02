import unittest

from hrm_flash.prompt_builder import Source, build_prompt_for_mode, normalize_mode


class TestThreeModes(unittest.TestCase):
    def setUp(self) -> None:
        self.q = "Was ist der Stand?"
        self.sources = [
            Source(sid="s0001", txt="Alpha Fakt"),
            Source(sid="s0002", txt="Beta Fakt"),
        ]

    def test_mode_normalization_and_default(self):
        self.assertEqual(normalize_mode(None), "mixed")
        self.assertEqual(normalize_mode("silent"), "mixed")
        self.assertEqual(normalize_mode("deepseek"), "deepseek_only")
        self.assertEqual(normalize_mode("retrieval"), "retrieval")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            normalize_mode("unknown_mode")

    def test_retrieval_mode_keeps_sources_visible(self):
        p = build_prompt_for_mode(self.q, self.sources, mode="retrieval")
        self.assertIn("[SOURCES]", p)
        self.assertIn("[s0001] Alpha Fakt", p)
        self.assertIn("[s0002] Beta Fakt", p)
        self.assertIn("cite", p.lower())

    def test_mixed_mode_hides_source_ids(self):
        p = build_prompt_for_mode(self.q, self.sources, mode="mixed")
        self.assertIn("[INTERNAL_CONTEXT]", p)
        self.assertIn("Alpha Fakt", p)
        self.assertNotIn("s0001", p)
        self.assertNotIn("s0002", p)
        self.assertIn("Never mention retrieval", p)

    def test_deepseek_only_has_no_retrieval_context(self):
        p = build_prompt_for_mode(self.q, self.sources, mode="deepseek_only")
        self.assertIn("[USER]", p)
        self.assertIn(self.q, p)
        self.assertNotIn("[SOURCES]", p)
        self.assertNotIn("[INTERNAL_CONTEXT]", p)


if __name__ == "__main__":
    unittest.main()
