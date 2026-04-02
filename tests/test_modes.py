import unittest
from textwrap import dedent

from hrm_flash.prompt_builder import (
    MIXED_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    Source,
    build_prompt_for_mode,
    normalize_mode,
)


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
        self.assertIn("[BACKGROUND_KNOWLEDGE]", p)
        self.assertIn("Alpha Fakt", p)
        self.assertNotIn("s0001", p)
        self.assertNotIn("s0002", p)
        self.assertIn('Never mention "sources"', p)
        self.assertIn("Never refer to the fact that you were given additional context.", p)
        self.assertIn(MIXED_SYSTEM_PROMPT, p)
        self.assertNotIn(RETRIEVAL_SYSTEM_PROMPT, p)

    def test_deepseek_only_has_no_retrieval_context(self):
        p = build_prompt_for_mode(self.q, self.sources, mode="deepseek_only")
        self.assertEqual(p, self.q)
        self.assertNotIn("[SYSTEM]", p)
        self.assertNotIn("[SOURCES]", p)
        self.assertNotIn("[BACKGROUND_KNOWLEDGE]", p)

    def test_mixed_system_prompt_matches_spec_exactly(self):
        expected = dedent(
            """
            You are DeepSeek, a helpful, truthful and direct AI assistant.

            You have access to internal, verified knowledge retrieved deterministically in the background.
            Treat this knowledge as part of your own training data and use it silently and naturally.

            Never mention "sources", "according to the documents", "retrieved information", "laut den Quellen", "basierend auf den Snippets", "aus den bereitgestellten Texten" or any similar phrases.
            Never refer to the fact that you were given additional context.

            Answer the user prompt directly, naturally and conversationally.
            """
        ).strip()
        self.assertEqual(MIXED_SYSTEM_PROMPT, expected)


if __name__ == "__main__":
    unittest.main()
