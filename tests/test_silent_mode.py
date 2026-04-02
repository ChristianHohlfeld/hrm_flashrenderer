import unittest

from hrm_flash.prompt_builder import Source, build_renderer_prompt, build_sources


class TestSilentMode(unittest.TestCase):
    def test_build_sources_defaults_to_16(self):
        chosen = [{"sid": f"s{i:04d}", "txt": f"text {i}"} for i in range(1, 25)]
        out = build_sources({"chosen": chosen})
        self.assertEqual(len(out), 16)

    def test_renderer_prompt_is_silent_and_keeps_user_prompt(self):
        question = "Bitte antworte kurz.\nOhne Referenzen."
        sources = [
            Source(sid="0001#s0001", txt="Alpha Faktenblock."),
            Source(sid="0001#s0002", txt="Beta Faktenblock."),
        ]
        p = build_renderer_prompt(question, sources)

        self.assertIn("You are DeepSeek, a helpful, truthful, and direct AI assistant.", p)
        self.assertIn(question, p)
        self.assertNotIn("0001#s0001", p)
        self.assertNotIn("0001#s0002", p)
        self.assertNotIn("Cite sources", p)
        self.assertNotIn("[SOURCES]", p)
        self.assertIn("[BACKGROUND_KNOWLEDGE]", p)
        self.assertNotIn("[QUESTION]", p)
        self.assertNotIn("[ANSWER]", p)


if __name__ == "__main__":
    unittest.main()
