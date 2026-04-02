import unittest

from hrm_flash import router


class TestRouterMaxModelMode(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_backends = dict(router.STATE.backends)
        router.STATE.backends = {
            "solo_22gb": router.Backend(name="solo_22gb", base_url="http://127.0.0.1:8081"),
            "nvlink_pair": router.Backend(name="nvlink_pair", base_url="http://127.0.0.1:8081"),
            "solo_3080": router.Backend(name="solo_3080", base_url="http://127.0.0.1:8083"),
        }

    def tearDown(self) -> None:
        router.STATE.backends = self._saved_backends

    def test_collapsed_mode_prefers_max_model_lane_by_default(self):
        selected = router._primary_backend(
            prompt_tokens=32,
            requested_max_new_tokens=96,
            route_hint=None,
            prefer_backend=None,
        )
        self.assertEqual(selected, "nvlink_pair")

    def test_explicit_fast_hint_still_uses_fast_lane(self):
        selected = router._primary_backend(
            prompt_tokens=32,
            requested_max_new_tokens=96,
            route_hint="fast",
            prefer_backend=None,
        )
        self.assertEqual(selected, "solo_3080")

    def test_explicit_prefer_backend_wins(self):
        selected = router._primary_backend(
            prompt_tokens=32,
            requested_max_new_tokens=96,
            route_hint=None,
            prefer_backend="solo_22gb",
        )
        self.assertEqual(selected, "solo_22gb")


if __name__ == "__main__":
    unittest.main()
