"""Dependency-light tests for SpecSteer's OpenAI recent-context policy."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_specsteer/vllm_0_28/asymspec_serving.py"
)
SPEC = spec_from_file_location("asymspec_serving", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fits_at_most(count):
    return lambda messages: len(messages) <= count


class AsymSpecServingTests(unittest.TestCase):
    def test_plain_single_user_is_retained(self):
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(MODULE.select_recent_messages(messages, fits_at_most(1)), messages)

    def test_system_is_pinned_and_oldest_turns_are_dropped(self):
        messages = [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new"},
        ]
        self.assertEqual(
            MODULE.select_recent_messages(messages, fits_at_most(2)),
            [messages[0], messages[-1]],
        )

    def test_newest_complete_turns_are_retained(self):
        messages = [
            {"role": "developer", "content": "pinned"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "one answer"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "two answer"},
        ]
        self.assertEqual(
            MODULE.select_recent_messages(messages, fits_at_most(3)),
            [messages[0], *messages[3:]],
        )

    def test_tool_call_and_tool_result_are_atomic(self):
        messages = [
            {"role": "system", "content": "pinned"},
            {"role": "user", "content": "lookup"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "tool", "content": "result", "tool_call_id": "x"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "latest"},
        ]
        self.assertEqual(
            MODULE.select_recent_messages(messages, fits_at_most(2)),
            [messages[0], messages[-1]],
        )
        self.assertEqual(MODULE.select_recent_messages(messages, fits_at_most(6)), messages)

    def test_oversized_newest_turn_fails_cleanly(self):
        with self.assertRaisesRegex(MODULE.AsymSpecContextError, "newest user interaction"):
            MODULE.select_recent_messages(
                [{"role": "system", "content": "pinned"}, {"role": "user", "content": "x"}],
                fits_at_most(1),
            )

    def test_orphaned_tool_history_is_rejected(self):
        with self.assertRaisesRegex(MODULE.AsymSpecContextError, "begin with a user"):
            MODULE.select_recent_messages(
                [{"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}],
                fits_at_most(10),
            )

    def test_text_only_rejects_content_parts_but_allows_tool_call_none_content(self):
        self.assertTrue(MODULE.is_text_only([{"role": "assistant", "content": None}]))
        self.assertFalse(
            MODULE.is_text_only(
                [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
            )
        )

    def test_client_cannot_supply_server_owned_full_prompt_key(self):
        self.assertTrue(
            MODULE.has_reserved_extra_arg({MODULE.RESERVED_EXTRA_ARG: [1, 2]})
        )
        self.assertFalse(MODULE.has_reserved_extra_arg({"unrelated": 1}))
