# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MMMU Pro answer parser and prompt-mode switch.

Guards two fixes:

1. ``_ANSWER_RE`` + reverse-scan in ``parse_multi_choice_response``: the
   default MMMU parser scanned forward, which caused CoT / thinking-mode
   outputs to pick up an earlier-appearing letter (e.g. from "option A is
   wrong because...") instead of the final ``Answer: X`` line.  The new
   reverse scan walks lines bottom-up and returns the first regex match,
   so the canonical final-answer line wins.

2. ``MMMU_PRO_PROMPT_MODE`` env variable: switches the prompt suffix
   between the MMMU-Benchmark's ``direct/standard`` template (default)
   and ``cot/standard`` (opt-in).  The latter adds +10-25 pp on smaller
   models by asking for "Answer: $LETTER" on the final line.
"""

from __future__ import annotations

import importlib
import os


# Reload the module under test whenever the env flips, since the suffix is
# captured at import time.
def _reload_utils(mode: str | None):
    if mode is None:
        os.environ.pop("MMMU_PRO_PROMPT_MODE", None)
    else:
        os.environ["MMMU_PRO_PROMPT_MODE"] = mode
    from tensorrt_llm.evaluate.lm_eval_tasks.mmmu_pro import utils

    importlib.reload(utils)
    return utils


# ---------------------------------------------------------------------------
# _ANSWER_RE + reverse-scan in parse_multi_choice_response
# ---------------------------------------------------------------------------


def test_cot_final_answer_line_wins():
    """The final 'Answer: X' line wins over earlier letters in the chain.

    This is the main reason thinking-mode went from 51% to 76% on 26B —
    the forward scanner was latching onto a random "A" inside the reasoning
    before ever reaching the final answer line.
    """
    utils = _reload_utils(None)
    resp = (
        "Let me think step by step.\n"
        "Option A is wrong because foo.\n"
        "Option B is wrong because bar.\n"
        "Option C is correct.\n"
        "Answer: C"
    )
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "C"


def test_final_answer_with_parentheses():
    """Models sometimes emit 'Answer: (C)' — regex tolerates parens."""
    utils = _reload_utils(None)
    resp = "Reasoning...\nAnswer: (C)"
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "C"


def test_final_answer_case_insensitive():
    """The regex is case-insensitive for the 'answer' keyword."""
    utils = _reload_utils(None)
    resp = "Thinking...\nanswer: D"
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "D"


def test_final_answer_keyword_is():
    """'Answer is X' form (without colon) also matches."""
    utils = _reload_utils(None)
    resp = "The answer is B."
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "B"


def test_final_answer_letter_out_of_choice_set_is_ignored():
    """Out-of-set regex match must not short-circuit the parser.

    If the final-answer regex matches a letter outside all_choices, the
    parser must fall back to the legacy scan instead of returning an
    invalid letter.
    """
    utils = _reload_utils(None)
    resp = "I think the answer is Z.\nBut actually A"
    # Only A/B/C/D are valid — Z must not win.
    result = utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {})
    assert result in {"A", "B", "C", "D"}  # Some valid letter, not Z.


def test_no_final_answer_falls_back_to_legacy_scan():
    """Fallback path: responses without 'Answer: X' use the legacy scan.

    We keep the upstream MMMU parser intact so non-CoT responses still
    get a best-effort letter match.
    """
    utils = _reload_utils(None)
    resp = "I choose (B) because it matches."
    # Legacy scan should still pick a letter.
    result = utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {})
    assert result in {"A", "B", "C", "D"}


def test_reverse_scan_picks_last_answer_line_across_multiple():
    """Last of several 'Answer: X' lines wins.

    Matches how the model typically self-corrects: it writes an initial
    guess, then a correction, and the final line is authoritative.
    """
    utils = _reload_utils(None)
    resp = "Answer: A\nWait, let me reconsider.\nAnswer: B"
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "B"


# ---------------------------------------------------------------------------
# MMMU_PRO_PROMPT_MODE env var
# ---------------------------------------------------------------------------


def test_default_mode_is_direct():
    """Unset env => direct/standard suffix (backward-compatible default)."""
    utils = _reload_utils(None)
    assert utils._MODE == "direct"
    assert "letter" in utils._PROMPT_SUFFIX.lower()
    assert "step by step" not in utils._PROMPT_SUFFIX.lower()


def test_mode_cot_switches_suffix():
    """MMMU_PRO_PROMPT_MODE=cot => cot/standard suffix (think step-by-step).

    This is the suffix the HF Gemma4 blog numbers appear to use — it adds
    the 'Answer: $LETTER' final-line instruction, which pairs with the
    reverse-scan parser above.
    """
    utils = _reload_utils("cot")
    assert utils._MODE == "cot"
    assert "step by step" in utils._PROMPT_SUFFIX.lower()
    assert "answer: $letter" in utils._PROMPT_SUFFIX.lower()


def test_mode_unknown_value_defaults_to_direct():
    """Unrecognized values => fall back to direct (defensive)."""
    utils = _reload_utils("something-else")
    assert utils._MODE == "something-else"
    # Anything not 'cot' picks the direct suffix.
    assert "letter" in utils._PROMPT_SUFFIX.lower()
    assert "step by step" not in utils._PROMPT_SUFFIX.lower()


def test_mode_cot_included_in_example_format():
    """MULTI_CHOICE_EXAMPLE_FORMAT must embed the cot suffix when mode=cot."""
    utils = _reload_utils("cot")
    try:
        assert "step by step" in utils.MULTI_CHOICE_EXAMPLE_FORMAT.lower()
    finally:
        # Restore module state for other tests running in the same session.
        _reload_utils(None)
