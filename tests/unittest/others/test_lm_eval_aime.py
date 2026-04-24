# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the local AIME task's ``utils.py``.

Covers the pure answer-extraction / normalization helpers used by the
``aime26`` task (``tensorrt_llm/evaluate/lm_eval_tasks/aime/``). The
helpers are mirrored from lm-evaluation-harness so the behaviour must
stay byte-compatible with upstream ``aime24`` / ``aime25`` scoring.
"""

from __future__ import annotations

from tensorrt_llm.evaluate.lm_eval_tasks.aime.utils import (
    is_equiv,
    last_boxed_only_string,
    process_results,
    remove_boxed,
    strip_string,
)

# ---------------------------------------------------------------------------
# last_boxed_only_string / remove_boxed
# ---------------------------------------------------------------------------


def test_last_boxed_only_string_plain():
    r"""Canonical \boxed{N} is returned verbatim including prefix and brace."""
    assert last_boxed_only_string("The answer is \\boxed{42}.") == "\\boxed{42}"


def test_last_boxed_only_string_nested_braces():
    r"""Nested braces (\boxed{\frac{1}{2}}) are balanced and preserved."""
    s = "Final: \\boxed{\\frac{1}{2}}"
    assert last_boxed_only_string(s) == "\\boxed{\\frac{1}{2}}"


def test_last_boxed_only_string_takes_last():
    r"""Rightmost \boxed wins; AIME outputs often restate candidates first."""
    s = "First guess \\boxed{7}, but actually \\boxed{42}."
    assert last_boxed_only_string(s) == "\\boxed{42}"


def test_last_boxed_only_string_no_boxed_returns_none():
    r"""No \boxed and no \fbox returns None so $...$ fallback can run."""
    assert last_boxed_only_string("The answer is 42.") is None


def test_last_boxed_only_string_space_variant():
    r"""\boxed N (space, no braces) returns up to the terminating $."""
    s = "Answer: \\boxed 42$ end."
    out = last_boxed_only_string(s)
    assert out is not None
    assert out.startswith("\\boxed ")
    assert "42" in out


def test_remove_boxed_strips_wrapper():
    assert remove_boxed("\\boxed{42}") == "42"


def test_remove_boxed_preserves_nested_latex():
    assert remove_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


# ---------------------------------------------------------------------------
# is_equiv / strip_string
# ---------------------------------------------------------------------------


def test_is_equiv_integer_exact():
    assert is_equiv("42", "42") is True


def test_is_equiv_whitespace_tolerant():
    """Leading, trailing, and internal whitespace is normalized away."""
    assert is_equiv(" 42 ", "42") is True
    assert is_equiv("4 2", "42") is True


def test_is_equiv_frac_sugar_expands():
    r"""\frac12 (compact sugar) compares equal to \frac{1}{2}."""
    assert is_equiv("\\frac12", "\\frac{1}{2}") is True


def test_is_equiv_distinct_values():
    assert is_equiv("42", "43") is False


def test_strip_string_kills_spaces_and_newlines():
    assert strip_string("  4\n2 ") == "42"


def test_strip_string_drops_leading_varname():
    """``x=42`` style prefixes are dropped when the LHS is short (<=2 chars)."""
    assert strip_string("k=42") == "42"


# ---------------------------------------------------------------------------
# process_results end-to-end
# ---------------------------------------------------------------------------


def _doc(answer) -> dict:
    """AIME doc with MathArena lowercase ``answer`` field."""
    return {"problem_idx": 1, "problem": "...", "answer": answer}


def test_process_results_boxed_correct():
    doc = _doc(42)
    results = ["Working... therefore \\boxed{42}."]
    assert process_results(doc, results) == {"exact_match": 1}


def test_process_results_boxed_wrong():
    doc = _doc(42)
    results = ["Working... therefore \\boxed{7}."]
    assert process_results(doc, results) == {"exact_match": 0}


def test_process_results_boxed_overrides_dollar_delimited():
    r"""When both $...$ and \boxed{} appear, \boxed{} wins."""
    doc = _doc(42)
    # The $...$ span captures 'decoy=7' but \boxed{42} must override it.
    results = ["Candidate $decoy=7$ but final \\boxed{42}."]
    assert process_results(doc, results) == {"exact_match": 1}


def test_process_results_dollar_fallback_when_no_boxed():
    r"""Without \boxed, fall back to content between first and last $."""
    doc = _doc(42)
    results = ["The answer is $42$."]
    assert process_results(doc, results) == {"exact_match": 1}


def test_process_results_answer_key_case_insensitive():
    """``answer`` key match is case-insensitive against dataset schema drift."""
    doc = {"problem": "...", "Answer": 42}
    results = ["\\boxed{42}"]
    assert process_results(doc, results) == {"exact_match": 1}
