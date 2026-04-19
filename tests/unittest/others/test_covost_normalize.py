# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CoVoST 2 prediction normalizer.

Gemma4 instruct occasionally prepends ``Translation:`` (or wraps outputs in
quotes) even when told "respond with only the translation, no other text".
The normalizer strips those wrappers so BLEU 1-gram precision matches the
raw reference text format, closing a sizable portion of the zh-CN→en gap.
"""

from tensorrt_llm.evaluate.covost2 import CoVoST2


def test_strip_common_prefixes_case_insensitive():
    norm = CoVoST2._normalize_prediction
    assert norm("Translation: Hello world") == "Hello world"
    assert norm("translation: Hello world") == "Hello world"
    assert norm("TRANSLATION: Hello world") == "Hello world"
    assert norm("Translated: Bonjour") == "Bonjour"
    assert norm("The translation is: Guten Tag") == "Guten Tag"
    assert norm("English translation: See you.") == "See you."
    assert norm("Here is the translation: Adiós") == "Adiós"


def test_strip_outer_quotes():
    norm = CoVoST2._normalize_prediction
    assert norm('"Hello world."') == "Hello world."
    assert norm("'Bonjour.'") == "Bonjour."
    # Smart quotes (U+201C U+201D)
    assert norm("\u201cHello.\u201d") == "Hello."
    # Only strip quotes when the whole string is quoted.
    assert norm('She said "hi" to me.') == 'She said "hi" to me.'


def test_preserves_unprefixed_text():
    norm = CoVoST2._normalize_prediction
    assert norm("Hello world") == "Hello world"
    assert norm("  Hello world  ") == "Hello world"


def test_strip_composite_prefix_plus_quotes():
    """Prefix strip must run before the quote strip.

    Gemma4 sometimes emits ``Translation: "Hello world."``.
    """
    norm = CoVoST2._normalize_prediction
    assert norm('Translation: "Hello world."') == "Hello world."


def test_preserves_internal_colons():
    """Only strip the prefix at the very start."""
    norm = CoVoST2._normalize_prediction
    # "he said: hi" should not match "translation:" so it's preserved as-is.
    assert norm("he said: hi") == "he said: hi"


def test_strip_bom_and_zero_width():
    """Strip leading BOM / zero-width spaces.

    These occasionally appear on Unicode-heavy decode paths.
    """
    norm = CoVoST2._normalize_prediction
    assert norm("\ufeffHello") == "Hello"
    assert norm("\u200bHello") == "Hello"
