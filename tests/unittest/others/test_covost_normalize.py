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


# ---------------------------------------------------------------------------
# Tests for ``_extract_translation`` — the HF AST-format post-processor.
# ---------------------------------------------------------------------------
#
# The HF AST prompt ("transcribe, then translate") instructs the model to
# output the transcription first and then ``"{TARGET_LANGUAGE}: <translation>"``.
# BLEU must score the translation only — not the transcription — so we look
# for the language-name marker and return the text after it.  Falls back to
# the generic normalizer when the marker is missing (model disobeyed the
# format, thinking-mode chain-of-thought, empty output, etc.).


def test_extract_translation_basic_ast_format():
    """Standard HF AST response: transcription, then 'TARGET: translation'."""
    extract = CoVoST2._extract_translation
    response = "Hello world\n\nChinese: 你好世界"
    assert extract(response, "Chinese") == "你好世界"


def test_extract_translation_marker_case_insensitive():
    """Language-name matching ignores case — models lowercase occasionally."""
    extract = CoVoST2._extract_translation
    assert extract("Hola\n\nenglish: Hello", "English") == "Hello"
    assert extract("Hola\n\nENGLISH: Hello", "English") == "Hello"


def test_extract_translation_picks_last_marker():
    """Last marker wins under multiple occurrences.

    Thinking chains and self-correction lines can mention the target
    language multiple times — the final occurrence is the canonical
    translation.
    """
    extract = CoVoST2._extract_translation
    response = (
        "Thinking: the speaker says hello\n"
        "Chinese: 错误的翻译\n"
        "\n"
        "Actually let me retry.\n"
        "Chinese: 你好"
    )
    assert extract(response, "Chinese") == "你好"


def test_extract_translation_falls_back_to_normalize_when_no_marker():
    """If the model ignored the format, fall back to generic normalization."""
    extract = CoVoST2._extract_translation
    # Plain response without the AST marker.
    assert extract("Translation: Hello world", "Chinese") == "Hello world"


def test_extract_translation_stops_at_double_newline():
    """Translation region ends at the next double-newline.

    Trailing chain-of-thought after the translation must not be
    included in the BLEU input.
    """
    extract = CoVoST2._extract_translation
    response = "Hola\n\nEnglish: Hello\n\nAdditional explanation goes here."
    assert extract(response, "English") == "Hello"


def test_extract_translation_empty_input():
    """Empty or None-like response shouldn't crash — return empty string."""
    extract = CoVoST2._extract_translation
    assert extract("", "English") == ""


def test_extract_translation_normalizes_after_marker():
    """Extracted segment still runs through _normalize_prediction.

    Leading quotes and prefixes get stripped on the translation side too.
    """
    extract = CoVoST2._extract_translation
    response = 'Hola\n\nEnglish: "Hello world."'
    assert extract(response, "English") == "Hello world."


def test_prompt_text_uses_hf_ast_format():
    """Regression: CoVoST prompt must use the HF AST transcribe+translate form.

    Documented in the Gemma4 model card.  The old 'translate only' form
    under-performed substantially on non-Latin source languages because
    the model had no transcription step to ground the translation on.
    """
    cov = object.__new__(CoVoST2)
    cov.src_name = "English"
    cov.tgt_name = "Chinese"
    prompt = cov._prompt_text()
    # HF AST structure: transcribe + translate, with explicit marker line.
    assert "Transcribe" in prompt
    assert "translate" in prompt.lower()
    assert "Chinese:" in prompt  # target-language marker that _extract_translation keys off
    assert "English" in prompt  # source language
