# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the multimodal wrapper's interleaved ``content_parts`` construction.

Guards the MMMU Pro multi-image regression: without interleaved content,
multi-image prompts (``"Consider <image 1>. What does <image 2> show?"``)
lose answer-grounding because all images get bulk-prepended before the
text.  The wrapper now produces an interleaved content_parts list for
OPENAI-format chat templates so ``_build_openai_content`` emits a
correctly-ordered OpenAI content list.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tensorrt_llm.evaluate.lm_eval import LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER, MultimodalLmEvalWrapper
from tensorrt_llm.inputs.content_format import ContentFormat


def _make_wrapper(model_type: str = "gemma4") -> MultimodalLmEvalWrapper:
    fake_llm = MagicMock()
    fake_llm.tokenizer = MagicMock()
    fake_llm.input_processor = MagicMock()
    fake_llm.input_processor.processor = MagicMock()
    with patch.object(MultimodalLmEvalWrapper, "_get_model_type", return_value=model_type):
        return MultimodalLmEvalWrapper(
            fake_llm,
            sampling_params=None,
            streaming=False,
            model_type=model_type,
        )


def _call_apply(wrapper, text: str, *, content_format: ContentFormat):
    """Run apply_chat_template against a stubbed trtllm_apply_chat_template.

    Returns the conversation dict that was built.  The real HF chat
    template requires an actual tokenizer; we only care about the
    conversation structure the wrapper constructs before it hands off.
    """
    chat_history = [{"role": "user", "content": text}]
    captured = {}

    def _fake_trtllm_apply(**kwargs):
        captured.update(kwargs)
        return "<stub>"

    with (
        patch(
            "tensorrt_llm.evaluate.lm_eval.resolve_hf_chat_template",
            return_value="<stub-template>",
        ),
        patch(
            "tensorrt_llm.evaluate.lm_eval._resolve_content_format",
            return_value=content_format,
        ),
        patch("tensorrt_llm.evaluate.lm_eval.trtllm_apply_chat_template", _fake_trtllm_apply),
    ):
        wrapper.apply_chat_template(chat_history)

    assert captured, "trtllm_apply_chat_template was not invoked"
    convs = captured["conversation"]
    assert len(convs) == 1
    return convs[0]


def test_single_image_does_not_interleave():
    """Single-image prompts never need interleaving.

    content_parts stays absent so the existing BEFORE_TEXT default keeps working.
    """
    wrapper = _make_wrapper()
    text = f"What is in {LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER}?"
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    assert conv.get("content_parts") is None
    assert LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER not in conv["content"]


def test_multi_image_openai_builds_content_parts():
    """Multi-image + OPENAI template carries the original interleaving in content_parts.

    ``_build_openai_content`` then emits media entries at the correct positions.
    """
    wrapper = _make_wrapper()
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"Consider {ph}. What does {ph} show?"
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    parts = conv.get("content_parts")
    assert parts is not None, "expected interleaved content_parts for multi-image OPENAI prompt"

    # Expected: ["Consider ", image, ". What does ", image, " show?"]
    kinds = [("text" if isinstance(p, str) else p["type"]) for p in parts]
    assert kinds == ["text", "image", "text", "image", "text"]
    # image parts keep an ascending media_index so downstream code can
    # correlate them with the images list.
    media_parts = [p for p in parts if isinstance(p, dict)]
    assert [p["media_index"] for p in media_parts] == [0, 1]


def test_multi_image_string_format_skips_interleave():
    """STRING-format chat templates skip the interleaving path.

    Placeholders are inserted into the flat text via
    ``add_multimodal_placeholders`` instead, so ``content_parts`` stays absent.
    """
    wrapper = _make_wrapper()
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"{ph} vs {ph}: what changed?"
    with patch(
        "tensorrt_llm.evaluate.lm_eval.add_multimodal_placeholders",
        return_value="<placeholders><placeholders> vs : what changed?",
    ):
        conv = _call_apply(wrapper, text, content_format=ContentFormat.STRING)
    assert conv.get("content_parts") is None


def test_trailing_text_after_last_image_preserved():
    """Text that follows the last image must be preserved verbatim.

    Otherwise the question suffix ('Answer:') is dropped before it reaches
    the model.
    """
    wrapper = _make_wrapper()
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"Compare {ph} with {ph}. Answer with a letter."
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    parts = conv["content_parts"]
    # Last part must be the trailing text.
    assert isinstance(parts[-1], str)
    assert parts[-1].endswith("Answer with a letter.")


def test_leading_image_no_empty_text_segment():
    """Leading ``<image>`` placeholders do not emit an empty-string text part.

    content_parts must begin with the image entry itself.
    """
    wrapper = _make_wrapper()
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"{ph} {ph} Answer?"
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    parts = conv["content_parts"]
    assert parts, "expected non-empty content_parts"
    assert isinstance(parts[0], dict) and parts[0]["type"] == "image"
    # Empty text segments must not be inserted.  Whitespace-only segments
    # (e.g. " " between two adjacent ``<image>`` placeholders) are preserved
    # because they faithfully reflect the user's prompt.
    assert all((not isinstance(p, str)) or p != "" for p in parts)
