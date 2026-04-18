# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``LmEvalWrapper`` sampling-param handling.

In particular, exercises the ``sampling_override`` flag introduced so that
model-card sampling recipes (e.g. Gemma4 26B: temperature=1.0 / top_p=0.95
/ top_k=64) can override the greedy defaults baked into lm-eval-harness
task YAMLs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper
from tensorrt_llm.sampling_params import SamplingParams


def _make_wrapper(
    sampling_params: SamplingParams | None = None,
    sampling_override: bool = False,
) -> LmEvalWrapper:
    """Build a wrapper with a fake llm.

    We only exercise ``_get_sampling_params`` which doesn't touch the llm
    object.
    """
    fake_llm = MagicMock()
    fake_llm.tokenizer = MagicMock()
    return LmEvalWrapper(
        fake_llm,
        sampling_params=sampling_params,
        sampling_override=sampling_override,
    )


def test_greedy_default_from_task_yaml():
    """Default (no sampling override): task yaml gen_kwargs win.

    Mirrors the original behaviour: lm-eval GPQA yaml sets temperature=0.0 to
    force greedy, and that has to keep working even when the caller supplies
    a default SamplingParams with temperature=0 from the CLI.
    """
    sp = SamplingParams(max_tokens=256)  # default temperature=0
    wrapper = _make_wrapper(sampling_params=sp, sampling_override=False)
    gen_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "until": ["</s>"],
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 0.0
    assert out.top_p == 1.0
    assert out.stop == ["</s>"]


def test_sampling_override_cli_wins_on_temperature_and_top_p():
    """sampling_override=True: CLI sampling params win over yaml.

    CLI temperature / top_p / top_k must NOT be clobbered by the task yaml's
    greedy gen_kwargs when the override flag is on.
    """
    sp = SamplingParams(
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        seed=1234,
    )
    wrapper = _make_wrapper(sampling_params=sp, sampling_override=True)
    # Task yaml tries to force greedy
    gen_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "until": ["</s>"],
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 1.0  # CLI wins
    assert out.top_p == 0.95  # CLI wins
    assert out.top_k == 64  # preserved from CLI
    assert out.seed == 1234  # preserved from CLI
    # stop tokens from task yaml are still respected
    assert out.stop == ["</s>"]


def test_sampling_override_still_respects_max_tokens_from_yaml():
    """sampling_override only touches temperature / top_p.

    max_gen_toks from the yaml (if any) must still take precedence so
    per-task output budgets behave as documented.
    """
    sp = SamplingParams(
        max_tokens=256,  # CLI default
        temperature=1.0,
        top_p=0.95,
    )
    wrapper = _make_wrapper(sampling_params=sp, sampling_override=True)
    gen_kwargs = {
        "temperature": 0.0,
        "max_gen_toks": 512,  # task-specific cap
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 1.0  # CLI wins
    assert out.max_tokens == 512  # task yaml wins


def test_sampling_override_no_cli_falls_back_to_yaml():
    """No-override path keeps the pre-existing behaviour.

    If the CLI doesn't supply any sampling knobs, the wrapper falls back to
    the task yaml's gen_kwargs populating SamplingParams.
    """
    wrapper = _make_wrapper(sampling_params=None, sampling_override=False)
    gen_kwargs = {
        "temperature": 0.0,
        "max_gen_toks": 256,
        "until": ["</s>"],
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 0.0
    assert out.max_tokens == 256
    assert out.stop == ["</s>"]
