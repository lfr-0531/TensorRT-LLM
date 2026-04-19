# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CoVoST 2 speech-translation benchmark.

Evaluates multimodal (audio-capable) LLMs on the CoVoST 2 test split using
pre-packaged audio from the ``fixie-ai/covost2`` mirror (no manual Common
Voice download required) and BLEU scoring via sacrebleu.

Gemma4's HF blog reports CoVoST scores for the E2B/E4B checkpoints, so this
task is added here for apples-to-apples comparison.
"""

from typing import Iterable, List, Optional, Tuple, Union

import click
import datasets
import numpy as np

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator

# Human-readable language names for the CoVoST 2 codes.  Used in the
# translation prompt to disambiguate the target language name for the model.
_LANG_NAMES = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cy": "Welsh",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "lv": "Latvian",
    "mn": "Mongolian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "sl": "Slovenian",
    "sv-SE": "Swedish",
    "ta": "Tamil",
    "tr": "Turkish",
    "zh-CN": "Chinese",
}


def _split_lang_pair(pair: str) -> Tuple[str, str]:
    """Parse a CoVoST config name like 'en_de' or 'en_zh-CN' into (src, tgt)."""
    src, tgt = pair.split("_", 1)
    return src, tgt


class CoVoST2(Evaluator):
    """Speech-translation benchmark.

    Each sample consists of an audio clip in a source language and a gold
    reference translation in the target language.  The model is prompted to
    translate the clip and scored with BLEU.
    """

    DEFAULT_DATASET = "fixie-ai/covost2"

    def __init__(
        self,
        lang_pair: str = "en_de",
        dataset_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        random_seed: int = 0,
        apply_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            random_seed=random_seed,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
            output_dir=output_dir,
        )
        self.lang_pair = lang_pair
        self.src_lang, self.tgt_lang = _split_lang_pair(lang_pair)
        self.src_name = _LANG_NAMES.get(self.src_lang, self.src_lang)
        self.tgt_name = _LANG_NAMES.get(self.tgt_lang, self.tgt_lang)

        if dataset_path is None:
            dataset_path = self.DEFAULT_DATASET
        self.data = datasets.load_dataset(
            dataset_path, lang_pair, split="test", trust_remote_code=True
        )
        self.data = self.data.shuffle(random_seed)
        if num_samples is None:
            self.num_samples = self.data.num_rows
        else:
            self.num_samples = min(num_samples, self.data.num_rows)

        self._bleu_scorer = None

    def _prompt_text(self) -> str:
        return (
            f"Translate the following {self.src_name} audio to {self.tgt_name}. "
            f"Respond with only the {self.tgt_name} translation, no other text."
        )

    def generate_samples(self) -> Iterable[tuple]:
        prompt_text = self._prompt_text()
        for i, sample in enumerate(self.data):
            if i >= self.num_samples:
                break
            audio_field = sample.get("audio")
            if audio_field is None:
                continue
            audio_array = audio_field.get("array")
            if audio_array is None:
                continue
            audio_array = np.asarray(audio_array, dtype=np.float32)
            reference = sample["translation"]
            # For multimodal audio requests we pass a dict with both the text
            # prompt and the audio payload; the input processor recognises
            # ``multi_modal_data`` and routes the audio tensor through the
            # audio tower.  Using a full ConversationMessage is unnecessary
            # because ``apply_chat_template`` is handled here.
            mm_input = {
                "prompt": prompt_text,
                "multi_modal_data": {"audio": [audio_array]},
            }
            yield mm_input, None, reference

    def do_apply_chat_template(self, llm, prompt):
        """Override to handle multimodal audio prompts.

        ``Evaluator.do_apply_chat_template`` assumes a pure-text prompt.  For
        audio requests we wrap the text in a ``<|audio|>`` placeholder so the
        Gemma4 chat template inserts the audio soft tokens in the right place.
        The multimodal data is carried separately in the dict returned by
        ``generate_samples``.
        """
        if isinstance(prompt, dict):
            text = prompt["prompt"]
            wrapped_text = f"<|audio|>\n{text}"
            # Route through the base implementation for the chat wrapping,
            # then re-attach the multimodal data to the returned dict.
            templated = super().do_apply_chat_template(llm, wrapped_text)
            prompt["prompt"] = templated
            return prompt
        return super().do_apply_chat_template(llm, prompt)

    def _ensure_bleu(self):
        if self._bleu_scorer is None:
            try:
                import sacrebleu

                self._bleu_scorer = sacrebleu
            except ImportError as e:
                raise ImportError(
                    "CoVoST2 evaluation requires sacrebleu for BLEU scoring. "
                    "Install it with `pip install sacrebleu`."
                ) from e

    def compute_score(
        self, outputs: List[RequestOutput], references: List[str], *auxiliaries
    ) -> float:
        self._ensure_bleu()
        predictions = [o.outputs[0].text.strip() for o in outputs]
        # sacrebleu expects [[ref1_a, ref1_b, ...], [ref2_a, ...]] where the
        # outer list is over reference streams; we only have a single stream.
        bleu = self._bleu_scorer.corpus_bleu(
            predictions, [references], tokenize=self._sacrebleu_tokenizer()
        )
        score = bleu.score
        logger.info(
            f"CoVoST2 [{self.lang_pair}] corpus BLEU = {score:.2f} "
            f"(sys_len={bleu.sys_len}, ref_len={bleu.ref_len}, "
            f"precisions={bleu.precisions})"
        )
        return score

    def _sacrebleu_tokenizer(self) -> str:
        # Use BLEU tokenizers matching the target language to match the
        # conventions in the CoVoST literature.
        mapping = {
            "zh-CN": "zh",
            "ja": "ja-mecab",
        }
        return mapping.get(self.tgt_lang, "13a")

    @click.command("covost2")
    @click.option(
        "--lang_pair",
        type=str,
        default="en_de",
        help="CoVoST 2 language pair (e.g., en_de, en_zh-CN, de_en). "
        "See fixie-ai/covost2 for the full list.",
    )
    @click.option(
        "--dataset_path",
        type=str,
        default=None,
        help="Dataset path on HF hub or local. Defaults to fixie-ai/covost2 (pre-packaged audio).",
    )
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset.",
    )
    @click.option("--random_seed", type=int, default=0, help="Random seed for dataset shuffling.")
    @click.option(
        "--apply_chat_template",
        is_flag=True,
        default=True,
        help="Whether to apply chat template (default: True for Gemma4).",
    )
    @click.option("--system_prompt", type=str, default=None, help="System prompt.")
    @click.option("--max_input_length", type=int, default=4096, help="Maximum prompt length.")
    @click.option("--max_output_length", type=int, default=256, help="Maximum generation length.")
    @click.option(
        "--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)."
    )
    @click.option("--output_dir", type=str, default=None, help="Directory to save the task infos.")
    @click.pass_context
    @staticmethod
    def command(
        ctx,
        lang_pair: str,
        dataset_path: Optional[str],
        num_samples: Optional[int],
        random_seed: int,
        apply_chat_template: bool,
        system_prompt: Optional[str],
        max_input_length: int,
        max_output_length: int,
        temperature: float,
        output_dir: Optional[str],
    ) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length,
            temperature=temperature,
        )
        evaluator = CoVoST2(
            lang_pair=lang_pair,
            dataset_path=dataset_path,
            num_samples=num_samples,
            random_seed=random_seed,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
            output_dir=output_dir,
        )
        evaluator.evaluate(llm, sampling_params)
        llm.shutdown()
