"""Guard the Llama 3.1 prompt format.

If templating silently drifts (a transformers upgrade, a different tokenizer),
every activation we collect afterwards is off-distribution and the whole study
is invalid. These tests are cheap -- tokenizer only, no weights -- so run them.

    .venv/bin/python -m pytest tests/ -v

Uses an ungated mirror of the exact Llama 3.1 8B Instruct tokenizer so the test
runs without a licence approval. Set LLAMA_TOKENIZER_ID to the gated repo id to
check against the canonical source.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate import build_chat_prompt, eos_token_ids  # noqa: E402

TOKENIZER_ID = os.environ.get("LLAMA_TOKENIZER_ID", "unsloth/Meta-Llama-3.1-8B-Instruct")

EOT_ID = 128009  # <|eot_id|> in the Llama 3.1 vocabulary
BOS_ID = 128000  # <|begin_of_text|>


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def test_stop_tokens_include_eot(tokenizer):
    """Generation must stop on <|eot_id|>, not just <|end_of_text|>."""
    assert EOT_ID in eos_token_ids(tokenizer)


def test_prompt_has_all_three_headers(tokenizer):
    prompt = build_chat_prompt(tokenizer, user="Hello", system="You are helpful.")
    for role in ("system", "user", "assistant"):
        assert f"<|start_header_id|>{role}<|end_header_id|>" in prompt


def test_prompt_ends_on_assistant_header(tokenizer):
    """Without a trailing assistant header the model continues the user turn."""
    prompt = build_chat_prompt(tokenizer, user="Hello", system="You are helpful.")
    assert prompt.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")


def test_bos_is_not_duplicated(tokenizer):
    """The template emits BOS, so we must tokenize with add_special_tokens=False."""
    prompt = build_chat_prompt(tokenizer, user="Hello", system="You are helpful.")
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert ids.count(BOS_ID) == 1


def test_system_prompt_is_optional(tokenizer):
    prompt = build_chat_prompt(tokenizer, user="Hello")
    assert "<|start_header_id|>user<|end_header_id|>" in prompt


def test_history_is_preserved_in_order(tokenizer):
    history = [
        {"role": "user", "content": "FIRST_TURN"},
        {"role": "assistant", "content": "REPLY_TURN"},
    ]
    prompt = build_chat_prompt(tokenizer, user="SECOND_TURN", system="S", history=history)
    assert prompt.index("FIRST_TURN") < prompt.index("REPLY_TURN") < prompt.index("SECOND_TURN")


def test_base_model_tokenizer_is_rejected(tokenizer):
    """A base (non-Instruct) tokenizer has no chat template -- fail loudly."""

    class NoTemplate:
        chat_template = None

    with pytest.raises(ValueError, match="chat_template"):
        build_chat_prompt(NoTemplate(), user="Hello")
