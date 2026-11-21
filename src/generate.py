"""Chat-formatted generation for Llama 3.1 Instruct.

The one thing that is easy to get wrong here is the prompt format. Llama 3.1
Instruct was trained with a specific header/EOT structure; feeding it a raw
string instead produces plausible-looking but off-distribution behaviour, which
would quietly poison every activation we collect later. So we always go through
`tokenizer.apply_chat_template`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

log = logging.getLogger(__name__)

# Llama 3.1 ends an assistant turn with <|eot_id|>, not the base <|end_of_text|>.
# Newer transformers sets this in generation_config, but we pass it explicitly so
# behaviour does not depend on the version installed.
_LLAMA_EOT = "<|eot_id|>"


@dataclass
class GenerationResult:
    prompt_text: str
    completion: str
    num_prompt_tokens: int
    num_completion_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.num_completion_tokens / self.seconds if self.seconds > 0 else 0.0


def build_chat_prompt(
    tokenizer: Any,
    user: str,
    system: str | None = None,
    history: Sequence[dict] | None = None,
) -> str:
    """Render messages into the model's exact instruct format."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    if tokenizer.chat_template is None:
        raise ValueError(
            "This tokenizer has no chat_template -- it is probably a base model, "
            "not an Instruct model. Check the model id."
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # emit the assistant header so it replies
    )


def eos_token_ids(tokenizer: Any) -> list[int]:
    """All ids that should terminate an assistant turn."""
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    # End-of-turn tokens of the chat families used here (Llama 3, Qwen2.5). Unknown tokens map
    # to None or to unk_token_id depending on the tokenizer, so guard against a bogus id.
    for tok in (_LLAMA_EOT, "<|im_end|>", "<|endoftext|>"):
        eot = tokenizer.convert_tokens_to_ids(tok)
        if eot is not None and eot != tokenizer.unk_token_id and eot not in ids:
            ids.append(eot)
    return ids


@torch.inference_mode()
def generate_chat(
    model: Any,
    tokenizer: Any,
    user: str,
    system: str | None = None,
    *,
    history: Sequence[dict] | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = 0,
) -> GenerationResult:
    """Generate one assistant reply. Greedy by default, for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)

    prompt_text = build_chat_prompt(tokenizer, user=user, system=system, history=history)

    # add_special_tokens=False: the template already emitted <|begin_of_text|>,
    # so letting the tokenizer add another would double the BOS.
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    num_prompt_tokens = int(inputs["input_ids"].shape[-1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": eos_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start = time.perf_counter()
    output_ids = model.generate(**inputs, **gen_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Slice off the prompt so we return only what the model actually wrote.
    new_ids = output_ids[0, num_prompt_tokens:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return GenerationResult(
        prompt_text=prompt_text,
        completion=completion,
        num_prompt_tokens=num_prompt_tokens,
        num_completion_tokens=int(new_ids.shape[-1]),
        seconds=elapsed,
    )
