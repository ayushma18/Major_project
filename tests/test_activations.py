"""Correctness tests for activation capture.

The point of these is that a hook can be wired to the wrong module and still
produce a tensor of exactly the right shape. Shape checks would pass; the
research built on top would be quietly wrong. So we check *values* against two
independent references:

  1. `output_hidden_states=True`, which transformers computes without our hooks.
  2. The residual-stream identity  resid_post == resid_mid + mlp_out.

Runs on a 135M model on CPU -- no GPU, no gated access, a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import (  # noqa: E402
    ActivationCapture,
    CaptureConfig,
    collect_activations,
    get_layers,
    load_activation_set,
    save_activation_set,
)
from src.generate import build_chat_prompt  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    return model, tok


@pytest.fixture(scope="module")
def batch(model_and_tokenizer):
    _model, tok = model_and_tokenizer
    prompts = [
        build_chat_prompt(tok, user="What is the capital of France?", system="You are helpful."),
        build_chat_prompt(tok, user="Name a colour.", system="You are helpful."),
    ]
    return tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)


def test_layer_stack_is_found(model_and_tokenizer):
    model, _ = model_and_tokenizer
    layers = get_layers(model)
    assert len(layers) == model.config.num_hidden_layers


@pytest.fixture(scope="module")
def reference_hidden_states(model_and_tokenizer, batch):
    model, _ = model_and_tokenizer
    with torch.inference_mode():
        return model(**batch, output_hidden_states=True, use_cache=False).hidden_states


@pytest.fixture(scope="module")
def captured_resid_post(model_and_tokenizer, batch):
    model, _ = model_and_tokenizer
    cfg = CaptureConfig(sites=("resid_post",), token_position="last")
    with torch.inference_mode(), ActivationCapture(model, cfg) as cap:
        cap.set_attention_mask(batch["attention_mask"])
        model(**batch, use_cache=False)
        return cap.stacked()["resid_post"]  # [n_layers, batch, hidden]


def test_resid_post_matches_output_hidden_states(
    model_and_tokenizer, reference_hidden_states, captured_resid_post
):
    """The decisive test: our hook must equal transformers' own hidden states.

    `hidden_states` has length n_layers+1 -- element 0 is the embedding output,
    so the output of decoder layer i is hidden_states[i+1].

    The final layer is excluded here and checked separately: transformers applies
    the model's final RMSNorm before storing the last entry.
    """
    model, _ = model_and_tokenizer
    n_layers = model.config.num_hidden_layers
    assert captured_resid_post.shape[0] == n_layers

    for i in range(n_layers - 1):
        expected = reference_hidden_states[i + 1][:, -1, :]  # last position, left-padded
        torch.testing.assert_close(captured_resid_post[i], expected, rtol=1e-4, atol=1e-4)


def test_final_layer_is_the_raw_stream_not_the_normed_one(
    model_and_tokenizer, reference_hidden_states, captured_resid_post
):
    """We capture the pre-RMSNorm stream at the last layer, on purpose.

    Steering vectors are added to the residual stream, so directions have to be
    computed in that same un-normalised space. This test pins the choice down:
    applying the final norm ourselves must reproduce transformers' last entry,
    and the raw value must NOT already equal it.
    """
    model, _ = model_and_tokenizer
    raw_last = captured_resid_post[-1]
    normed_reference = reference_hidden_states[-1][:, -1, :]

    with torch.inference_mode():
        torch.testing.assert_close(
            model.model.norm(raw_last), normed_reference, rtol=1e-4, atol=1e-4
        )

    # Guard against the norm becoming a no-op and making the test vacuous.
    assert not torch.allclose(raw_last, normed_reference, rtol=1e-3, atol=1e-3)


def test_residual_stream_identity(model_and_tokenizer, batch):
    """resid_post == resid_mid + mlp_out, for every layer.

    This confirms resid_mid and mlp_out are hooked to the right side of the MLP
    block -- shape alone cannot distinguish them.
    """
    model, _ = model_and_tokenizer
    cfg = CaptureConfig(sites=("resid_post", "resid_mid", "mlp_out"), token_position="last")

    with torch.inference_mode(), ActivationCapture(model, cfg) as cap:
        cap.set_attention_mask(batch["attention_mask"])
        model(**batch, use_cache=False)
        acts = cap.stacked()

    torch.testing.assert_close(
        acts["resid_post"], acts["resid_mid"] + acts["mlp_out"], rtol=1e-4, atol=1e-4
    )


def test_all_positions_shape_and_last_agrees(model_and_tokenizer, batch):
    """token_position='all' must contain 'last' at index -1."""
    model, _ = model_and_tokenizer

    with torch.inference_mode(), ActivationCapture(
        model, CaptureConfig(sites=("resid_post",), token_position="all")
    ) as cap:
        cap.set_attention_mask(batch["attention_mask"])
        model(**batch, use_cache=False)
        all_pos = cap.stacked()["resid_post"]

    with torch.inference_mode(), ActivationCapture(
        model, CaptureConfig(sites=("resid_post",), token_position="last")
    ) as cap:
        cap.set_attention_mask(batch["attention_mask"])
        model(**batch, use_cache=False)
        last_pos = cap.stacked()["resid_post"]

    assert all_pos.ndim == 4  # [layer, batch, seq, hidden]
    assert all_pos.shape[2] == batch["input_ids"].shape[1]
    torch.testing.assert_close(all_pos[:, :, -1, :], last_pos)


def test_mean_ignores_left_padding(model_and_tokenizer):
    """A padded sequence must give the same mean as the same sequence unpadded.

    Without masking, left padding drags the mean toward the pad embedding, which
    would make every mean-pooled vector a function of batch composition.
    """
    model, tok = model_and_tokenizer
    text = build_chat_prompt(tok, user="What is the capital of France?", system="You are helpful.")
    short = build_chat_prompt(tok, user="Hi.", system="You are helpful.")

    cfg = CaptureConfig(sites=("resid_post",), token_position="mean")

    def capture(enc):
        with torch.inference_mode(), ActivationCapture(model, cfg) as cap:
            cap.set_attention_mask(enc["attention_mask"])
            model(**enc, use_cache=False)
            return cap.stacked()["resid_post"]

    alone = capture(tok([text], return_tensors="pt", add_special_tokens=False))
    # Batched with a shorter prompt, so `text` is unpadded but `short` is padded.
    padded = capture(tok([text, short], return_tensors="pt", padding=True, add_special_tokens=False))

    torch.testing.assert_close(alone[:, 0, :], padded[:, 0, :], rtol=1e-4, atol=1e-4)


def test_layer_subset_is_respected(model_and_tokenizer, batch):
    model, _ = model_and_tokenizer
    cfg = CaptureConfig(sites=("resid_post",), layers=(0, 5, 10), token_position="last")
    with torch.inference_mode(), ActivationCapture(model, cfg) as cap:
        cap.set_attention_mask(batch["attention_mask"])
        model(**batch, use_cache=False)
        assert cap.stacked()["resid_post"].shape[0] == 3
        assert cap.selected_layers == (0, 5, 10)


def test_negative_layer_index_wraps(model_and_tokenizer):
    model, _ = model_and_tokenizer
    n = model.config.num_hidden_layers
    cap = ActivationCapture(model, CaptureConfig(layers=(-1, -2)))
    assert cap.selected_layers == (n - 1, n - 2)


def test_hooks_are_removed_on_exit(model_and_tokenizer, batch):
    """A leaked hook would corrupt every later forward pass in the process."""
    model, _ = model_and_tokenizer
    layers = get_layers(model)
    before = len(layers[0]._forward_hooks)

    with ActivationCapture(model, CaptureConfig(sites=("resid_post",))) as cap:
        assert len(layers[0]._forward_hooks) > before
        with torch.inference_mode():
            model(**batch, use_cache=False)

    assert len(layers[0]._forward_hooks) == before


def test_hooks_are_removed_even_on_exception(model_and_tokenizer):
    model, _ = model_and_tokenizer
    layers = get_layers(model)
    before = len(layers[0]._forward_hooks)

    with pytest.raises(RuntimeError):
        with ActivationCapture(model, CaptureConfig(sites=("resid_post",))):
            raise RuntimeError("boom")

    assert len(layers[0]._forward_hooks) == before


def test_batching_does_not_change_results(model_and_tokenizer):
    """Batch size must be a performance knob only, never a correctness one."""
    model, tok = model_and_tokenizer
    prompts = [
        build_chat_prompt(tok, user=q, system="You are helpful.")
        for q in ["What is 2+2?", "Name a colour.", "Where is Rome?", "Say hello."]
    ]
    cfg = CaptureConfig(sites=("resid_post",), token_position="last")

    one = collect_activations(model, tok, prompts, config=cfg, batch_size=1, progress=False)
    four = collect_activations(model, tok, prompts, config=cfg, batch_size=4, progress=False)

    torch.testing.assert_close(
        one.site("resid_post"), four.site("resid_post"), rtol=1e-4, atol=1e-4
    )


def test_save_load_round_trip(model_and_tokenizer, tmp_path):
    model, tok = model_and_tokenizer
    prompts = [build_chat_prompt(tok, user="What is 2+2?", system="You are helpful.")]
    acts = collect_activations(
        model,
        tok,
        prompts,
        config=CaptureConfig(sites=("resid_post", "attn_out"), layers=(0, 1), token_position="last"),
        provenance={"model_id": MODEL_ID},
        progress=False,
    )

    out = save_activation_set(acts, tmp_path / "run")
    back = load_activation_set(out)

    assert back.layer_indices == acts.layer_indices
    assert back.provenance["model_id"] == MODEL_ID
    assert len(back.prompts) == 1
    for site in acts.activations:
        assert torch.equal(acts.activations[site], back.activations[site])


def test_layer_accessor_uses_absolute_indices(model_and_tokenizer, batch):
    model, tok = model_and_tokenizer
    prompts = [build_chat_prompt(tok, user="Hi.", system="S")]
    acts = collect_activations(
        model, tok, prompts, config=CaptureConfig(layers=(3, 7)), progress=False
    )
    assert acts.layer("resid_post", 7).shape == (1, model.config.hidden_size)
    with pytest.raises(KeyError):
        acts.layer("resid_post", 4)  # captured 3 and 7, not 4


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="Unknown site"):
        CaptureConfig(sites=("not_a_site",))
    with pytest.raises(ValueError, match="token_position"):
        CaptureConfig(token_position="middle")


def test_out_of_range_layer_is_rejected(model_and_tokenizer):
    model, _ = model_and_tokenizer
    with pytest.raises(IndexError):
        ActivationCapture(model, CaptureConfig(layers=(9999,)))
