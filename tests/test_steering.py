"""Correctness tests for src/steering.py on a tiny random Llama (CPU, no downloads).

The checks that matter:
  * Directional ablation leaves NO component along r anywhere in the residual stream.
  * Hook ablation equals weight orthogonalisation (Arditi et al.'s equivalent form).
  * Activation addition shifts the chosen layer's output by exactly coeff * v and
    changes nothing upstream.
  * Hooks are removed on exit, and generation with the KV cache matches a no-cache run.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from src.activations import ActivationCapture, CaptureConfig  # noqa: E402
from src.steering import AblateDirection, AddVector, Compose, cosine, orthogonalize  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 97, (2, 9))
    return model, ids


def _resid(model, ids, sites=("resid_post", "resid_mid")):
    with ActivationCapture(model, CaptureConfig(sites=sites, token_position="all")) as cap:
        model(input_ids=ids, use_cache=False)
        return cap.stacked()


def test_ablation_removes_direction_everywhere(tiny):
    model, ids = tiny
    r = torch.randn(64)
    r_hat = r / r.norm()
    with AblateDirection(model, r):
        acts = _resid(model, ids)
    for site, t in acts.items():
        comp = (t.float() @ r_hat).abs().max().item()
        assert comp < 1e-4, f"{site} keeps a component of {comp} along r"


def test_ablation_equals_weight_orthogonalisation(tiny):
    model, ids = tiny
    r = torch.randn(64)
    r_hat = r / r.norm()
    with torch.no_grad():
        with AblateDirection(model, r):
            hook_logits = model(input_ids=ids, use_cache=False).logits

        import copy

        m2 = copy.deepcopy(model)
        P = torch.eye(64) - torch.outer(r_hat, r_hat)
        m2.model.embed_tokens.weight.copy_(m2.model.embed_tokens.weight @ P)
        for layer in m2.model.layers:
            layer.self_attn.o_proj.weight.copy_(P @ layer.self_attn.o_proj.weight)
            layer.mlp.down_proj.weight.copy_(P @ layer.mlp.down_proj.weight)
        w_logits = m2(input_ids=ids, use_cache=False).logits
    assert torch.allclose(hook_logits, w_logits, atol=1e-4), (hook_logits - w_logits).abs().max()


def test_add_vector_shifts_exactly_one_layer(tiny):
    model, ids = tiny
    base = _resid(model, ids, sites=("resid_post",))["resid_post"]
    v = torch.randn(64)
    with AddVector(model, layer=1, vector=v, coeff=2.5):
        steered = _resid(model, ids, sites=("resid_post",))["resid_post"]
    # Upstream untouched.
    assert torch.allclose(base[0], steered[0], atol=1e-6)
    # Layer 1 output shifted by exactly 2.5 v at every position.
    diff = steered[1] - base[1]
    assert torch.allclose(diff, (2.5 * v).expand_as(diff), atol=1e-5)
    # Downstream changed.
    assert not torch.allclose(base[2], steered[2], atol=1e-4)


def test_add_vector_normalize(tiny):
    model, ids = tiny
    base = _resid(model, ids, sites=("resid_post",))["resid_post"]
    v = 7.0 * torch.randn(64)
    with AddVector(model, layer=0, vector=v, coeff=3.0, normalize=True):
        steered = _resid(model, ids, sites=("resid_post",))["resid_post"]
    norms = (steered[0] - base[0]).norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 3.0), atol=1e-4)


def test_zero_coeff_is_identity_and_hooks_removed(tiny):
    model, ids = tiny
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits
        with AddVector(model, layer=2, vector=torch.randn(64), coeff=0.0):
            same = model(input_ids=ids, use_cache=False).logits
        with AblateDirection(model, torch.randn(64)):
            pass
        after = model(input_ids=ids, use_cache=False).logits
    assert torch.allclose(ref, same, atol=1e-6)
    assert torch.allclose(ref, after, atol=1e-6)
    for layer in model.model.layers:
        assert not layer._forward_hooks and not layer._forward_pre_hooks
        assert not layer.self_attn._forward_hooks and not layer.mlp._forward_hooks


def test_hooks_removed_on_exception(tiny):
    model, _ = tiny
    with pytest.raises(RuntimeError):
        with Compose(AddVector(model, 1, torch.randn(64), 1.0), AblateDirection(model, torch.randn(64))):
            raise RuntimeError("boom")
    for layer in model.model.layers:
        assert not layer._forward_hooks and not layer._forward_pre_hooks


def test_generation_with_cache_matches_no_cache(tiny):
    """Steering must act identically on prefill+decode with the KV cache and on full
    recomputation, otherwise generated tokens would be steered inconsistently."""
    model, ids = tiny
    v = torch.randn(64)
    r = torch.randn(64)
    with torch.no_grad(), Compose(AddVector(model, 1, v, 1.5), AblateDirection(model, r)):
        out = model.generate(input_ids=ids[:1], max_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=0)
        seq = ids[:1]
        for _ in range(6):
            logits = model(input_ids=seq, use_cache=False).logits[:, -1]
            seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, seq)


def test_orthogonalize_and_cosine():
    torch.manual_seed(1)
    v, r = torch.randn(32), torch.randn(32)
    o = orthogonalize(v, r)
    assert abs(cosine(o, r)) < 1e-5
    assert abs(o.norm().item() - v.norm().item()) < 1e-4
    o2 = orthogonalize(v, r, keep_norm=False)
    assert torch.allclose(o2 + (v @ (r / r.norm())) * (r / r.norm()), v, atol=1e-5)


def test_add_vector_rows_matches_per_row_runs(tiny):
    from src.steering import AddVectorRows

    model, ids = tiny
    deltas = torch.randn(2, 64)
    with torch.no_grad():
        with AddVectorRows(model, 1, deltas):
            both = model(input_ids=ids, use_cache=False).logits
        for row in range(2):
            with AddVector(model, 1, deltas[row], 1.0):
                single = model(input_ids=ids[row : row + 1], use_cache=False).logits
            assert torch.allclose(both[row], single[0], atol=1e-5)


def test_add_vector_masked_only_touches_masked_positions(tiny):
    from src.steering import AddVectorMasked

    model, ids = tiny
    base = _resid(model, ids, sites=("resid_post",))["resid_post"]
    mask = torch.zeros(ids.shape, dtype=torch.bool)
    mask[:, 2:5] = True
    v = torch.randn(64)
    with AddVectorMasked(model, 1, v, mask, on_decode=False):
        steered = _resid(model, ids, sites=("resid_post",))["resid_post"]
    diff = steered[1] - base[1]
    assert torch.allclose(diff[:, 2:5], v.expand(2, 3, 64), atol=1e-5)
    assert torch.allclose(diff[:, :2], torch.zeros(2, 2, 64), atol=1e-6)
    assert torch.allclose(diff[:, 5:], torch.zeros(2, ids.shape[1] - 5, 64), atol=1e-6)
    # positions after the mask still change at LATER layers (attention carries it forward)
    assert not torch.allclose(steered[2][:, 5:], base[2][:, 5:], atol=1e-4)
