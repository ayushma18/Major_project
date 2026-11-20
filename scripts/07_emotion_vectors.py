"""E04: extract the emotion vectors from the story corpus and check them on held-out stories.

    python scripts/07_emotion_vectors.py --model llama-8b

Two vectors per emotion and layer (both stored in artifacts/vectors/<model>/):
    emotion_caa_<e>       paired CAA (proposal eq. 5.1): mean over the 50 train pairs of
                          (story with emotion) - (its neutral twin)
    emotion_centered_<e>  one-vs-mean: mean of e's emotional train stories minus the mean of
                          all 19 emotions' emotional train stories (as in Anthropic's emotion
                          work); also defined for `neutral` from the standalone neutral stories

Held-out checks (pairs 51 to 60 of every emotion, never used to build a vector):
    pair_acc[l]        fraction of held-out pairs whose emotional story projects higher on
                       v_e than its neutral twin (chance 0.5)
    diff_top1[l]       19-way: is the held-out difference (emo - neutral) closest (cosine) to
                       its own emotion's CAA vector (chance 1/19)
    story_top1[l]      19-way on single stories: (h - grand mean) closest to own centered vector
Plus the emotion-by-emotion cosine matrix and a 2-D PCA at the chosen readout layer,
with the correlation of PC1 with a valence label (positive / negative emotions).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.runtime import load, write_json  # noqa: E402
from src.vectors import VectorLibrary, slug, span_means  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VALENCE = {"afraid": -1, "angry": -1, "anxious": -1, "calm": 1, "desperate": -1, "enthusiastic": 1, "exasperated": -1,
           "fearful": -1, "gloomy": -1, "grateful": 1, "guilty": -1, "happy": 1, "hopeful": 1, "hostile": -1, "loving": 1,
           "nervous": -1, "proud": 1, "reflective": 0, "sad": -1}
AROUSAL = {"afraid": 1, "angry": 1, "anxious": 1, "calm": -1, "desperate": 1, "enthusiastic": 1, "exasperated": 1,
           "fearful": 1, "gloomy": -1, "grateful": -1, "guilty": 0, "happy": 1, "hopeful": 0, "hostile": 1, "loving": -1,
           "nervous": 1, "proud": 0, "reflective": -1, "sad": -1}


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def pearson(a: list[float], b: list[float]) -> float:
    ta, tb = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    ta, tb = ta - ta.mean(), tb - tb.mean()
    return float((ta @ tb) / (ta.norm() * tb.norm()).clamp(min=1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--readout-layer", type=int, default=None, help="layer for the cosine matrix and PCA (default: best story_top1)")
    args = ap.parse_args()
    t0 = time.time()
    model, tok, prov = load(args.model)
    lib = VectorLibrary(prov["model_name"])
    emotions = data.available_emotions()
    cache = ROOT / "outputs" / "activations" / slug(prov["model_name"])
    cache.mkdir(parents=True, exist_ok=True)

    H: dict[str, dict[str, torch.Tensor]] = {}
    for e in emotions:
        rows = data.load_stories(e)
        print(f"{e}: {len(rows)} pairs")
        emo = span_means(model, tok, [r["emotional"] for r in rows], batch_size=args.batch_size)
        neu = span_means(model, tok, [r["neutral"] for r in rows], batch_size=args.batch_size)
        split = torch.tensor([r["split"] == "train" for r in rows])
        H[e] = {"emo": emo, "neu": neu, "train": split}
    neutral_rows = [__import__("json").loads(l) for l in open(data.STORIES / "neutral.jsonl", encoding="utf-8") if l.strip()] if (data.STORIES / "neutral.jsonl").exists() else []
    neutral_h = span_means(model, tok, [r["text"] for r in neutral_rows], batch_size=args.batch_size) if neutral_rows else None

    from safetensors.torch import save_file

    save_file({f"{e}_{k}": v.contiguous() for e, d in H.items() for k, v in d.items() if k != "train"}, str(cache / "story_means.safetensors"))

    n_layers = next(iter(H.values()))["emo"].shape[1]
    caa = {e: (d["emo"][d["train"]] - d["neu"][d["train"]]).mean(0) for e, d in H.items()}  # [L, H]
    emo_means = {e: d["emo"][d["train"]].mean(0) for e, d in H.items()}
    grand = torch.stack(list(emo_means.values())).mean(0)
    centered = {e: m - grand for e, m in emo_means.items()}
    if neutral_h is not None:
        centered["neutral"] = neutral_h[:50].mean(0) - grand
    for e in emotions:
        lib.save(f"emotion_caa_{e}", caa[e], {"emotion": e, "kind": "paired CAA", "n_pairs": int(H[e]["train"].sum()), "space": "resid_post[layer]", "shape": list(caa[e].shape)})
        lib.save(f"emotion_centered_{e}", centered[e], {"emotion": e, "kind": "one-vs-mean of 19 emotions", "space": "resid_post[layer]"})
    # The paired CAA vectors share one large common component ("emotional vs flat
    # narration"). Store it, and each emotion's SPECIFIC part with that component removed.
    caa_units = torch.stack([caa[e] / caa[e].norm(dim=-1, keepdim=True) for e in emotions])  # [E, L, H]
    common = caa_units.mean(0)  # [L, H]
    common_hat = common / common.norm(dim=-1, keepdim=True)
    lib.save("emotion_common", common, {"kind": "mean of the 19 unit paired-CAA vectors per layer", "space": "resid_post[layer]"})
    for e in emotions:
        proj = (caa[e] * common_hat).sum(-1, keepdim=True) * common_hat
        lib.save(f"emotion_specific_{e}", caa[e] - proj, {"emotion": e, "kind": "paired CAA minus its projection on emotion_common", "space": "resid_post[layer]"})
    if "neutral" in centered:
        lib.save("emotion_centered_neutral", centered["neutral"], {"emotion": "neutral", "kind": "neutral standalone minus mean of 19 emotions"})

    # Held-out validation, every layer.
    E = emotions
    caa_stack = torch.stack([caa[e] for e in E])  # [E, L, H]
    cen_stack = torch.stack([centered[e] for e in E])
    pair_acc, diff_top1, story_top1 = [], [], []
    for l in range(n_layers):
        ok_pair = tot = ok_diff = ok_story = n_story = 0
        cu = unit(caa_stack[:, l])
        ce = unit(cen_stack[:, l])
        for i, e in enumerate(E):
            d = H[e]
            ho = ~d["train"]
            ve = cu[i]
            pe = d["emo"][ho, l] @ ve
            pn = d["neu"][ho, l] @ ve
            ok_pair += int((pe > pn).sum())
            tot += int(ho.sum())
            diffs = unit(d["emo"][ho, l] - d["neu"][ho, l])
            ok_diff += int(((diffs @ cu.T).argmax(-1) == i).sum())
            xs = unit(d["emo"][ho, l] - grand[l])
            ok_story += int(((xs @ ce.T).argmax(-1) == i).sum())
            n_story += int(ho.sum())
        pair_acc.append(ok_pair / tot)
        diff_top1.append(ok_diff / tot)
        story_top1.append(ok_story / n_story)
    readout = args.readout_layer if args.readout_layer is not None else int(torch.tensor(story_top1).argmax())

    # Geometry at the readout layer.
    U = unit(caa_stack[:, readout])
    cos_matrix = (U @ U.T).tolist()
    X = cen_stack[:, readout]
    Xc = X - X.mean(0)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    pcs = (Xc @ Vh[:2].T)
    explained = (S[:2] ** 2 / (S**2).sum()).tolist()
    pc1, pc2 = pcs[:, 0].tolist(), pcs[:, 1].tolist()
    val = [VALENCE[e] for e in E]
    aro = [AROUSAL[e] for e in E]
    report = {
        "model": prov["model_name"], "emotions": E, "n_layers": n_layers, "readout_layer": readout,
        "heldout": {"pair_acc": pair_acc, "diff_top1": diff_top1, "story_top1": story_top1, "chance_top1": 1 / len(E)},
        "cos_matrix_caa": cos_matrix,
        "pca_centered": {"pc1": pc1, "pc2": pc2, "explained": explained, "corr_pc1_valence": pearson(pc1, val), "corr_pc2_arousal": pearson(pc2, aro), "corr_pc1_arousal": pearson(pc1, aro), "corr_pc2_valence": pearson(pc2, val)},
        "vector_norms_caa": {e: caa[e].norm(dim=-1).tolist() for e in E},
        "shared_component": {"mean_offdiag_cos_caa": [float(((caa_units[:, l] @ caa_units[:, l].T).sum() - len(E)) / (len(E) * (len(E) - 1))) for l in range(n_layers)],
                             "mean_cos_to_common": [float((caa_units[:, l] @ common_hat[l]).mean()) for l in range(n_layers)]},
        "seconds": time.time() - t0, "provenance": prov,
    }
    write_json(ROOT / "results" / "raw" / f"E04_{args.model}.json", report)
    best = max(range(n_layers), key=lambda l: diff_top1[l])
    print(f"held-out pair accuracy at best layer {best}: {pair_acc[best]:.2f}; diff top-1 {diff_top1[best]:.2f}; story top-1 max {max(story_top1):.2f} at layer {readout} (chance {1 / len(E):.2f})")
    print(f"PC1 vs valence r = {report['pca_centered']['corr_pc1_valence']:.2f}; PC2 vs arousal r = {report['pca_centered']['corr_pc2_arousal']:.2f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
