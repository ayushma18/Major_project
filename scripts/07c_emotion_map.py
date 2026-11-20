"""E04c: the emotion map after removing neutral-text directions (Anthropic 2026 method).

    python scripts/07c_emotion_map.py --model llama-8b --layer 12

Anthropic (Sofroniew et al. 2026) project out the principal components of NEUTRAL text that
explain 50% of its variance before analysing emotion vectors, which removes topic and style
differences. Here the neutral twins of all 19 emotions are the neutral text. Reports, at one
layer, the PCA of the cleaned one-vs-mean emotion vectors and its correlation with valence
and arousal labels, before and after cleaning. CPU only, reads the cached story means.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.runtime import ALIASES, write_json  # noqa: E402
from src.vectors import slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
ev = __import__("importlib").import_module("07_emotion_vectors")


def pca(X: torch.Tensor, k: int = 2):
    Xc = X - X.mean(0)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:k].T, (S[:k] ** 2 / (S**2).sum()).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--var", type=float, default=0.5)
    args = ap.parse_args()
    name = ALIASES.get(args.model, args.model)
    T = load_file(str(ROOT / "outputs" / "activations" / slug(name) / "story_means.safetensors"))
    E = [e for e in data.EMOTIONS if f"{e}_emo" in T]
    L = args.layer
    emo = {e: T[f"{e}_emo"][:50, L] for e in E}          # train split
    neu = torch.cat([T[f"{e}_neu"][:, L] for e in E])     # all neutral twins
    nc = neu - neu.mean(0)
    _, S, Vh = torch.linalg.svd(nc, full_matrices=False)
    cum = (S**2).cumsum(0) / (S**2).sum()
    k = int((cum < args.var).sum()) + 1
    P = Vh[:k]                                             # [k, H]
    means = torch.stack([emo[e].mean(0) for e in E])
    cen = means - means.mean(0)
    cleaned = cen - (cen @ P.T) @ P
    out = {"model": name, "layer": L, "emotions": E, "neutral_pcs_removed": k, "variance_target": args.var}
    for tag, X in (("raw", cen), ("cleaned", cleaned)):
        pcs, expl = pca(X)
        val = [ev.VALENCE[e] for e in E]
        aro = [ev.AROUSAL[e] for e in E]
        r = {"pc1_valence": ev.pearson(pcs[:, 0].tolist(), val), "pc2_arousal": ev.pearson(pcs[:, 1].tolist(), aro),
             "pc1_arousal": ev.pearson(pcs[:, 0].tolist(), aro), "pc2_valence": ev.pearson(pcs[:, 1].tolist(), val)}
        out[tag] = {"explained": expl, "corr": r, "pc1": pcs[:, 0].tolist(), "pc2": pcs[:, 1].tolist()}
        print(f"{tag:8s} PC1 {100 * expl[0]:.0f}% r(valence) {r['pc1_valence']:+.2f} r(arousal) {r['pc1_arousal']:+.2f} | PC2 {100 * expl[1]:.0f}% r(arousal) {r['pc2_arousal']:+.2f} r(valence) {r['pc2_valence']:+.2f}")
    print(f"neutral PCs removed: {k}")
    write_json(ROOT / "results" / "raw" / f"E04c_{args.model}_L{L}.json", out)


if __name__ == "__main__":
    main()
