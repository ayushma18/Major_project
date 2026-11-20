"""E04b: do the emotion vectors recognise emotion in an INDEPENDENT test set?

    python scripts/07b_ood_eval.py --model llama-8b

The vectors were built from long third-person story pairs written by GPT agents. This test
uses data/stories/ood/validation.jsonl: 228 short first-person texts (monologue, speech,
messages, diary lines; 12 per emotion) written by a different model (Claude), on everyday
topics, never naming the emotion. A vector that only captured the training corpus's
vocabulary or scenarios should fail here.

For every layer, each text's mean activation (minus the mean over all 228 texts) is assigned
to the emotion whose vector has the highest cosine. Reported per vector type (centered,
specific, caa): 19-way top-1 accuracy by layer, the best layer, and a per-emotion confusion
summary at that layer. Chance is 1/19 = 5.3%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.runtime import load, write_json  # noqa: E402
from src.vectors import VectorLibrary, span_means  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    model, tok, prov = load(args.model)
    lib = VectorLibrary(prov["model_name"])
    rows = [json.loads(l) for l in open(ROOT / "data" / "stories" / "ood" / "validation.jsonl", encoding="utf-8") if l.strip()]
    E = [e for e in data.EMOTIONS if lib.exists(f"emotion_centered_{e}")]
    rows = [r for r in rows if r["emotion"] in E]
    H = span_means(model, tok, [r["text"] for r in rows], wrapper="Read this short text.\n\n{text}", batch_size=args.batch_size, skip_first=1)  # [N, L, H]
    labels = torch.tensor([E.index(r["emotion"]) for r in rows])
    X = H - H.mean(0, keepdim=True)
    n_layers = H.shape[1]
    out = {"model": prov["model_name"], "n_texts": len(rows), "emotions": E, "chance": 1 / len(E), "types": {}}
    for kind in ("centered", "specific", "caa"):
        if not all(lib.exists(f"emotion_{kind}_{e}") for e in E):
            continue
        V = torch.stack([lib.load(f"emotion_{kind}_{e}")[0] for e in E])  # [E, L, H]
        acc = []
        preds_by_layer = []
        for l in range(n_layers):
            pred = (unit(X[:, l]) @ unit(V[:, l]).T).argmax(-1)
            acc.append(float((pred == labels).float().mean()))
            preds_by_layer.append(pred)
        best = int(torch.tensor(acc).argmax())
        pred = preds_by_layer[best]
        per = {e: float((pred[labels == i] == i).float().mean()) for i, e in enumerate(E)}
        confused = {}
        for i, e in enumerate(E):
            wrong = pred[(labels == i) & (pred != i)].tolist()
            if wrong:
                top = max(set(wrong), key=wrong.count)
                confused[e] = E[top]
        out["types"][kind] = {"top1_by_layer": acc, "best_layer": best, "best_top1": acc[best], "per_emotion_at_best": per, "most_confused_with": confused}
        print(f"{kind:9s} best layer {best}: 19-way accuracy {acc[best]:.2f} (chance {1 / len(E):.2f})")
    # Headline numbers for the interface: the centered vectors (one-vs-mean, closest to Anthropic's method).
    main_kind = "centered" if "centered" in out["types"] else next(iter(out["types"]))
    out["best_top1"] = out["types"][main_kind]["best_top1"]
    out["top1_by_layer"] = out["types"][main_kind]["top1_by_layer"]
    out["headline_kind"] = main_kind
    write_json(ROOT / "results" / "raw" / f"E04b_{args.model}.json", out)


if __name__ == "__main__":
    main()
