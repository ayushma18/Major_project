"""Figures for the paper, drawn from results/raw (rates only, no completions).

    .venv/bin/python paper/make_figures.py

Writes paper/figures/*.png (200 dpi) and *.pdf. A figure whose data is missing is skipped.
Palette: the dataviz reference categorical slots 1-3 (validated, light mode); the aqua slot sits
below 3:1 on the surface, so every series carries a direct label or a legend.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, BAND, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["STIX Two Text", "STIXGeneral"], "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def load(name):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_main(m="llama-8b"):
    """Every condition at c = 0.5 against the random band."""
    st = load(f"E07_stats_{m}.json")
    if not st:
        return
    base, mu, sd = st["baseline_asr"], st["random"]["asr_mean"], st["random"]["asr_sd"]
    groups = [("random\ndirections", [(k, v) for k, v in st["random"]["per_seed"].items()]),
              ("paired emotion\n(proposal)", [(e, r["asr"]) for e, r in st["kinds"].get("emotion", {}).items()]),
              ("paired, refusal\npart removed", [(e, r["asr"]) for e, r in st["kinds"].get("perp", {}).items()]),
              ("specific part", [(e, r["asr"]) for e, r in st["kinds"].get("specific", {}).items()]),
              ("paired,\nsubtracted", [(e, r["asr"]) for e, r in st["kinds"].get("negative", {}).items()])]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.axhspan(100 * (mu - 2 * sd), 100 * (mu + 2 * sd), color=BAND, alpha=0.55, lw=0, zorder=0)
    ax.axhline(100 * mu, color=MUTED, lw=1, zorder=1)
    ax.axhline(100 * base, color=INK, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.45, 100 * (mu + 2 * sd) - 0.8, f"grey band: random mean {100 * mu:.1f}% ± 2 sd", ha="left", va="top", fontsize=7.5, color=MUTED)
    ax.text(-0.45, 100 * base - 0.8, f"no injection {100 * base:.1f}%", ha="left", va="top", fontsize=7.5, color=INK)
    hi = mu + 2 * sd
    for i, (label, pts) in enumerate(groups):
        pts = sorted(pts, key=lambda x: x[1])
        n = len(pts)
        labels = []
        for j, (name, y) in enumerate(pts):
            x = i + (j - (n - 1) / 2) * (0.5 / max(n - 1, 1))
            above = i > 0 and y > hi
            ax.scatter(x, 100 * y, s=22, color=ORANGE if above else (MUTED if i == 0 else BLUE), edgecolor=SURFACE, linewidth=1.2, zorder=3)
            if above:
                labels.append([name, x, 100 * y, 100 * y])
        # spread labels so none overlap (at least 2.4 points apart), keeping their order
        for k in range(1, len(labels)):
            labels[k][3] = max(labels[k][3], labels[k - 1][3] + 2.4)
        for name, x, y, ty in labels:
            ax.annotate(name, (x, y), xytext=(i + 0.3, ty), textcoords="data", fontsize=7, color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5) if abs(ty - y) > 0.5 else None)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.5, 4.75)
    ax.set_ylabel("judged harmful (%)")
    ax.set_title("Llama 3.1 8B, layer 12, c = 0.5: each dot is one vector, 159 HarmBench behaviours", fontsize=9, loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    abl = st.get("other", {}).get("ablate-refusal", {}).get("asr")
    if abl:
        ax.text(-0.45, 58, f"refusal direction ablated: {100 * abl:.1f}% (off the scale)", fontsize=7.5, color=MUTED, va="top")
    ax.set_ylim(0, 60)
    save(fig, "fig1_main")


def fig_cosine(m="llama-8b", L=12, c="0.5"):
    rows = []
    for kind, title in (("emotion", "paired vectors"), ("specific", "specific parts")):
        d = load(f"E11_{m}_{kind}_L{L}_c{c}.json")
        if d:
            rows.append((title, d))
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(6.8, 2.9), sharey=True)
    axes = axes if len(rows) > 1 else [axes]
    for ax, (title, d) in zip(axes, rows):
        xs = [r["cos"] for r in d["rows"]]
        ys = [100 * r["delta_asr"] for r in d["rows"]]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.scatter(xs, ys, s=22, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        top = sorted(sorted(d["rows"], key=lambda r: -abs(r["delta_asr"]))[:4], key=lambda r: r["cos"])
        for k, r in enumerate(top):  # alternate above and below so neighbours do not collide
            ax.annotate(r["emotion"], (r["cos"], 100 * r["delta_asr"]), xytext=(3, 5 if k % 2 == 0 else -10), textcoords="offset points", fontsize=7, color=INK)
        s = d["delta_asr"]
        ax.set_title(f"{title}\nSpearman rho = {s['spearman_rho']:+.2f}, permutation p = {s['perm_p']:.2f}", fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("cos(vector, refusal direction) at layer 12")
    axes[0].set_ylabel("change in judged harmful (points)")
    save(fig, "fig2_cosine")


def fig_sleeper(m="llama-8b", drift_layer="14"):
    """Top: turn-4 judged-harmful rate by injection timing. Bottom: paired drift at benign turns 2 and 3."""
    single, J = load(f"E12_{m}_single.json"), (load(f"judged_{m}_multiturn-single.json") or {}).get("conditions", {})
    J1 = (load(f"judged_{m}_harmbench-test.json") or {}).get("conditions", {})
    drift = load(f"E12_{m}_single_drift.json") or {}
    if not single or "none" not in J:
        return
    vecs = {}
    for stem, cnd in single["conditions"].items():
        if cnd.get("vector") and cnd.get("mode"):
            vecs.setdefault(cnd["vector"], {})[cnd["mode"]] = stem
    names = {"random:2@12:1": "random direction\n(seed 2, c = 1)", "specific:happy@12:0.5": "happy,\nspecific part", "emotion:loving@12:-0.5": "minus loving\n(paired)",
             "emotion:happy@12:0.5": "happy\n(paired)"}
    vs = [v for v in names if v in vecs and any(J.get(vecs[v].get(k, "")) for k in ("early", "same", "early-full", "same-full"))]
    if not vs:
        return
    has_drift = bool(drift.get("vectors")) and all(v in drift["vectors"] for v in vs)
    fig, axes = plt.subplots(2 if has_drift else 1, 1, figsize=(6.8, 5.8 if has_drift else 3.4), gridspec_kw={"height_ratios": [1.3, 1]} if has_drift else None)
    ax = axes[0] if has_drift else axes
    # hue = when the vector is injected; solid = every position of that turn, hatched = text only
    MODES = [("early", BLUE, True, "turn 1, text only"), ("early-full", BLUE, False, "turn 1, every position"),
             ("same", ORANGE, True, "harmful turn, text only"), ("same-full", ORANGE, False, "harmful turn, every position")]
    w = 0.18
    for i, v in enumerate(vs):
        single_turn = J1.get(v, {}).get("asr")
        for k, (mode, col, hatch, lab) in enumerate(MODES):
            jm = J.get(vecs[v].get(mode, ""))
            if not jm:
                continue
            y, x = 100 * jm["asr"], i + (k - 1.5) * (w + 0.02)
            ax.bar(x, y, width=w, color=SURFACE if hatch else col, edgecolor=col, hatch="////" if hatch else None, linewidth=1.0,
                   label=lab if i == 0 else None, zorder=3)
            ax.text(x, y + 0.8, f"{y:.0f}", ha="center", fontsize=6.5, color=INK)
        if single_turn is not None:
            ax.scatter(i + 0.5, 100 * single_turn, marker="D", s=24, color=AQUA, edgecolor=SURFACE, linewidth=1, zorder=4, label="single turn, no conversation (Fig. 1)" if i == 0 else None)
    ax.axhline(100 * J["none"]["asr"], color=INK, lw=1, ls=(0, (4, 3)), label=f"no injection ({100 * J['none']['asr']:.1f}%)")
    ax.set_xticks(range(len(vs)), [names[v] for v in vs])
    ax.set_ylabel("judged harmful at turn 4 (%)")
    ax.set_title(f"(a) Three benign turns, then one harmful request ({single['n_harmful']} conversations)", fontsize=9, loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 95)
    ax.legend(fontsize=7, loc="upper center", ncols=3, handlelength=1.4, columnspacing=1.0)
    if has_drift:
        bx = axes[1]
        gap = drift["harmful_gap"][drift_layer]
        bx.axhline(0, color=MUTED, lw=0.8)
        # blue = turn-1 injection as in panel (a); hollow = text only, solid = every turn-1 position
        series = [("", True, "o", 2, "text only, turn 2"), ("", True, "s", 3, "text only, turn 3"),
                  ("|full", False, "o", 2, "every position, turn 2"), ("|full", False, "s", 3, "every position, turn 3")]
        for i, v in enumerate(vs):
            for k, (suffix, hollow, mk, t, lab) in enumerate(series):
                per = drift["vectors"].get(v + suffix)
                if not per:
                    continue
                r = {x["turn"]: x for x in per[drift_layer]}[t]
                y, e = 100 * r["mean_diff"] / gap, 100 * 1.96 * r["se"] / gap
                x = i + (k - 1.5) * 0.17
                bx.errorbar(x, y, yerr=e, fmt=mk, ms=5, color=BLUE, mfc=SURFACE if hollow else BLUE, mec=BLUE, elinewidth=1, capsize=2, zorder=3,
                            label=lab if i == 0 else None)
        bx.set_xticks(range(len(vs)), [names[v] for v in vs])
        bx.set_ylabel("shift, % of the harmful-turn jump")
        bx.set_title(f"(b) Turn-1 injection minus none: refusal-direction projection at later benign turns (layer {drift_layer}, 95% CI, {drift['n_scripts']} scripts)",
                     fontsize=8.5, loc="left", color=INK)
        bx.grid(axis="x", visible=False)
        bx.set_xlim(ax.get_xlim())
        bx.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncols=4, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(h_pad=1.6)
    save(fig, "fig3_sleeper")


def fig_replication():
    a, b = load("E07_stats_llama-8b.json"), load("E07_stats_qwen2.5-7b.json")
    if not (a and b):
        return
    cats = [("no injection", lambda s: s["baseline_asr"]), ("random", lambda s: s["random"]["asr_mean"])]
    for kind, lab in (("emotion", "paired\nemotion"), ("specific", "specific\npart"), ("negative", "paired,\nsubtracted")):
        cats.append((lab, lambda s, k=kind: sum(x["asr"] for x in s["kinds"][k].values()) / len(s["kinds"][k]) if k in s["kinds"] else None))
    cats.append(("refusal\nablated", lambda s: s.get("other", {}).get("ablate-refusal", {}).get("asr")))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    w = 0.36
    for k, (s, col, lab) in enumerate(((a, BLUE, "Llama 3.1 8B"), (b, ORANGE, "Qwen2.5 7B"))):
        for i, (_, f) in enumerate(cats):
            y = f(s)
            if y is None:
                continue
            ax.bar(i + (k - 0.5) * (w + 0.03), 100 * y, width=w, color=col, label=lab if i == 0 else None, zorder=3)
            ax.text(i + (k - 0.5) * (w + 0.03), 100 * y + 1, f"{100 * y:.0f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cats)), [c[0] for c in cats])
    ax.set_ylabel("mean judged harmful (%)")
    ax.set_title("Both models, c = 0.5 one layer after each model's refusal layer", fontsize=9, loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "fig4_replication")


if __name__ == "__main__":
    fig_main()
    fig_cosine()
    fig_sleeper()
    fig_replication()
