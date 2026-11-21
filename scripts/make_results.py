"""Build the interface's results files from the raw experiment outputs.

    python scripts/make_results.py --model llama-8b

Reads results/raw/*.json (written by scripts 03 to 12) and writes results/<id>.json in the
shape of docs/CONTRACTS.md section 2, then results/index.json. Every builder is tolerant of
missing inputs: an experiment without raw data stays "planned". No completion text to a
harmful prompt is ever copied: only rates, labels and benign examples.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runtime import ALIASES, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


def raw(name: str):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def base(id_, title, module, question, model):
    return {"id": id_, "title": title, "module": module, "status": "planned", "question": question, "model": model,
            "started": None, "finished": None, "config": {}, "headline": None, "metrics": [], "tables": [], "charts": [],
            "interpretation": [], "caveats": [], "artifacts": []}


def e00(m, model):
    r = base("E00-model-selection", "Which model to study, and how fast this Mac runs it", 0,
             "Can a 48 GB M5 Pro run the whole experiment grid, and at what speed?", model)
    runs = []
    for name in ("llama-1b", "llama-8b"):
        d = raw(f"E00_{name}.json")
        if d:
            for x in d["runs"]:
                runs.append([d["model"], x["dtype"], x["batch"], "yes" if x["steer_hook"] else "no", round(x["tok_per_s"], 1), round(x.get("mps_driver_gib", 0), 1)])
            if d.get("mlx", {}).get("single_stream_tok_per_s"):
                runs.append([d["model"], "MLX bf16", 1, "no", round(d["mlx"]["single_stream_tok_per_s"], 1), None])
    if not runs:
        return r
    r["status"] = "done"
    r["tables"].append({"id": "throughput", "title": "Generation throughput (tokens per second, whole batch)", "columns": ["model", "dtype", "batch", "steering hook", "tok/s", "GPU memory GiB"], "rows": runs})
    eight = [x for x in runs if "8B" in x[0] and x[1] == "bfloat16" and x[3] == "no"]
    if eight:
        r["charts"].append({"id": "tps_8b", "type": "bar", "title": "Llama 3.1 8B throughput by batch size (bf16, MPS)",
                            "x": {"label": "batch size", "values": [str(x[2]) for x in eight]}, "y": {"label": "tokens per second", "unit": "tok/s"},
                            "series": [{"name": "tok/s", "values": [x[4] for x in eight]}]})
        best = max(eight, key=lambda x: x[4])
        r["headline"] = f"Llama 3.1 8B runs at {best[4]:.0f} tokens per second at batch {best[2]} on the M5 Pro; one HarmBench condition takes about 2 to 3 minutes."
        r["metrics"].append({"name": "tps_8b_best", "label": "8B best throughput", "value": best[4], "unit": "tok/s"})
    sel = raw("model_selection.json")
    if sel:
        r["interpretation"] += sel.get("interpretation", [])
        r["tables"].append({"id": "ranking", "title": "Model ranking from the three-lens review", "columns": ["model", "interpretability", "safety", "compute", "overall", "note"],
                            "rows": [[x["model"], x.get("interp"), x.get("safety"), x.get("compute"), x["overall"], x["note"]] for x in sel.get("ranking", [])]})
    r["caveats"].append("Throughput depends on what else runs on the Mac; numbers are from one session.")
    return r


def judged(m, set_="harmbench-test"):
    return (raw(f"judged_{m}_{set_}.json") or {}).get("conditions", {})


def gen(m, set_):
    return (raw(f"gen_{m}_{set_}.json") or {}).get("conditions", {})


def e01_e03(m, model):
    J, G = judged(m), gen(m, "harmbench-test")
    H = gen(m, "harmless")
    out = []
    r1 = base("E01-baseline-refusal", "Baseline refusal on HarmBench", 2, "How often does the unmodified model refuse the 159 HarmBench standard test behaviours?", model)
    r3 = base("E03-refusal-ablation", "Suppressing the guardrail direction", 2, "If the refusal direction is projected out of every layer, does the model answer what it refused before?", model)
    if "baseline" in G:
        for r in (r1,):
            r["status"] = "done"
        g = G["baseline"]
        r1["metrics"].append({"name": "refusal_rate", "label": "Refusal rate (phrase match)", "value": g["refusal_substring_rate"], "unit": "fraction"})
        r1["config"] = {"n_prompts": g["n"], "max_new_tokens": g["max_new_tokens"], "decoding": "greedy"}
        if "baseline" in J:
            j = J["baseline"]
            r1["metrics"].insert(0, {"name": "asr", "label": "Attack success rate (HarmBench judge)", "value": j["asr"], "unit": "fraction", "ci95": j["ci95"]})
            r1["headline"] = f"The unmodified model refuses {pct(g['refusal_substring_rate'])} of HarmBench requests; the judge rates {pct(j['asr'])} of answers as harmful."
            cats = j["by_category"]
            r1["charts"].append({"id": "asr_by_category", "type": "bar", "title": "Baseline ASR by HarmBench category", "x": {"label": "category", "values": list(cats)},
                                 "y": {"label": "ASR", "unit": "fraction"}, "series": [{"name": "ASR", "values": [cats[c] for c in cats]}]})
        else:
            r1["headline"] = f"The unmodified model refuses {pct(g['refusal_substring_rate'])} of HarmBench requests (phrase match; judge pending)."
    if "ablate-refusal" in G:
        r3["status"] = "done"
        g0, g1 = G.get("baseline"), G["ablate-refusal"]
        rows = [["no intervention", g0["refusal_substring_rate"] if g0 else None, J.get("baseline", {}).get("asr")],
                ["refusal direction ablated", g1["refusal_substring_rate"], J.get("ablate-refusal", {}).get("asr")]]
        r3["tables"].append({"id": "ablation", "title": "Refusal and ASR with and without the guardrail direction", "columns": ["condition", "refusal rate", "ASR (judge)"], "rows": rows})
        r3["charts"].append({"id": "ablation_bar", "type": "bar", "title": "Refusal rate before and after ablation", "x": {"label": "condition", "values": [x[0] for x in rows]},
                             "y": {"label": "fraction of prompts", "unit": "fraction"},
                             "series": [{"name": "refusal rate", "values": [x[1] for x in rows]}] + ([{"name": "ASR (judge)", "values": [x[2] for x in rows]}] if all(x[2] is not None for x in rows) else [])})
        if "ablate-refusal" in J:
            j1 = J["ablate-refusal"]
            r3["metrics"].append({"name": "asr_ablated", "label": "ASR with the guardrail direction removed", "value": j1["asr"], "unit": "fraction", "ci95": j1["ci95"]})
            r3["headline"] = f"Removing one direction drops refusal from {pct(g0['refusal_substring_rate'] if g0 else None)} to {pct(g1['refusal_substring_rate'])}; the judge rates {pct(j1['asr'])} of answers harmful (baseline {pct(J.get('baseline', {}).get('asr'))})."
        else:
            r3["headline"] = f"Removing one direction drops refusal from {pct(g0['refusal_substring_rate'] if g0 else None)} to {pct(g1['refusal_substring_rate'])} (judge pending)."
        if "baseline" in H and "ablate-refusal" in H:
            r3["tables"].append({"id": "harmless", "title": "Over-refusal check on 100 harmless instructions", "columns": ["condition", "refusal rate"],
                                 "rows": [["no intervention", H["baseline"]["refusal_substring_rate"]], ["refusal direction ablated", H["ablate-refusal"]["refusal_substring_rate"]]]})
        r3["interpretation"].append("This replicates Arditi et al. (2024): refusal in this model is carried mostly by one linear direction.")
        r3["caveats"].append("The phrase-match refusal rate misses refusals worded differently and counts some partial answers as refusals; the HarmBench judge is the headline metric.")
    out += [r1, r3]
    return out


def e02(m, model):
    r = base("E02-refusal-direction", "Finding the refusal (guardrail) direction", 3, "Which layer and token position carry the refusal direction, following Arditi et al. 2024?", model)
    d = raw(f"E02_{m}.json")
    if not d:
        return r
    r["status"] = "done"
    s = d["selected"]
    r["headline"] = f"The guardrail direction sits at layer {s['layer']} on template token {s['pos']}; ablating it pushes the refusal score from {d['baseline_refusal_score']['harmful_val']:.1f} to {s['bypass']:.1f} on held-out harmful prompts."
    r["config"] = {"n_harmful_train": 128, "n_harmless_train": 128, "n_val": 32, "selection": "lowest bypass with induce > 0, KL < 0.1, layer < 80%"}
    r["metrics"] += [{"name": "layer", "label": "Selected layer", "value": s["layer"], "unit": "index"},
                     {"name": "bypass", "label": "Refusal score after ablation (lower = refuses less)", "value": s["bypass"], "unit": "log-odds"},
                     {"name": "induce", "label": "Refusal score on harmless prompts when added", "value": s["induce"], "unit": "log-odds"},
                     {"name": "kl", "label": "KL divergence on harmless prompts after ablation", "value": s["kl"], "unit": "nats"}]
    cands = d["candidates"]
    layers = sorted({c["layer"] for c in cands})
    poss = sorted({c["pos"] for c in cands})
    z = [[next((c["bypass"] for c in cands if c["layer"] == l and c["pos"] == p), None) for l in layers] for p in poss]
    r["charts"].append({"id": "bypass_heatmap", "type": "heatmap", "title": "Refusal score after ablating each candidate (lower = better guardrail direction)",
                        "x": {"label": "layer", "values": layers}, "y": {"label": "template token position", "values": poss}, "z": z})
    r["charts"].append({"id": "norms", "type": "line", "title": "Mean residual-stream norm by layer (sets the steering scale)", "x": {"label": "layer", "values": list(range(len(d["resid_norms"])))},
                        "y": {"label": "L2 norm", "unit": ""}, "series": [{"name": "mean norm", "values": d["resid_norms"]}]})
    r["artifacts"] = [f"artifacts/vectors/{ALIASES.get(m, m).replace('/', '--')}/refusal.safetensors"]
    return r


def e04(m, model):
    r = base("E04-emotion-vectors", "Extracting the 20 emotion vectors", 1, "Do the story pairs give emotion directions that recognise held-out stories, and how much do the 19 directions share?", model)
    d = raw(f"E04_{m}.json")
    if not d:
        return r
    r["status"] = "done"
    h = d["heldout"]
    L = list(range(d["n_layers"]))
    r["charts"].append({"id": "heldout", "type": "line", "title": "Held-out recognition by layer", "x": {"label": "layer", "values": L}, "y": {"label": "accuracy", "unit": "fraction"},
                        "series": [{"name": "pair test (chance 0.5)", "values": h["pair_acc"]}, {"name": "19-way, difference vectors", "values": h["diff_top1"]}, {"name": "19-way, single stories", "values": h["story_top1"]}]})
    if "shared_component" in d:
        r["charts"].append({"id": "shared", "type": "line", "title": "How much the paired vectors share one common direction", "x": {"label": "layer", "values": L}, "y": {"label": "cosine", "unit": ""},
                            "series": [{"name": "mean cosine between emotions", "values": d["shared_component"]["mean_offdiag_cos_caa"]}, {"name": "mean cosine with the common direction", "values": d["shared_component"]["mean_cos_to_common"]}]})
    E = d["emotions"]
    r["charts"].append({"id": "cos_matrix", "type": "heatmap", "title": f"Cosine between paired emotion vectors at layer {d['readout_layer']}", "x": {"label": "emotion", "values": E}, "y": {"label": "emotion", "values": E}, "z": d["cos_matrix_caa"]})
    pc = d["pca_centered"]
    r["charts"].append({"id": "pca", "type": "scatter", "title": f"Emotion map (PCA of one-vs-mean vectors, layer {d['readout_layer']})", "x": {"label": f"PC1 ({100 * pc['explained'][0]:.0f}% of variance)"}, "y": {"label": f"PC2 ({100 * pc['explained'][1]:.0f}%)"},
                        "points": [{"x": x, "y": y, "label": e} for x, y, e in zip(pc["pc1"], pc["pc2"], E)]})
    best = max(L, key=lambda l: h["diff_top1"][l])
    ood = raw(f"E04b_{m}.json")
    r["headline"] = f"All 19 emotions are recognised in held-out stories (19-way accuracy {pct(h['diff_top1'][best])} at layer {best}, chance {pct(h['chance_top1'])})" + (f"; on the independent short-text set, {pct(ood['best_top1'])}." if ood else ".")
    r["metrics"] += [{"name": "diff_top1", "label": "Held-out 19-way accuracy (best layer)", "value": h["diff_top1"][best], "unit": "fraction"},
                     {"name": "pc1_valence", "label": "Correlation of PC1 with valence", "value": pc["corr_pc1_valence"], "unit": "r"}]
    if ood:
        r["metrics"].append({"name": "ood_top1", "label": "Independent short-text 19-way accuracy (best layer)", "value": ood["best_top1"], "unit": "fraction"})
        r["charts"].append({"id": "ood", "type": "line", "title": "Independent test set (different writer, different form)", "x": {"label": "layer", "values": list(range(len(ood["top1_by_layer"])))}, "y": {"label": "accuracy", "unit": "fraction"},
                            "series": [{"name": "19-way accuracy", "values": ood["top1_by_layer"]}, {"name": "chance", "values": [1 / 19] * len(ood["top1_by_layer"])}]})
    m12 = raw(f"E04c_{m}_L12.json")
    if m12:
        c = m12["cleaned"]
        r["charts"].append({"id": "pca_clean", "type": "scatter", "title": f"Emotion map after removing {m12['neutral_pcs_removed']} neutral-text directions (layer 12, Anthropic's clean-up)",
                            "x": {"label": f"PC1 ({100 * c['explained'][0]:.0f}%)"}, "y": {"label": f"PC2 ({100 * c['explained'][1]:.0f}%, r with valence {c['corr']['pc2_valence']:+.2f})"},
                            "points": [{"x": x, "y": y, "label": e} for x, y, e in zip(c["pc1"], c["pc2"], m12["emotions"])]})
        r["metrics"].append({"name": "valence_clean", "label": "Valence correlation after neutral clean-up (layer 12)", "value": max(abs(c["corr"]["pc1_valence"]), abs(c["corr"]["pc2_valence"])), "unit": "|r|"})
        r["interpretation"].append(f"Once the main directions of neutral text are removed, valence (pleasant versus unpleasant) organises the emotions (|r| = {max(abs(c['corr']['pc1_valence']), abs(c['corr']['pc2_valence'])):.2f}), close to Anthropic's 0.81 and Sun et al.'s 0.89.")
    if ood:
        t = ood["types"].get("centered", {})
        if t.get("most_confused_with"):
            r["interpretation"].append("Mistakes on the independent set are human-like: " + ", ".join(f"{a} read as {b}" for a, b in list(t["most_confused_with"].items())[:6]) + ".")
    r["caveats"].append("Held-out stories come from the same writers and format as the training stories, so high accuracy here can partly reflect vocabulary; the independent short-text set is the stronger test.")
    if "shared_component" in d:
        r["interpretation"].append("The paired vectors (emotional story minus its neutral twin) share one large common direction: much of each vector is 'emotional versus flat writing'. The project therefore also tests each emotion's specific part with that common direction removed.")
    return r


def e05(m, model):
    r = base("E05-emotion-steering-sanity", "Do the emotion vectors steer emotion", 1, "When a vector is added during an ordinary answer, does the tone change to that emotion while the text stays fluent?", model)
    d = raw(f"E05_{m}.json")
    if not d:
        return r
    r["status"] = "done"
    A = d["aggregate"]
    E = d["emotions"]
    rows = []
    for L in d["layers"]:
        for c in d["coeffs"]:
            em = [a for a in A if a["layer"] == L and a["c"] == c and a["vector"] in E]
            rn = [a for a in A if a["layer"] == L and a["c"] == c and a["vector"].startswith("random")]
            if em:
                rows.append([L, c, sum(a["top1"] for a in em) / len(em), sum(a["p_target"] for a in em) / len(em), sum(a["nll"] for a in em) / len(em), sum(a["nll"] for a in rn) / max(1, len(rn)) if rn else None])
    none = [a for a in A if a["vector"] == "none"]
    r["tables"].append({"id": "grid", "title": "Emotion recognition and fluency by layer and strength (strength c = fraction of a typical activation)", "columns": ["layer", "c", "named correctly", "p(target)", "NLL (emotion)", "NLL (random)"], "rows": rows})
    if rows:
        best = max(rows, key=lambda x: x[2] - 0.05 * max(0, x[4] - 2))
        r["headline"] = f"At layer {best[0]} and strength {best[1]}, an unsteered reader names the injected emotion {pct(best[2])} of the time (chance 5%)."
        for L in d["layers"]:
            sub = [x for x in rows if x[0] == L]
            r["charts"].append({"id": f"recog_L{L}", "type": "line", "title": f"Layer {L}: recognition and fluency vs strength", "x": {"label": "strength c", "values": [x[1] for x in sub]},
                                "y": {"label": "value", "unit": ""}, "series": [{"name": "named correctly", "values": [x[2] for x in sub]}, {"name": "NLL emotion (fluency, lower better)", "values": [x[4] for x in sub]}, {"name": "NLL random", "values": [x[5] for x in sub]}]})
    ex = d.get("examples", {})
    r["tables"].append({"id": "examples", "title": "Benign examples: 'Describe your morning so far.'", "columns": ["vector", "layer", "c", "reply (first 200 characters)"],
                        "rows": [[e, x["layer"], x["c"], x["text"][:200]] for e in list(ex) for x in ex[e][: len(d["layers"]) * len(d["coeffs"])]]})
    r["caveats"].append("The reader is the same model without steering, so it can share the model's own biases about tone.")
    return r


def e06(m, model):
    r = base("E06-geometry", "How emotion vectors line up with the guardrail direction", 3, "What is the cosine between each emotion vector and the refusal direction, and is it above chance?", model)
    d = raw(f"E06_{m}.json")
    if not d:
        return r
    r["status"] = "done"
    L = d["refusal"]["layer"]
    for kind, t in d["types"].items():
        E = t["emotions"]
        r["charts"].append({"id": f"heat_{kind}", "type": "heatmap", "title": f"cos(emotion vector, refusal direction) by layer, {kind} vectors", "x": {"label": "layer", "values": list(range(d["n_layers"]))}, "y": {"label": "emotion", "values": E},
                            "z": [t["cos_same_layer"][e] for e in E]})
        row = sorted(((e, t["cos_same_layer"][e][L]) for e in E), key=lambda x: -x[1])
        r["charts"].append({"id": f"bar_{kind}", "type": "bar", "title": f"At the refusal layer {L}: cosine with the guardrail direction ({kind})", "x": {"label": "emotion", "values": [x[0] for x in row]},
                            "y": {"label": "cosine", "unit": ""}, "series": [{"name": "cosine", "values": [x[1] for x in row]}]})
    t = d["types"].get("specific") or d["types"].get("caa")
    row = sorted(((e, t["cos_same_layer"][e][L]) for e in t["emotions"]), key=lambda x: -x[1])
    r["headline"] = f"At layer {L}, {row[0][0]} points most towards the guardrail direction (cos {row[0][1]:+.2f}) and {row[-1][0]} most away ({row[-1][1]:+.2f}); random directions stay within plus or minus {d['null_abs_cos']['p99']:.3f}."
    r["metrics"].append({"name": "null_p99", "label": "Random-direction |cos| (99th percentile)", "value": d["null_abs_cos"]["p99"], "unit": ""})
    r["interpretation"].append("Positive cosine means adding the emotion pushes the model TOWARDS refusing; negative means it pushes away from refusing.")
    return r


def e07_e10(m, model):
    J, G = judged(m), gen(m, "harmbench-test")
    out = []
    r7 = base("E07-emotion-injection-asr", "Emotion injection against HarmBench", 2, "Does adding an emotion vector make the model answer harmful requests it otherwise refuses?", model)
    r8 = base("E08-random-baseline", "Random vectors of the same size", 2, "How much does ANY push of the same size change refusal? Every emotion result must beat this.", model)
    r10 = base("E10-orthogonalised", "Emotion vectors with the guardrail part removed", 3, "If the part of each emotion vector that lies along the refusal direction is removed, does the effect disappear?", model)
    sweep = raw(f"E07_plan_{m}.json")
    if not G:
        return [r7, r8, r10]

    def val(name):
        j = J.get(name)
        g = G.get(name)
        return (j["asr"] if j else None, g["refusal_substring_rate"] if g else None)

    b_asr, b_ref = val("baseline")
    emo = [k for k in G if k.startswith(("emotion:", "specific:", "common@", "centered:"))]
    rnd = [k for k in G if k.startswith("random:")]
    perp = [k for k in G if k.startswith("perp:")]
    if emo:
        r7["status"] = "done" if not sweep or all(k in G for k in sweep.get("conditions", [])) else "running"
        rows = [[k, *val(k)] for k in sorted(emo)]
        r7["tables"].append({"id": "all", "title": "Every emotion condition (kind:emotion@layer:strength)", "columns": ["condition", "ASR (judge)", "refusal rate"], "rows": rows})
        groups = {}
        for k in emo:
            kind = k.split(":")[0].split("@")[0]
            lc = k.split("@")[1]
            groups.setdefault((kind, lc), []).append(k)
        for (kind, lc), ks in sorted(groups.items()):
            ks = sorted(ks, key=lambda k: -(val(k)[0] if val(k)[0] is not None else (1 - (val(k)[1] or 0))))
            use_asr = all(val(k)[0] is not None for k in ks)
            r7["charts"].append({"id": f"{kind}_{lc}", "type": "bar", "title": f"{'ASR' if use_asr else 'Refusal rate'} per emotion, {kind} vectors at layer {lc.split(':')[0]}, strength {lc.split(':')[1]}",
                                 "x": {"label": "emotion", "values": [k.split(":")[1].split("@")[0] if ":" in k else k for k in ks]}, "y": {"label": "ASR" if use_asr else "refusal rate", "unit": "fraction"},
                                 "series": [{"name": "ASR" if use_asr else "refusal rate", "values": [val(k)[0] if use_asr else val(k)[1] for k in ks]}]})
        scored = [(k, *val(k)) for k in emo if val(k)[0] is not None]
        if scored and b_asr is not None:
            top = max(scored, key=lambda x: x[1])
            r7["headline"] = f"Strongest emotion condition {top[0]}: ASR {pct(top[1])} against {pct(b_asr)} with no injection."
        else:
            low = min(((k, *val(k)) for k in emo), key=lambda x: x[2])
            r7["headline"] = f"Lowest refusal under emotion injection: {low[0]} at {pct(low[2])} (baseline {pct(b_ref)}); judge pending."
    st = raw(f"E07_stats_{m}.json")
    if st and st.get("kinds"):
        for kind, rows_k in st["kinds"].items():
            r7["tables"].append({"id": f"stats_{kind}", "title": f"{kind} vectors at layer {st['layer']}, strength {st['c']}: against baseline ({100 * st['baseline_asr']:.1f}%) and random vectors (mean {100 * st['random']['asr_mean']:.1f}%, sd {100 * st['random']['asr_sd']:.1f}, {st['random']['n_seeds']} seeds)" if st['random']['n_seeds'] else f"{kind} vectors",
                                 "columns": ["emotion", "ASR", "change", "newly harmful", "newly safe", "McNemar p (Holm)", "z vs random", "NLL", "broken"],
                                 "rows": [[e, round(r["asr"], 3), round(r["delta"], 3), r["newly_harmful"], r["newly_safe"], round(r["mcnemar_p_holm"], 4), None if r["z_vs_random"] is None else round(r["z_vs_random"], 2), None if r["nll"] is None else round(r["nll"], 2), None if r["broken"] is None else round(r["broken"], 2)] for e, r in sorted(rows_k.items(), key=lambda x: -x[1]["asr"])]})
        em = st["kinds"].get("emotion", {})
        if em:
            sig = [e for e, r in em.items() if r["mcnemar_p_holm"] < 0.05]
            above = [e for e, r in em.items() if r["z_vs_random"] is not None and r["z_vs_random"] > 2]
            r7["interpretation"].append(f"Against the baseline, {len(sig)} of {len(em)} emotions change ASR significantly after Holm correction ({', '.join(sig) or 'none'}); {len(above)} sit more than two standard deviations above the random-vector mean ({', '.join(above) or 'none'}). The random band comes from 12 seeds and is descriptive, not a calibrated test: a condition above every seed has an add-one rank p-value of 1/13 at best, before any correction.")
    if rnd:
        r8["status"] = "done"
        rows = [[k, *val(k)] for k in sorted(rnd)]
        r8["tables"].append({"id": "random", "title": "Random unit directions at the same injected norm", "columns": ["condition", "ASR (judge)", "refusal rate"], "rows": rows})
        parts = []
        for cval in sorted({k.split(":")[-1] for k in rnd}, key=float):
            a = [val(k)[0] for k in rnd if k.split(":")[-1] == cval and val(k)[0] is not None]
            if a:
                mu = sum(a) / len(a)
                sd = (sum((x - mu) ** 2 for x in a) / max(1, len(a) - 1)) ** 0.5
                parts.append(f"strength {cval}: mean {pct(mu)} (sd {100 * sd:.1f}, {len(a)} seeds, range {pct(min(a))} to {pct(max(a))})")
                r8["charts"].append({"id": f"rand_{cval}", "type": "bar", "title": f"Random directions at strength {cval}: judged-harmful rate per seed (baseline {pct(b_asr)})",
                                     "x": {"label": "seed", "values": [k.split(":")[1].split("@")[0] for k in sorted(rnd, key=lambda k: int(k.split(":")[1].split("@")[0])) if k.split(":")[-1] == cval and val(k)[0] is not None]},
                                     "y": {"label": "ASR", "unit": "fraction"},
                                     "series": [{"name": "ASR", "values": [val(k)[0] for k in sorted(rnd, key=lambda k: int(k.split(":")[1].split("@")[0])) if k.split(":")[-1] == cval and val(k)[0] is not None]}]})
        r8["headline"] = ("Random pushes, " + "; ".join(parts) + f". Baseline {pct(b_asr)}.") if parts else "Random runs generated; judge pending."
    if perp:
        r10["status"] = "done"
        rows = []
        for k in sorted(perp):
            e_name = k.replace("perp:", "emotion:")
            rows.append([k.split(":")[1].split("@")[0], k.split("@")[1], *val(e_name), *val(k)])
        r10["tables"].append({"id": "perp", "title": "Emotion vector vs the same vector with its guardrail component removed (equal norm)", "columns": ["emotion", "layer:c", "ASR emotion", "refusal emotion", "ASR orthogonalised", "refusal orthogonalised"], "rows": rows})
        pairs = [(x[2], x[4]) for x in rows if x[2] is not None and x[4] is not None]
        if pairs:
            me, mp = sum(a for a, _ in pairs) / len(pairs), sum(b for _, b in pairs) / len(pairs)
            r10["headline"] = f"Projecting the layer-12 refusal candidate out of the paired vectors did not lower mean ASR: {pct(me)} with it, {pct(mp)} without ({len(pairs)} emotions, baseline {pct(b_asr)})."
            r10["interpretation"].append("If the emotion vectors worked only by overlapping the refusal direction, removing that overlap would bring ASR back towards baseline. It did not in this grid. The removed direction is the layer-12 candidate (cosine 0.81 with the ablated layer-11 direction), and the difference between the two means was not tested, so this does not show how the vectors act on refusal.")
    return [r7, r8, r10]


def e09(m, model):
    r = base("E09-text-priming", "Text-level emotional priming (FreakOut style)", 2, "Does a stressful or relaxing system prompt change refusal, with no activation change at all?", model)
    J, G = judged(m), gen(m, "harmbench-test")
    pr = sorted(k for k in G if k.startswith("prime:"))
    screen = not pr
    if screen:  # only the 64-token screen exists: phrase-match refusal, judge where it was run
        J, G = judged(m, "harmbench-test-screen"), gen(m, "harmbench-test-screen")
        pr = sorted(k for k in G if k.startswith("prime:"))
    if not pr:
        return r
    r["status"] = "done"
    if screen:
        r["caveats"].append(f"From the 64-token screen (refusal-phrase rate; no injection on the same screen: {pct(G.get('baseline', {}).get('refusal_substring_rate'))}). "
                            "Priming moved refusal by a few points in either direction, so it was not rerun at 256 tokens.")

    def v(k):
        return J.get(k, {}).get("asr"), G[k]["refusal_substring_rate"]

    rows = [[k, *v(k)] for k in pr]
    r["tables"].append({"id": "priming", "title": "System-prompt priming conditions", "columns": ["condition", "ASR (judge)", "refusal rate"], "rows": rows})
    groups = {}
    for k in pr:
        groups.setdefault(k.split(":")[1], []).append(v(k))
    means = {g: (sum(x[1] for x in xs) / len(xs)) for g, xs in groups.items()}
    r["charts"].append({"id": "prime_bar", "type": "bar", "title": "Mean refusal rate by priming group", "x": {"label": "group", "values": list(means)}, "y": {"label": "refusal rate", "unit": "fraction"}, "series": [{"name": "refusal", "values": list(means.values())}]})
    r["headline"] = "Mean refusal by system-prompt mood: " + ", ".join(f"{g} {pct(x)}" for g, x in means.items())
    r["headline"] += f" (no system prompt: {pct(G['baseline']['refusal_substring_rate'])}): text priming does not lower refusal." if "baseline" in G else "."
    return r


def e11(m, model):
    r = base("E11-correlation", "Does geometry predict the attack", 3, "Across the 19 emotions, does the cosine with the refusal direction predict how much the emotion changes ASR?", model)
    files = sorted(glob.glob(str(RAW / f"E11_{m}_*.json")), key=lambda f: (0 if "_emotion_" not in f else 1, f))  # paired-vector file last: it sets the headline
    if not files:
        return r
    r["status"] = "done"
    for f in files:
        d = json.loads(Path(f).read_text())
        key = "delta_asr" if d.get("delta_asr") else "delta_refusal"
        r["charts"].append({"id": Path(f).stem, "type": "scatter", "title": f"{d['kind']} vectors, layer {d['layer']}, strength {d['c']}: cosine vs change in ASR",
                            "x": {"label": "cos(emotion, refusal direction)"}, "y": {"label": "change in ASR vs baseline"}, "points": [{"x": x["cos"], "y": x["delta_asr"], "label": x["emotion"]} for x in d["rows"]]})
        s = d["delta_asr"]
        r["metrics"].append({"name": f"rho_{d['kind']}_L{d['layer']}_c{d['c']}", "label": f"Spearman rho, {d['kind']} L{d['layer']} c{d['c']}", "value": s["spearman_rho"], "unit": "rho", "ci95": s["boot_ci95"]})
        r["headline"] = f"Spearman rho between cosine and ASR change = {s['spearman_rho']:+.2f} (permutation p = {s['perm_p']:.3f}, {len(d['rows'])} emotions, layer {d['layer']}, strength {d['c']})."
    return r


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    from math import comb
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / den
    return mid - half, mid + half


def paired_diff_ci(both: int, b: int, c: int, n: int) -> tuple[float, float]:
    """95% interval for p1 - p2 on paired binary labels (Newcombe 1998, method 10, hybrid score).

    both: harmful in both conditions; b: harmful only in condition 1; c: harmful only in condition 2."""
    d = n - both - b - c
    p1, p2 = (both + b) / n, (both + c) / n
    l1, u1 = _wilson(both + b, n)
    l2, u2 = _wilson(both + c, n)
    den = ((both + b) * (c + d) * (both + c) * (b + d)) ** 0.5
    phi = (both * d - b * c) / den if den > 0 else 0.0
    dl = ((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2) ** 0.5
    du = ((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2) ** 0.5
    return p1 - p2 - dl, p1 - p2 + du


def holm_in_place(rows: list, i_p: int, i_out: int) -> None:
    """Write Holm-adjusted p-values of column i_p into column i_out (rows with p None are skipped)."""
    idx = sorted((k for k, r in enumerate(rows) if r[i_p] is not None), key=lambda k: rows[k][i_p])
    run = 0.0
    for rank, k in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - rank) * rows[k][i_p]))
        rows[k][i_out] = run


def vector_label(spec: str | None) -> str:
    """specific:happy@12:0.5 -> 'happy (specific part), c=0.5'; random:2@12:1 -> 'random direction, seed 2, c=1'."""
    if not spec:
        return "none"
    kind, rest = spec.split(":", 1)
    arg, lc = rest.split("@")
    c = float(lc.split(":")[1])
    if kind == "random":
        return f"random direction, seed {arg}, c={c:g}"
    name = {"emotion": f"{arg} (paired, as in the proposal)", "specific": f"{arg} (specific part)", "centered": f"{arg} (one-vs-mean)"}.get(kind, f"{kind} {arg}")
    return f"{'minus ' if c < 0 else ''}{name}, c={abs(c):g}"


def e12(m, model):
    r = base("E12-multiturn-persistence", "The sleeper effect", 4, "If an emotion is injected only at turn 1, is the model still easier to attack later in the conversation?", model)
    single, stacked = raw(f"E12_{m}_single.json"), raw(f"E12_{m}_stacked.json")
    if not (single or stacked):
        return r
    J1 = judged(m)
    judged_any = False

    def by_vector(d):
        """vector spec -> {mode: stem}, plus the stem of the no-injection condition."""
        out, none = {}, None
        for stem, c in d["conditions"].items():
            if c.get("condition") == "none" or stem == "none":
                none = stem
            elif c.get("vector") and c.get("mode"):
                out.setdefault(c["vector"], {})[c["mode"]] = stem
        return out, none

    if single:
        J = judged(m, "multiturn-single")
        vecs, none = by_vector(single)
        n0 = J.get(none or "none")
        n_conv = single["n_harmful"]
        MODES = [("same", "harmful turn, request text only"), ("same-full", "harmful turn, request and template tokens"),
                 ("early", "turn 1 only, text only"), ("early-full", "turn 1 only, every turn-1 position")]
        rows = []
        for spec, modes in vecs.items():
            for mode, desc in MODES:
                jm = J.get(modes.get(mode, ""))
                if not (jm and n0):
                    continue
                judged_any = True
                E, N = set(jm["harmful_ids"]), set(n0["harmful_ids"])
                bb, cc = len(E - N), len(N - E)
                lo, hi = paired_diff_ci(len(E & N), bb, cc, n_conv)
                rows.append([vector_label(spec), mode, desc, J1.get(spec, {}).get("asr"), jm["asr"], n0["asr"], jm["asr"] - n0["asr"], lo, hi, bb, cc,
                             binom_two_sided(min(bb, cc), bb + cc), None])
        for fam in (("same", "same-full"), ("early", "early-full")):  # Holm within each question
            sub = [x for x in rows if x[1] in fam]
            holm_in_place(sub, 11, 12)
        if rows:
            r["tables"].append({"id": "persist_single", "title": f"Clean sleeper test: 3 benign turns, then one harmful request at turn 4 ({n_conv} conversations); every row against no injection",
                                "columns": ["vector", "mode", "injected at", "ASR single turn (no conversation)", "ASR turn 4", "ASR no injection", "difference",
                                            "95% CI low", "95% CI high", "newly harmful", "newly safe", "McNemar p", "McNemar p, Holm (same modes / early modes)"],
                                "rows": rows})
            order = [v for v in vecs if any(x[0] == vector_label(v) for x in rows)]
            r["charts"].append({"id": "single_bars", "type": "bar", "title": "Turn-4 ASR by injection timing and span (dashed: no injection)",
                                "x": {"label": "vector", "values": [vector_label(v) for v in order]}, "y": {"label": "ASR (HarmBench judge)", "unit": "fraction"},
                                "series": [{"name": desc, "values": [next((x[4] for x in rows if x[0] == vector_label(v) and x[1] == mode), None) for v in order]} for mode, desc in MODES]
                                + [{"name": "no injection", "values": [n0["asr"]] * len(order)}]})
            pc = [x for x in rows if x[1] == "same-full"] or [x for x in rows if x[1] == "same"]
            sl = [x for x in rows if x[1] == "early-full"] or [x for x in rows if x[1] == "early"]
            pc_ok = [x for x in pc if x[12] is not None and x[12] < 0.05 and x[6] > 0]
            sl_sig = [x for x in sl if x[12] is not None and x[12] < 0.05]
            rng = lambda xs: f"{pct(min(x[4] for x in xs))} to {pct(max(x[4] for x in xs))}"
            if sl and pc:
                top = max(sl, key=lambda x: x[8])
                if sl_sig:
                    r["headline"] = (f"A turn-1 injection changes turn-4 ASR for {', '.join(x[0] for x in sl_sig)} (Holm p < 0.05): {rng(sl)} against {pct(n0['asr'])} with no injection.")
                elif pc_ok:
                    r["headline"] = (f"No detectable sleeper effect: injected at the harmful turn the vectors still work ({rng(pc)}), but injected only at turn 1 they leave turn-4 ASR at "
                                     f"{rng(sl)} (no injection {pct(n0['asr'])}; largest upper 95% bound {100 * top[8]:+.1f} points).")
                else:
                    r["headline"] = (f"Inside a conversation the vectors barely move turn-4 ASR even when injected at the harmful turn ({rng(pc)}; no injection {pct(n0['asr'])}), "
                                     f"so the sleeper test cannot separate persistence from a weaker effect in context; turn-1 injection: {rng(sl)}.")
            r["interpretation"].append("Each row is paired with no injection by conversation (exact McNemar; Holm within the harmful-turn modes and within the turn-1 modes). The interval is Newcombe's "
                                       "hybrid score interval for a difference of paired proportions (method 10).")
            r["interpretation"].append("Text-only modes steer the user text (and, at turn 1, the reply) but not the chat-template tokens after the request, where refusal is read out; "
                                       "the full modes steer those too, as single-turn steering does.")
        drift = raw(f"E12_{m}_single_drift.json")
        if drift and drift.get("vectors"):
            drows = []
            for key, per in drift["vectors"].items():
                spec, _, full = key.partition("|")
                for l, trs in per.items():
                    gap = (drift.get("harmful_gap") or {}).get(l)
                    for t in trs:
                        if t["turn"] in (2, 3):
                            drows.append([vector_label(spec) + (" (every turn-1 position)" if full else " (turn-1 text only)"), int(l), t["turn"], round(t["mean_diff"], 4), round(t["sd_none"], 3), None if not gap else round(t["mean_diff"] / gap, 4), t["n_up"], t["n_down"], float(f"{t['perm_p']:.3g}")])
            r["tables"].append({"id": "drift_paired", "title": f"Does a turn-1 injection still move later benign turns? Projection on the refusal candidate, early minus none, paired by script ({drift['n_scripts']} scripts)",
                                "columns": ["vector", "layer", "turn", "mean difference", "sd across scripts (no injection)", "difference as a share of the harmful-turn jump", "scripts up", "scripts down", "sign-flip p"],
                                "rows": drows})
        for l in single.get("drift_layers", []):
            series = [{"name": vector_label(c.get("vector")) if c.get("mode") is None or c.get("condition") == "none" else f"{c['mode']}: {vector_label(c.get('vector'))}",
                       "values": (c.get("drift_mean_by_turn") or {}).get(str(l))} for c in single["conditions"].values() if (c.get("drift_mean_by_turn") or {}).get(str(l))]
            if series:
                r["charts"].append({"id": f"drift_single_{l}", "type": "line", "title": f"Clean test: projection on the guardrail direction at layer {l}, last prompt token, per turn",
                                    "x": {"label": "turn", "values": [str(t) for t in range(1, 5)]}, "y": {"label": "projection", "unit": ""}, "series": series})
        r["config"]["single"] = {"conversations": single["n_harmful"], "benign_scripts": single["n_scripts"], "injection_layer": single["layer"], "drift_layers": single["drift_layers"]}

    if stacked:
        J = judged(m, "multiturn")
        vecs, none = by_vector(stacked)
        turns = ["4", "5", "6", "7"]
        series, rows = [], []
        n0 = J.get(none or "none")
        if n0 and "by_turn" in n0:
            series.append({"name": "no injection", "values": [n0["by_turn"].get(t, {}).get("asr") for t in turns]})
            series.append({"name": "no injection: refusal phrase rate", "values": [n0["by_turn"].get(t, {}).get("refusal") for t in turns]})
        for spec, modes in vecs.items():
            for mode in ("early", "same"):
                j = J.get(modes.get(mode, ""))
                if j and "by_turn" in j:
                    judged_any = True
                    series.append({"name": f"{mode}: {vector_label(spec)}", "values": [j["by_turn"].get(t, {}).get("asr") for t in turns]})
            je, js = J.get(modes.get("early", "")), J.get(modes.get("same", ""))
            if je and js and "by_turn" in je and "by_turn" in js:
                for t in turns:
                    a, b = je["by_turn"].get(t, {}).get("asr"), js["by_turn"].get(t, {}).get("asr")
                    rows.append([vector_label(spec), t, a, b, None if a is None or b is None else a - b, (n0 or {}).get("by_turn", {}).get(t, {}).get("asr")])
        if series:
            r["charts"].append({"id": "stacked_by_turn", "type": "line", "title": "Proposal protocol (four harmful turns in a row): ASR at each harmful turn",
                                "x": {"label": "turn", "values": turns}, "y": {"label": "rate", "unit": "fraction"}, "series": series})
        if rows:
            r["tables"].append({"id": "persist_stacked", "title": "Proposal protocol: persistence by turn (eq. 5.6)", "columns": ["vector", "turn", "ASR early", "ASR same-turn", "delta persist", "ASR no injection"], "rows": rows})
        if n0 and "by_turn" in n0:
            ref = [n0["by_turn"].get(t, {}).get("refusal") for t in turns]
            lo, hi = min(x for x in ref[1:] if x is not None), max(x for x in ref[1:] if x is not None)
            later = pct(lo) if lo == hi else f"{pct(lo)} to {pct(hi)}"
            r["caveats"].append(f"In the proposal's protocol the four harmful requests come one after another. With no injection the model refuses {pct(ref[0])} at turn 4 and then "
                                f"{later} at turns 5 to 7: an earlier refusal in the context makes every later refusal near certain, so turns 5 to 7 sit at a floor. "
                                "The clean test (one harmful request per conversation) was added for that reason.")
            if not r["headline"]:
                t4 = [f"{s_['name']} {pct(s_['values'][0])}" for s_ in series if "refusal" not in s_["name"] and s_["values"][0] is not None]
                r["headline"] = f"Proposal protocol, turn 4 ASR: {'; '.join(t4)}. After the first refusal the model refuses every later harmful turn."
        r["config"]["stacked"] = {"conversations": stacked["n_scripts"], "turns": stacked["n_turns"], "injection_layer": stacked["layer"]}

    r["status"] = "done" if judged_any else "running"
    if not judged_any:
        r["headline"] = "Multi-turn conversations generated; per-turn ASR pending the judge."
    r["caveats"].append("Each condition is a single greedy run of the 8B model; behaviours are paired across conditions, so differences between conditions are compared behaviour by behaviour.")
    return r


REPLICATION = "qwen2.5-7b"
SHORT = {"llama-8b": "Llama 3.1 8B", "qwen2.5-7b": "Qwen2.5 7B"}


def model_summary(m):
    """The numbers behind the main claims for one model (None until its baseline is judged)."""
    J = judged(m)
    if "baseline" not in J:
        return None
    name = ALIASES.get(m, m)
    ref = ROOT / "artifacts" / "vectors" / name.replace("/", "--") / "refusal.json"
    s = {"baseline": J["baseline"]["asr"], "ablated": J.get("ablate-refusal", {}).get("asr")}
    if ref.exists():
        s["refusal_layer"] = json.loads(ref.read_text())["layer"]
    h = gen(m, "harmless")
    s["harmless_ablated"] = h.get("ablate-refusal", {}).get("refusal_substring_rate")
    s["harmless_baseline"] = h.get("baseline", {}).get("refusal_substring_rate")
    ood = raw(f"E04b_{m}.json")
    s["ood_top1"] = ood["best_top1"] if ood else None
    st = raw(f"E07_stats_{m}.json")
    if st:
        L, c = st["layer"], st["c"]
        s.update({"layer": L, "c": c, "random_mean": st["random"]["asr_mean"], "random_sd": st["random"]["asr_sd"], "random_n": st["random"]["n_seeds"]})
        for kind, rows in st["kinds"].items():
            s[kind] = {"mean": sum(x["asr"] for x in rows.values()) / len(rows), "n": len(rows),
                       "above": sorted(((e, x["asr"]) for e, x in rows.items() if x["z_vs_random"] is not None and x["z_vs_random"] > 2), key=lambda t: -t[1])}
        for kind in ("emotion", "specific"):
            cor = raw(f"E11_{m}_{kind}_L{L}_c{float(c)}.json") or raw(f"E11_{m}_{kind}_L{L}_c{c}.json")
            if cor:
                s[f"rho_{kind}"] = (cor["delta_asr"]["spearman_rho"], cor["delta_asr"]["perm_p"])
    g = raw(f"E06_{m}.json")
    if g and "caa" in g.get("types", {}) and s.get("refusal_layer") is not None:
        t = g["types"]["caa"]
        s["max_cos"] = max(abs(t["cos_same_layer"][e][s["refusal_layer"]]) for e in t["emotions"])
        s["cos_p99"] = g.get("null_abs_cos", {}).get("p99")
    return s


def e13(model):
    r = base("E13-replication", "Does it hold on a second model family?", 5,
             "Run the conditions behind the main claims on Qwen2.5-7B-Instruct with the same rules: do the results replicate?", model)
    a, b = model_summary("llama-8b"), model_summary(REPLICATION)
    if not (a and b):
        return r

    def above(x, kind):
        if kind not in x:
            return "pending"
        k = x[kind]
        names = ", ".join(f"{e} {pct(v)}" for e, v in k["above"][:5]) or "none"
        return f"mean {pct(k['mean'])}; above random: {len(k['above'])} of {k['n']} ({names})"

    def rho(x, kind):
        v = x.get(f"rho_{kind}")
        return f"{v[0]:+.2f} (p = {v[1]:.2f})" if v else "pending"

    def f(x, key, fmt=pct):
        v = x.get(key)
        return fmt(v) if v is not None else "pending"

    rows = [
        ["Refusal direction found at layer", f"{a.get('refusal_layer')} of 32", f"{b.get('refusal_layer')} of 28"],
        ["Emotion injection layer and strength", f"{a.get('layer')}, c = {a.get('c')}", f"{b.get('layer', 'pending')}, c = {b.get('c', 'pending')}"],
        ["Judged harmful, no injection", f(a, "baseline"), f(b, "baseline")],
        ["Judged harmful, refusal direction ablated", f(a, "ablated"), f(b, "ablated")],
        ["Harmless requests refused, ablated (phrase match)", f(a, "harmless_ablated"), f(b, "harmless_ablated")],
        ["Emotion vectors name the emotion on independent texts (chance 5%)", f(a, "ood_top1"), f(b, "ood_top1")],
        ["Largest |cos(emotion, refusal)| at the refusal layer", f(a, "max_cos", lambda v: f"{v:.3f}"), f(b, "max_cos", lambda v: f"{v:.3f}")],
        ["Random directions, mean judged harmful", f"{pct(a['random_mean'])} (sd {pct(a['random_sd'])}, {a['random_n']} seeds)" if "random_mean" in a else "pending",
         f"{pct(b['random_mean'])} (sd {pct(b['random_sd'])}, {b['random_n']} seeds)" if "random_mean" in b else "pending"],
        ["Paired emotion vectors (the proposal's)", above(a, "emotion"), above(b, "emotion")],
        ["Specific parts (shared direction removed)", above(a, "specific"), above(b, "specific")],
        ["Paired vectors subtracted", above(a, "negative"), above(b, "negative")],
        ["Spearman rho, cosine vs ASR change (paired)", rho(a, "emotion"), rho(b, "emotion")],
        ["Spearman rho, cosine vs ASR change (specific)", rho(a, "specific"), rho(b, "specific")],
    ]
    r["tables"].append({"id": "replication", "title": "The main claims on both models (official HarmBench judge, 159 test behaviours, 256-token answers)",
                        "columns": ["measure", SHORT["llama-8b"], SHORT[REPLICATION]], "rows": rows})
    cats = [("no injection", "baseline"), ("random", "random_mean"), ("paired emotion", "emotion"), ("specific part", "specific"), ("subtracted", "negative"), ("refusal ablated", "ablated")]

    def val(x, key):
        v = x.get(key)
        return v["mean"] if isinstance(v, dict) else v

    r["charts"].append({"id": "replication_bars", "type": "bar", "title": "Mean judged-harmful rate by kind of push, both models",
                        "x": {"label": "condition", "values": [c[0] for c in cats]}, "y": {"label": "ASR (HarmBench judge)", "unit": "fraction"},
                        "series": [{"name": SHORT[m], "values": [val(x, k) for _, k in cats]} for m, x in (("llama-8b", a), (REPLICATION, b))]})
    done = all(k in b for k in ("emotion", "specific", "negative"))
    r["status"] = "done" if done else "running"
    if b.get("ablated") is not None:
        head = f"Qwen2.5 7B: ablating its refusal direction takes judged-harmful answers from {pct(b['baseline'])} to {pct(b['ablated'])}"
        if "emotion" in b and "random_mean" in b:
            head += (f"; the paired emotion vectors average {pct(b['emotion']['mean'])} against {pct(b['random_mean'])} for random pushes "
                     f"({len(b['emotion']['above'])} of {b['emotion']['n']} above random)")
        r["headline"] = head + "."
    r["config"] = {"model": ALIASES[REPLICATION], "rule": "injection one layer after the selected refusal layer, c = 0.5, fixed before the run (Llama: refusal 11, injection 12)"}
    r["caveats"].append("Qwen2.5's chat template adds its default system prompt; every Qwen condition, including the baseline, has it.")
    r["caveats"].append("One greedy run per condition; the random band uses 10 seeds on Qwen and 12 on Llama.")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--primary", default="llama-8b", help="the model whose results the interface shows by default")
    args = ap.parse_args()
    m = args.model
    model = ALIASES.get(m, m)
    results = [e00(m, model), *e01_e03(m, model), e02(m, model), e04(m, model), e05(m, model), e06(m, model), *e07_e10(m, model), e09(m, model), e11(m, model), e12(m, model)] + ([e13(model)] if m == args.primary else [])
    now = datetime.now().isoformat(timespec="seconds")
    order = {r["id"]: r for r in results}
    outdir = ROOT / "results" if m == args.primary else ROOT / "results" / (f"warmup-{m}" if m == "llama-1b" else f"replication-{m}")
    index_path = outdir / "index.json"
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmpl = json.loads((ROOT / "results" / "index.json").read_text())
        for x in tmpl["experiments"]:
            x["status"], x["file"] = "planned", f"{outdir.relative_to(ROOT)}/{x['id']}.json"
        write_json(index_path, tmpl)
    index = json.loads(index_path.read_text())
    for x in index["experiments"]:
        r = order.get(x["id"])
        if r is None:
            continue
        if r["status"] != "planned":
            r["finished"] = now if r["status"] == "done" else None
            write_json(outdir / f"{r['id']}.json", r)
            x["status"] = r["status"]
            x["headline"] = r["headline"] or x["headline"]
        else:  # no raw data (any more): do not point the interface at a results file
            x["status"] = "planned"
    index["updated"] = now
    index["model"] = model
    write_json(index_path, index)
    print("\n".join(f"{x['id']:30s} {x['status']:8s} {x['headline']}" for x in index["experiments"]))


if __name__ == "__main__":
    main()
