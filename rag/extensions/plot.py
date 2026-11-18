#!/usr/bin/env python3
"""Make the two report figures."""
import argparse, csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def ci(x, B=5000):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    m = x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def group(path, key):
    g = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for k, v in r.items():
            f = fnum(v)
            if f is not None:
                g[r[key]][k].append(f)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth")
    ap.add_argument("--context")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    out = Path(a.out_dir)
    if a.depth:
        g = group(a.depth, "condition")
        Ks = sorted(int(c[1:]) for c in g)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for m, lab in (("recall", "Recall@K"), ("ndcg", "NDCG@K"), ("hit", "Hit@K")):
            mu, lo, hi = zip(*[ci(g[f"K{k}"][m]) for k in Ks])
            ax[0].plot(Ks, mu, marker="o", label=lab)
            ax[0].fill_between(Ks, lo, hi, alpha=0.2)
        ax[0].set_xscale("log"); ax[0].set_xlabel("K (retrieval depth)"); ax[0].set_ylabel("score")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        tok = [np.mean(g[f"K{k}"]["ctx_chars"]) / 4 for k in Ks]
        lat = [np.median(g[f"K{k}"]["retr_ms"]) for k in Ks]
        ax[1].plot(Ks, tok, marker="s", color="tab:red")
        ax[1].set_xscale("log"); ax[1].set_xlabel("K"); ax[1].set_ylabel("context tokens (chars/4)", color="tab:red")
        ax2 = ax[1].twinx()
        ax2.plot(Ks, lat, marker="^", color="tab:green"); ax2.set_ylabel("retrieval ms (median)", color="tab:green")
        ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "fig_depth.png", dpi=300); print("wrote fig_depth.png")
    if a.context:
        g = group(a.context, "condition")
        ns = sorted((int(c[1:]), c) for c in g if re.fullmatch(r"n\d+", c))
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for m, lab in (("combined", "combined score"), ("correctness", "correctness / 4"),
                       ("grounding", "grounding / 4"), ("completeness", "completeness / 4")):
            vals = []
            for n, c in ns:
                x = g[c].get(m, [])
                x = [v / 4 for v in x] if m != "combined" else x
                vals.append(ci(x))
            mu, lo, hi = zip(*vals)
            ax[0].plot([n for n, _ in ns], mu, marker="o", label=lab)
            if m == "combined":
                ax[0].fill_between([n for n, _ in ns], lo, hi, alpha=0.2)
        for ref, style in (("oracle", "--"), ):
            if ref in g and g[ref].get("combined"):
                ax[0].axhline(np.mean(g[ref]["combined"]), linestyle=style, color="k", label="oracle context")
        ax[0].set_xscale("log"); ax[0].set_xlabel("records passed to the LLM (n)")
        ax[0].set_ylabel("judge score"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
        tok = [np.mean(g[c]["prompt_tokens"]) for _, c in ns]
        lat = [np.median(g[c]["gen_ms"]) / 1000 for _, c in ns]
        ax[1].plot([n for n, _ in ns], tok, marker="s", color="tab:red")
        ax[1].set_xscale("log"); ax[1].set_xlabel("n"); ax[1].set_ylabel("prompt tokens", color="tab:red")
        ax2 = ax[1].twinx(); ax2.plot([n for n, _ in ns], lat, marker="^", color="tab:green")
        ax2.set_ylabel("generation latency, s (median)", color="tab:green"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "fig_context.png", dpi=300); print("wrote fig_context.png")


if __name__ == "__main__":
    main()
