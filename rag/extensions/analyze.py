#!/usr/bin/env python3
"""Mean and bootstrap 95% CI per condition, plus paired differences against a reference condition.
  """
import argparse, csv
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    data = defaultdict(lambda: defaultdict(dict))
    conds = []
    with open(a.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = r.get("condition") or r.get("cond")
            q = r.get("qid") or r.get("query_id")
            if c not in conds:
                conds.append(c)
            for m in a.metrics:
                try:
                    data[m][c][q] = float(r[m])
                except (KeyError, TypeError, ValueError):
                    pass

    def ci(x):
        x = np.asarray(x, float)
        if x.size == 0:
            return float("nan"), float("nan"), float("nan")
        mm = x[rng.integers(0, x.size, size=(a.boot, x.size))].mean(axis=1)
        return x.mean(), *np.percentile(mm, [2.5, 97.5])

    for m in a.metrics:
        print(f"\n### {m}\n")
        print(f"| condition | n | mean [95% CI] | paired diff vs {a.ref} [95% CI] |")
        print("|---|---|---|---|")
        ref = data[m].get(a.ref, {})
        for c in conds:
            d = data[m].get(c, {})
            mu, lo, hi = ci(list(d.values()))
            common = sorted(set(d) & set(ref))
            if c != a.ref and common:
                dm, dl, dh = ci([d[q] - ref[q] for q in common])
                ds = f"{dm:+.3f} [{dl:+.3f}, {dh:+.3f}] (n={len(common)})"
            else:
                ds = "-"
            print(f"| {c} | {len(d)} | {mu:.3f} [{lo:.3f}, {hi:.3f}] | {ds} |")


if __name__ == "__main__":
    main()
