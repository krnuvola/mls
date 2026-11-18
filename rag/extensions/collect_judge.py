#!/usr/bin/env python3
"""Merge judge scores into the per-query context-study log.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def read_rows(path):
    try:
        if path.suffix == ".jsonl":
            return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return obj
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return []
        if path.suffix in (".csv", ".tsv"):
            with path.open(newline="", encoding="utf-8-sig") as f:
                return list(csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ","))
    except Exception:
        return []
    return []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def map_columns(row):
    cols = {}
    for k, v in flatten(row).items():
        lk, val = k.lower(), num(v)
        if val is None:
            continue
        for name, keys in (("correctness", ("correct",)), ("grounding", ("ground",)),
                           ("completeness", ("complete",))):
            if any(s in lk for s in keys) and name not in cols:
                cols[name] = val
        if ("combined" in lk or lk.split(".")[-1] in ("score", "normalized_score", "normalised_score", "overall")) \
                and "combined" not in cols:
            cols["combined"] = val
    return cols


def load_judge(d):
    best = None
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rows = read_rows(p)
        scored = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            qid = r.get("query_id") or r.get("qid")
            cols = map_columns(r)
            if qid and {"correctness", "grounding", "completeness"} <= set(cols):
                scored.setdefault(str(qid), []).append(cols)
        if scored and (best is None or len(scored) > len(best[1])):
            best = (p, scored)
    if best is None:
        return None, {}
    p, scored = best
    avg = {}
    for q, lst in scored.items():
        keys = set().union(*[set(c) for c in lst])
        avg[q] = {k: float(np.mean([c[k] for c in lst if k in c])) for k in keys}
        if "combined" not in avg[q]:
            avg[q]["combined"] = (avg[q]["correctness"] + avg[q]["grounding"] + avg[q]["completeness"]) / 12
    return p, avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--conds", required=True)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--judge-pattern", default="judge_{cond}_{split}")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    log = rd / f"context_study_{a.split}.csv"
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    judged = {}
    for c in a.conds.split(","):
        d = rd / a.judge_pattern.format(cond=c, split=a.split)
        p, scores = load_judge(d)
        if p is None:
            print(f"[{c}] no per-query scores found under {d}", file=sys.stderr)
            if d.exists():
                print("   files there:", [str(x.relative_to(d)) for x in d.rglob('*') if x.is_file()][:20], file=sys.stderr)
            continue
        print(f"[{c}] {len(scores)} queries from {p.name}")
        judged[c] = scores
    if not judged:
        sys.exit("No judge output found. Paste `ls -R` of one judge directory and `head` of its per-query file.")
    out_rows = []
    for r in rows:
        sc = judged.get(r["condition"], {}).get(r["qid"], {})
        out_rows.append({**r, **{k: sc.get(k, "") for k in ("correctness", "grounding", "completeness", "combined")}})
    outp = rd / f"context_study_judged_{a.split}.csv"
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
