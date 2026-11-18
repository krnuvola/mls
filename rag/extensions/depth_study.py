"""Stage A - retrieval depth.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.metrics import recall_at_k, ndcg_at_k


def load_cache(path):
    cache = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cache[r["query_id"]] = r["transcript"]
    return cache


def gap_size(scores, delta, nmax=20):
    """Adaptive context size: keep the records scoring within `delta` of the top score."""
    return max(1, sum(1 for s in scores[:nmax] if s >= scores[0] - delta))


def boot(x, rng, B=5000):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--transcript-cache", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=20, help="timing repeats per query and K")
    ap.add_argument("--ks", default="1,2,3,5,10,20,50,100")
    ap.add_argument("--deltas", default="0.02,0.05,0.1,0.15,0.2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Ks = [int(k) for k in a.ks.split(",")]
    deltas = [float(d) for d in a.deltas.split(",")]
    rng = np.random.default_rng(0)

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(a.split)
    qrels = data.load_qrels(a.split)
    cache = load_cache(a.transcript_cache)
    missing = [i["query_id"] for i in items if i["query_id"] not in cache]
    if missing:
        sys.exit(f"{len(missing)} queries have no cached transcript (e.g. {missing[:3]}). "
                 "Run evaluate_rag.py with --transcript-cache first.")

    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=a.device)
    retr = ExactDenseRetriever(data_dir / "embeddings" / "corpus_enc_a.npy",
                               data_dir / "embeddings" / "corpus_enc_a_ids.json")
    import time
    rows, pools, failures = [], {}, []
    retr.retrieve(encoder.encode("warm up"), k=100)  # warm caches

    for qi, item in enumerate(items, 1):
        qid = item["query_id"]
        rel = qrels.get(qid, {})
        if not rel:
            continue
        t0 = time.perf_counter()
        vec = encoder.encode(cache[qid])
        enc_ms = (time.perf_counter() - t0) * 1000
        ret = retr.retrieve(vec, k=max(Ks + [100]))
        ranked, scores = ret["record_ids"], ret["scores"]
        pools[qid] = (ranked, scores, rel)
        if recall_at_k(ranked, rel, 100) == 0:
            failures.append((qid, cache[qid], len(rel)))
        for _ in range(50):  # warm up after the encoder so early K values are not penalised
            retr.retrieve(vec, k=100)
        for K in Ks:
            lat = [retr.retrieve(vec, k=K)["latency_ms"] for _ in range(a.repeats)]
            top = ranked[:K]
            nrel_in = len(set(top) & set(rel))
            rows.append({
                "condition": f"K{K}", "qid": qid, "K": K, "n_rel": len(rel),
                "recall": recall_at_k(ranked, rel, K), "hit": float(nrel_in > 0),
                "precision": nrel_in / K, "ndcg": ndcg_at_k(ranked, rel, K),
                "ctx_chars": sum(len(data.record_text(r)) for r in top),
                "retr_ms": float(np.median(lat)), "encode_ms": enc_ms})
        if qi % 25 == 0:
            print(f"  {qi}/{len(items)} queries", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n## Retrieval depth ({a.split}, {len(pools)} queries, mean 95% bootstrap CI)\n")
    print("| K | Recall | Hit@K | Precision | NDCG@K | context tokens (chars/4) | retrieval ms (median) |")
    print("|---|---|---|---|---|---|---|")
    for K in Ks:
        sel = [r for r in rows if r["K"] == K]
        def m(k):
            mu, lo, hi = boot([r[k] for r in sel], rng)
            return f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"
        print(f"| {K} | {m('recall')} | {m('hit')} | {m('precision')} | {m('ndcg')} | "
              f"{np.mean([r['ctx_chars'] for r in sel]) / 4:.0f} | {np.median([r['retr_ms'] for r in sel]):.2f} |")
    print(f"\nencoder time per query (median): {np.median([r['encode_ms'] for r in rows]):.1f} ms")

    print("\n## Adaptive selection from the top-20 pool (retrieval only)\n")
    print("| delta | mean n | recall | recall, fixed n | precision | precision, fixed n |")
    print("|---|---|---|---|---|---|")
    for d in deltas:
        ns, rec, rec_f, pre, pre_f = [], [], [], [], []
        for qid, (ranked, scores, rel) in pools.items():
            n = gap_size(scores, d)
            ns.append(n)
        nfix = max(1, int(round(np.mean(ns))))
        for qid, (ranked, scores, rel) in pools.items():
            n = gap_size(scores, d)
            rec.append(recall_at_k(ranked, rel, n)); pre.append(len(set(ranked[:n]) & set(rel)) / n)
            rec_f.append(recall_at_k(ranked, rel, nfix)); pre_f.append(len(set(ranked[:nfix]) & set(rel)) / nfix)
        print(f"| {d} | {np.mean(ns):.2f} (median {np.median(ns):.0f}; 10th-90th pct {np.percentile(ns, 10):.0f}-{np.percentile(ns, 90):.0f}) | {np.mean(rec):.3f} | {np.mean(rec_f):.3f} (n={nfix}) | "
              f"{np.mean(pre):.3f} | {np.mean(pre_f):.3f} |")

    r5 = [r for r in rows if r["K"] == 5]
    if r5:
        print(f"\nQueries whose gold records are all missed at K=5: {sum(1 for r in r5 if r['hit'] == 0)}/{len(r5)}")
    print(f"Queries with Recall@100 = 0 (retrieval or ASR failure): {len(failures)}/{len(pools)}")
    fpath = out.with_suffix(".failures.txt")
    fpath.write_text("\n".join(f"{q}\tn_rel={n}\t{t}" for q, t, n in failures), encoding="utf-8")
    print(f"wrote {out} and {fpath}")


if __name__ == "__main__":
    main()
