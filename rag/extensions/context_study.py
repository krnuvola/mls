#!/usr/bin/env python3
"""Stage B/C - how much context to pass to the LLM """
import argparse, csv, json, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever

COND = re.compile(r"^(?:n(?P<n>\d+)|(?P<oracle>oracle)|gap(?P<d>\d*\.\d+))(?:r(?P<rep>\d+))?$")
FIELDS = ["condition", "qid", "split", "n_ctx", "ctx_recall", "ctx_precision", "ctx_chars",
          "prompt_tokens", "completion_tokens", "usage_source", "gen_ms", "retr_ms",
          "embed_ms", "answer_chars", "transcript"]


def load_cache(path):
    cache = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cache[r["query_id"]] = r["transcript"]
    return cache


def gap_size(scores, delta, nmax=20):
    return max(1, sum(1 for s in scores[:nmax] if s >= scores[0] - delta))


def select_ids(cond, ranked, scores, rel):
    m = COND.match(cond)
    if not m:
        raise SystemExit(f"bad condition name: {cond}")
    if m["n"]:
        return ranked[:int(m["n"])]
    if m["oracle"]:
        return sorted(rel, key=lambda r: (-rel[r], r))[:10]
    return ranked[:gap_size(scores, float(m["d"]))]


def make_generator():
    from src.generator import ELMGenerator

    class UsageGenerator(ELMGenerator):
        """Same prompt, model and temperature as ELMGenerator; also records token usage."""
        def __init__(self):
            super().__init__()
            self._tls = threading.local()
            orig = self.client.chat.completions.create

            def wrapped(*args, **kw):
                r = orig(*args, **kw)
                self._tls.usage = getattr(r, "usage", None)
                return r
            self.client.chat.completions.create = wrapped

        def generate_usage(self, prompt, retries=3):
            for attempt in range(retries):
                try:
                    self._tls.usage = None
                    out = self.generate(prompt)
                    u = self._tls.usage
                    out["prompt_tokens"] = getattr(u, "prompt_tokens", None)
                    out["completion_tokens"] = getattr(u, "completion_tokens", None)
                    return out
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    print(f"  ELM error ({type(e).__name__}); retrying", flush=True)
                    time.sleep(3 * (attempt + 1))
    return UsageGenerator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--conditions", default="n1,n3,n5,n10,n20,oracle")
    ap.add_argument("--transcript-cache", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    for c in conds:
        if not COND.match(c):
            sys.exit(f"bad condition name: {c}")
    pool_k = max([20] + [int(COND.match(c)["n"]) for c in conds if COND.match(c)["n"]])

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(a.split)
    if a.limit:
        items = items[:a.limit]
    qrels = data.load_qrels(a.split)
    cache = load_cache(a.transcript_cache)
    items = [i for i in items if i["query_id"] in cache and qrels.get(i["query_id"])]
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=a.device)
    retr = ExactDenseRetriever(data_dir / "embeddings" / "corpus_enc_a.npy",
                               data_dir / "embeddings" / "corpus_enc_a_ids.json")
    builder = SimpleContextBuilder()

    pre = {}
    for it in items:
        qid = it["query_id"]
        t0 = time.perf_counter()
        vec = encoder.encode(cache[qid])
        emb_ms = (time.perf_counter() - t0) * 1000
        ret = retr.retrieve(vec, k=pool_k)
        pre[qid] = dict(item=it, ranked=ret["record_ids"], scores=ret["scores"],
                        retr_ms=ret["latency_ms"], emb_ms=emb_ms, rel=qrels[it["query_id"]])

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"context_study_{a.split}.csv"

    def prep(cond, qid):
        p = pre[qid]
        ids = select_ids(cond, p["ranked"], p["scores"], p["rel"])
        built = builder.build(cache[qid], ids, data)
        hit = len(set(ids) & set(p["rel"]))
        return ids, built, hit

    if a.dry_run:
        print(f"{len(items)} queries; conditions {conds}\n")
        print("| condition | mean n | mean context chars | est. prompt tokens | ctx recall | ctx precision |")
        print("|---|---|---|---|---|---|")
        for c in conds:
            ns, ch, rc, pr = [], [], [], []
            for qid in pre:
                ids, built, hit = prep(c, qid)
                ns.append(len(ids)); ch.append(sum(len(t) for t in built["contexts"]))
                rc.append(hit / len(pre[qid]["rel"])); pr.append(hit / len(ids))
            print(f"| {c} | {np.mean(ns):.2f} | {np.mean(ch):.0f} | {(np.mean(ch) + 400) / 4:.0f} | "
                  f"{np.mean(rc):.3f} | {np.mean(pr):.3f} |")
        return

    gen = make_generator()
    done = set()
    for c in conds:
        p = out_dir / f"ctx_{c}_{a.split}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add((c, json.loads(line)["query_id"]))
    jobs = [(c, q) for c in conds for q in pre if (c, q) not in done]
    random.Random(a.seed).shuffle(jobs)  # interleave conditions so ELM load drift is not confounded
    print(f"{len(jobs)} generations to run ({len(done)} already done)", flush=True)
    new_csv = not csv_path.exists()
    lock = threading.Lock()
    counter = [0]
    fcsv = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if new_csv:
        writer.writeheader()

    def run(job):
        c, qid = job
        p = pre[qid]
        ids, built, hit = prep(c, qid)
        res = gen.generate_usage(built["prompt"])
        pt, ct = res["prompt_tokens"], res["completion_tokens"]
        src = "api"
        if pt is None:
            pt, src = len(built["prompt"]) // 4, "estimate"
        row = {"query_id": qid, "split": a.split, "audio_file": p["item"]["audio_file"],
               "transcript": cache[qid], "answer": res["answer"], "retrieved_record_ids": ids,
               "timing": {"asr_ms": 0.0, "embedding_ms": p["emb_ms"], "retrieval_ms": p["retr_ms"],
                          "generation_ms": res["latency_ms"],
                          "end_to_end_ms": p["emb_ms"] + p["retr_ms"] + res["latency_ms"]}}
        crow = {"condition": c, "qid": qid, "split": a.split, "n_ctx": len(ids),
                "ctx_recall": hit / len(p["rel"]), "ctx_precision": hit / len(ids),
                "ctx_chars": sum(len(t) for t in built["contexts"]), "prompt_tokens": pt,
                "completion_tokens": ct, "usage_source": src, "gen_ms": res["latency_ms"],
                "retr_ms": p["retr_ms"], "embed_ms": p["emb_ms"],
                "answer_chars": len(res["answer"]), "transcript": cache[qid]}
        with lock:
            with (out_dir / f"ctx_{c}_{a.split}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.writerow(crow)
            fcsv.flush()
            counter[0] += 1
            print(f"[{counter[0]}/{len(jobs)}] {c} {qid} n={len(ids)} tokens={pt} gen={res['latency_ms']:.0f}ms", flush=True)

    if a.workers > 1:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(run, jobs))
    else:
        for j in jobs:
            run(j)
    fcsv.close()
    print(f"done. candidate files: {out_dir}/ctx_<condition>_{a.split}.jsonl ; per-query log: {csv_path}")


if __name__ == "__main__":
    main()
