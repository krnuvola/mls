

"""Generate answer candidates using the audio-to-answer baseline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import random

from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever


class CachedASR:
    """Return fixed transcripts from a previous ASR evaluation."""

    def __init__(self, cache_path, items, data_dir):
        transcripts = {}

        with Path(cache_path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    query_id = row["query_id"]

                    if query_id in transcripts:
                        raise ValueError(
                            f"Duplicate cached query: {query_id}"
                        )

                    transcripts[query_id] = row["transcript"]

        self.by_audio = {}

        for item in items:
            query_id = item["query_id"]

            if query_id not in transcripts:
                raise ValueError(
                    f"Missing cached transcript: {query_id}"
                )

            transcript = transcripts[query_id]

            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(
                    f"Empty cached transcript: {query_id}"
                )

            audio_path = (data_dir / item["audio_file"]).resolve()
            self.by_audio[audio_path] = transcript

    def transcribe(self, audio_path):
        return self.by_audio[Path(audio_path).resolve()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--split",
        choices=["practice", "dev"],
        required=True,
    )

    ap.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents retrieved",
    )

    ap.add_argument(
        "--context-n",
        type=int,
        default=None,
        help="Number of top-ranked documents supplied to the generator",
    )

    ap.add_argument(
        "--transcript-cache",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--device",
        default=None,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Diagnostic subset only; selects the first N queries",
    )

    ap.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many queries without replacement",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible query sampling",
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if args.k < 1:
        ap.error("--k must be positive")

    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    if args.sample_size is not None and args.sample_size < 1:
        ap.error("--sample-size must be positive")

    if args.sample_size is not None and args.limit is not None:
        ap.error("--sample-size and --limit cannot be used together")

    if args.context_n is not None and not (
        1 <= args.context_n <= args.k
    ):
        ap.error("--context-n must be between 1 and --k")

    if args.output.exists():
        ap.error(
            "Output exists; choose a new filename "
            "to preserve prior results"
        )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)

    items = data.audio_items(args.split)
    total_available = len(items)

    # ---------------------------------------------------------
    # Select queries
    # ---------------------------------------------------------

    if args.sample_size is not None:
        if args.sample_size > total_available:
            ap.error(
                f"--sample-size {args.sample_size} exceeds "
                f"available queries ({total_available})"
            )

        rng = random.Random(args.seed)
        items = rng.sample(items, args.sample_size)

    elif args.limit is not None:
        items = items[:args.limit]

    # Validate unique query IDs.
    selected_ids = [item["query_id"] for item in items]

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate query IDs in selected dataset")

    print("=" * 60)
    print("RAG GENERATION CONFIGURATION")
    print("=" * 60)
    print(f"Split:              {args.split}")
    print(f"Available queries:  {total_available}")
    print(f"Selected queries:   {len(items)}")
    print(f"Retrieval K:        {args.k}")
    print(
        f"Context N:          "
        f"{args.context_n if args.context_n is not None else args.k}"
    )
    print(
        f"Sampling seed:      "
        f"{args.seed if args.sample_size is not None else 'N/A'}"
    )
    print(f"Transcript cache:   {args.transcript_cache}")
    print(f"Output:             {args.output}")
    print("=" * 60, flush=True)

    # ---------------------------------------------------------
    # Initialise ASR
    # ---------------------------------------------------------

    if args.transcript_cache is not None:
        asr = CachedASR(
            args.transcript_cache,
            items,
            data_dir,
        )
    else:
        asr = ASRAdapter()
        asr.prepare()

    # ---------------------------------------------------------
    # Initialise RAG pipeline
    # ---------------------------------------------------------

    pipeline = RAGPipeline(
        asr,
        LocalEncoder(
            ROOT / "models/enc_a",
            device=args.device,
        ),
        ExactDenseRetriever(
            data_dir / "embeddings/corpus_enc_a.npy",
            data_dir / "embeddings/corpus_enc_a_ids.json",
        ),
        data,
        SimpleContextBuilder(),
        ELMGenerator(),
    )

    # ---------------------------------------------------------
    # Generate answers
    # ---------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("x", encoding="utf-8") as f:
        for index, item in enumerate(items, start=1):
            result = pipeline.answer_audio(
                data_dir / item["audio_file"],
                k=args.k,
                context_n=args.context_n,
            )

            row = {
                "query_id": item["query_id"],
                "split": args.split,
                "audio_file": item["audio_file"],
                "transcript": result["transcript"],
                "answer": result["answer"],
                "retrieved_record_ids": result["retrieval"]["record_ids"],
                "retrieval_k": args.k,
                "context_n": (
                    args.context_n
                    if args.context_n is not None
                    else args.k
                ),
                "timing": result["timing"],
                "generation_usage": result["generation_usage"],
            }

            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            f.flush()

            print(
                f"[{index}/{len(items)}] "
                f"{item['query_id']}",
                flush=True,
            )

    print("=" * 60)
    print("GENERATION COMPLETE")
    print(f"Queries processed: {len(items)}")
    print(f"Output saved to:   {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
