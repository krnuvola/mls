from __future__ import annotations

import time


class RAGPipeline:
    def __init__(
        self,
        asr,
        encoder,
        retriever,
        data,
        context_builder,
        generator,
    ):
        self.asr = asr
        self.encoder = encoder
        self.retriever = retriever
        self.data = data
        self.context_builder = context_builder
        self.generator = generator

    def answer_audio(self, audio_path, k: int = 5, context_n: int | None = None):
        if k < 1:
            raise ValueError("k must be positive")
        if context_n is not None and not (1 <= context_n <= k):
            raise ValueError("context_n must be between 1 and k")

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio_path)
        asr_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        query_vector = self.encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        retrieval = self.retriever.retrieve(query_vector, k=k)

        selected_ids = retrieval["record_ids"]
        if context_n is not None:
            selected_ids = selected_ids[:context_n]

        built = self.context_builder.build(
            transcript,
            selected_ids,
            self.data,
        )

        generation = self.generator.generate(built["prompt"])

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "transcript": transcript,
            "retrieval": retrieval,
            "contexts": built["contexts"],
            "answer": generation["answer"],
            "generation_usage": {
                "model": generation.get("model"),
                "prompt_tokens": generation.get("prompt_tokens"),
                "completion_tokens": generation.get("completion_tokens"),
                "total_tokens": generation.get("total_tokens"),
            },
            "timing": {
                "asr_ms": asr_ms,
                "embedding_ms": embedding_ms,
                "retrieval_ms": retrieval["latency_ms"],
                "generation_ms": generation["latency_ms"],
                "end_to_end_ms": total_ms,
            },
        }
