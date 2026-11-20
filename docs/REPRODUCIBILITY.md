# Experimental Reproducibility

This document distinguishes recorded experimental results from
commands that have been independently re-executed.

## 1. Hardware and environment

Recorded ASR experiments used:

- GPU: NVIDIA H200 MIG 1g.18gb
- Compute capability: sm_90
- PyTorch: 2.10.0+cu128 (recorded in the original ASR log)
- CUDA toolkit: verify the exact nvcc version for each run
- Audio: hw1-asr/test_audio.wav, 3.50 seconds
- Prompt: 59 tokens
- Generated output: 13 tokens

Run GPU experiments on an allocated teaching-cluster compute node,
not the login node.

The CUDA toolkit version, installed dependencies, Git commit,
and model revision should be recorded for future experiments.

## 2. CUDA kernel correctness

Source:
- hw1-asr/csrc/task1_rope.cu
- hw1-asr/csrc/task2_reduce.cu
- hw1-asr/csrc/task3_gemm.cu
- hw1-asr/csrc/task4_attention.cu

Tests:
- hw1-asr/tests/

Previous testing reported 33/33 passing cases.

Revalidate on the target GPU using the build and test instructions
in hw1-asr/README.md.

## 3. GEMM tile-size experiment

Recorded outputs:
- experiments/task3/tile8.txt
- experiments/task3/tile16.txt
- experiments/task3/tile32.txt

All three logs report 10/10 correctness tests passing.

The experiment compares TILE=8, TILE=16, and TILE=32.
Larger tiles generally perform better on large matrix workloads,
while the smaller tile is faster for the recorded M=1 lm_head case.

These logs are historical evidence. Exact source snapshots,
build commands, compiler flags, and run conditions should be
documented before claiming full independent reproducibility.

## 4. End-to-end ASR

Benchmark entry point:
- hw1-asr/benchmark.py

The recommended launcher is:
- hw1-asr/run.sh

The benchmark supports:
- --cuda
- --audio
- --warmup
- --runs
- --profile

Recorded results:

| Configuration | Mean latency (ms) | Peak memory (GiB) |
|---|---:|---:|
| PyTorch baseline | 1339.2 | 8.90 |
| CUDA RoPE/reductions, PyTorch GEMM/attention | 1296.2 | 8.90 |
| PyTorch GEMM, CUDA attention | 2015.5 | 8.59 |
| All custom CUDA | 5867.1 | 8.56 |

Recorded logs:
- experiments/asr/end_to_end_pytorch.txt
- experiments/asr/end_to_end_cuda_matched.txt
- experiments/asr/ablation_torch_gemm.txt
- experiments/asr/ablation_torch_gemm_attention.txt
- experiments/asr/profile_cuda_all.txt
- experiments/asr/profile_hybrid.txt

All four recorded configurations produced the expected
transcription and reported PASS.

The all-CUDA profiling log attributes approximately 84.5%
of profiled kernel time to linear operations and 15.0%
to attention.

## 5. RAG retrieval and context-size experiments

Implementation entry points:
- rag/extensions/depth_study.py
- rag/extensions/context_study.py
- rag/scripts/generate_answers.py
- rag/scripts/evaluate_answers.py

The context and depth studies require a transcript cache.
Do not substitute a different cache without documenting the change.

### 5.1 Development-set study

Results are stored in:
- rag/results/

These include:
- retrieval depth measurements
- context-size experiments
- oracle-context comparison
- adaptive-context experiments
- answer-judge summaries

The exact settings and query alignment for the final
development-set study must be verified against these files.

### 5.2 Practice-set study

Main summary:
- experiments/rag/main_context_size_results.json

Configuration:
- Split: practice
- Queries: 30 sampled from 350
- Sampling seed: 42
- Retrieval K: 20
- Context N: 3, 5, 10, 20
- Generator: utter-project/EuroLLM-22B-Instruct-2512
- Judge: gpt-4.1
- Judge repetitions: 1

| Context N | Mean answer quality | Mean prompt tokens | Mean generation latency (s) |
|---|---:|---:|---:|
| 3 | 0.6472 | 708.6 | 2.99 |
| 5 | 0.6139 | 1310.8 | 3.85 |
| 10 | 0.6944 | 2497.2 | 4.09 |
| 20 | 0.6417 | 5024.3 | 4.61 |

The results are exploratory and should not be interpreted
as statistically significant differences without paired analysis.

Candidate answers:
- experiments/rag/main_k20_n3_seed42.jsonl
- experiments/rag/main_k20_n5_seed42.jsonl
- experiments/rag/main_k20_n10_seed42.jsonl
- experiments/rag/main_k20_n20_seed42.jsonl

Judging outputs:
- experiments/rag/main_k20_n3_gpt41_evaluation/
- experiments/rag/main_k20_n5_gpt41_evaluation/
- experiments/rag/main_k20_n10_gpt41_evaluation/
- experiments/rag/main_k20_n20_gpt41_evaluation/

### 5.3 Full practice retrieval

SLURM entry point:
- experiments/rag/run_full_practice.sbatch

Recorded output:
- experiments/rag/practice_350_metrics.json

## 6. Reproducibility limitations

- Historical logs do not consistently capture Git commits.
- Some experimental commands are not yet recorded verbatim.
- Model and dataset revisions may affect future runs.
- ELM generation and judging depend on external service availability.
- Judge outputs can vary across requests and model revisions.
- The 30-query practice study is distinct from the development-set study.
- Single-run or low-repeat timings do not capture all system variance.

Do not overwrite historical experiment outputs when validating
reproduction. Write new results to separate paths and compare them.
