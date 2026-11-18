#!/bin/bash
#SBATCH --job-name=mls-rag-practice
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/rag/logs/practice_%j.out
#SBATCH --error=experiments/rag/logs/practice_%j.err

set -euo pipefail

cd "$HOME/edin-mls-26-autumn"

eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate "$HOME/.conda/envs/mls-asr"

export RAG_ASR_PYTHON="$HOME/.conda/envs/mls-asr/bin/python"
source "$HOME/mls-rag-env/bin/activate"

module add cuda
export ASR_CUDA=none
export TMPDIR="$HOME/.cache/mls-cuda-tmp"
mkdir -p "$TMPDIR"

python -u rag/scripts/evaluate_rag.py \
  --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --device cpu \
  --k-max 100 \
  --output experiments/rag/practice_350_metrics.json
