#!/usr/bin/env bash
# Minimal working example for the SecureCodeRL pipeline.
#
# Exercises the full code path on a tiny problem set so reviewers can verify
# the artifact is functional without budgeting the ~24h V100 run that
# reproduces paper Table 3. Runs in roughly 10 to 30 minutes on a V100 16 GB
# and roughly 30 to 90 minutes on CPU.
#
# This script does NOT reproduce the paper's headline numbers. It exercises:
#   1. Dependency import (torch, transformers, peft, bandit)
#   2. APPS+ data preparation pipeline (4 prompts, capped at 2 tests each)
#   3. SFT for 1 epoch on the prepared data
#   4. PPO for 3 episodes against the SFT checkpoint
#   5. Evaluation on 4 prompts with both partial-credit and binary reward
#   6. Bandit static-analysis path on a sample generation
#
# A successful run prints "MINIMAL EXAMPLE: PASS" and exits 0. Any failure
# in the pipeline prints "MINIMAL EXAMPLE: FAIL <stage>" and exits non-zero.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORK_DIR="${WORK_DIR:-$ROOT/.minimal_example_work}"
mkdir -p "$WORK_DIR"

fail() { echo "MINIMAL EXAMPLE: FAIL $1"; exit 1; }
stage() { echo; echo "[$(date +%H:%M:%S)] >>> $1"; }

# Stage 1. Dependency import check.
stage "1/6 dependency import check"
python3 - <<'PY' || { echo "MINIMAL EXAMPLE: FAIL dependency-import"; exit 1; }
import importlib, sys
required = ["torch", "transformers", "peft", "bandit"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print(f"missing: {missing}", file=sys.stderr); sys.exit(1)
print("dependency import OK")
PY

# Stage 2. Data preparation (very small).
stage "2/6 prepare 4 APPS+ prompts with 2 tests each"
python3 prepare_ppo_data.py \
    --output "$WORK_DIR/mini_prompts.json" \
    --max_problems 4 \
    --max_tests 2 \
    > "$WORK_DIR/prep.log" 2>&1 || fail "prepare-ppo-data"
test -s "$WORK_DIR/mini_prompts.json" || fail "prepare-ppo-data-empty-output"

# Stage 3. SFT for 1 epoch on the prepared sample.
stage "3/6 SFT, 1 epoch on a 4-sample subset"
# We need a tiny SFT dataset. The script reads JSONL from data/sft/{train,val}.jsonl
# by default; reuse those if present, otherwise synthesize a 4-line stub.
SFT_DATA_DIR="$WORK_DIR/sft_data"
mkdir -p "$SFT_DATA_DIR"
if [[ -s "$ROOT/data/sft/train.jsonl" ]]; then
    head -n 4 "$ROOT/data/sft/train.jsonl" > "$SFT_DATA_DIR/train.jsonl"
    head -n 2 "$ROOT/data/sft/val.jsonl"   > "$SFT_DATA_DIR/val.jsonl"
else
    # synthesize a deterministic micro-dataset (will not produce a useful model
    # but exercises the SFT code path)
    cat > "$SFT_DATA_DIR/train.jsonl" <<'JSONL'
{"prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n", "completion": "    return a + b\n"}
{"prompt": "def sub(a, b):\n    \"\"\"Return a - b.\"\"\"\n", "completion": "    return a - b\n"}
{"prompt": "def mul(a, b):\n    \"\"\"Return a * b.\"\"\"\n", "completion": "    return a * b\n"}
{"prompt": "def neg(a):\n    \"\"\"Return -a.\"\"\"\n", "completion": "    return -a\n"}
JSONL
    head -n 2 "$SFT_DATA_DIR/train.jsonl" > "$SFT_DATA_DIR/val.jsonl"
fi

python3 train_sft_stdin.py \
    --data_dir "$SFT_DATA_DIR" \
    --output_dir "$WORK_DIR/sft" \
    --epochs 1 \
    --batch_size 1 \
    --gradient_accumulation 1 \
    --max_seq_length 512 \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_steps 100 \
    > "$WORK_DIR/sft.log" 2>&1 || fail "sft-training"
test -d "$WORK_DIR/sft/final" || fail "sft-no-final-checkpoint"

# Stage 4. PPO for 3 episodes against the SFT checkpoint.
stage "4/6 PPO, 3 episodes against the SFT checkpoint"
python3 train_ppo.py \
    --sft_checkpoint "$WORK_DIR/sft/final" \
    --prompts_file "$WORK_DIR/mini_prompts.json" \
    --episodes 3 \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --max_new_tokens 64 \
    --output_dir "$WORK_DIR/ppo" \
    > "$WORK_DIR/ppo.log" 2>&1 || fail "ppo-training"

# Stage 5. Evaluation with both partial-credit and binary reward paths.
stage "5/6 evaluate on 4 prompts (partial-credit + binary)"
python3 evaluate_dual_metrics.py \
    --checkpoint "$WORK_DIR/sft/final" \
    --prompts_file "$WORK_DIR/mini_prompts.json" \
    --num_samples 4 \
    --output_dir "$WORK_DIR/eval" \
    > "$WORK_DIR/eval.log" 2>&1 || fail "evaluation"
test -s "$WORK_DIR/eval/evaluation_results.json" || fail "evaluation-no-output"

# Stage 6. Bandit security analysis path.
stage "6/6 Bandit security analysis on a sample generation"
python3 - <<'PY' || { echo "MINIMAL EXAMPLE: FAIL bandit-runner"; exit 1; }
from rl_training.bandit_runner import BanditRunner
sample_code = """
import os
def run(cmd):
    os.system(cmd)  # MEDIUM/HIGH severity, should be detected
print("hello")
"""
runner = BanditRunner(use_bandit=True)
findings = runner.analyze(sample_code)
v = runner.compute_severity(findings)
rsec = runner.compute_rsec(findings)
print(f"findings={len(findings)} V={v:.3f} R_sec={rsec:.3f}")
# Sanity: at least one MEDIUM/HIGH finding for os.system; V > 0; R_sec < 1
assert len(findings) >= 1, "expected at least one Bandit finding for os.system"
assert v > 0.0, "expected V > 0 with a HIGH finding"
assert rsec < 1.0, "expected R_sec < 1.0 when V > 0"
print("Bandit path OK")
PY

echo
echo "MINIMAL EXAMPLE: PASS"
echo "Work products under: $WORK_DIR"
echo "  - mini_prompts.json   (4 APPS+ prompts, 2 tests each)"
echo "  - sft/                (1-epoch SFT checkpoint)"
echo "  - ppo/                (3-episode PPO checkpoint)"
echo "  - eval/               (evaluation_results.json on 4 prompts)"
echo "  - *.log               (per-stage logs)"
