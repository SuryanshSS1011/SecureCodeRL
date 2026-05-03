# experiment_results/

20-prompt pilot evaluation from December 12, 2025 on the SFT, PPO-simple,
PPO-fresh, and PPO-continue checkpoints. These numbers are **not** the
LCTES '26 Table 3 results. Table 3 is a separate 100-prompt evaluation on the
same checkpoints, run on a single NVIDIA V100 16 GB.

The files in this directory are kept as an audit trail of the smaller pilot
that informed the experiment design. They are not load-bearing for the paper.

## Inventory

| File | What it is |
|------|------------|
| `EXPERIMENT_SUMMARY.md` | Run configuration (episodes, batch, lr, eval samples) |
| `evaluation_summary.csv` | Per-model aggregate metrics (20 samples each) |
| `evaluation_detailed.md` | Failure-mode breakdown and per-model outcome counts |
| `qualitative_examples.md` | Selected generation examples |
| `experiment_log.txt` | Pipeline timing log |
| `baseline/` | Pre-training baseline evaluation output |
| `final_comparison/` | Post-training comparative evaluation output |
| `data/` | Per-prompt evaluation traces |

## Reproducing at the paper's 100-prompt scale

```bash
python evaluate_dual_metrics.py \
    --checkpoints \
        ./checkpoints/sft_stdin/best \
        ./checkpoints/ppo/ppo/best \
        ./checkpoints/ppo_continue/ppo/best \
        ./checkpoints/ppo_fresh/ppo/best \
    --prompts_file data/prompts/ppo_prompts_with_tests.json \
    --num_samples 100 \
    --seed 42
```

Expected runtime is 30 to 45 minutes per checkpoint on a V100 16 GB. See the
top-level `README.md` quick-start for the full training pipeline that produces
the checkpoints in the first place.
