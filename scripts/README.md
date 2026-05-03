# scripts/

Helper scripts that exercise the SecureCodeRL pipeline.

| Script | Purpose | Runtime |
|--------|---------|---------|
| `minimal_example.sh` | Exercises every stage of the pipeline (data prep, SFT, PPO, evaluation, Bandit) on a 4-prompt subset. Prints `MINIMAL EXAMPLE: PASS` on success. Does not reproduce paper numbers. | ~10–30 min on V100 16 GB; ~30–90 min on CPU |

For full reproduction at the paper's 100-prompt scale, see the top-level
[`README.md`](../README.md) and [`run_both_experiments.sh`](../run_both_experiments.sh).
