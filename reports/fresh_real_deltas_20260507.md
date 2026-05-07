# Fresh-Real Delta Report (2026-05-07)

Baseline fresh-real acc=0.522 macro_f1=0.389 rows=23

| run | seed | holdout_rows | acc_delta | macro_f1_delta | MOVE_TO_CHAIR_f1_delta | CHECK_TABLE_f1_delta | regressions |
|---|---:|---:|---:|---:|---:|---:|---|
| exp_h32_cap_seed42 | - | 23 | +0.000 | +0.020 | -0.350 | +0.226 | MOVE_TO_CHAIR |
| sweep_h32_over | sweep_freshIJ_h32_20260507_140750 | 23 | +0.087 | +0.103 | +0.083 | +0.121 | none |
| sweep_h32_over | sweep_freshIJ_h32_20260507_140750 | 23 | +0.087 | +0.046 | -0.306 | +0.264 | MOVE_TO_CHAIR |
| sweep_h32_over | sweep_freshIJ_h32_20260507_140750 | 23 | +0.000 | -0.002 | -0.350 | +0.000 | MOVE_TO_CHAIR |
| sweep_h48_over | sweep_freshIJ_h48hpc12_20260507_140925 | 39 | +0.017 | +0.166 | -0.306 | +0.137 | MOVE_TO_CHAIR |
| sweep_h48_over | sweep_freshIJ_h48hpc12_20260507_140925 | 39 | +0.042 | +0.150 | -0.194 | +0.092 | MOVE_TO_CHAIR |
| sweep_h48_over | sweep_freshIJ_h48hpc12_20260507_140925 | 39 | +0.068 | +0.176 | -0.013 | +0.238 | MOVE_TO_CHAIR |

Stop-rule flag: any run with mixed signs across classes is treated as unstable and should stay in data-refresh-only loop.