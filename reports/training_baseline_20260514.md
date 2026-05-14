# Reasoning Training Baseline - 2026-05-14

This snapshot preserves the best-performing local reasoning baseline from the May 14, 2026 data refresh cycle.

## Promoted baseline metrics

- Test accuracy: `0.7590`
- Test macro F1: `0.7567`
- `MOVE_TO_CHAIR` F1: `0.6834`
- `CHECK_TABLE` F1: `0.6869`
- Fresh-real accuracy: `0.5897`
- Fresh-real macro F1: `0.5868`

## Data batches kept from this cycle

- `data/raw/media_labeled_img2267_2_20260514_120100.csv`
- `data/raw/media_labeled_img2269_20260514_122418.csv`
- `data/raw/media_labeled_img2270_20260514_124341.csv`
- `data/raw/media_labeled_img2271_20260514_124341.csv`
- `data/raw/media_labeled_img2272_20260514_130217.csv`
- `data/raw/media_labeled_img2273_20260514_130217.csv`
- `data/raw/media_labeled_img2274_20260514_130217.csv`
- `data/raw/media_labeled_img2275_20260514_130217.csv`

## Notes

- Later experimental batches were useful for analysis, but they did not beat this baseline under the strict promotion gates.
- `scripts/run_reasoning_training_pipeline.sh` now protects `reports/metrics_promoted_baseline.json` from pruning so comparison runs do not lose the baseline.
