# RUNBOOK — 5G-NIDD realistic evaluation

Internal operational guide. For the scientific description see [README.md](README.md).

All computation runs on the Windows workstation (60 cores, 171.8 GB RAM, no
GPU). There is no SSH dispatch, no scheduler, and no shared filesystem: every
stage is a local subprocess pool driven by `scripts/run_campaign.py`.

---

## 1. Pre-flight checks (5 min)

```bash
python -c "import numpy, pandas, sklearn, lightgbm, xgboost, scipy, psutil; print('imports ok')"
```

Verify the source data and the derived master table:

```bash
python -c "import pandas as pd; d=pd.read_parquet('data/master_5g_nidd.parquet'); print(d.shape); print(d['y_multi'].value_counts().sort_index().to_dict()); print(d['BS'].value_counts().to_dict())"
```

Expected exactly:

| Check | Expected value |
|---|---|
| master shape | `(1215890, 93)` — 89 features + `y_multi`, `y_binary`, `BS`, `capture_day` |
| class counts | `{0: 477737, 1: 140812, 2: 1155, 3: 9721, 4: 20043, 5: 73124, 6: 20052, 7: 457340, 8: 15906}` |
| base stations | `{1: 728316, 2: 487574}` |

The class counts must match the dataset paper's Table XXII exactly. If they do
not, the master was built from the wrong source file — see section 9.

Rebuild the master from the author-provided file when needed:

```bash
python -m src.preprocess
```

Unit tests:

```bash
python -m pytest tests/ -q
```

Expected: `44 passed` in under 20 seconds. No data or network required.

## 2. Smoke test (~3 min)

One cell per runner, at a subsample small enough to finish in seconds:

```bash
python -m src.run_experiment --model lightgbm --split cross_station --seed 42 --features full --train-bs 1 --subsample 30000 --out scratch/smoke_experiment.json
```

```bash
python -m src.run_stacking --base-set trees_mlp --split cross_station --seed 42 --features full --train-bs 1 --subsample 30000 --k-folds 3 --out-dir scratch/smoke_stack
```

```bash
python -m src.run_shift_analysis --seed 42 --train-bs 1 --features full --models lightgbm,lr --subsample 30000 --reliance-subsample 3000 --reliance-repeats 1 --out scratch/smoke_shift.json
```

```bash
python -m src.run_latency --model lightgbm --split cross_station --seed 42 --features full --train-bs 1 --subsample 30000 --batch-sizes 1,64,1024 --out scratch/smoke_latency.json
```

Smoke numbers are **not** representative: at 30k the cross-station split has
only about 9,300 flows per side, so macro-F1 runs 0.01 to 0.02 higher than at
the 300k working size. Smoke checks wiring, not results.

Always write smoke output outside `results/`. A smoke file left in
`results/latency/` will be treated as a completed job by the campaign driver
and silently skipped.

```bash
rm -rf scratch/
```

## 3. Full run (~6-9 h total)

Stages are independent and resumable; each skips jobs whose output already
exists.

```bash
python scripts/run_campaign.py --stage grid --workers 3
```

```bash
python scripts/run_campaign.py --stage decomposition --workers 3
```

```bash
python scripts/run_campaign.py --stage shift --workers 3
```

```bash
python scripts/run_campaign.py --stage stacking --workers 3
```

```bash
python scripts/run_campaign.py --stage latency --workers 1
```

| Stage | Jobs | Output | Wall clock (3 workers) |
|---|---|---|---|
| grid | 850 | `results/metrics_windows/` | ~1.5 h |
| decomposition | 400 | `results/decomposition/` | ~2-3 h |
| shift | 20 | `results/shift/` | ~2 h |
| stacking | 60 | `results/stacking/` (5 files each) | ~4-7 h |
| latency | 55 | `results/latency/` | ~2 h, single worker |

The false-positive diagnostic is not a campaign stage; it is a single
command that trains three detectors in the severe direction and takes about
four minutes.

**Ordering matters.** `latency` must run last, alone, with `--workers 1`, on an
otherwise idle machine. Tail percentiles measured while other jobs compete for
cores describe the contention, not the serving path.

Options:

- **Subset**: `--limit N` runs the first N pending jobs.
- **Dry run**: `--dry-run` lists what would run without starting anything.
- **Force**: `--rerun` ignores existing outputs.
- **Resume after a crash**: rerun the same command. Completed outputs are
  detected and skipped; a partially written JSON is not, so delete any file
  whose job is listed as failed in the log before resuming.

## 4. Monitoring

```bash
tail -f results/campaign_grid_console.log
```

Count completed jobs and failures:

```bash
python -c "import json; rows=[json.loads(l) for l in open('results/campaign_grid.jsonl',encoding='utf-8')]; bad=[r for r in rows if r['returncode']]; print(len(rows),'done |',len(bad),'failed'); [print(r['job'], r.get('stderr_tail','')[-300:]) for r in bad[:5]]"
```

Per-job wall clock at the 300k working size, three workers:

| Model | Seconds per job | Notes |
|---|---|---|
| LightGBM | 15-22 | fastest |
| XGBoost | 20-30 | |
| Random Forest | 25-40 | |
| Logistic Regression | 24-45 | scaler plus lbfgs |
| MLP | 120-220 | single-threaded in scikit-learn; dominates the tail of every stage |
| Stacking (one base fit, five fusion rules) | 600-1500 | five folds plus a refit per base |

Resource ceiling: each model uses `n_jobs=16`, so three workers occupy about 48
of 60 cores. Do not raise `--workers` above 3 while other work is running on
the machine.

## 5. Post-run verification

Artifact counts:

```bash
python -c "from pathlib import Path; [print(f'{d.name:16} {len(list(d.glob(\"*.json\")))}') for d in Path('results').iterdir() if d.is_dir()]"
```

| Directory | Expected file count |
|---|---|
| `metrics_windows/` | 850 |
| `decomposition/` | 400 |
| `shift/` | 20 JSON + 20 CSV |
| `stacking/` | 300 (60 jobs x 5 fusion rules) |
| `latency/` | 55 |

Headline numbers, 300k subsample, ten seeds:

| Protocol | Model | Expected macro-F1 | Expected binary F1 |
|---|---|---|---|
| random | any | 0.993 - 0.998 | > 0.999 |
| temporal | LightGBM | 0.93 - 0.99 per seed, mean ~0.967 | > 0.99 |
| cross-station, class-matched, BS1 to BS2 | LightGBM | ~0.93 | ~0.89 |
| cross-station, class-matched, BS2 to BS1 | LightGBM | ~0.79 | ~0.92 |
| cross-station, class-matched, bidirectional mean | LightGBM | ~0.86 | |
| cross-station, raw both sides, BS2 to BS1 | LightGBM | ~0.70 | **~0.69** |
| cross-station, raw both sides, BS2 to BS1 | LR | ~0.84 | **~0.69** |
| prior control, BS2 | LightGBM | ~0.98 | ~0.999 |
| holdout-attack | any | 0.11 - 0.12 | n/a |

The two bold cells are the reproduction of the published cross-station regime
(Wang et al. report binary F1 of 0.671 to 0.687). If they come out far from
0.69, the raw protocol is not being applied — check that `--train-bs` reached
the split, since only splits listed in `STATION_SPLITS` receive it.

Environment check against the previous run:

```bash
python -m src.compare_environments --top 10
```

Expected: structural fields identical in every paired cell; ten-seed means
within 0.008 macro-F1 on the full feature set and 0.017 on importance-selected
subsets; individual seeds may differ by up to 0.109 on the random split and
0.100 on the temporal split.

## 6. Analysis workflow

### 6a. Aggregation and paper artifacts

Reduce the per-cell records to the tables and figures the manuscript includes:

```bash
python -m src.aggregate_decomposition --metric binary_f1
```

```bash
python -m src.aggregate_shift
```

```bash
python -m src.aggregate_stacking
```

```bash
python -m src.aggregate_latency
```

```bash
python -m src.compare_environments --top 6
```

Then regenerate everything the manuscript inputs. Both scripts read only from
`results/`, so no number can enter the paper without a file behind it:

```bash
python -m src.make_paper_tables && python -m src.make_paper_figures
```

The false-positive profile quoted in the mechanism section is produced by:

```bash
python -m src.diagnose_false_positives --models lightgbm,rf,lr --train-bs 2
```

Build the release package once the pipeline has finished:

```bash
python scripts/build_submission.py --date YYYY-MM-DD
```

### 6b. Critical comparisons

| Question | Where the answer comes from |
|---|---|
| Is the reported cross-station collapse real? | `decomposition_summary_effects.csv`, test-composition effect on binary F1 versus on TPR and FPR |
| Why do tree models degrade more than linear ones? | `shift_summary_per_model.csv`, reliance-weighted shift against macro-F1 drop |
| Does the ensemble beat its best member? | `stacking_summary.csv`, `stack_minus_best_mean` with its Wilcoxon p-value |
| Can a latency service-level objective be met? | `latency_summary.csv`, p95 and p99 columns at batch 1 |
| Is the XGBoost single-flow cost real? | `latency_summary_xgboost_overhead.csv`, `dmatrix_share` |

### 6c. Figures

To be regenerated after the campaign: split comparison, feature trade-off,
per-class heatmap, SHAP comparison (existing four), plus new figures for the
factorial decomposition, the reliance-versus-shift scatter, and the latency
tail sweep.

### 6d. Red flags — do NOT submit if any of these hold

- Master table class counts differ from the dataset paper's Table XXII.
- `compare_environments` reports any structural mismatch (`n_train`, `n_test`,
  `n_features`, `num_classes` differing for the same cell).
- Any campaign stage log contains a non-zero return code that was not
  investigated and resolved.
- Binary F1 under the raw-raw protocol is outside 0.60 to 0.75: the
  reproduction of the published regime is the load-bearing evidence for C1.
- Binary FPR under the raw-raw protocol exceeds 0.01. The claim that the
  collapse is a composition effect depends on the false-positive rate staying
  negligible; if it does not, the claim is wrong and must be withdrawn.
- Fewer than ten seeds present in any cell that carries a claim.
- Any latency figure produced with `--workers` greater than 1.
- Any number in the manuscript that cannot be traced to a file under
  `results/`.

## 7. Hand-off

The public mirror is released as
`github.com/tvquynh/5g-nidd-realistic-eval`. Before re-release, sync `src/`,
`configs/`, `scripts/`, `tests/`, and the aggregated CSV and JSON outputs.
Never sync `data/` — the dataset is redistributed by its authors, not by us.

## 8. Backup

Lite (about 40 MB): `results/*.csv`, `results/*.json`, `results/campaign_*.jsonl`,
`docs/`, `paper/`.

Full (about 400 MB): all of `results/` including the per-cell JSON files.

The master parquet (40 MB) is derived and can always be rebuilt from
`Encoded.csv` with `python -m src.preprocess`; it does not need backing up.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Encoded.csv expected 1215890 rows, got N` | Pointed at `Combined.csv` or a truncated copy | Fix `encoded_csv` in `configs/paths.yaml` |
| Feature count is 165 instead of 89 | Stale master from the earlier custom preprocessing pipeline | Delete `data/master_5g_nidd.parquet` and rerun `python -m src.preprocess` |
| `train_bs` appears as `None` for a station split | The split name is missing from `STATION_SPLITS` in the runner | Add it to the set in all three runners |
| Campaign reports a job complete but the JSON is empty | A previous run was interrupted mid-write | Delete the file and rerun the stage |
| Latency p99 is many times p50 across all models | Other jobs were running during the measurement | Rerun the latency stage alone with `--workers 1` |
| `ValueError: n_splits cannot be greater than the number of members in each class` | A rare class has fewer members than the fold count | `_safe_kfold` caps the fold count; check that it is being used |
| Results differ from a previous run on the same machine | Should not happen: LightGBM is bit-identical here | Confirm the master was not rebuilt with a different library stack |
| Merge fails on `train_bs` dtype in `compare_environments` | One side serialized `null`, the other `1.0` | Already normalized in `load_dir`; check for hand-edited JSON |
