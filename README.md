# Cross-station evaluation of machine-learning intrusion detection on 5G-NIDD

Reproducibility artifact: the pipeline, the per-cell result records, and the
aggregated tables behind a study of what the cross-station evaluation protocol
actually measures.

With the attack catalog fixed, training a detector on one base station and
scoring it on another is the standard way to argue that a detector generalizes.
This study takes that protocol apart and asks what the resulting number is a
property of: the detector, or the scoring set.

## What is measured

**A decomposition of cross-station degradation.** Training-side and scoring-side
class composition are varied independently in a two-by-two design, with a
same-station control in which only the scoring priors change. The unbalanced
cell reproduces the regime reported in the literature; the design then
attributes the degradation to its sources. Both directions, ten seeds, five
detectors, one paired Wilcoxon signed-rank test per effect.

**A mechanism for the residual concept shift.** Per-feature population stability
index and Kolmogorov-Smirnov statistics, computed within class so that differing
priors do not leak into the measurement, paired with model-agnostic permutation
reliance for each detector. Detectors that concentrate reliance on features that
move between stations are the ones that lose accuracy.

**Why fusion does not rescue a shifted detector.** Out-of-fold stacking with five
fusion rules ablated. The diagnostic is the rank correlation between the
out-of-fold ordering of the base learners, which is what any meta-learner is
fitted on, and their ordering on the shifted scoring set.

**Deployment operating points.** Per-sample latency at the 50th, 95th and 99th
percentiles across a batch sweep, parameter counts, serialized model size, peak
resident memory, and a breakdown separating input-conversion overhead from the
cost of evaluating a model.

## Key finding

Under the unbalanced protocol used in the published literature, binary F1 in the
BS2-to-BS1 direction falls to about 0.69. Class-matching the scoring set raises
it to about 0.92 with essentially no dependence on how the detector was trained,
while the binary false-positive rate stays near 0.0003 throughout. A substantial
part of the reported collapse is the response of a prevalence-sensitive metric
to the composition of the scoring set. A real but much smaller concept shift
remains, and macro-F1 under the fully matched protocol still falls from about
0.997 in-station to 0.79-0.93 depending on direction and detector.

## Dataset

5G-NIDD, captured on the 5G Test Network at the University of Oulu: 1,215,890
flows, 9 classes (benign plus 8 attack types), two physically separate Pico base
stations, two capture days. Licensed CC BY 4.0 and distributed by its authors.
**This repository does not redistribute it.**

Experiments use the authors' published ML-ready file (`Encoded.csv`, per their
`dataload.txt`), which yields 89 features after the column drops they specify.
Using the authors' own preprocessing rather than a custom drop list is a
deliberate choice: it removes the most station-unstable raw fields at the source
and makes the results comparable to work that follows the same recipe.

## Getting the data

Place `Encoded.csv` where `configs/paths.yaml` expects:

```
data/raw/combined_encoded/Encoded.csv
```

Everything else in `configs/paths.yaml` is relative to the repository root and
needs no editing.

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11. CPU only; no GPU is used or required.

## Run

One stage at a time. Stages are independent and resumable; each skips jobs whose
output already exists.

```bash
python scripts/run_campaign.py --stage grid          --workers 3
python scripts/run_campaign.py --stage decomposition --workers 3
python scripts/run_campaign.py --stage shift         --workers 3
python scripts/run_campaign.py --stage stacking      --workers 3
python scripts/run_campaign.py --stage latency       --workers 1
```

The latency stage must run alone with a single worker: tail latency measured
against competing jobs is not a latency measurement.

`RUNBOOK.md` carries the pre-flight checks, the smoke test, monitoring, and the
post-run verification in full.

## Runtime

Measured on 60 CPU cores with no GPU, three concurrent jobs, each model using 16
threads.

| Stage | Jobs | Wall clock |
|---|---|---|
| Accuracy grid | 850 | ~1.5 h |
| Decomposition | 400 | ~2-3 h |
| Shift and reliance | 20 | ~2 h |
| Stacking | 60 base fits, 300 records | ~4-7 h |
| Latency | 55 | ~2 h, single worker |

A single LightGBM cell at the 300k working size takes 15 to 22 seconds. The
shallow network is the slowest lightweight model at 120 to 220 seconds and
dominates the tail of every stage.

## Results layout

| Directory | Records | Behind |
|---|---|---|
| `results/metrics_windows/` | 850 | the accuracy grid |
| `results/decomposition/` | 400 | the factorial decomposition and the prior control |
| `results/stacking/` | 300 | the fusion sweep |
| `results/latency/` | 55 | latency and complexity profiles |
| `results/shift/` | 40 | shift and reliance analyses |

Aggregated tables sit at the top level of `results/` as `.csv` and `.json`, and
the campaign job logs as `campaign_*.jsonl`.

## A note on the shared code base

The loader, the splits and the model factories are shared with a separate
open-set rejection study by the same group, which lives in its own repository.
The two studies report no result that depends on the other and share no table or
figure. This repository carries the full pipeline so that it stands alone, but
only the result records behind this study.

## Authors

Van-Quynh Trinh (first author, Posts and Telecommunications Institute of
Technology), Trong-Thua Huynh (corresponding author, Posts and
Telecommunications Institute of Technology), De-Thu Huynh (The Saigon
International University), Ngoc-Hieu Le (Posts and Telecommunications Institute
of Technology).

Funded by the Posts and Telecommunications Institute of Technology, Vietnam.

## License

MIT, see `LICENSE`. The 5G-NIDD dataset is not covered by it and remains under
its own CC BY 4.0 terms.
