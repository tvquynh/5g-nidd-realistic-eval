"""Job driver for the full experiment campaign on a single workstation.

Every job is an independent ``python -m src.<runner>`` subprocess. The driver
keeps a bounded number of them in flight, skips jobs whose output already
exists so an interrupted campaign resumes where it stopped, and appends one
line per job to a JSONL log.

Usage:
    python scripts/run_campaign.py --stage grid --workers 3
    python scripts/run_campaign.py --stage decomposition --workers 3
    python scripts/run_campaign.py --stage stacking --workers 3
    python scripts/run_campaign.py --stage shift --workers 3
    python scripts/run_campaign.py --stage latency --workers 1
    python scripts/run_campaign.py --stage grid --dry-run

Stage ``latency`` must run with a single worker on an otherwise idle machine:
its tail-latency percentiles are meaningless under competing load.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
LIGHTWEIGHT_MODELS = ["lightgbm", "xgboost", "rf", "lr"]
FEATURE_SETS = ["full", "top50", "top20", "top20_stable"]
SUBSAMPLE = 300_000

# (split name, train_bs) pairs. Cross-station is run in both directions.
SPLIT_SETTINGS = [
    ("random", None),
    ("temporal", None),
    ("cross_station", 1),
    ("cross_station", 2),
    ("holdout_attack", None),
]


@dataclass
class Job:
    name: str
    argv: List[str]
    outputs: List[Path] = field(default_factory=list)

    def is_done(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)


def _bs_part(train_bs: Optional[int]) -> str:
    return f"_bs{train_bs}" if train_bs else ""


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def build_grid_jobs(results: Path) -> List[Job]:
    """Re-run the lightweight classifier grid in the current environment.

    Output goes to ``results/metrics_windows`` rather than ``results/metrics``
    so that the previously published set stays intact and the two can be
    compared cell by cell.
    """
    out_dir = results / "metrics_windows"
    jobs: List[Job] = []

    for model in LIGHTWEIGHT_MODELS:
        for split, train_bs in SPLIT_SETTINGS:
            for features in FEATURE_SETS:
                for seed in SEEDS:
                    out = out_dir / (f"{model}_{split}{_bs_part(train_bs)}"
                                     f"_{features}_seed{seed}.json")
                    argv = [
                        "-m", "src.run_experiment",
                        "--model", model, "--split", split, "--seed", str(seed),
                        "--features", features, "--subsample", str(SUBSAMPLE),
                        "--out", str(out),
                    ]
                    if train_bs:
                        argv += ["--train-bs", str(train_bs)]
                    jobs.append(Job(out.stem, argv, [out]))

    # The shallow network is the slowest of the lightweight models; it is run
    # on the full feature set only, matching the published configuration.
    for split, train_bs in SPLIT_SETTINGS:
        for seed in SEEDS:
            out = out_dir / f"mlp_{split}{_bs_part(train_bs)}_full_seed{seed}.json"
            argv = [
                "-m", "src.run_experiment",
                "--model", "mlp", "--split", split, "--seed", str(seed),
                "--features", "full", "--subsample", str(SUBSAMPLE),
                "--out", str(out),
            ]
            if train_bs:
                argv += ["--train-bs", str(train_bs)]
            jobs.append(Job(out.stem, argv, [out]))

    return jobs


def build_stacking_jobs(results: Path) -> List[Job]:
    """Stacked ensembles with every fusion rule evaluated on one base fit.

    Both base sets are run under cross-station shift, where the question of
    whether fusion can exploit its best member is decided. The in-distribution
    and mild-shift settings are run with the conventional base set only.
    """
    from src.stacking import META_LEARNERS

    out_dir = results / "stacking"
    plan = [
        ("cross_station", 1, ["trees_mlp", "diverse"]),
        ("cross_station", 2, ["trees_mlp", "diverse"]),
        ("random", None, ["trees_mlp"]),
        ("temporal", None, ["trees_mlp"]),
    ]

    jobs: List[Job] = []
    for split, train_bs, base_sets in plan:
        for base_set in base_sets:
            for seed in SEEDS:
                outputs = [
                    out_dir / (f"stack_{base_set}_{meta}_{split}{_bs_part(train_bs)}"
                               f"_full_seed{seed}.json")
                    for meta in META_LEARNERS
                ]
                argv = [
                    "-m", "src.run_stacking",
                    "--base-set", base_set, "--split", split, "--seed", str(seed),
                    "--features", "full", "--subsample", str(SUBSAMPLE),
                ]
                if train_bs:
                    argv += ["--train-bs", str(train_bs)]
                jobs.append(Job(f"stack_{base_set}_{split}{_bs_part(train_bs)}_seed{seed}",
                                argv, outputs))
    return jobs


def build_shift_jobs(results: Path) -> List[Job]:
    """Feature-shift and reliance analysis, both cross-station directions."""
    out_dir = results / "shift"
    jobs: List[Job] = []
    for train_bs in (1, 2):
        for seed in SEEDS:
            out = out_dir / f"shift_bs{train_bs}_full_seed{seed}.json"
            argv = [
                "-m", "src.run_shift_analysis",
                "--seed", str(seed), "--train-bs", str(train_bs),
                "--features", "full", "--subsample", str(SUBSAMPLE),
                "--models", "lightgbm,xgboost,rf,lr,mlp",
                "--out", str(out),
            ]
            jobs.append(Job(out.stem, argv, [out]))
    return jobs


def build_latency_jobs(results: Path, seeds: Optional[List[int]] = None) -> List[Job]:
    """Latency, model size, and memory profiles.

    Run with a single worker: the percentiles describe the serving path, and a
    machine running other jobs would report its own contention instead.
    """
    out_dir = results / "latency"
    seeds = seeds or SEEDS[:5]
    jobs: List[Job] = []

    for model in LIGHTWEIGHT_MODELS + ["mlp"]:
        for features in ("full", "top20"):
            for seed in seeds:
                out = out_dir / (f"latency_{model}_cross_station_bs1"
                                 f"_{features}_seed{seed}.json")
                argv = [
                    "-m", "src.run_latency",
                    "--model", model, "--split", "cross_station", "--train-bs", "1",
                    "--seed", str(seed), "--features", features,
                    "--subsample", str(SUBSAMPLE), "--out", str(out),
                ]
                jobs.append(Job(out.stem, argv, [out]))

    # The stack is profiled on the full feature set only: its serving cost is
    # the sum of its bases, which the per-model rows already decompose.
    for seed in seeds:
        out = out_dir / f"latency_stack_trees_mlp_lr_cross_station_bs1_full_seed{seed}.json"
        argv = [
            "-m", "src.run_latency",
            "--model", "stack", "--stack-base-set", "trees_mlp", "--stack-meta", "lr",
            "--split", "cross_station", "--train-bs", "1", "--seed", str(seed),
            "--features", "full", "--subsample", str(SUBSAMPLE), "--out", str(out),
        ]
        jobs.append(Job(out.stem, argv, [out]))

    return jobs


def build_decomposition_jobs(results: Path) -> List[Job]:
    """Factorial decomposition of the cross-station degradation.

    The training-side and test-side class compositions are varied
    independently, plus a same-station control in which only the test priors
    change. Together with the fully matched cell already produced by the grid
    stage, this gives the complete two-by-two design in both directions.
    Full feature set only: the decomposition is about protocol, not about
    feature budget.
    """
    out_dir = results / "decomposition"
    protocols = [
        "cross_station_raw_raw",
        "cross_station_raw_matched",
        "cross_station_matched_raw",
        "prior_control",
    ]
    models = LIGHTWEIGHT_MODELS + ["mlp"]

    jobs: List[Job] = []
    for protocol in protocols:
        for model in models:
            for train_bs in (1, 2):
                for seed in SEEDS:
                    out = out_dir / f"{model}_{protocol}_bs{train_bs}_full_seed{seed}.json"
                    argv = [
                        "-m", "src.run_experiment",
                        "--model", model, "--split", protocol, "--seed", str(seed),
                        "--features", "full", "--train-bs", str(train_bs),
                        "--subsample", str(SUBSAMPLE), "--out", str(out),
                    ]
                    jobs.append(Job(out.stem, argv, [out]))
    return jobs


def build_openset_jobs(results: Path) -> List[Job]:
    """Open-set rejection rules, one base fit per (base, seed).

    Each job fits its base classifier once and writes one record per rejection
    rule, so rules that are mathematically equivalent are compared on identical
    probabilities rather than on separately trained models.
    """
    out_dir = results / "openset_v2"
    plan = [("lightgbm", SEEDS), ("xgboost", SEEDS),
            ("tabnet", SEEDS[:5]), ("ftt", SEEDS[:5])]
    methods = ("none", "msp", "energy", "mahalanobis", "knn")

    jobs: List[Job] = []
    for base, seeds in plan:
        for seed in seeds:
            outputs = [out_dir / f"openset_{base}_{m}_full_seed{seed}.json"
                       for m in methods]
            argv = ["-m", "src.run_open_set_all",
                    "--base", base, "--seed", str(seed),
                    "--features", "full", "--subsample", str(SUBSAMPLE)]
            jobs.append(Job(f"openset_{base}_seed{seed}", argv, outputs))
    return jobs


def build_openset_detail_jobs(results: Path) -> List[Job]:
    """Per-attack rejectability and the operating-point sweep, one fit per cell."""
    out_dir = results / "openset_detail"
    plan = [("lightgbm", SEEDS), ("xgboost", SEEDS),
            ("tabnet", SEEDS[:5]), ("ftt", SEEDS[:5])]
    jobs: List[Job] = []
    for base, seeds in plan:
        for seed in seeds:
            out = out_dir / f"detail_{base}_full_seed{seed}.json"
            argv = ["-m", "src.run_open_set_detail", "--base", base,
                    "--seed", str(seed), "--features", "full",
                    "--subsample", str(SUBSAMPLE)]
            jobs.append(Job(f"detail_{base}_seed{seed}", argv, [out]))
    return jobs


def build_openset_temperature_jobs(results: Path) -> List[Job]:
    """Probability-space energy across temperatures, one base fit per cell.

    Ordered cheapest base first so the tree results land while the deep bases
    are still fitting.
    """
    out_dir = results / "openset_temperature"
    plan = [("lightgbm", SEEDS), ("xgboost", SEEDS),
            ("tabnet", SEEDS[:5]), ("ftt", SEEDS[:5])]
    jobs: List[Job] = []
    for base, seeds in plan:
        for seed in seeds:
            out = out_dir / f"temp_{base}_full_seed{seed}.json"
            argv = ["-m", "src.run_open_set_temperature", "--base", base,
                    "--seed", str(seed), "--features", "full",
                    "--subsample", str(SUBSAMPLE)]
            jobs.append(Job(f"temp_{base}_seed{seed}", argv, [out]))
    return jobs


def build_openset_destinations_jobs(results: Path) -> List[Job]:
    """Where unflagged novel flows land, and how confident the base is on them.

    Cheapest base first, so the tree results are available while the deep
    models are still fitting.
    """
    out_dir = results / "openset_destinations"
    plan = [("lightgbm", SEEDS), ("xgboost", SEEDS),
            ("tabnet", SEEDS[:5]), ("ftt", SEEDS[:5])]
    jobs: List[Job] = []
    for base, seeds in plan:
        for seed in seeds:
            out = out_dir / f"dest_{base}_full_seed{seed}.json"
            argv = ["-m", "src.run_open_set_destinations", "--base", base,
                    "--seed", str(seed), "--features", "full",
                    "--subsample", str(SUBSAMPLE)]
            jobs.append(Job(f"dest_{base}_seed{seed}", argv, [out]))
    return jobs


STAGES = {
    "grid": build_grid_jobs,
    "decomposition": build_decomposition_jobs,
    "openset": build_openset_jobs,
    "openset_detail": build_openset_detail_jobs,
    "openset_temperature": build_openset_temperature_jobs,
    "openset_destinations": build_openset_destinations_jobs,
    "stacking": build_stacking_jobs,
    "shift": build_shift_jobs,
    "latency": build_latency_jobs,
}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_job(job: Job, log_path: Path, log_lock) -> Dict:
    started = time.time()
    completed = subprocess.run(
        [sys.executable] + job.argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    record = {
        "job": job.name,
        "returncode": completed.returncode,
        "seconds": round(time.time() - started, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs_present": [p.exists() for p in job.outputs],
    }
    if completed.returncode != 0:
        record["stderr_tail"] = completed.stderr[-2000:]
    with log_lock:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent subprocesses; each model uses n_jobs=16")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="ignore existing outputs")
    ap.add_argument("--limit", type=int, default=0, help="run at most N jobs")
    args = ap.parse_args()

    import threading
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    profile = os.environ.get("NIDD_PROFILE", "local")
    results = Path(cfg[profile]["results"])

    jobs = STAGES[args.stage](results)
    pending = jobs if args.rerun else [j for j in jobs if not j.is_done()]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Stage {args.stage}: {len(jobs)} jobs total, {len(pending)} to run, "
          f"{len(jobs) - len(pending)} already complete")
    if args.stage == "latency" and args.workers > 1:
        print("WARNING: latency percentiles require a single worker on an idle machine")

    if args.dry_run:
        for job in pending[:20]:
            print("  ", job.name)
        if len(pending) > 20:
            print(f"   ... and {len(pending) - 20} more")
        return

    if not pending:
        print("Nothing to do.")
        return

    log_path = results / f"campaign_{args.stage}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()

    started = time.time()
    failures: List[Dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, log_path, log_lock): job for job in pending}
        for future in as_completed(futures):
            record = future.result()
            done += 1
            status = "ok " if record["returncode"] == 0 else "FAIL"
            if record["returncode"] != 0:
                failures.append(record)
            elapsed = time.time() - started
            # rate is wall-clock seconds per completed job and already
            # reflects the worker count, so it is not divided again.
            rate = elapsed / done
            remaining = (len(pending) - done) * rate
            print(f"[{done}/{len(pending)}] {status} {record['job']} "
                  f"({record['seconds']}s) | eta ~{remaining / 60:.0f} min", flush=True)

    print(f"\nStage {args.stage} finished in {(time.time() - started) / 60:.1f} min "
          f"({len(failures)} failures)")
    for record in failures[:10]:
        print(f"  FAILED {record['job']}: {record.get('stderr_tail', '')[-400:]}")
    print(f"Log: {log_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
