# xLM-STITCH

A minimal standalone reproduction repo for the current MLM/CLM interpretability paper.

This repo is intentionally trimmed to the code needed to:
- use the bundled canonical dataset splits for the paper
- run the benchmark experiments that feed the paper tables and figures
- materialize reusable full hidden-state caches for probe, lens, and PCA tasks
- generate the paper-facing numeric tables and figure PNGs
- compare reproduced numbers against a reference snapshot from the main research repo

## Scope

Reproduced paper artifacts:
- MLM tables: depth, readability/repair, control-token change rates
- CLM tables: depth/readout, local repair, control-position change rates
- Figures: MLM readout depth, MLM local repair, CLM next-token readout, CLM shifted recovery, CLM local repair mirror, MLM PCA panel, CLM PCA panel

Not reproduced here:
- Overleaf text and bibliography
- legacy exploratory tasks not used in the current paper
- historical reports from the main repo

## Environment

This repo ships with a copied `.venv` from the main repo so it can be run immediately on the same machine.

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

## Install / dataset source

1. Activate `.venv`.
2. The canonical paper splits are already bundled in `datasets/splits/`:
- `smoke.jsonl`
- `dev.jsonl`
- `benchmark.jsonl`

Optional: if you need to rebuild them on the same machine, run:
```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_bookcorpus_splits.py
```
The rebuild path expects either local BookCorpusOpen availability or internet access.

## Smoke run

Runs one MLM and one CLM end-to-end, then builds paper artifacts and a drift report.
```powershell
.\.venv\Scripts\python.exe .\scripts\run_paper_repro.py smoke --device cuda
```

## Full run

Runs the full benchmark paper pipeline for all paper models, builds paper artifacts, and writes a drift report.
```powershell
.\.venv\Scripts\python.exe .\scripts\run_paper_repro.py full --device cuda
```

## Outputs

Benchmark results:
- `results/exploratory/phase1_benchmark/`
- `results/exploratory/phase2_benchmark/`
- `results/exploratory/phase4_benchmark/`

Paper-facing artifacts:
- `paper/figures/`
- `paper/tables/`

Drift report:
- `reports/drift/paper_drift_report.md`

## Notes on efficiency

- Probe feature caches are materialized once per model and reused by:
  - `probe_holdout` MLP
  - MLM `lens_decoding`
  - PCA figure generation
  - MLM transfer refresh
- CLM benchmark uses the trimmed-prefix dataset variant from the paper.
- This repo does not read or reuse experiment artifacts from the main research repo.
