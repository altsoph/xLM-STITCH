# xLM-STITCH

A standalone reproduction repo for the MLM/CLM interpretability paper.

This repo contains the code needed to:
- use the bundled canonical dataset splits for the paper
- run the benchmark experiments that feed the paper tables and figures
- materialize reusable full hidden-state caches for probe, lens, and PCA tasks
- generate the paper-facing numeric tables and figure PNGs

## Scope

Reproduced paper artifacts:
- MLM tables: depth, readability/repair, control-token change rates
- CLM tables: depth/readout, local repair, control-position change rates
- Figures: MLM readout depth, MLM local repair, CLM next-token readout, CLM shifted recovery, CLM local repair mirror, MLM PCA panel, CLM PCA panel

## Environment

Use requirements.txt to build the `.venv`.

## Install / dataset source

1. Activate `.venv`.
2. The splits are already bundled in `datasets/splits/`:
- `smoke.jsonl`
- `dev.jsonl`
- `benchmark.jsonl`

## Smoke run

Runs one MLM and one CLM end-to-end, then builds paper artifacts.
```shell
python ./scripts/run_paper_repro.py smoke --device cuda
```

## Full run

Runs the full benchmark paper pipeline for all paper models, builds paper artifacts.
```shell
python ./scripts/run_paper_repro.py full --device cuda
```

## Outputs

Paper-facing artifacts:
- `paper/figures/`
- `paper/tables/`

## Notes on efficiency

- Probe feature caches are materialized once per model and reused by:
  - `probe_holdout` MLP
  - MLM `lens_decoding`
  - PCA figure generation
  - MLM transfer refresh
- CLM benchmark uses the trimmed-prefix dataset variant from the paper.
- This repo does not read or reuse experiment artifacts from the main research repo.
