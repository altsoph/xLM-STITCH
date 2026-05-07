from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _paper_common import build_clm_configs, build_mlm_configs, filter_for_smoke
from analysis.probes.feature_cache import materialize_probe_feature_cache, recommended_probe_cache_batch_size, recommended_probe_cache_storage_dtype
from experiments.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run xLM-STITCH paper reproduction pipeline')
    ap.add_argument('mode', choices=['smoke', 'full'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--model-precision', default='fp32')
    ap.add_argument('--tokenizer-batch-size', type=int, default=128)
    return ap.parse_args()


def run_one(cfg) -> None:
    runner = ExperimentRunner(cfg, base_dir=ROOT)
    try:
        runner.setup()
        runner.run()
    finally:
        runner.teardown()


def ensure_probe_caches(configs, args: argparse.Namespace) -> None:
    for cfg in configs:
        if cfg.task_family != 'probe_holdout':
            continue
        storage_dtype = recommended_probe_cache_storage_dtype(cfg.model.name)
        batch_size = recommended_probe_cache_batch_size(cfg.model.name)
        materialize_probe_feature_cache(cfg, ROOT, device_override=args.device, model_precision_override=args.model_precision, storage_dtype=storage_dtype, batch_size=batch_size, tokenizer_batch_size=args.tokenizer_batch_size, force=False)


def run_postprocessing(smoke: bool) -> None:
    py = str(ROOT / '.venv' / 'Scripts' / 'python.exe')
    scripts = []
    if not smoke:
        scripts.append(['scripts/refresh_mlm_probe_holdout_transfer_torch.py'])
    scripts.extend([
        ['scripts/build_paper_tables_current.py'] + (['--smoke'] if smoke else []),
        ['scripts/build_paper_figures_current.py'] + (['--smoke'] if smoke else []),
    ])
    if not smoke:
        scripts.append(['scripts/compare_paper_reference.py'])
    for cmd in scripts:
        subprocess.run([py] + cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    split = 'smoke' if args.mode == 'smoke' else 'benchmark'
    configs = build_mlm_configs(split, args.device) + build_clm_configs(split, args.device)
    if args.mode == 'smoke':
        configs = filter_for_smoke(configs)
    ensure_probe_caches(configs, args)
    for cfg in configs:
        print(f'RUN {cfg.run_id}')
        run_one(cfg)
    run_postprocessing(smoke=(args.mode == 'smoke'))
    print(json.dumps({'mode': args.mode, 'split': split, 'runs': len(configs)}, indent=2))


if __name__ == '__main__':
    main()
