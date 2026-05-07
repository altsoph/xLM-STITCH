"""Run a single experiment from a YAML config or command-line args.

Usage:
    python run_experiment.py --config configs/runs/example.yaml
    python run_experiment.py --model bert-base-uncased --task clean_passthrough
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.schema import DatasetConfig, ModelConfig, RunConfig, load_run_config
from experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--config", help="Path to YAML run config")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--task", default="clean_passthrough")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/exploratory")
    args = parser.parse_args()

    if args.config:
        config = load_run_config(args.config)
    else:
        short = args.model.split("/")[-1].replace("-", "_")
        config = RunConfig(
            run_id=f"{short}_{args.task}_s{args.seed}",
            phase="adhoc",
            model=ModelConfig(name=args.model, device=args.device),
            dataset=DatasetConfig(name="handcrafted_debug", split="smoke"),
            task_family=args.task,
            seed=args.seed,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

    runner = ExperimentRunner(config, base_dir=Path(__file__).resolve().parent)
    try:
        runner.setup()
        metrics = runner.run()
        print(f"\nRun complete: {config.run_id}")
        print(f"Metrics: {metrics.metrics}")
    except Exception as e:
        print(f"Run failed: {e}", file=sys.stderr)
        raise
    finally:
        runner.teardown()


if __name__ == "__main__":
    main()
