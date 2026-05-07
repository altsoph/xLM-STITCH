from __future__ import annotations

import argparse
import json
from pathlib import Path

from _paper_common import CLM_MODELS, MLM_MODELS, SMOKE_CLM_MODELS, SMOKE_MLM_MODELS, TABLE_DIR, clm_run_dir, load_metrics, max_layer_value, mlm_run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    return ap.parse_args()


def _pct(x: float, digits: int = 1) -> float:
    return round(100 * x, digits)


def build_tables(smoke: bool = False) -> dict:
    mlm_models = [m for m in MLM_MODELS if (not smoke or m['name'] in SMOKE_MLM_MODELS)]
    clm_models = [m for m in CLM_MODELS if (not smoke or m['name'] in SMOKE_CLM_MODELS)]
    tables = {}
    tables['model_inventory'] = {
        'headers': ['Family', 'Model', 'Layers', 'ParamsM'],
        'rows': (
            [['MLM', m['display'], m['layers'], m['params_m']] for m in mlm_models] +
            [['CLM', m['display'], m['layers'], m['params_m']] for m in clm_models]
        ),
    }
    mlm_repair_rows = []
    for m in mlm_models:
        lens = load_metrics(mlm_run_dir(m, 'lens_decoding', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        clean = load_metrics(mlm_run_dir(m, 'clean_passthrough', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        single = load_metrics(mlm_run_dir(m, 'single_corrupt_repair', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        swap = load_metrics(mlm_run_dir(m, 'swap_repair', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        dist = load_metrics(mlm_run_dir(m, 'distant_swap_repair', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        p = single['unmasked_restored_rate']
        mlm_repair_rows.append([m['display'], _pct(max_layer_value(lens['avg_retention_by_layer_ns'])), _pct(clean['avg_change_rate']), _pct(p), _pct(swap['both_swap_positions_restored_rate']), _pct(dist['dist8_both_restored_rate']), _pct(p*p)])
    tables['mlm_repair'] = {'headers': ['Model', 'Final readout', 'Clean change', 'Single repair', 'Adj swap both', 'Dist-8 both', 'Sim indep both'], 'rows': mlm_repair_rows}
    mlm_depth_rows = []
    for m in mlm_models:
        probe = load_metrics(mlm_run_dir(m, 'probe_holdout', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        mlm_depth_rows.append([m['display'], round(100 * probe['holdout_layer_mlp_acc'], 2), _pct(probe['content_to_function_transfer_acc']), _pct(probe['function_to_content_transfer_acc'])])
    tables['mlm_depth'] = {'headers': ['Model', 'MLP holdout', 'Content->Function', 'Function->Content'], 'rows': mlm_depth_rows}
    mlm_ctl_rows = []
    for m in mlm_models:
        ctl = load_metrics(mlm_run_dir(m, 'special_token_intervention', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        mlm_ctl_rows.append([m['display'], _pct(ctl['mask_top1_changed_rate_freeze_cls']), _pct(ctl['mask_top1_changed_rate_freeze_sep']), _pct(ctl['mask_top1_changed_rate_freeze_both']), _pct(ctl['mask_top1_changed_rate_zero_cls']), _pct(ctl['mask_top1_changed_rate_zero_sep']), _pct(ctl['mask_top1_changed_rate_zero_both']), _pct(ctl['mask_top1_changed_rate_zero_ordinary_mean'])])
    tables['mlm_control_change'] = {'headers': ['Model', 'Freeze CLS', 'Freeze SEP', 'Freeze both', 'Zero CLS', 'Zero SEP', 'Zero both', 'Zero ordinary'], 'rows': mlm_ctl_rows}
    clm_readout_rows = []
    for m in clm_models:
        probe = load_metrics(clm_run_dir(m, 'probe_holdout', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        tuned = load_metrics(clm_run_dir(m, 'decoder_tuned_lens', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        lens = load_metrics(clm_run_dir(m, 'lens_decoding', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        clm_readout_rows.append([m['display'], round(100 * probe['holdout_layer_mlp_acc'], 2), _pct(tuned['late_tuned_lastvis_retention']), _pct(tuned['late_tuned_lastvis_m1_retention']), _pct(max_layer_value(lens['avg_retention_by_layer_ns']))])
    tables['clm_readout'] = {'headers': ['Model', 'MLP holdout', 'Late next-token readout', 'Late right-edge recovery', 'Final shifted-input recovery'], 'rows': clm_readout_rows}
    clm_ctl_rows = []
    for m in clm_models:
        ctl = load_metrics(clm_run_dir(m, 'decoder_control_intervention', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        clm_ctl_rows.append([m['display'], _pct(1.0 - ctl['gen_freeze_first']), _pct(1.0 - ctl['gen_zero_first']), _pct(1.0 - ctl['gen_zero_ordinary_mean'])])
    tables['clm_control_change'] = {'headers': ['Model', 'Freeze first', 'Zero first', 'Zero ordinary'], 'rows': clm_ctl_rows}
    clm_repair_rows = []
    for m in clm_models:
        single = load_metrics(clm_run_dir(m, 'single_corrupt_repair', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        swap = load_metrics(clm_run_dir(m, 'swap_repair', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        dist = load_metrics(clm_run_dir(m, 'distant_swap_repair', 'benchmark' if not smoke else 'smoke') / 'metrics.json')
        p = single['unmasked_restored_rate']
        clm_repair_rows.append([m['display'], _pct(p), _pct(swap['both_swap_positions_restored_rate']), _pct(dist['dist8_both_restored_rate']), _pct(p*p)])
    tables['clm_repair'] = {'headers': ['Model', 'Single-token recovery', 'Adjacent two-token recovery', 'Distance-8 two-token recovery', 'Simulated independent double recovery'], 'rows': clm_repair_rows}
    return tables


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_tables(smoke=args.smoke)
    suffix = '_smoke' if args.smoke else ''
    (TABLE_DIR / f'paper_tables_current{suffix}.json').write_text(json.dumps(tables, indent=2), encoding='utf-8')
    lines = ['# Paper Tables']
    for name, table in tables.items():
        lines.append(''); lines.append(f'## {name}')
        headers = table['headers']; rows = table['rows']
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')
        for row in rows:
            lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    (TABLE_DIR / f'paper_tables_current{suffix}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'tables': list(tables.keys()), 'output_dir': str(TABLE_DIR), 'smoke': args.smoke}, indent=2))


if __name__ == '__main__':
    main()
