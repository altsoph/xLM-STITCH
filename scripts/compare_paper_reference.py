from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'reports' / 'drift'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    suffix = '_smoke' if args.smoke else ''
    table_path = ROOT / 'paper' / 'tables' / f'paper_tables_current{suffix}.json'
    ref_path = ROOT / 'reference' / 'paper_reference.json'
    tables = json.loads(table_path.read_text(encoding='utf-8'))
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ['# Paper Drift Report', '']
    summary = {}
    for table_name, cur_table in tables.items():
        if table_name not in ref or table_name == 'model_inventory':
            continue
        ref_table = ref[table_name]
        ref_rows = {tuple(row[:2]) if table_name == 'model_inventory' else row[0]: row for row in ref_table['rows']}
        drifts = []
        for cur_row in cur_table['rows']:
            row_name = tuple(cur_row[:2]) if table_name == 'model_inventory' else cur_row[0]
            if row_name not in ref_rows:
                continue
            ref_row = ref_rows[row_name]
            for idx in range(1, min(len(ref_row), len(cur_row))):
                rv = ref_row[idx]; cv = cur_row[idx]
                if isinstance(rv, (int, float)) and isinstance(cv, (int, float)):
                    drifts.append((abs(cv-rv), row_name, cur_table['headers'][idx], rv, cv))
        drifts.sort(reverse=True)
        summary[table_name] = drifts[:20]
        lines.append(f'## {table_name}')
        for drift, row_name, col_name, rv, cv in drifts[:20]:
            lines.append(f'- {row_name} / {col_name}: ref={rv} current={cv} drift={drift:.4f}')
        if not drifts:
            lines.append('- no overlapping numeric rows')
        lines.append('')
    (OUT_DIR / f'paper_drift_report{suffix}.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT_DIR / f'paper_drift_report{suffix}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'report_md': str(OUT_DIR / f'paper_drift_report{suffix}.md'), 'smoke': args.smoke}, indent=2))


if __name__ == '__main__':
    main()
