from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from _paper_common import CLM_MODELS, FIG_DIR, MLM_MODELS, SMOKE_CLM_MODELS, SMOKE_MLM_MODELS, clm_run_dir, load_metrics, mlm_run_dir
from analysis.probes.feature_cache import load_cache_metadata


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    return ap.parse_args()


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def _cache_dir_for_probe(run_dir: Path) -> Path:
    return run_dir / 'probe_feature_cache_full_v1'


def _pca_points(cache_dir: Path, per_layer: int = 200, seed: int = 42):
    meta = load_cache_metadata(cache_dir)
    hs = np.load(cache_dir / meta['files']['hidden_states'], mmap_mode='r')
    total_tokens = int(meta['total_tokens'])
    num_layers = int(meta['num_layers'])
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    base_idx = np.arange(total_tokens)
    sample_n = min(per_layer, total_tokens)
    for layer_idx in range(num_layers):
        choose = rng.choice(base_idx, size=sample_n, replace=False)
        rows.append(np.asarray(hs[choose, layer_idx, :], dtype=np.float32))
        labels.extend([layer_idx] * sample_n)
    X = np.concatenate(rows, axis=0)
    coords = PCA(n_components=3, random_state=seed).fit_transform(X)
    return coords, np.array(labels)


def _panel_pca(models, run_dir_fn, out_path: Path, nrows: int, ncols: int):
    fig = plt.figure(figsize=(15.0, 9.0 if nrows == 2 else 12.0))
    cmap = plt.get_cmap('tab20')
    for idx, model in enumerate(models, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
        coords, labels = _pca_points(_cache_dir_for_probe(run_dir_fn(model)))
        for layer in np.unique(labels):
            pts = coords[labels == layer]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4, alpha=0.55, color=cmap(int(layer) % 20))
        ax.set_title(model['display'], fontsize=12, pad=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    total = nrows * ncols
    for idx in range(len(models) + 1, total + 1):
        ax = fig.add_subplot(nrows, ncols, idx)
        ax.axis('off')
    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    _save(fig, out_path)


def main() -> None:
    args = parse_args()
    mlm_models = [m for m in MLM_MODELS if (not args.smoke or m['name'] in SMOKE_MLM_MODELS)]
    clm_models = [m for m in CLM_MODELS if (not args.smoke or m['name'] in SMOKE_CLM_MODELS)]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # fig12
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), sharey=True)
    main_models = [m for m in mlm_models if m['display'] != 'ModernBERT']
    modern = next((m for m in mlm_models if m['display'] == 'ModernBERT'), None)
    colors = {'ALBERT':'#4c72b0','BERT':'#55a868','RoBERTa':'#c44e52','XLM-R':'#8172b2','ModernBERT':'#dd8452'}
    for model in main_models:
        metrics = load_metrics(mlm_run_dir(model, 'lens_decoding', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')
        layers = sorted(map(int, metrics['avg_retention_by_layer_ns'].keys())); vals = [metrics['avg_retention_by_layer_ns'][str(k)] for k in layers]
        axes[0].plot(layers, vals, linewidth=2.2, color=colors.get(model['display'], '#333333'), label=model['display'])
    if modern is not None:
        metrics = load_metrics(mlm_run_dir(modern, 'lens_decoding', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')
        layers = sorted(map(int, metrics['avg_retention_by_layer_ns'].keys())); vals = [metrics['avg_retention_by_layer_ns'][str(k)] for k in layers]
        axes[1].plot(layers, vals, linewidth=2.4, color=colors['ModernBERT'], label='ModernBERT')
        axes[1].set_title('ModernBERT', fontsize=14, pad=8)
        axes[1].legend(frameon=False, fontsize=9, loc='lower right')
    axes[0].set_title(', '.join(m['display'] for m in main_models), fontsize=14, pad=8)
    axes[0].set_ylabel('Exact-match readout')
    for ax in axes:
        ax.set_xlabel('Layer'); ax.set_ylim(0.0, 1.02); ax.grid(True, alpha=0.25)
    if main_models: axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    _save(fig, FIG_DIR / ('fig12_mlm_readout_depth_smoke.png' if args.smoke else 'fig12_mlm_readout_depth.png'))

    # fig01
    colors = sns.color_palette('tab10', len(mlm_models))
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8))
    for color, model in zip(colors, mlm_models):
        swap = load_metrics(mlm_run_dir(model, 'swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')
        dist = load_metrics(mlm_run_dir(model, 'distant_swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')
        axes[0].plot([1,3,5,8],[swap['both_swap_positions_restored_rate'], dist['dist3_both_restored_rate'], dist['dist5_both_restored_rate'], dist['dist8_both_restored_rate']], linewidth=2.0, color=color, label=model['display'])
    x = np.arange(len(mlm_models)) * 1.3; width = 0.18; single_vals=[]; sim_vals=[]; actual_vals=[]; dist8_vals=[]
    for model in mlm_models:
        single = load_metrics(mlm_run_dir(model, 'single_corrupt_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')['unmasked_restored_rate']
        actual = load_metrics(mlm_run_dir(model, 'swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')['both_swap_positions_restored_rate']
        dist8 = load_metrics(mlm_run_dir(model, 'distant_swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')['dist8_both_restored_rate']
        single_vals.append(single); sim_vals.append(single**2); actual_vals.append(actual); dist8_vals.append(dist8)
    axes[1].bar(x - 1.5*width, single_vals, width=width, color='#4c72b0'); axes[1].bar(x - 0.5*width, sim_vals, width=width, color='#55a868'); axes[1].bar(x + 0.5*width, actual_vals, width=width, color='#c44e52'); axes[1].bar(x + 1.5*width, dist8_vals, width=width, color='#8172b2')
    axes[0].set_xticks([1,3,5,8]); axes[0].set_xlabel('Distance between swapped tokens'); axes[0].set_ylabel('Both swapped tokens restored'); axes[0].set_ylim(0,0.5); axes[0].grid(True, alpha=0.25)
    axes[1].set_xticks(x); axes[1].set_xticklabels([m['display'] for m in mlm_models], rotation=35, ha='right'); axes[1].set_ylabel('Restoration probability'); axes[1].set_ylim(0,0.5); axes[1].grid(True, axis='y', alpha=0.25)
    _save(fig, FIG_DIR / ('fig01_mlm_local_repair_smoke.png' if args.smoke else 'fig01_mlm_local_repair.png'))

    # fig04 fig05 fig09
    def _grid_lines(models, task, fields, colors, out_name):
        fig, axes = plt.subplots(max(1, (len(models)+2)//3), min(3, max(1, len(models))), figsize=(15.0, 5.0 if len(models)<=3 else 11.0), sharey=True)
        axes = np.array(axes).reshape(-1)
        for idx, model in enumerate(models):
            ax = axes[idx]
            metrics = load_metrics(clm_run_dir(model, task, 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')
            layers = sorted(map(int, metrics[fields[0]].keys()))
            for field, color, ls in colors:
                vals = [metrics[field][str(k)] for k in layers]
                ax.plot(layers, vals, linewidth=1.8, linestyle=ls, color=color)
            ax.set_title(model['display'], fontsize=10)
            ax.set_xlabel('Layer' if idx // 3 == ((len(models)-1)//3) else '')
            ax.set_ylim(0,1.05); ax.grid(True, alpha=0.25)
        for j in range(len(models), len(axes)):
            axes[j].axis('off')
        if len(models): axes[0].set_ylabel('Retention')
        fig.subplots_adjust(hspace=0.30)
        _save(fig, FIG_DIR / out_name)

    _grid_lines(clm_models, 'decoder_tuned_lens', ['tuned_lastvis_retention_by_layer','raw_lastvis_retention_by_layer'], [('tuned_lastvis_retention_by_layer','#1f77b4','-'), ('raw_lastvis_retention_by_layer','#d62728','--')], 'fig04_clm_next_token_readout_smoke.png' if args.smoke else 'fig04_clm_next_token_readout.png')
    _grid_lines(clm_models, 'decoder_tuned_lens', ['tuned_lastvis_retention_by_layer','tuned_lastvis_m1_retention_by_layer'], [('tuned_lastvis_retention_by_layer','#1f77b4','-'), ('tuned_lastvis_m1_retention_by_layer','#2ca02c','-')], 'fig05_clm_shifted_recovery_smoke.png' if args.smoke else 'fig05_clm_shifted_recovery.png')

    colors = sns.color_palette('tab10', len(clm_models))
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4))
    for color, model in zip(colors, clm_models):
        swap = load_metrics(clm_run_dir(model, 'swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')
        dist = load_metrics(clm_run_dir(model, 'distant_swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')
        axes[0].plot([1,3,5,8],[swap['both_swap_positions_restored_rate'], dist['dist3_both_restored_rate'], dist['dist5_both_restored_rate'], dist['dist8_both_restored_rate']], linewidth=2.0, color=color)
    x = np.arange(len(clm_models)) * 1.2; width = 0.16; single_vals=[]; sim_vals=[]; actual_vals=[]; dist8_vals=[]
    for model in clm_models:
        single = load_metrics(clm_run_dir(model, 'single_corrupt_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')['unmasked_restored_rate']
        actual = load_metrics(clm_run_dir(model, 'swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')['both_swap_positions_restored_rate']
        dist8 = load_metrics(clm_run_dir(model, 'distant_swap_repair', 'benchmark' if not args.smoke else 'smoke') / 'metrics.json')['dist8_both_restored_rate']
        single_vals.append(single); sim_vals.append(single**2); actual_vals.append(actual); dist8_vals.append(dist8)
    axes[1].bar(x - 1.5*width, single_vals, width=width, color='#4c72b0'); axes[1].bar(x - 0.5*width, sim_vals, width=width, color='#55a868'); axes[1].bar(x + 0.5*width, actual_vals, width=width, color='#c44e52'); axes[1].bar(x + 1.5*width, dist8_vals, width=width, color='#8172b2')
    axes[0].set_xlabel('Distance between swapped tokens'); axes[0].set_ylabel('Both swapped tokens restored'); axes[0].set_xticks([1,3,5,8]); axes[0].set_ylim(0,0.5); axes[0].grid(True, alpha=0.25)
    axes[1].set_ylabel('Restoration probability'); axes[1].set_xticks(x); axes[1].set_xticklabels([m['display'] for m in clm_models], rotation=35, ha='right'); axes[1].set_ylim(0,0.5); axes[1].grid(True, axis='y', alpha=0.25)
    _save(fig, FIG_DIR / ('fig09_clm_local_repair_mirror_smoke.png' if args.smoke else 'fig09_clm_local_repair_mirror.png'))

    _panel_pca(mlm_models, lambda m: mlm_run_dir(m, 'probe_holdout', 'benchmark' if not args.smoke else 'smoke'), FIG_DIR / ('fig03_mlm_pca_depth_panel_smoke.png' if args.smoke else 'fig03_mlm_pca_depth_panel.png'), 2, 3 if len(mlm_models) > 1 else 1)
    _panel_pca(clm_models, lambda m: clm_run_dir(m, 'probe_holdout', 'benchmark' if not args.smoke else 'smoke'), FIG_DIR / ('fig11_clm_pca_depth_panel_smoke.png' if args.smoke else 'fig11_clm_pca_depth_panel.png'), 3 if len(clm_models) > 3 else 1, 3 if len(clm_models) > 1 else 1)
    print(json.dumps({'output_dir': str(FIG_DIR), 'smoke': args.smoke}, indent=2))


def _panel_pca(models, run_dir_fn, out_path, nrows, ncols):
    fig = plt.figure(figsize=(15.0, 9.0 if nrows > 1 else 5.0))
    cmap = plt.get_cmap('tab20')
    for idx, model in enumerate(models, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
        coords, labels = _pca_points(_cache_dir_for_probe(run_dir_fn(model)))
        for layer in np.unique(labels):
            pts = coords[labels == layer]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4, alpha=0.55, color=cmap(int(layer) % 20))
        ax.set_title(model['display'], fontsize=12, pad=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    total = nrows * ncols
    for idx in range(len(models)+1, total+1):
        ax = fig.add_subplot(nrows, ncols, idx); ax.axis('off')
    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    _save(fig, out_path)


def _cache_dir_for_probe(run_dir: Path) -> Path:
    return run_dir / 'probe_feature_cache_full_v1'


def _pca_points(cache_dir: Path, per_layer: int = 200, seed: int = 42):
    meta = load_cache_metadata(cache_dir)
    hs = np.load(cache_dir / meta['files']['hidden_states'], mmap_mode='r')
    total_tokens = int(meta['total_tokens'])
    num_layers = int(meta['num_layers'])
    rng = np.random.default_rng(seed)
    rows = []; labels = []; base_idx = np.arange(total_tokens); sample_n = min(per_layer, total_tokens)
    for layer_idx in range(num_layers):
        choose = rng.choice(base_idx, size=sample_n, replace=False)
        rows.append(np.asarray(hs[choose, layer_idx, :], dtype=np.float32)); labels.extend([layer_idx] * sample_n)
    X = np.concatenate(rows, axis=0)
    coords = PCA(n_components=3, random_state=seed).fit_transform(X)
    return coords, np.array(labels)


if __name__ == '__main__':
    main()
