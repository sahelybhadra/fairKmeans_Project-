#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Analysis: Effect of Lambda on Fairness-Utility Trade-off
========================================================================
Stanley's algorithm (fairkmeans_prc) — varying lambda from 0.1 to 0.9.

Fixed:
  window_size = 3
  k per dataset: crime=4, compas=6, adult=6, motor=4
  5 runs per (dataset, lambda) combination

Output:
  lambda_analysis_summary.txt   — per-lambda averages
  lambda_tradeoff.png           — 2x2 Fairness-Utility plots
"""

# Fix Windows encoding issues
import sys
import io

from src import init, metrics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from importlib import import_module
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src import init
from src.algorithms import fairkmeans_prc

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS = ['crime', 'compas', 'adult', 'motor']
CLUSTERS_CONFIG = {
    'crime':  4,
    'compas': 6,
    'adult':  6,
    'motor':  4,
}
LAMBDAS= {
    'crime':  0.4,
    'compas': 0.5,
    'adult':  0.3,
    'motor':  0.5,
}
#LAMBDAS = 0.5 #[round(v, 1) for v in np.arange(0.1, 1.0, 0.1)]   # 0.1 … 0.9
WINDOW_SIZE = [1,2,3,5,10, 15, 20]
N_RUNS = 5
INIT_METHOD = 'kmeans_plusplus'
MAX_ITER = 100

SUMMARY_FILE = open('window_analysis_summary_emd.txt', 'w', encoding='utf-8')


def log(msg):
    print(msg)
    print(msg, file=SUMMARY_FILE)
    SUMMARY_FILE.flush()


# ── Per-dataset experiment ────────────────────────────────────────────────────
def run_dataset(dataset_name, n_clusters,lam):
    log("\n" + "=" * 80)
    log(f"DATASET: {dataset_name.upper()}   k={n_clusters}   Lambda{lam}")
    log("=" * 80)

    # Load data once
    dataset_module = import_module(f"src.datasets.{dataset_name}")
    _, X, s, n_nonsensitive = dataset_module.load()
    log(f"  Loaded {len(X)} points")

    # Results indexed by lambda
    results = {}   # lambda_ -> {'utility': [...], 'fairness': [...]}

    for w in WINDOW_SIZE:
        log(f"\n  window_size = {w}")
        utility_runs = []
        fairness_runs = []
        fairness3_runs = []
        emd_runs = []

        for run_idx in range(N_RUNS):
            log(f"    run {run_idx + 1}/{N_RUNS} ...", )
            t0 = time.time()

            init_centroids = init.load(
                dataset_name=dataset_name,
                n_clusters=n_clusters,
                init_method=INIT_METHOD,
                random_state=run_idx
            )

            c, centroids, info, stats = fairkmeans_prc.run(
                X=X,
                n_nonsensitive=n_nonsensitive,
                s=s,
                n_clusters=n_clusters,
                init_centroids=init_centroids,
                dataset_name=dataset_name,
                init_method=INIT_METHOD,
                random_state=run_idx,
                window_size=w,
                lambda_=lam,
                max_iter=MAX_ITER,
                export=False
            )
            scores = metrics.evaluate(X=X, n_nonsensitive=n_nonsensitive, s=s, c=c, centroids=centroids,n_clusters=n_clusters, window_size=3)
            # info has 'utility loss' and 'fairness loss' as columns
            u = float(info['utility loss'].iloc[0])
            f = float(info['fairness loss'].iloc[0])
            f3= float(scores['pooling window loss (size=3)'])
            emd = float(scores['max emd'])
            utility_runs.append(u)
            fairness_runs.append(f)
            fairness3_runs.append(f3)
            emd_runs.append(emd)
            elapsed = time.time() - t0
            log(f"      utility={u:.5f}     fairness={f:.5f}  fairness3={f3:.5f}  emd={emd:.5f}     ({elapsed:.1f}s)")

        results[w] = {
            'utility_mean': np.mean(utility_runs),
            'utility_std':  np.std(utility_runs),
            'fairness_mean': np.mean(fairness_runs),
            'fairness_std':  np.std(fairness_runs),
            'fairness3_mean': np.mean(fairness3_runs),
            'fairness3_std':  np.std(fairness3_runs),
            'emd_mean': np.mean(emd_runs),
            'emd_std':  np.std(emd_runs),
        }
        log(f"    → avg utility={results[w]['utility_mean']:.5f} ± {results[w]['utility_std']:.5f}"
            f"   avg fairness={results[w]['fairness_mean']:.5f} ± {results[w]['fairness_std']:.5f}"
            f"   avg fairness3={results[w]['fairness3_mean']:.5f} ± {results[w]['fairness3_std']:.5f}"
            f"   avg emd={results[w]['emd_mean']:.5f} ± {results[w]['emd_std']:.5f}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_tradeoff(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fairness-Utility Trade-off in Stanley's Algorithm\n"
                 "(k and lambda fixed, window_size varied)",
                 fontsize=14, fontweight='bold')

    for ax, dataset_name in zip(axes.flat, DATASETS):
        res = all_results[dataset_name]
        lambdas = sorted(res.keys())

        u_mean = np.array([res[l]['utility_mean']  for l in lambdas])
        u_std  = np.array([res[l]['utility_std']   for l in lambdas])
        f_mean = np.array([res[l]['fairness_mean'] for l in lambdas])
        f_std  = np.array([res[l]['fairness_std']  for l in lambdas])

        ax2 = ax.twinx()

        # K-Means (utility) loss — left axis, blue
        ax.plot(lambdas, u_mean, 'b-o', label='K-Means Loss (Utility)', linewidth=2)
        ax.fill_between(lambdas, u_mean - u_std, u_mean + u_std,
                        color='blue', alpha=0.15)
        ax.set_ylabel('K-Means Loss (Utility)', color='blue', fontsize=11)
        ax.tick_params(axis='y', labelcolor='blue')

        # Fairness loss — right axis, red dashed
        ax2.plot(lambdas, f_mean, 'r--x', label='Fairness Loss', linewidth=2)
        ax2.fill_between(lambdas, f_mean - f_std, f_mean + f_std,
                         color='red', alpha=0.12)
        ax2.set_ylabel('Fairness Loss', color='red', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title(f'{dataset_name.upper()}  (k={CLUSTERS_CONFIG[dataset_name]})',
                     fontsize=12)
        ax.set_xlabel('The window size ', fontsize=11)
        ax.set_xticks(lambdas)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'window_tradeoff.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    log(f"\nPlot saved to: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    start = time.time()
    log(f"\n{'='*80}")
    log("  STANLEY'S ALGORITHM — window size HYPERPARAMETER ANALYSIS")
    log(f"{'='*80}")
    log(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    #log(f"Lambdas: {LAMBDAS}")
    log(f"Runs per setting: {N_RUNS}")
    log(f"Datasets: {DATASETS}")

    all_results = {}
    for dataset_name in DATASETS:
        n_clusters = CLUSTERS_CONFIG[dataset_name]
        lam = LAMBDAS[dataset_name]
        all_results[dataset_name] = run_dataset(dataset_name, n_clusters,lam)
        df = pd.DataFrame.from_dict(all_results, orient="index")
        df.to_csv("WindowLengthAnalysis.csv")
        
    # Summary table
    log("\n\n" + "=" * 80)
    log("SUMMARY TABLE (mean ± std over 5 runs)")
    log("=" * 80)
    for dataset_name in DATASETS:
        log(f"\n{dataset_name.upper()}:")
        log(f"  {'WINDOW SIZE':>8}  {'Utility Mean':>14}  {'Utility Std':>12}  "
            f"{'Fairness Mean':>14}  {'Fairness Std':>12}")
        log(f"  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*12}")
        for w in sorted(all_results[dataset_name].keys()):
            r = all_results[dataset_name]
            log(f"  {w :> 8.1f}  {r['utility_mean']:>14.6f}  {r['utility_std']:>12.6f}  "
                f"{r['fairness_mean']:>14.6f}  {r['fairness_std']:>12.6f}")
    df = pd.DataFrame.from_dict(all_results, orient="index")
    df.to_csv("WindowLengthAnalysis.csv")
    plot_tradeoff(all_results)

    elapsed = time.time() - start
    log(f"\nTotal time: {elapsed:.1f}s  ({elapsed/3600:.2f}h)")
    log(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    SUMMARY_FILE.close()
