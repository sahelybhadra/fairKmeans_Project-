#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced Greedy with Swap Optimization Benchmarking Script
===========================================================
Runs Algorithm 2 (Balanced Greedy + Swap) 5 times on each dataset.
Compares improvement over Algorithm 1 (Balanced Greedy alone).

Configuration:
- Crime: k=4, window_size=3
- Compas: k=6, window_size=3
- Adult: k=6, window_size=3
"""

# Fix Windows encoding issues
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from importlib import import_module
from pandas import DataFrame, Series
from tabulate import tabulate
import time

from src import init, metrics
from src.algorithms import balanced_greedy_swap
from src.timer import Timer

# Create log files for output
SUMMARY_FILE = open('balanced_greedy_swap_summary.txt', 'w', encoding='utf-8')
ITERATIONS_FILE = open('balanced_greedy_swap_iterations.txt', 'w', encoding='utf-8')

def log_print(message, log_file=None, **kwargs):
    """Print to console and optionally to log file"""
    print(message, **kwargs)
    if log_file:
        print(message, file=log_file, **kwargs)
        log_file.flush()

def log_both(message, **kwargs):
    """Print to both summary and iterations files"""
    log_print(message, SUMMARY_FILE, **kwargs)
    log_print(message, ITERATIONS_FILE, **kwargs)

def log_summary_only(message, **kwargs):
    """Print to summary file only"""
    log_print(message, SUMMARY_FILE, **kwargs)

def log_iterations_only(message, **kwargs):
    """Print to iterations file only"""
    log_print(message, ITERATIONS_FILE, **kwargs)

# %% Configuration
#DATASETS = ['crime', 'compas', 'adult']
DATASETS = [ 'adult']
CLUSTERS_CONFIG = {
    'adult': 6,
    'crime': 4,
    'compas': 6
}
N_RUNS = 1  # Running 5 times for proper averaging
INIT_METHOD = 'kmeans_plusplus'
EPSILON = 0.1
GAMMA = 1.1
WINDOW_SIZE = 3
MAX_ITER = 100
MAX_SWAP_ITERATIONS = 100

def benchmark_dataset(dataset_name, n_clusters):
    """Benchmark a single dataset."""
    
    log_both("\n" + "="*80)
    log_both(f"BENCHMARKING: {dataset_name.upper()} Dataset (k={n_clusters})")
    log_both("="*80)
    log_both(f"Parameters: k={n_clusters}, epsilon={EPSILON}, gamma={GAMMA}, window_size={WINDOW_SIZE}")
    log_both(f"Max iterations: {MAX_ITER}, Max swap iterations: {MAX_SWAP_ITERATIONS}")
    
    # Load dataset
    log_both(f"\n[1/3] Loading {dataset_name} dataset...")
    dataset_module = import_module(f"src.datasets.{dataset_name}")
    df, X, s, n_nonsensitive = dataset_module.load()
    log_both(f"  [OK] Loaded: {len(X)} points")
    
    all_run_results = []
    all_run_times = []
    all_phase1_costs = []
    all_phase2_costs = []
    all_improvements = []
    all_n_swaps = []
    
    log_both(f"\n[2/3] Running algorithm {N_RUNS} times...")
    for run_idx in range(N_RUNS):
        run_start = time.time()
        
        log_both(f"\n  {'-'*60}")
        log_both(f"  Run {run_idx + 1}/{N_RUNS}")
        log_both(f"  {'-'*60}")
        
        # Load centroids
        random_state = run_idx
        init_centroids = init.load(
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            init_method=INIT_METHOD,
            random_state=random_state
        )
        
        # Run algorithm
        timer = Timer()
        timer.resume()
        
        assignments, final_centroids, info, stats = balanced_greedy_swap.run(
            X=X,
            n_nonsensitive=n_nonsensitive,
            s=s,
            n_clusters=n_clusters,
            init_centroids=init_centroids,
            dataset_name=dataset_name,
            init_method=INIT_METHOD,
            random_state=random_state,
            epsilon=EPSILON,
            gamma=GAMMA,
            window_size=WINDOW_SIZE,
            max_iter=MAX_ITER,
            max_swap_iterations=MAX_SWAP_ITERATIONS,
            verbose=True
        )
        df = pd.DataFrame({
            's': s,
            'c': assignments
        })

        # Save to CSV
        df.to_csv('output.csv', index=False)
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        c=assignments
        # Plot the overall dataset distribution (dashed black)
        sns.kdeplot(data=s, fill=False, color="black", linestyle="--", label="dataset")  # data1 as in data 2 we dont have age column. Also it will not affect as it drawing for whole dataset

        # Plot each cluster's age distribution
        clusters = sorted(c.unique())
        palette = sns.color_palette("tab10", len(clusters))

        for i, j in enumerate(clusters):
            subset = s[c == j]  # here for which points we will need original dataset or data1
            sns.kdeplot(subset, fill=False, color=palette[i], label=f"cluster_{i} ({len(subset)} objects)")

        plt.xlabel("age")
        plt.ylabel("proportion")
        plt.title("Age Distribution : Clusters vs Full Dataset")
        plt.legend()
        plt.tight_layout()
#plt.show()
        plt.savefig("./result/motor/Plot_bera_K4L5_1.png")
        plt.close()
        timer.pause()
        execution_time = timer.elapsed / 1e9  # Convert nanoseconds to seconds
        
        # Evaluate metrics
        scores = metrics.evaluate(
            X=X,
            n_nonsensitive=n_nonsensitive,
            s=s,
            c=assignments,
            centroids=final_centroids,
            n_clusters=n_clusters,
            window_size=WINDOW_SIZE
        )
        
        # Log individual run metrics to iterations file
        log_iterations_only(f"\n  {'='*60}")
        log_iterations_only(f"  METRICS - {dataset_name.upper()} Run {run_idx + 1}/{N_RUNS}")
        log_iterations_only(f"  {'='*60}")
        scores_with_time = scores.copy()
        scores_with_time['Time'] = execution_time
        scores_with_time['Phase1_Cost'] = info['initial_cost']
        scores_with_time['Phase2_Cost'] = info['final_cost']
        scores_with_time['Cost_Improvement'] = info['cost_improvement']
        scores_with_time['N_Swaps'] = info['phase2_swaps']
        log_iterations_only(tabulate(scores_with_time.to_frame(), headers=['Metric', 'Value'], tablefmt='presto'))
        
        all_run_results.append(scores)
        all_run_times.append(execution_time)
        all_phase1_costs.append(info['initial_cost'])
        all_phase2_costs.append(info['final_cost'])
        all_improvements.append(info['cost_improvement'])
        all_n_swaps.append(info['phase2_swaps'])
        
        run_elapsed = time.time() - run_start
        log_both(f"\n  [OK] Run {run_idx + 1} completed in {execution_time:.2f} seconds")
        log_both(f"  [OK] Phase 1: {info['phase1_iterations']} iterations, converged: {info['phase1_converged']}")
        log_both(f"  [OK] Phase 2: {info['phase2_swaps']} swaps in {info['phase2_iterations']} iterations")
        log_both(f"  [OK] Cost improvement: {info['cost_improvement']:.5e}")
    
    # Compute statistics
    log_both(f"\n[3/3] Computing statistics...")
    results_df = pd.DataFrame(all_run_results)
    results_df['Time'] = all_run_times
    results_df['Phase1_Cost'] = all_phase1_costs
    results_df['Phase2_Cost'] = all_phase2_costs
    results_df['Cost_Improvement'] = all_improvements
    results_df['N_Swaps'] = all_n_swaps
    
    mean_scores = results_df.mean()
    std_scores = results_df.std()
    
    # Format results
    formatted_results = {}
    for metric in mean_scores.index:
        mean_val = mean_scores[metric]
        std_val = std_scores[metric]
        
        if metric in ['Time', 'N_Swaps']:
            formatted_results[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
        elif abs(mean_val) < 0.01 or abs(mean_val) > 1000:
            formatted_results[metric] = f"{mean_val:.3e} ± {std_val:.3e}"
        else:
            formatted_results[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
    
    # Display results
    log_both("\n" + "="*80)
    log_both(f"RESULTS for {dataset_name.upper()} (Average of {N_RUNS} runs)")
    log_both("="*80)
    
    result_table = pd.DataFrame({
        'Metric': list(formatted_results.keys()),
        'Balanced Greedy + Swap': list(formatted_results.values())
    })
    
    log_both(tabulate(result_table, headers='keys', tablefmt='grid', showindex=False))
    
    return mean_scores, std_scores, formatted_results

# %% Main
if __name__ == "__main__":
    start_time = time.time()
    
    log_both(f"\n{'='*80}")
    log_both(f" ")
    log_both(f"   BALANCED GREEDY + SWAP OPTIMIZATION BENCHMARKING")
    log_both(f" ")
    log_both(f"{'='*80}")
    log_both(f"\nStarted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_both(f"Running {N_RUNS} iterations per dataset")
    log_both(f"Datasets: {', '.join(DATASETS)}")
    log_both(f"\nConfiguration:")
    for dataset in DATASETS:
        k = CLUSTERS_CONFIG[dataset]
        log_both(f"  {dataset}: k={k}, epsilon={EPSILON}, gamma={GAMMA}, window_size={WINDOW_SIZE}")
    log_both(f"  Max iterations: Phase 1={MAX_ITER}, Phase 2={MAX_SWAP_ITERATIONS}")
    
    all_results = {}
    
    for dataset_name in DATASETS:
        n_clusters = CLUSTERS_CONFIG[dataset_name]
        
        try:
            mean_scores, std_scores, formatted = benchmark_dataset(dataset_name, n_clusters)
            all_results[dataset_name] = {
                'mean': mean_scores,
                'std': std_scores,
                'formatted': formatted
            }
        except Exception as e:
            log_both(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    total_time = time.time() - start_time
    
    log_both(f"\n{'='*80}")
    log_both(f" ")
    log_both(f"   FINAL SUMMARY")
    log_both(f" ")
    log_both(f"{'='*80}")
    
    if 'crime' in all_results:
        log_both(f"\n{'='*80}")
        log_both("TABLE 1: CRIME DATASET (k=4)")
        log_both('='*80)
        crime_formatted = all_results['crime']['formatted']
        for metric, value in crime_formatted.items():
            log_both(f"{metric:.<55} {value}")
    
    if 'compas' in all_results:
        log_both(f"\n{'='*80}")
        log_both("TABLE 2: COMPAS DATASET (k=6)")
        log_both('='*80)
        compas_formatted = all_results['compas']['formatted']
        for metric, value in compas_formatted.items():
            log_both(f"{metric:.<55} {value}")
    
    if 'adult' in all_results:
        log_both(f"\n{'='*80}")
        log_both("TABLE 3: ADULT DATASET (k=6)")
        log_both('='*80)
        adult_formatted = all_results['adult']['formatted']
        for metric, value in adult_formatted.items():
            log_both(f"{metric:.<55} {value}")
    
    log_both(f"\n{'='*80}")
    log_both(f"   BENCHMARKING COMPLETE!")
    log_both(f"{'='*80}")
    log_both(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    log_both(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_both(f"\nResults saved to:")
    log_both(f"  - balanced_greedy_swap_summary.txt (overall results)")
    log_both(f"  - balanced_greedy_swap_iterations.txt (iteration-by-iteration details)")
    log_both("\nCopy these results to add to your comparison tables!")
    log_both(f"These values are averages over {N_RUNS} runs.\n")
    
    # Close log files
    SUMMARY_FILE.close()
    ITERATIONS_FILE.close()
