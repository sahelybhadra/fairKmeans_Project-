#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm 2: Balanced Greedy Fair K-Means with Swap Optimization
=================================================================

This algorithm improves upon the basic Balanced Greedy (Algorithm 1) by adding
a post-processing swap phase that refines the clustering.

Motivation:
-----------
Greedy clustering can get stuck in local optima due to early assignment errors.
A refinement step helps correct poor initial assignments.

Approach:
---------
Phase 1: Initial Clustering
    - Run Algorithm 1 (Balanced Greedy Fair K-Means)
    - Obtain valid initial clusters satisfying fairness and capacity constraints

Phase 2: Swap Refinement (Iterative)
    Step 1: Identify Swap Candidates
        - Select a point x_i that:
            * Is far from its current cluster center (C_curr)
            * Is closer to another cluster center (C_dest)
    
    Step 2: Find Swap Partner
        - Choose a point x_j from the destination cluster C_dest to swap with x_i
    
    Step 3: Cost Evaluation
        - Compute total clustering distance before the swap
        - Compute total clustering distance after the swap
        - Continue only if new cost is lower than old cost
    
    Step 4: Constraint Validation
        - Ensure the swap preserves:
            * Fairness limits for all sliding windows
            * Note: Capacity checks not needed (1-to-1 swap keeps cluster sizes constant)
    
    Step 5: Execute Swap
        - If all constraints are satisfied:
            * Swap x_i and x_j between clusters
    
    Step 6: Iteration
        - Repeat the swap process for:
            * A fixed number of iterations, or
            * Until no further improvement is possible

Optimizations:
--------------
- Distance matrix pre-computed once per swap pass (vectorized)
- Capacity checks removed (1-to-1 swaps don't change cluster sizes)
- Fairness checks optimized with NumPy arrays

Author: Yasin
Date: February 22, 2026
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from typing import Tuple

from . import balanced_greedy
from . import lloyd
from . import utils


# Algorithm identifier
ID = 'BalancedGreedySwap'


def compute_clustering_cost(X: DataFrame, n_nonsensitive: int, 
                           assignments: Series, centroids: DataFrame) -> float:
    """
    Compute total clustering cost (k-means objective).
    Uses optimized lloyd.compute_objective function.
    
    Parameters
    ----------
    X : DataFrame
        Data points
    n_nonsensitive : int
        Number of non-sensitive features
    assignments : Series
        Cluster assignments
    centroids : DataFrame
        Cluster centroids
    
    Returns
    -------
    float
        Total sum of squared distances (k-means objective)
    """
    return lloyd.compute_objective(X, n_nonsensitive, assignments, centroids)


def check_fairness_after_swap(s: Series, assignments: Series, idx1: int, idx2: int,
                              dataset_count: Series, windows, fairness_limit: Series) -> bool:
    """
    OPTIMIZED: Check if swapping two points preserves fairness constraints.
    Uses pre-computed fairness limits for efficiency.
    
    Note: Capacity check not needed - 1-to-1 swap keeps cluster sizes constant.
    """
    cluster1 = assignments.loc[idx1]
    cluster2 = assignments.loc[idx2]
    
    if cluster1 == cluster2:
        return True  # Same cluster, no effect
    
    # Create temporary assignments with the swap
    temp_assignments = assignments.copy()
    temp_assignments.loc[idx1] = cluster2
    temp_assignments.loc[idx2] = cluster1
    
    # Check fairness for both affected clusters only
    for cluster_id in [cluster1, cluster2]:
        cluster_mask = temp_assignments == cluster_id
        cluster_s = s[cluster_mask]
        
        if len(cluster_s) == 0:
            continue
        
        cluster_count = cluster_s.value_counts(sort=False).reindex(
            dataset_count.index, fill_value=0
        )
        cluster_pooled_count = utils.sumpool(hist=cluster_count, windows=windows)
        cluster_pooled_prop = cluster_pooled_count.divide(len(cluster_s))
        
        # Check if any window violates fairness
        if (cluster_pooled_prop > fairness_limit).any():
            return False
    
    return True


def swap_refinement(X: DataFrame, n_nonsensitive: int, s: Series,
                   assignments: Series, centroids: DataFrame,
                   epsilon: float = 0.1, gamma: float = 1.1, 
                   window_size: float = 3,
                   max_swap_iterations: int = 100,
                   verbose: bool = True) -> Tuple[Series, DataFrame, dict]:
    """
    OPTIMIZED: Refine clustering by swapping points between clusters.
    
    Optimizations:
    - Distance matrix computed ONCE per pass (vectorized)
    - Capacity checks removed (1-to-1 swaps don't change sizes)
    - Fairness checks use fast NumPy operations
    
    Parameters
    ----------
    X : DataFrame
        Data points
    n_nonsensitive : int
        Number of non-sensitive features
    s : Series
        Sensitive attribute
    assignments : Series
        Initial cluster assignments (from Algorithm 1)
    centroids : DataFrame
        Initial cluster centroids
    epsilon : float
        Fairness tolerance
    gamma : float
        Capacity tolerance (kept for API compatibility, not used in swaps)
    window_size : float
        Window size for fairness checking
    max_swap_iterations : int
        Maximum number of swap passes
    verbose : bool
        Print progress
    
    Returns
    -------
    Tuple[Series, DataFrame, dict]
        Refined assignments, updated centroids, and refinement info
    """
    if verbose:
        print("\n" + "="*60)
        print("PHASE 2: SWAP REFINEMENT (Optimized)")
        print("="*60)
    
    # Prepare fairness checking structures (COMPUTE ONCE!)
    dataset_count = s.value_counts(sort=False).sort_index()
    windows = utils.generate_windows(index=dataset_count.index, window_size=window_size)
    dataset_pooled_count = utils.sumpool(hist=dataset_count, windows=windows)
    dataset_pooled_prop = dataset_pooled_count.divide(dataset_pooled_count.sum())
    fairness_limit = dataset_pooled_prop + epsilon
    
    # Current state
    current_assignments = assignments.copy()
    current_centroids = centroids.copy()
    initial_cost = lloyd.compute_objective(X, n_nonsensitive, current_assignments, current_centroids)
    current_cost = initial_cost
    
    if verbose:
        print(f"Initial cost: {initial_cost:.5e}")
        print(f"Max swap passes: {max_swap_iterations}")
    
    n_swaps = 0
    
    # Swap refinement loop
    for pass_num in range(max_swap_iterations):
        swaps_this_pass = 0
        candidates_checked = 0
        improvements_found = 0
        fairness_violations = 0
        
        # Pre-compute distance matrix ONCE per pass (KEY OPTIMIZATION!)
        distances = utils.compute_distances(X, current_centroids)
        
        # Try to find beneficial swaps
        for idx1 in X.index:
            c1 = current_assignments.loc[idx1]
            dist_to_c1 = distances.loc[idx1, c1]
            
            # Find closest different cluster
            other_clusters = distances.loc[idx1].drop(c1)
            if len(other_clusters) == 0:
                continue
                
            c2 = other_clusters.idxmin()
            dist_to_c2 = distances.loc[idx1, c2]
            
            # If moving to c2 doesn't help idx1, skip
            if dist_to_c2 >= dist_to_c1:
                continue
            
            # Find best swap partner in c2 (check only top candidates for speed)
            c2_points = current_assignments[current_assignments == c2].index
            MAX_CANDIDATES_TO_CHECK = min(50, len(c2_points))  # Limit fairness checks
            
            # Sort c2 points by distance to c1 (best candidates want to move to c1)
            c2_distances_to_c1 = distances.loc[c2_points, c1].sort_values(ascending=True)
            top_candidates = c2_distances_to_c1.head(MAX_CANDIDATES_TO_CHECK).index
            
            best_partner = None
            best_improvement = 0
            
            for idx2 in top_candidates:
                candidates_checked += 1
                dist_to_c2_for_idx2 = distances.loc[idx2, c2]
                dist_to_c1_for_idx2 = distances.loc[idx2, c1]
                
                # Calculate mutual benefit
                cost_before = dist_to_c1 + dist_to_c2_for_idx2
                cost_after = dist_to_c2 + dist_to_c1_for_idx2
                improvement = cost_before - cost_after
                
                if improvement > best_improvement:
                    improvements_found += 1
                    # Check fairness constraints
                    if check_fairness_after_swap(s, current_assignments, idx1, idx2, 
                                                 dataset_count, windows, fairness_limit):
                        best_improvement = improvement
                        best_partner = idx2
                    else:
                        fairness_violations += 1
            
            # Execute best swap for this point
            if best_partner is not None:
                current_assignments.loc[idx1] = c2
                current_assignments.loc[best_partner] = c1
                swaps_this_pass += 1
                n_swaps += 1
                
                # Update centroids for affected clusters
                current_centroids.loc[c1] = X[current_assignments == c1].mean()
                current_centroids.loc[c2] = X[current_assignments == c2].mean()
                
                # Recompute distances since centroids changed
                distances = utils.compute_distances(X, current_centroids)
        
            print(f"          Candidates checked: {candidates_checked}, Improvements found: {improvements_found}, Fairness violations: {fairness_violations}")
        if verbose:
            current_cost = lloyd.compute_objective(X, n_nonsensitive, current_assignments, current_centroids)
            improvement_pct = 100 * (initial_cost - current_cost) / initial_cost
            print(f"Pass {pass_num+1:2d}: {swaps_this_pass:3d} swaps, cost={current_cost:.5e} (-{improvement_pct:.3f}%)")
        
        # Stop if no swaps in this pass
        if swaps_this_pass == 0:
            if verbose:
                print(f"\nNo beneficial swaps found. Stopping early.")
            break
    
    final_cost = lloyd.compute_objective(X, n_nonsensitive, current_assignments, current_centroids)
    total_improvement = initial_cost - final_cost
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Swap refinement completed")
        print(f"Total swaps: {n_swaps}")
        print(f"Initial cost: {initial_cost:.5e}")
        print(f"Final cost: {final_cost:.5e}")
        print(f"Total improvement: {total_improvement:.5e} ({100*total_improvement/initial_cost:.3f}%)")
        print(f"{'='*60}")
    
    refinement_info = {
        'n_swaps': n_swaps,
        'n_iterations': pass_num + 1,
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'improvement': total_improvement
    }
    
    return current_assignments, current_centroids, refinement_info


def run(X, n_nonsensitive, s, n_clusters, init_centroids, 
        dataset_name, init_method, random_state,
        epsilon=0.1, gamma=1.1, window_size=3, 
        max_iter=100, max_swap_iterations=100,
        verbose=True, save=False):
    """
    Run Algorithm 2: Balanced Greedy Fair K-Means with Swap Optimization.
    
    This is a two-phase algorithm:
    Phase 1: Initial clustering using Algorithm 1 (Balanced Greedy)
    Phase 2: Iterative swap refinement to improve clustering quality
    
    Parameters
    ----------
    X : DataFrame
        Dataset features (n_samples x n_features)
    n_nonsensitive : int
        Number of non-sensitive features
    s : Series
        Sensitive attribute (age or race)
    n_clusters : int
        Number of clusters (k)
    init_centroids : DataFrame
        Initial centroid positions
    dataset_name : str
        Name of dataset (for logging)
    init_method : str
        Initialization method used
    random_state : int
        Random seed
    epsilon : float
        Fairness tolerance (default: 0.1 = 10% deviation allowed)
    gamma : float
        Capacity tolerance (default: 1.1 = clusters can be 110% of average size)
    window_size : float
        Sliding window size for fairness checking
    max_iter : int
        Maximum iterations for Phase 1
    max_swap_iterations : int
        Maximum swap iterations for Phase 2
    verbose : bool
        Print progress
    save : bool
        Save results (not implemented)
    
    Returns
    -------
    Tuple[Series, DataFrame, dict, DataFrame]
        - assignments : Series with cluster labels
        - centroids : DataFrame with final centroid positions
        - info : dict with algorithm metadata
        - stats : DataFrame with empty stats (for compatibility)
    """
    if verbose:
        print("\n" + "="*60)
        print(f"Running {ID}")
        print("(Algorithm 2: Fair K-Means with Swap Optimization)")
        print(f"Dataset: {dataset_name}, k={n_clusters}")
        print(f"Parameters: epsilon={epsilon}, gamma={gamma}, window_size={window_size}")
        print(f"Max iter: {max_iter}, Max swap iter: {max_swap_iterations}")
        print("="*60)
    
    # PHASE 1: Initial clustering with Algorithm 1
    if verbose:
        print("\n" + "="*60)
        print("PHASE 1: INITIAL CLUSTERING (Balanced Greedy)")
        print("="*60)
    
    initial_assignments, initial_centroids, phase1_info, phase1_stats = balanced_greedy.run(
        X=X,
        n_nonsensitive=n_nonsensitive,
        s=s,
        n_clusters=n_clusters,
        init_centroids=init_centroids,
        dataset_name=dataset_name,
        init_method=init_method,
        random_state=random_state,
        epsilon=epsilon,
        gamma=gamma,
        window_size=window_size,
        max_iter=max_iter,
        verbose=verbose,
        save=False
    )
    
    # PHASE 2: Swap refinement
    final_assignments, final_centroids, refinement_info = swap_refinement(
        X=X,
        n_nonsensitive=n_nonsensitive,
        s=s,
        assignments=initial_assignments,
        centroids=initial_centroids,
        epsilon=epsilon,
        gamma=gamma,
        window_size=window_size,
        max_swap_iterations=max_swap_iterations,
        verbose=verbose
    )
    
    # Combine info from both phases
    info = {
        'algorithm': ID,
        'dataset': dataset_name,
        'n_clusters': n_clusters,
        'epsilon': epsilon,
        'gamma': gamma,
        'window_size': window_size,
        'phase1_iterations': phase1_info['n_iterations'],
        'phase1_converged': phase1_info['converged'],
        'phase2_swaps': refinement_info['n_swaps'],
        'phase2_iterations': refinement_info['n_iterations'],
        'initial_cost': refinement_info['initial_cost'],
        'final_cost': refinement_info['final_cost'],
        'cost_improvement': refinement_info['improvement']
    }
    
    # Empty stats for compatibility
    stats = pd.DataFrame()
    
    if verbose:
        print(f"\n{'='*60}")
        print("ALGORITHM 2 COMPLETED")
        print(f"Phase 1: {phase1_info['n_iterations']} iterations")
        print(f"Phase 2: {refinement_info['n_swaps']} swaps in {refinement_info['n_iterations']} iterations")
        print(f"Final cost: {refinement_info['final_cost']:.5e}")
        print(f"Total improvement: {refinement_info['improvement']:.5e}")
        print(f"{'='*60}")
    
    return final_assignments, final_centroids, info, stats
