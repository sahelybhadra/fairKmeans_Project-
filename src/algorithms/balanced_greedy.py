#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm 1: Balanced Greedy Fair K-Means

Implements fair clustering using an Expectation-Maximization framework where
fairness is enforced through sliding windows on continuous sensitive attributes
and capacity constraints ensure balanced cluster sizes.

Attributes
----------
ID : str
    Algorithm identifier.
EPSILON : float
    Fairness tolerance (alpha). Maximum deviation from global distribution.
GAMMA : float
    Capacity tolerance. Maximum cluster size relative to n/k.
WINDOW_SIZE : int
    Size of sliding windows for continuous sensitive attribute.
MAX_ITER : int
    Maximum number of iterations.

Routine Listings
----------------
generate_sliding_windows()
    Create sliding windows over unique values of sensitive attribute.
compute_fairness_limits()
    Compute maximum allowed ratio for each window.
check_if_adding_point_is_fair()
    Verify if adding a point violates fairness constraints.
assign_all_points()
    E-step: Greedy assignment with fairness and capacity constraints.
update_centroids()
    M-step: Update cluster centers based on current assignments.
run()
    Main routine to execute the Balanced Greedy algorithm.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from . import lloyd, utils
from ..timer import Timer


# %% Algorithm identifier and default parameters
ID = 'BalancedGreedy'
EPSILON = 0.1         # fairness tolerance (alpha)
GAMMA = 1.1           # capacity tolerance
WINDOW_SIZE = 3       # sliding window size
MAX_ITER = 100        # maximum iterations


# %% Generate sliding windows
def generate_sliding_windows(s, window_size):
    """
    Create sliding windows over unique values of sensitive attribute.
    
    Example: ages [20, 21, 22, 23, 24] with window_size=3 creates
    windows [20-22], [21-23], [22-24].
    
    Parameters
    ----------
    s : Series
        Sensitive attribute values
    window_size : int
        Number of consecutive values per window
    
    Returns
    -------
    IntervalIndex
        Sliding windows as intervals
    """
    # Get all unique age values and sort them
    unique_ages = np.sort(s.unique())
    
    # If we don't have enough unique ages for a window
    if len(unique_ages) < window_size:
        # Just create one big window from min to max age
        intervals = pd.IntervalIndex.from_tuples(
            [(unique_ages.min(), unique_ages.max())], 
            closed='both'
        )
    else:
        # Create sliding windows
        windows = []
        for i in range(len(unique_ages) - window_size + 1):
            left = unique_ages[i]
            right = unique_ages[i + window_size - 1]
            windows.append((left, right))
        
        intervals = pd.IntervalIndex.from_tuples(windows, closed='both')
    
    return intervals


# %% Compute fairness limits
def compute_fairness_limits(s, windows, epsilon):
    """
    Compute maximum allowed ratio of points for each window.
    
    Formula: limit = (global_ratio + epsilon)
    
    If 10% of all points fall in window [20-22], then each cluster
    should have at most (10% + epsilon) points from [20-22].
    
    Parameters
    ----------
    s : Series
        Sensitive attribute values
    windows : IntervalIndex
        Sliding windows
    epsilon : float
        Fairness tolerance
    
    Returns
    -------
    Series
        Fairness limit for each window
    """
    n_total = len(s)
    limits = {}
    
    for window in windows:
        # Count how many people fall in this age window
        in_window = s.apply(lambda x: window.left <= x <= window.right)
        
        # Calculate global ratio for this window
        global_ratio = in_window.sum() / n_total
        
        # Fairness limit = global ratio + epsilon
        limits[window] = global_ratio + epsilon
    
    return Series(limits)


# %% Check fairness constraint
def check_if_adding_point_is_fair(point_age, current_cluster_ages, windows, limits):
    """
    Check if adding a point to a cluster violates fairness constraints.
    
    For each window, verifies that the ratio after adding the point
    would not exceed the fairness limit.
    
    Parameters
    ----------
    point_age : float
        Age of the point to add
    current_cluster_ages : Series
        Ages of points already in the cluster
    windows : IntervalIndex
        Sliding windows
    limits : Series
        Fairness limits for each window
    
    Returns
    -------
    bool
        True if adding point maintains fairness, False otherwise
    """
    # If cluster is empty, always fair to add
    if len(current_cluster_ages) == 0:
        return True
    
    # Temporarily add the new point - use numpy array for speed
    if len(current_cluster_ages) > 0:
        temp_ages = np.append(current_cluster_ages.values, point_age)
    else:
        temp_ages = np.array([point_age])
    
    cluster_size = len(temp_ages)
    
    # Check each window - use vectorized operations
    for window, limit in limits.items():
        # Count how many points in cluster fall in this window (vectorized)
        in_window = (temp_ages >= window.left) & (temp_ages <= window.right)
        cluster_ratio = in_window.sum() / cluster_size
        
        # If ratio exceeds limit, it's unfair
        if cluster_ratio > limit:
            return False
    
    return True


# %% E-STEP: Assignment with constraints
def assign_all_points(X, s, centroids, n_clusters, epsilon, gamma, window_size, random_state=None):
    """
    E-STEP: Greedy assignment with fairness and capacity constraints.
    
    This is the "Expectation" step in the EM framework. Each point is
    assigned to the nearest cluster that satisfies both capacity and
    fairness constraints. Points are processed in random order to ensure
    fairness and avoid order bias.
    
    Algorithm:
    1. Shuffle all points (prevents bias toward early points)
    2. For each point (in random order):
       a. Sort clusters by distance (nearest first)
       b. Try to assign to nearest cluster satisfying:
          - Capacity constraint: cluster size < (n/k) × gamma
          - Fairness constraint: window ratios within epsilon
       c. Fallback: If all clusters violate constraints, assign to emptiest
    
    Parameters
    ----------
    X : DataFrame
        Dataset features
    s : Series
        Sensitive attribute (e.g., age)
    centroids : DataFrame
        Current cluster centers
    n_clusters : int
        Number of clusters (k)
    epsilon : float
        Fairness tolerance
    gamma : float
        Capacity tolerance
    window_size : int
        Sliding window size
    random_state : int, optional
        Random seed for shuffling
    
    Returns
    -------
    Series
        Cluster assignment for each point
    """
    n_points = len(X)
    
    # Step 1: Compute distance from each point to all centroids
    distances = utils.compute_distances(X=X, centroids=centroids)
    
    # Step 2: Generate sliding windows
    windows = generate_sliding_windows(s, window_size)
    
    # Step 3: Compute fairness limits
    limits = compute_fairness_limits(s, windows, epsilon)
    
    # Step 4: Calculate maximum capacity per cluster
    # Formula: (total_points / k) * gamma
    # Example: 1000 points, 5 clusters, gamma=1.1 -> max = 220 points per cluster
    max_capacity = int(np.ceil((n_points / n_clusters) * gamma))
    
    # Step 5: Initialize tracking structures
    assignments = Series(index=X.index, dtype=int, name='cluster')
    cluster_counts = Series(0, index=range(n_clusters))
    
    # Step 6: CRITICAL - Shuffle points to avoid order bias
    # Without shuffling, the first points always get the "best" spots
    # Shuffling ensures fairness in the process itself
    if random_state is not None:
        shuffled_indices = X.index.to_series().sample(frac=1, random_state=random_state).values
    else:
        shuffled_indices = X.index.to_series().sample(frac=1).values
    
    # Step 7: GREEDY ASSIGNMENT - Process each point
    for idx in shuffled_indices:
        point_age = s.loc[idx]
        
        # Get distances to all clusters, sorted nearest first
        point_distances = distances.loc[idx].sort_values()
        
        assigned = False
        
        # Try clusters in order of increasing distance (greedy choice)
        for cluster_id in point_distances.index:
            
            # Constraint Check 1: Capacity
            if cluster_counts.loc[cluster_id] >= max_capacity:
                continue  # Cluster is full, try next
            
            # Constraint Check 2: Fairness
            # Get all ages currently in this cluster
            mask = assignments == cluster_id
            current_ages = s[mask] if mask.any() else Series(dtype=float)
            
            # Check if adding this point would violate fairness
            # (check_if_adding_point_is_fair returns True for empty clusters)
            if check_if_adding_point_is_fair(point_age, current_ages, windows, limits):
                # Both constraints satisfied! Make the assignment
                assignments.loc[idx] = cluster_id
                cluster_counts.loc[cluster_id] += 1
                assigned = True
                break  # Move to next point
        
        # FALLBACK STRATEGY: Handle "orphan" points
        # If a point violates constraints for all clusters, we must still assign it
        # Strategy: Assign to the emptiest cluster (minimizes overall violation)
        if not assigned:
            emptiest = cluster_counts.idxmin()
            assignments.loc[idx] = emptiest
            cluster_counts.loc[emptiest] += 1
    
    return assignments


# %% M-STEP: Update centroids
def update_centroids(X, assignments, n_clusters, old_centroids=None):
    """
    M-STEP: Update cluster centroids based on current assignments.
    
    This is the "Maximization" step in the EM framework. Each cluster
    center is recomputed as the mean of all points assigned to it.
    Empty clusters retain their previous centroids to avoid numerical issues.
    
    Parameters
    ----------
    X : DataFrame
        Dataset features
    assignments : Series
        Current cluster assignments
    n_clusters : int
        Number of clusters (k)
    old_centroids : DataFrame, optional
        Previous centroids (used if cluster becomes empty)
    
    Returns
    -------
    DataFrame
        Updated centroids
    """
    # Compute mean of each cluster
    new_centroids = X.groupby(assignments).mean().reindex(index=range(n_clusters))
    
    # Handle empty clusters: keep old centroid if provided
    if old_centroids is not None:
        empty_clusters = new_centroids.isnull().any(axis=1)
        if empty_clusters.any():
            new_centroids.loc[empty_clusters] = old_centroids.loc[empty_clusters]
    
    return new_centroids


# %% Run algorithm
def run(X, n_nonsensitive, s, n_clusters, init_centroids, 
        dataset_name, init_method, random_state,
        epsilon=EPSILON, gamma=GAMMA, window_size=WINDOW_SIZE, 
        max_iter=MAX_ITER, verbose=True, save=False):
    """
    Run the Balanced Greedy Fair K-Means algorithm using EM framework.
    
    Algorithm (Expectation-Maximization):
    ---------------------------------------
    1. Initialize centroids using init_centroids (e.g., k-means++)
    2. Loop until convergence:
       a. E-STEP: Assign points to nearest cluster satisfying constraints
       b. M-STEP: Update centroids as mean of assigned points
       c. Check convergence (no reassignments)
    
    Parameters
    ----------
    X : DataFrame
        Dataset features (n_samples x n_features)
    n_nonsensitive : int
        Number of non-sensitive features
    s : Series
        Sensitive attribute (age)
    n_clusters : int
        Number of clusters (k)
    init_centroids : DataFrame
        Initial cluster centers
    dataset_name : str
        Dataset name for logging
    init_method : str
        Initialization method used (e.g., 'kmeans_plusplus')
    random_state : int
        Random seed for reproducibility
    epsilon : float, default=0.1
        Fairness tolerance (alpha)
    gamma : float, default=1.1
        Capacity tolerance (balance factor)
    window_size : int, default=3
        Size of sliding windows for age fairness
    max_iter : int, default=100
        Maximum number of iterations
    verbose : bool, default=True
        Print progress information
    save : bool, default=False
        Save results to file
    
    Returns
    -------
    assignments : Series
        Final cluster assignments for each point
    centroids : DataFrame
        Final cluster centers
    info : dict
        Algorithm metadata (runtime, convergence, etc.)
    stats : DataFrame
        Per-iteration statistics (objective, reassignments)
    """
    
    # Print starting message
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {ID} (Algorithm 1: Fair K-Means with EM Framework)")
        print(f"Dataset: {dataset_name}, k={n_clusters}")
        print(f"Parameters: epsilon={epsilon}, gamma={gamma}, window_size={window_size}")
        print(f"{'='*60}")
    
    # Start timer
    timer = Timer()
    timer.resume()
    
    # INITIALIZATION: Start with provided centroids (e.g., k-means++)
    centroids = init_centroids.copy()
    
    # Initialize assignments (will be computed in first E-step)
    assignments = Series(index=X.index, dtype=int, name='cluster')
    old_assignments = None
    
    # Track statistics for each iteration
    iteration_stats = []
    
    # ============================================
    # MAIN EM LOOP: Expectation-Maximization
    # ============================================
    for iteration in range(max_iter):
        
        # ----------------------------------------
        # E-STEP: Reassign points (Expectation)
        # ----------------------------------------
        # CRITICAL: Do this FIRST to use the init_centroids in iteration 0!
        # Points are shuffled and assigned to nearest cluster that satisfies constraints
        
        # Compute seed for this iteration (ensures reproducibility + variation)
        iter_seed = (random_state + iteration) if random_state is not None else None
        
        assignments = assign_all_points(
            X, s, centroids, n_clusters, epsilon, gamma, window_size,
            random_state=iter_seed
        )
        
        # ----------------------------------------
        # M-STEP: Update centroids (Maximization)
        # ----------------------------------------
        # Save old centroids to handle potential empty clusters
        centroids_before_update = centroids.copy()
        centroids = update_centroids(X, assignments, n_clusters, old_centroids=centroids_before_update)
        
        # ----------------------------------------
        # CONVERGENCE CHECK
        # ----------------------------------------
        # Check how many points changed clusters
        if old_assignments is not None:
            n_changes = (assignments != old_assignments).sum()
        else:
            n_changes = len(X)  # First iteration, everything is "new"
        
        old_assignments = assignments.copy()
        
        # Compute objective function (sum of squared distances)
        objective = lloyd.compute_objective(
            X=X, n_nonsensitive=n_nonsensitive, 
            c=assignments, centroids=centroids
        )
        
        # Record iteration statistics
        iteration_stats.append({
            'iteration': iteration,
            'objective': objective,
            'reassignments': n_changes
        })
        
        # Print progress
        if verbose and (iteration % 10 == 0 or n_changes == 0):
            print(f"Iter {iteration:3d}: objective={objective:.5e}, reassignments={n_changes:5d}")
        
        # Check if converged (no points changed cluster)
        # Skip check on iter 0 because n_changes is artificial
        if iteration > 0 and n_changes == 0:
            if verbose:
                print(f"\n*** Converged at iteration {iteration} ***")
            break
    
    # Stop timer
    timer.pause()
    
    # Create statistics DataFrame
    stats = pd.DataFrame(iteration_stats)
    
    # Create info dictionary
    info = {
        'algorithm': ID,
        'dataset': dataset_name,
        'n_clusters': n_clusters,
        'epsilon': epsilon,
        'gamma': gamma,
        'window_size': window_size,
        'n_iterations': len(stats),
        'converged': (n_changes == 0),
        'time': str(timer)
    }
    
    if verbose:
        print(f"\nCompleted in {timer}")
        print(f"Final objective: {objective:.5e}")
    
    return assignments, centroids, info, stats
