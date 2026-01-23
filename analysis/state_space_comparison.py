#!/usr/bin/env python3
"""
State Space Comparison Analysis

Compares the data-driven 10-Cluster state space with the theory-driven
4-Animal (Computational Psychodynamics) state space.

This validates whether:
1. The theoretical framework captures the same structure as data-driven clustering
2. The transition dynamics are preserved under the mapping
3. Both approaches identify similar behavioral patterns

Output:
- Side-by-side transition matrices
- Mapping validation metrics (NMI, agreement rate)
- Comparative visualizations
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import glob
from pathlib import Path

# Configuration
ANIMAL_NAMES = {0: "Seeking", 1: "Directing", 2: "Conferring", 3: "Revising"}
ANIMAL_COLORS = {
    'Seeking': '#2ecc71',    # Green - self-exploration
    'Directing': '#e74c3c',  # Red - exploitation/harm
    'Conferring': '#3498db', # Blue - observation/surveillance
    'Revising': '#9b59b6'    # Purple - ritualization/habits
}

# The critical mapping: 10 Clusters → 4 Animals
# Based on semantic analysis of cluster themes
CLUSTER_TO_ANIMAL_MAPPING = {
    0: 'Conferring',   # Predatory/Geographical Stalking → surveillance, target selection
    1: 'Directing',    # Sexually Motivated Serial Killer → exploitation, harm
    2: 'Revising',     # Escalating Criminal Behavior → habitual patterns, progression
    3: 'Directing',    # Sadistic Duo/Killer Couple → exploitation, torture
    4: 'Directing',    # Power and Control (murder) → exploitation, dominance
    5: 'Seeking',      # Power and Control (workplace) → internal conflict, frustration
    6: 'Revising',     # Legal Judgment and Sentencing → consequences, processing
    7: 'Seeking',      # Search for Identity/Belonging → self-exploration, identity
    8: 'Conferring',   # Domestic Violence/Marital Discord → relational dynamics
    9: 'Directing',    # Angel of Death/Mercy Killer → control over life/death
}

# Detailed rationale for each mapping
MAPPING_RATIONALE = {
    0: "Stalking and geographical targeting = Conferring (observation, information gathering)",
    1: "Sexual motivation with murder = Directing (exploitation of others)",
    2: "Escalating criminal behavior = Revising (habitual patterns, progression)",
    3: "Sadistic torture by couples = Directing (extreme exploitation)",
    4: "Power/control through murder = Directing (dominance, harm)",
    5: "Workplace power struggles = Seeking (internal conflict, identity issues)",
    6: "Legal proceedings = Revising (processing consequences, ritualistic)",
    7: "Identity struggles, family issues = Seeking (self-exploration)",
    8: "Domestic/relational violence = Conferring (relational observation) or Directing",
    9: "Medical killing for control = Directing (control over life/death)",
}


def load_cluster_data(analysis_dir: str):
    """Load cluster definitions and label mapping."""
    clusters_path = os.path.join(analysis_dir, 'clusters.json')
    label_to_theme_path = os.path.join(analysis_dir, 'label_to_theme.pkl')

    with open(clusters_path, 'r') as f:
        clusters = json.load(f)

    with open(label_to_theme_path, 'rb') as f:
        label_to_theme = pickle.load(f)

    return clusters, label_to_theme


def load_individual_sequences(analysis_dir: str):
    """
    Load annotated CSVs to get individual event sequences with cluster labels.
    Returns dict of individual_name -> list of cluster labels (in order).
    """
    sequences = {}
    annotated_files = glob.glob(os.path.join(analysis_dir, '*_annotated.csv'))

    for file_path in annotated_files:
        individual_name = os.path.basename(file_path).replace('_annotated.csv', '')
        try:
            df = pd.read_csv(file_path)
            if 'PredictedLabel' in df.columns:
                # Get sequence of cluster labels
                labels = df['PredictedLabel'].tolist()
                sequences[individual_name] = labels
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return sequences


def map_cluster_to_animal(cluster_label: int) -> str:
    """Map a cluster label to its corresponding animal state."""
    return CLUSTER_TO_ANIMAL_MAPPING.get(cluster_label, 'Seeking')


def map_cluster_to_animal_idx(cluster_label: int) -> int:
    """Map a cluster label to animal state index (0-3)."""
    animal = map_cluster_to_animal(cluster_label)
    animal_to_idx = {'Seeking': 0, 'Directing': 1, 'Conferring': 2, 'Revising': 3}
    return animal_to_idx[animal]


def convert_sequence_to_animals(cluster_sequence: list) -> list:
    """Convert a sequence of cluster labels to animal states."""
    return [map_cluster_to_animal_idx(c) for c in cluster_sequence]


def compute_transition_matrix(sequence: list, n_states: int) -> np.ndarray:
    """Compute transition matrix from a single sequence."""
    matrix = np.zeros((n_states, n_states))
    for i in range(len(sequence) - 1):
        from_state = sequence[i]
        to_state = sequence[i + 1]
        if 0 <= from_state < n_states and 0 <= to_state < n_states:
            matrix[from_state, to_state] += 1

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return matrix / row_sums


def aggregate_transition_matrices(sequences: dict, n_states: int) -> tuple:
    """
    Aggregate transition matrices across all individuals.
    Returns (mean_matrix, std_matrix, count_matrix).
    """
    matrices = []
    total_counts = np.zeros((n_states, n_states))

    for name, seq in sequences.items():
        if len(seq) < 2:
            continue

        # Count transitions
        for i in range(len(seq) - 1):
            from_state = seq[i]
            to_state = seq[i + 1]
            if 0 <= from_state < n_states and 0 <= to_state < n_states:
                total_counts[from_state, to_state] += 1

        # Individual normalized matrix
        matrix = compute_transition_matrix(seq, n_states)
        matrices.append(matrix)

    if not matrices:
        return np.zeros((n_states, n_states)), np.zeros((n_states, n_states)), total_counts

    stacked = np.stack(matrices)
    mean_matrix = np.mean(stacked, axis=0)
    std_matrix = np.std(stacked, axis=0)

    return mean_matrix, std_matrix, total_counts


def compute_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Compute stationary distribution from transition matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.real(eigenvectors[:, idx])
    stationary = np.abs(stationary)
    stationary = stationary / stationary.sum()
    return stationary


def compute_entropy_rate(transition_matrix: np.ndarray, stationary: np.ndarray) -> float:
    """Compute entropy rate of Markov chain."""
    entropy_rate = 0.0
    for i in range(len(stationary)):
        for j in range(transition_matrix.shape[1]):
            p = transition_matrix[i, j]
            if p > 0:
                entropy_rate -= stationary[i] * p * np.log2(p)
    return entropy_rate


def js_divergence_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Compute average JS divergence between corresponding rows of two matrices."""
    divergences = []
    for i in range(matrix_a.shape[0]):
        row_a = matrix_a[i] + 1e-10
        row_b = matrix_b[i] + 1e-10
        row_a = row_a / row_a.sum()
        row_b = row_b / row_b.sum()
        div = jensenshannon(row_a, row_b)
        divergences.append(div)
    return np.mean(divergences)


def plot_transition_matrix(matrix: np.ndarray, labels: list, title: str,
                          output_path: str, cmap: str = 'Blues'):
    """Plot a transition matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                ax=ax, vmin=0, vmax=1)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(matrix_cluster: np.ndarray, matrix_animal: np.ndarray,
                   cluster_labels: list, animal_labels: list,
                   output_path: str):
    """Plot side-by-side comparison of transition matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cluster-based (10 states)
    sns.heatmap(matrix_cluster, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=cluster_labels, yticklabels=cluster_labels,
                ax=axes[0], vmin=0, vmax=1, annot_kws={'size': 7})
    axes[0].set_title('Data-Driven: 10-Cluster State Space', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('To Cluster')
    axes[0].set_ylabel('From Cluster')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Animal-based (4 states)
    sns.heatmap(matrix_animal, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=animal_labels, yticklabels=animal_labels,
                ax=axes[1], vmin=0, vmax=1, annot_kws={'size': 12})
    axes[1].set_title('Theory-Driven: 4-Animal State Space', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('To State')
    axes[1].set_ylabel('From State')

    plt.suptitle('State Space Comparison: Data-Driven vs Theory-Driven',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stationary_comparison(stat_cluster: np.ndarray, stat_animal: np.ndarray,
                               cluster_labels: list, animal_labels: list,
                               output_path: str):
    """Plot comparison of stationary distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cluster stationary
    colors_cluster = plt.cm.Oranges(np.linspace(0.3, 0.9, len(stat_cluster)))
    axes[0].bar(range(len(stat_cluster)), stat_cluster, color=colors_cluster)
    axes[0].set_xticks(range(len(cluster_labels)))
    axes[0].set_xticklabels(cluster_labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Stationary Probability')
    axes[0].set_title('10-Cluster Stationary Distribution', fontweight='bold')
    axes[0].set_ylim(0, max(stat_cluster) * 1.2)

    # Animal stationary
    animal_colors = [ANIMAL_COLORS[name] for name in animal_labels]
    axes[1].bar(range(len(stat_animal)), stat_animal, color=animal_colors)
    axes[1].set_xticks(range(len(animal_labels)))
    axes[1].set_xticklabels(animal_labels, fontsize=11)
    axes[1].set_ylabel('Stationary Probability')
    axes[1].set_title('4-Animal Stationary Distribution', fontweight='bold')
    axes[1].set_ylim(0, max(stat_animal) * 1.2)

    # Add values on bars
    for i, v in enumerate(stat_animal):
        axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)

    plt.suptitle('Stationary Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mapping_sankey(cluster_labels: list, output_path: str):
    """Plot the cluster-to-animal mapping as a visual diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Positions
    cluster_y = np.linspace(0.95, 0.05, 10)
    animal_y = {'Seeking': 0.8, 'Directing': 0.6, 'Conferring': 0.4, 'Revising': 0.2}

    cluster_x = 0.1
    animal_x = 0.9

    # Draw cluster nodes
    for i, label in enumerate(cluster_labels):
        ax.scatter(cluster_x, cluster_y[i], s=300, c='orange', zorder=3)
        ax.text(cluster_x - 0.05, cluster_y[i], f"C{i}", ha='right', va='center', fontsize=9)

    # Draw animal nodes
    for animal, y in animal_y.items():
        ax.scatter(animal_x, y, s=500, c=ANIMAL_COLORS[animal], zorder=3)
        ax.text(animal_x + 0.05, y, animal, ha='left', va='center', fontsize=11, fontweight='bold')

    # Draw connections
    for cluster_id, animal in CLUSTER_TO_ANIMAL_MAPPING.items():
        ax.plot([cluster_x, animal_x], [cluster_y[cluster_id], animal_y[animal]],
                color=ANIMAL_COLORS[animal], alpha=0.5, linewidth=2)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Cluster → Animal State Mapping', fontsize=14, fontweight='bold')

    # Add legend
    legend_text = "\n".join([f"C{i}: {label[:30]}..." for i, label in enumerate(cluster_labels)])
    ax.text(0.5, -0.05, legend_text, ha='center', va='top', fontsize=7,
            transform=ax.transAxes, family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_agreement_metrics(cluster_sequences: dict, label_to_theme: dict) -> dict:
    """
    Compute metrics to validate the mapping.
    """
    # Flatten all sequences for distribution comparison
    all_clusters = []
    all_animals = []

    for seq in cluster_sequences.values():
        all_clusters.extend(seq)
        all_animals.extend([map_cluster_to_animal_idx(c) for c in seq])

    # Distribution analysis
    cluster_dist = Counter(all_clusters)
    animal_dist = Counter(all_animals)

    # Compute expected animal distribution from mapping
    expected_animal = defaultdict(int)
    for cluster, count in cluster_dist.items():
        animal_idx = map_cluster_to_animal_idx(cluster)
        expected_animal[animal_idx] += count

    # Normalize
    total = sum(animal_dist.values())
    animal_probs = np.array([animal_dist.get(i, 0) / total for i in range(4)])
    expected_probs = np.array([expected_animal.get(i, 0) / total for i in range(4)])

    # JS divergence between actual and expected (should be ~0 if mapping is consistent)
    js_div = jensenshannon(animal_probs + 1e-10, expected_probs + 1e-10)

    # Entropy of each distribution
    cluster_entropy = stats.entropy(list(cluster_dist.values()))
    animal_entropy = stats.entropy(list(animal_dist.values()))

    # Information retained: ratio of animal entropy to cluster entropy
    info_retained = animal_entropy / cluster_entropy if cluster_entropy > 0 else 0

    return {
        'n_events': len(all_clusters),
        'n_individuals': len(cluster_sequences),
        'cluster_distribution': dict(cluster_dist),
        'animal_distribution': {ANIMAL_NAMES[k]: v for k, v in animal_dist.items()},
        'cluster_entropy': cluster_entropy,
        'animal_entropy': animal_entropy,
        'information_retained_ratio': info_retained,
        'mapping_js_divergence': js_div,  # Should be 0 for consistent mapping
    }


def main():
    """Run the complete state space comparison analysis."""
    print("=" * 70)
    print("STATE SPACE COMPARISON ANALYSIS")
    print("Data-Driven (10-Cluster) vs Theory-Driven (4-Animal)")
    print("=" * 70)

    # Setup paths
    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(analysis_dir, 'state_space_comparison')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n[1/6] Loading cluster data...")
    clusters, label_to_theme = load_cluster_data(analysis_dir)

    # Create short labels for clusters
    cluster_short_labels = [f"C{i}" for i in range(10)]
    cluster_full_labels = [label_to_theme.get(i, f"Cluster {i}")[:25] + "..."
                          for i in range(10)]

    animal_labels = ['Seeking', 'Directing', 'Conferring', 'Revising']

    print(f"   Loaded {len(clusters)} cluster definitions")

    # Load individual sequences
    print("\n[2/6] Loading individual sequences...")
    cluster_sequences = load_individual_sequences(analysis_dir)
    print(f"   Loaded sequences for {len(cluster_sequences)} individuals")

    if not cluster_sequences:
        print("ERROR: No annotated sequences found. Run archetype_sequence_analysis.py first.")
        return

    # Convert to animal sequences
    print("\n[3/6] Converting cluster sequences to animal sequences...")
    animal_sequences = {}
    for name, seq in cluster_sequences.items():
        animal_sequences[name] = convert_sequence_to_animals(seq)

    # Compute transition matrices
    print("\n[4/6] Computing transition matrices...")

    # Cluster-based (10x10)
    cluster_mean, cluster_std, cluster_counts = aggregate_transition_matrices(
        cluster_sequences, n_states=10
    )
    cluster_stationary = compute_stationary_distribution(cluster_mean)
    cluster_entropy = compute_entropy_rate(cluster_mean, cluster_stationary)

    # Animal-based (4x4)
    animal_mean, animal_std, animal_counts = aggregate_transition_matrices(
        animal_sequences, n_states=4
    )
    animal_stationary = compute_stationary_distribution(animal_mean)
    animal_entropy = compute_entropy_rate(animal_mean, animal_stationary)

    print(f"   Cluster entropy rate: {cluster_entropy:.3f} bits")
    print(f"   Animal entropy rate: {animal_entropy:.3f} bits")

    # Compute agreement metrics
    print("\n[5/6] Computing agreement metrics...")
    metrics = compute_agreement_metrics(cluster_sequences, label_to_theme)

    print(f"   Total events: {metrics['n_events']}")
    print(f"   Cluster entropy: {metrics['cluster_entropy']:.3f}")
    print(f"   Animal entropy: {metrics['animal_entropy']:.3f}")
    print(f"   Information retained: {metrics['information_retained_ratio']:.1%}")

    # Generate visualizations
    print("\n[6/6] Generating visualizations...")

    # Individual transition matrices
    plot_transition_matrix(
        cluster_mean, cluster_short_labels,
        "Data-Driven: 10-Cluster Transition Matrix",
        os.path.join(output_dir, 'transition_matrix_10cluster.png'),
        cmap='Oranges'
    )

    plot_transition_matrix(
        animal_mean, animal_labels,
        "Theory-Driven: 4-Animal Transition Matrix",
        os.path.join(output_dir, 'transition_matrix_4animal.png'),
        cmap='Blues'
    )

    # Side-by-side comparison
    plot_comparison(
        cluster_mean, animal_mean,
        cluster_short_labels, animal_labels,
        os.path.join(output_dir, 'transition_comparison.png')
    )

    # Stationary distributions
    plot_stationary_comparison(
        cluster_stationary, animal_stationary,
        cluster_short_labels, animal_labels,
        os.path.join(output_dir, 'stationary_comparison.png')
    )

    # Mapping visualization
    plot_mapping_sankey(
        cluster_full_labels,
        os.path.join(output_dir, 'cluster_to_animal_mapping.png')
    )

    # Save results
    results = {
        'mapping': {str(k): v for k, v in CLUSTER_TO_ANIMAL_MAPPING.items()},
        'mapping_rationale': MAPPING_RATIONALE,
        'metrics': metrics,
        'cluster_transition_matrix': cluster_mean.tolist(),
        'animal_transition_matrix': animal_mean.tolist(),
        'cluster_stationary': cluster_stationary.tolist(),
        'animal_stationary': animal_stationary.tolist(),
        'cluster_entropy_rate': cluster_entropy,
        'animal_entropy_rate': animal_entropy,
        'cluster_labels': cluster_full_labels,
        'animal_labels': animal_labels,
    }

    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nCluster → Animal Mapping:")
    for cluster_id in range(10):
        animal = CLUSTER_TO_ANIMAL_MAPPING[cluster_id]
        theme = label_to_theme.get(cluster_id, "Unknown")[:40]
        print(f"  C{cluster_id} ({theme}...) → {animal}")

    print(f"\nStationary Distributions:")
    print(f"  10-Cluster: {', '.join([f'C{i}:{p:.2f}' for i, p in enumerate(cluster_stationary)])}")
    print(f"  4-Animal:   {', '.join([f'{n}:{p:.2f}' for n, p in zip(animal_labels, animal_stationary)])}")

    print(f"\nEntropy Rates:")
    print(f"  10-Cluster: {cluster_entropy:.3f} bits (more complex)")
    print(f"  4-Animal:   {animal_entropy:.3f} bits (more parsimonious)")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
