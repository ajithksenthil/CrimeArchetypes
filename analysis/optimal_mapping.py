#!/usr/bin/env python3
"""
Optimal State Space Mapping

Derives the data-driven optimal mapping from K-clusters to M-states
that maximizes information retention, then compares to theoretical mapping.

This demonstrates the methodology described in STATE_SPACE_METHODOLOGY.md
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import product
from scipy import stats
from scipy.special import comb
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from typing import Dict, List, Tuple
from dataclasses import dataclass

np.random.seed(42)

ANIMAL_NAMES = ['Seeking', 'Directing', 'Conferring', 'Revising']

# Theoretical mapping (from domain knowledge)
THEORETICAL_MAPPING = {
    0: 2,  # Stalking → Conferring
    1: 1,  # Sexual murder → Directing
    2: 3,  # Escalating crime → Revising
    3: 1,  # Sadistic → Directing
    4: 1,  # Power/control → Directing
    5: 0,  # Workplace → Seeking
    6: 3,  # Legal → Revising
    7: 0,  # Identity → Seeking
    8: 2,  # Domestic → Conferring
    9: 1,  # Angel of death → Directing
}


@dataclass
class MappingResult:
    """Container for mapping analysis results."""
    mapping: Dict[int, int]
    mapping_name: str
    information_retention: float
    mutual_information: float
    source_entropy: float
    target_entropy: float
    target_distribution: Dict[str, float]


def load_sequences(analysis_dir: str) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """Load cluster sequences and theme labels."""
    sequences = {}
    annotated_files = glob.glob(os.path.join(analysis_dir, '*_annotated.csv'))

    for file_path in annotated_files:
        name = os.path.basename(file_path).replace('_annotated.csv', '')
        try:
            df = pd.read_csv(file_path)
            if 'PredictedLabel' in df.columns:
                sequences[name] = [l for l in df['PredictedLabel'].tolist() if 0 <= l < 10]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Load cluster themes
    label_to_theme_path = os.path.join(analysis_dir, 'label_to_theme.pkl')
    with open(label_to_theme_path, 'rb') as f:
        label_to_theme = pickle.load(f)

    return sequences, label_to_theme


def flatten_sequences(sequences: Dict[str, List[int]]) -> List[int]:
    """Flatten all sequences into one list."""
    return [s for seq in sequences.values() for s in seq]


def apply_mapping(labels: List[int], mapping: Dict[int, int]) -> List[int]:
    """Apply a mapping to labels."""
    return [mapping.get(l, 0) for l in labels]


def compute_entropy(labels: List[int]) -> float:
    """Compute entropy in bits."""
    counts = list(Counter(labels).values())
    return stats.entropy(counts, base=2)


def compute_mutual_info_bits(labels_1: List[int], labels_2: List[int]) -> float:
    """Compute mutual information in bits."""
    # sklearn uses natural log, convert to bits
    mi_nats = mutual_info_score(labels_1, labels_2)
    return mi_nats / np.log(2)


def information_retention(source_labels: List[int], target_labels: List[int]) -> float:
    """
    Compute information retention ratio.

    I(S; T) / H(S)

    This measures what fraction of the source information is preserved
    in the target representation.
    """
    mi = compute_mutual_info_bits(source_labels, target_labels)
    h_source = compute_entropy(source_labels)
    return mi / h_source if h_source > 0 else 0


def evaluate_mapping(
    source_labels: List[int],
    mapping: Dict[int, int],
    mapping_name: str
) -> MappingResult:
    """Evaluate a mapping's information retention."""
    target_labels = apply_mapping(source_labels, mapping)

    mi = compute_mutual_info_bits(source_labels, target_labels)
    h_source = compute_entropy(source_labels)
    h_target = compute_entropy(target_labels)
    retention = mi / h_source if h_source > 0 else 0

    # Target distribution
    target_counts = Counter(target_labels)
    total = sum(target_counts.values())
    target_dist = {ANIMAL_NAMES[i]: target_counts.get(i, 0) / total
                   for i in range(4)}

    return MappingResult(
        mapping=mapping,
        mapping_name=mapping_name,
        information_retention=retention,
        mutual_information=mi,
        source_entropy=h_source,
        target_entropy=h_target,
        target_distribution=target_dist
    )


def find_optimal_mapping_exhaustive(
    source_labels: List[int],
    n_source: int = 10,
    n_target: int = 4
) -> Tuple[Dict[int, int], float]:
    """
    Find optimal mapping by exhaustive search.

    For 10 clusters → 4 states, there are 4^10 = 1,048,576 possible mappings.
    This is feasible to compute exhaustively.
    """
    print(f"\n   Searching {n_target}^{n_source} = {n_target**n_source:,} possible mappings...")

    best_mapping = {}
    best_retention = 0

    source_states = list(range(n_source))
    n_evaluated = 0

    for mapping_tuple in product(range(n_target), repeat=n_source):
        mapping = {s: t for s, t in zip(source_states, mapping_tuple)}
        target_labels = apply_mapping(source_labels, mapping)
        retention = information_retention(source_labels, target_labels)

        if retention > best_retention:
            best_retention = retention
            best_mapping = mapping.copy()

        n_evaluated += 1
        if n_evaluated % 100000 == 0:
            print(f"   Evaluated {n_evaluated:,} mappings, best so far: {best_retention:.4f}")

    return best_mapping, best_retention


def find_optimal_mapping_greedy(
    source_labels: List[int],
    n_source: int = 10,
    n_target: int = 4
) -> Tuple[Dict[int, int], float]:
    """
    Find approximately optimal mapping using greedy hill-climbing.

    Much faster than exhaustive search, but may find local optimum.
    """
    print("\n   Using greedy optimization...")

    # Initialize: assign each source to the most frequent target
    # in its neighborhood (based on co-occurrence)
    mapping = {s: s % n_target for s in range(n_source)}

    def get_retention(m):
        return information_retention(source_labels, apply_mapping(source_labels, m))

    best_retention = get_retention(mapping)
    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1

        for source in range(n_source):
            current_target = mapping[source]
            best_target = current_target
            best_local_retention = best_retention

            for target in range(n_target):
                if target == current_target:
                    continue

                mapping[source] = target
                retention = get_retention(mapping)

                if retention > best_local_retention:
                    best_local_retention = retention
                    best_target = target
                    improved = True

            mapping[source] = best_target
            best_retention = best_local_retention

        print(f"   Iteration {iteration}: retention = {best_retention:.4f}")

    return mapping, best_retention


def find_optimal_mapping_spectral(
    source_labels: List[int],
    sequences: Dict[str, List[int]],
    n_source: int = 10,
    n_target: int = 4
) -> Tuple[Dict[int, int], float]:
    """
    Find optimal mapping using spectral clustering on transition co-occurrence.

    Groups source states that have similar transition patterns.
    """
    print("\n   Using spectral clustering on transitions...")

    # Build transition co-occurrence matrix
    cooccur = np.zeros((n_source, n_source))

    for seq in sequences.values():
        for i in range(len(seq) - 1):
            if 0 <= seq[i] < n_source and 0 <= seq[i+1] < n_source:
                cooccur[seq[i], seq[i+1]] += 1

    # Symmetrize and normalize
    cooccur = cooccur + cooccur.T
    row_sums = cooccur.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cooccur_norm = cooccur / row_sums

    # Add small diagonal for numerical stability
    affinity = cooccur_norm + 0.01 * np.eye(n_source)

    # Spectral clustering
    clustering = SpectralClustering(n_clusters=n_target,
                                     affinity='precomputed',
                                     random_state=42)
    assignments = clustering.fit_predict(affinity)

    mapping = {i: int(assignments[i]) for i in range(n_source)}

    retention = information_retention(source_labels, apply_mapping(source_labels, mapping))

    return mapping, retention


def compute_all_mappings(
    source_labels: List[int],
    sequences: Dict[str, List[int]]
) -> Dict[str, MappingResult]:
    """Compute and compare all mapping approaches."""
    results = {}

    # 1. Theoretical mapping
    print("\n[1/4] Evaluating theoretical mapping...")
    results['theoretical'] = evaluate_mapping(
        source_labels, THEORETICAL_MAPPING, "Theoretical"
    )
    print(f"   Retention: {results['theoretical'].information_retention:.4f}")

    # 2. Greedy optimal
    print("\n[2/4] Finding greedy optimal mapping...")
    greedy_map, greedy_ret = find_optimal_mapping_greedy(source_labels)
    results['greedy_optimal'] = evaluate_mapping(
        source_labels, greedy_map, "Greedy Optimal"
    )
    print(f"   Retention: {results['greedy_optimal'].information_retention:.4f}")

    # 3. Spectral optimal
    print("\n[3/4] Finding spectral optimal mapping...")
    spectral_map, spectral_ret = find_optimal_mapping_spectral(
        source_labels, sequences
    )
    results['spectral_optimal'] = evaluate_mapping(
        source_labels, spectral_map, "Spectral Optimal"
    )
    print(f"   Retention: {results['spectral_optimal'].information_retention:.4f}")

    # 4. Exhaustive optimal (may take a minute)
    print("\n[4/4] Finding exhaustive optimal mapping...")
    exhaustive_map, exhaustive_ret = find_optimal_mapping_exhaustive(source_labels)
    results['exhaustive_optimal'] = evaluate_mapping(
        source_labels, exhaustive_map, "Exhaustive Optimal"
    )
    print(f"   Retention: {results['exhaustive_optimal'].information_retention:.4f}")

    return results


def plot_mapping_comparison(
    results: Dict[str, MappingResult],
    label_to_theme: Dict[int, str],
    output_dir: str
):
    """Visualize mapping comparisons."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Information retention comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    retentions = [results[n].information_retention for n in names]
    colors = ['#e74c3c' if n == 'theoretical' else '#3498db' for n in names]

    bars = ax.bar(names, retentions, color=colors, alpha=0.8)
    ax.set_ylabel('Information Retention')
    ax.set_title('Comparison of Mapping Approaches\n(Higher = Better Structure Preservation)')
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, ret in zip(bars, retentions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ret:.3f}', ha='center', fontsize=11, fontweight='bold')

    # Add baseline
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retention_comparison.png'), dpi=300)
    plt.close()

    # 2. Mapping visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for (name, result), ax in zip(results.items(), axes.flatten()):
        mapping = result.mapping

        # Create mapping matrix
        matrix = np.zeros((10, 4))
        for src, tgt in mapping.items():
            matrix[src, tgt] = 1

        sns.heatmap(matrix, annot=False, cmap='Blues', ax=ax,
                    xticklabels=ANIMAL_NAMES,
                    yticklabels=[f"C{i}" for i in range(10)],
                    cbar=False, linewidths=0.5)
        ax.set_title(f"{name.replace('_', ' ').title()}\nRetention: {result.information_retention:.3f}")
        ax.set_xlabel('Target State')
        ax.set_ylabel('Source Cluster')

    plt.suptitle('Mapping Visualizations: Which Clusters Map to Which States?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mapping_visualization.png'), dpi=300)
    plt.close()

    # 3. Detailed mapping table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    # Build table data
    headers = ['Cluster', 'Theme', 'Theoretical', 'Optimal', 'Match?']
    theoretical = results['theoretical'].mapping
    optimal = results['exhaustive_optimal'].mapping

    table_data = []
    for i in range(10):
        theme = label_to_theme.get(i, '')[:35] + '...'
        theo_state = ANIMAL_NAMES[theoretical[i]]
        opt_state = ANIMAL_NAMES[optimal[i]]
        match = '✓' if theoretical[i] == optimal[i] else '✗'
        table_data.append([f'C{i}', theme, theo_state, opt_state, match])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color cells
    for i in range(11):  # Including header
        for j in range(5):
            cell = table[i, j]
            if i == 0:
                cell.set_facecolor('#3498db')
                cell.set_text_props(color='white', fontweight='bold')
            elif j == 4:  # Match column
                if i > 0 and table_data[i-1][4] == '✓':
                    cell.set_facecolor('#90EE90')
                elif i > 0:
                    cell.set_facecolor('#FFB6C1')

    n_match = sum(1 for i in range(10) if theoretical[i] == optimal[i])
    ax.set_title(f'Theoretical vs Optimal Mapping Comparison\nAgreement: {n_match}/10 ({n_match*10}%)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.savefig(os.path.join(output_dir, 'mapping_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualizations to {output_dir}")


def print_detailed_results(results: Dict[str, MappingResult], label_to_theme: Dict[int, str]):
    """Print detailed analysis results."""
    print("\n" + "=" * 80)
    print("OPTIMAL MAPPING ANALYSIS RESULTS")
    print("=" * 80)

    # Summary table
    print("\n" + "-" * 80)
    print("INFORMATION RETENTION COMPARISON")
    print("-" * 80)
    print(f"{'Mapping':<25} {'Retention':>12} {'MI (bits)':>12} {'H(Target)':>12}")
    print("-" * 80)

    for name, result in sorted(results.items(), key=lambda x: -x[1].information_retention):
        print(f"{result.mapping_name:<25} {result.information_retention:>12.4f} "
              f"{result.mutual_information:>12.4f} {result.target_entropy:>12.4f}")

    # Best mapping details
    best_name = max(results.keys(), key=lambda x: results[x].information_retention)
    best = results[best_name]

    print("\n" + "-" * 80)
    print(f"OPTIMAL MAPPING DETAILS ({best.mapping_name})")
    print("-" * 80)

    for cluster_id in range(10):
        target_id = best.mapping[cluster_id]
        theme = label_to_theme.get(cluster_id, 'Unknown')[:40]
        animal = ANIMAL_NAMES[target_id]

        # Check if same as theoretical
        theo_target = THEORETICAL_MAPPING[cluster_id]
        match = "=" if target_id == theo_target else f"(theo: {ANIMAL_NAMES[theo_target]})"

        print(f"  C{cluster_id} → {animal:<12} {match:<20} | {theme}")

    # Comparison
    print("\n" + "-" * 80)
    print("THEORETICAL VS OPTIMAL COMPARISON")
    print("-" * 80)

    theoretical = results['theoretical']
    optimal = results['exhaustive_optimal']

    n_match = sum(1 for i in range(10)
                  if theoretical.mapping[i] == optimal.mapping[i])

    print(f"  Clusters with same mapping:    {n_match}/10 ({n_match*10}%)")
    print(f"  Theoretical retention:         {theoretical.information_retention:.4f} ({theoretical.information_retention*100:.1f}%)")
    print(f"  Optimal retention:             {optimal.information_retention:.4f} ({optimal.information_retention*100:.1f}%)")
    print(f"  Improvement from optimization: {(optimal.information_retention - theoretical.information_retention)*100:.2f} percentage points")

    # Target distributions
    print("\n" + "-" * 80)
    print("TARGET STATE DISTRIBUTIONS")
    print("-" * 80)
    print(f"{'State':<15} {'Theoretical':>15} {'Optimal':>15}")
    print("-" * 80)

    for state in ANIMAL_NAMES:
        theo_pct = theoretical.target_distribution[state] * 100
        opt_pct = optimal.target_distribution[state] * 100
        print(f"{state:<15} {theo_pct:>14.1f}% {opt_pct:>14.1f}%")

    print("\n" + "=" * 80)


def main():
    """Run optimal mapping analysis."""
    print("=" * 80)
    print("OPTIMAL STATE SPACE MAPPING ANALYSIS")
    print("=" * 80)

    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(analysis_dir, 'optimal_mapping')

    # Load data
    print("\nLoading data...")
    sequences, label_to_theme = load_sequences(analysis_dir)
    source_labels = flatten_sequences(sequences)
    print(f"Loaded {len(sequences)} individuals, {len(source_labels)} events")

    # Compute all mappings
    results = compute_all_mappings(source_labels, sequences)

    # Print detailed results
    print_detailed_results(results, label_to_theme)

    # Generate visualizations
    plot_mapping_comparison(results, label_to_theme, output_dir)

    # Save results
    save_results = {
        name: {
            'mapping': {str(k): int(v) for k, v in r.mapping.items()},
            'mapping_name': r.mapping_name,
            'information_retention': float(r.information_retention),
            'mutual_information': float(r.mutual_information),
            'source_entropy': float(r.source_entropy),
            'target_entropy': float(r.target_entropy),
            'target_distribution': {k: float(v) for k, v in r.target_distribution.items()}
        }
        for name, r in results.items()
    }

    results_path = os.path.join(output_dir, 'optimal_mapping_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
