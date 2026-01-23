#!/usr/bin/env python3
"""
Proper Null Model Tests for State Space Validation

This script implements the correct null hypotheses:

1. MAPPING NULL: Is the theoretical cluster→animal mapping better than random mappings?
   H0: Theoretical mapping = Random mapping (in terms of structure preservation)

2. SEQUENCE NULL: Do observed transition dynamics differ from shuffled sequences?
   H0: Transition matrix = Transition matrix of shuffled sequences

3. PREDICTIVE NULL: Does the state space improve next-state prediction over baseline?
   H0: State-based prediction = Marginal frequency prediction
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from typing import Dict, List, Tuple
from itertools import permutations

np.random.seed(42)

# Theoretical mapping
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

ANIMAL_NAMES = ['Seeking', 'Directing', 'Conferring', 'Revising']


def load_sequences(analysis_dir: str) -> Dict[str, List[int]]:
    """Load cluster sequences from annotated CSVs."""
    sequences = {}
    annotated_files = glob.glob(os.path.join(analysis_dir, '*_annotated.csv'))

    for file_path in annotated_files:
        name = os.path.basename(file_path).replace('_annotated.csv', '')
        try:
            df = pd.read_csv(file_path)
            if 'PredictedLabel' in df.columns:
                sequences[name] = df['PredictedLabel'].tolist()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return sequences


def apply_mapping(cluster_seq: List[int], mapping: Dict[int, int]) -> List[int]:
    """Apply a cluster→animal mapping to a sequence."""
    return [mapping.get(c, 0) for c in cluster_seq if 0 <= c < 10]


def compute_transition_matrix(sequence: List[int], n_states: int) -> np.ndarray:
    """Compute normalized transition matrix."""
    counts = np.zeros((n_states, n_states))
    for i in range(len(sequence) - 1):
        if 0 <= sequence[i] < n_states and 0 <= sequence[i+1] < n_states:
            counts[sequence[i], sequence[i+1]] += 1

    # Normalize with smoothing
    counts += 0.01
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def compute_entropy_rate(P: np.ndarray) -> float:
    """Compute entropy rate of Markov chain."""
    # Stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    pi = pi / pi.sum()

    # Entropy rate
    H = 0.0
    for i in range(len(pi)):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                H -= pi[i] * P[i, j] * np.log2(P[i, j])
    return H


def generate_random_mapping(n_clusters: int = 10, n_animals: int = 4) -> Dict[int, int]:
    """Generate a random cluster→animal mapping."""
    return {i: np.random.randint(0, n_animals) for i in range(n_clusters)}


def compute_mapping_metrics(
    cluster_sequences: Dict[str, List[int]],
    mapping: Dict[int, int]
) -> Dict:
    """Compute metrics for a given mapping."""
    # Apply mapping to all sequences
    animal_sequences = {
        name: apply_mapping(seq, mapping)
        for name, seq in cluster_sequences.items()
    }

    # Aggregate sequence
    all_clusters = [c for seq in cluster_sequences.values() for c in seq if 0 <= c < 10]
    all_animals = [mapping.get(c, 0) for c in all_clusters]

    # Mutual information (normalized)
    mi = mutual_info_score(all_clusters, all_animals)
    h_cluster = stats.entropy(list(Counter(all_clusters).values()))
    h_animal = stats.entropy(list(Counter(all_animals).values()))

    # Normalized MI
    nmi = mi / np.sqrt(h_cluster * h_animal) if h_cluster > 0 and h_animal > 0 else 0

    # Information retention
    info_retention = mi / h_cluster if h_cluster > 0 else 0

    # Transition matrix and entropy rate
    all_animal_seq = [a for seq in animal_sequences.values() for a in seq]
    P = compute_transition_matrix(all_animal_seq, 4)
    entropy_rate = compute_entropy_rate(P)

    # Transition predictability (1 - normalized entropy rate)
    max_entropy = np.log2(4)  # Maximum entropy for 4 states
    predictability = 1 - (entropy_rate / max_entropy)

    return {
        'nmi': nmi,
        'info_retention': info_retention,
        'entropy_rate': entropy_rate,
        'predictability': predictability,
        'h_animal': h_animal,
    }


def test_mapping_null(
    cluster_sequences: Dict[str, List[int]],
    theoretical_mapping: Dict[int, int],
    n_permutations: int = 10000
) -> Dict:
    """
    Test 1: Is the theoretical mapping better than random mappings?

    Null: Theoretical mapping performs same as random mappings.
    """
    print("\n[TEST 1] Mapping Null Hypothesis")
    print("-" * 50)

    # Observed metrics with theoretical mapping
    observed = compute_mapping_metrics(cluster_sequences, theoretical_mapping)

    # Null distribution: random mappings
    null_nmi = []
    null_info = []
    null_pred = []

    for i in range(n_permutations):
        random_map = generate_random_mapping()
        metrics = compute_mapping_metrics(cluster_sequences, random_map)
        null_nmi.append(metrics['nmi'])
        null_info.append(metrics['info_retention'])
        null_pred.append(metrics['predictability'])

        if (i + 1) % 1000 == 0:
            print(f"   Completed {i+1}/{n_permutations} permutations...")

    null_nmi = np.array(null_nmi)
    null_info = np.array(null_info)
    null_pred = np.array(null_pred)

    # P-values (one-tailed: observed > null)
    p_nmi = np.mean(null_nmi >= observed['nmi'])
    p_info = np.mean(null_info >= observed['info_retention'])
    p_pred = np.mean(null_pred >= observed['predictability'])

    # Effect sizes (Cohen's d)
    d_nmi = (observed['nmi'] - np.mean(null_nmi)) / np.std(null_nmi)
    d_info = (observed['info_retention'] - np.mean(null_info)) / np.std(null_info)
    d_pred = (observed['predictability'] - np.mean(null_pred)) / np.std(null_pred)

    results = {
        'test': 'Mapping Null',
        'hypothesis': 'Theoretical mapping = Random mapping',
        'n_permutations': n_permutations,
        'metrics': {
            'nmi': {
                'observed': float(observed['nmi']),
                'null_mean': float(np.mean(null_nmi)),
                'null_std': float(np.std(null_nmi)),
                'p_value': float(p_nmi),
                'effect_size': float(d_nmi),
                'significant': p_nmi < 0.05
            },
            'info_retention': {
                'observed': float(observed['info_retention']),
                'null_mean': float(np.mean(null_info)),
                'null_std': float(np.std(null_info)),
                'p_value': float(p_info),
                'effect_size': float(d_info),
                'significant': p_info < 0.05
            },
            'predictability': {
                'observed': float(observed['predictability']),
                'null_mean': float(np.mean(null_pred)),
                'null_std': float(np.std(null_pred)),
                'p_value': float(p_pred),
                'effect_size': float(d_pred),
                'significant': p_pred < 0.05
            }
        },
        'null_distributions': {
            'nmi': null_nmi.tolist(),
            'info_retention': null_info.tolist(),
            'predictability': null_pred.tolist()
        }
    }

    print(f"\n   Results:")
    print(f"   {'Metric':<20} {'Observed':>10} {'Null Mean':>10} {'p-value':>10} {'Effect d':>10}")
    print(f"   {'-'*60}")
    for metric in ['nmi', 'info_retention', 'predictability']:
        m = results['metrics'][metric]
        sig = '***' if m['p_value'] < 0.001 else '**' if m['p_value'] < 0.01 else '*' if m['p_value'] < 0.05 else ''
        print(f"   {metric:<20} {m['observed']:>10.4f} {m['null_mean']:>10.4f} {m['p_value']:>10.4f} {m['effect_size']:>9.2f} {sig}")

    return results


def test_sequence_null(
    cluster_sequences: Dict[str, List[int]],
    theoretical_mapping: Dict[int, int],
    n_permutations: int = 1000
) -> Dict:
    """
    Test 2: Do transition dynamics differ from shuffled sequences?

    Null: Observed transitions = Shuffled sequence transitions
    """
    print("\n[TEST 2] Sequence Null Hypothesis")
    print("-" * 50)

    # Apply mapping
    animal_sequences = {
        name: apply_mapping(seq, theoretical_mapping)
        for name, seq in cluster_sequences.items()
    }

    # Observed transition matrix
    all_seq = [a for seq in animal_sequences.values() for a in seq]
    P_observed = compute_transition_matrix(all_seq, 4)
    H_observed = compute_entropy_rate(P_observed)

    # Null: shuffle each sequence (breaks temporal structure)
    null_entropy = []
    null_js_div = []

    for i in range(n_permutations):
        shuffled_seqs = {
            name: list(np.random.permutation(seq))
            for name, seq in animal_sequences.items()
        }
        all_shuffled = [a for seq in shuffled_seqs.values() for a in seq]
        P_shuffled = compute_transition_matrix(all_shuffled, 4)

        null_entropy.append(compute_entropy_rate(P_shuffled))

        # JS divergence from observed
        js = np.mean([jensenshannon(P_observed[i], P_shuffled[i]) for i in range(4)])
        null_js_div.append(js)

        if (i + 1) % 200 == 0:
            print(f"   Completed {i+1}/{n_permutations} permutations...")

    null_entropy = np.array(null_entropy)
    null_js_div = np.array(null_js_div)

    # The observed entropy rate should be LOWER than shuffled (more predictable)
    # P-value: proportion of shuffled that are as low or lower than observed
    p_entropy = np.mean(null_entropy <= H_observed)

    # Effect size
    d_entropy = (np.mean(null_entropy) - H_observed) / np.std(null_entropy)

    results = {
        'test': 'Sequence Null',
        'hypothesis': 'Observed transitions = Shuffled transitions',
        'n_permutations': n_permutations,
        'observed_entropy_rate': float(H_observed),
        'null_entropy_mean': float(np.mean(null_entropy)),
        'null_entropy_std': float(np.std(null_entropy)),
        'p_value': float(p_entropy),
        'effect_size': float(d_entropy),
        'significant': p_entropy < 0.05,
        'interpretation': 'Lower entropy = more predictable transitions'
    }

    print(f"\n   Results:")
    print(f"   Observed Entropy Rate: {H_observed:.4f} bits")
    print(f"   Null Mean Entropy:     {np.mean(null_entropy):.4f} ± {np.std(null_entropy):.4f} bits")
    print(f"   p-value:               {p_entropy:.4f}")
    print(f"   Effect size (d):       {d_entropy:.2f}")

    if p_entropy < 0.05:
        print(f"   *** SIGNIFICANT: Observed sequences have lower entropy (more structured)")
    else:
        print(f"   Not significant at p < 0.05")

    return results


def test_predictive_null(
    cluster_sequences: Dict[str, List[int]],
    theoretical_mapping: Dict[int, int],
    n_bootstrap: int = 1000
) -> Dict:
    """
    Test 3: Does state-based prediction beat marginal frequency prediction?

    Compare: P(next_state | current_state) vs P(next_state) marginal
    """
    print("\n[TEST 3] Predictive Null Hypothesis")
    print("-" * 50)

    # Apply mapping
    animal_sequences = {
        name: apply_mapping(seq, theoretical_mapping)
        for name, seq in cluster_sequences.items()
    }

    all_seq = [a for seq in animal_sequences.values() for a in seq]

    # Markov prediction accuracy
    P = compute_transition_matrix(all_seq, 4)

    markov_correct = 0
    marginal_correct = 0
    n_predictions = 0

    # Marginal distribution
    marginal = Counter(all_seq)
    most_common = marginal.most_common(1)[0][0]
    marginal_probs = np.array([marginal.get(i, 0) for i in range(4)])
    marginal_probs = marginal_probs / marginal_probs.sum()

    for seq in animal_sequences.values():
        for i in range(len(seq) - 1):
            current = seq[i]
            actual_next = seq[i + 1]

            # Markov prediction: argmax P(next | current)
            markov_pred = np.argmax(P[current])

            # Marginal prediction: argmax P(state)
            marginal_pred = most_common

            if markov_pred == actual_next:
                markov_correct += 1
            if marginal_pred == actual_next:
                marginal_correct += 1

            n_predictions += 1

    markov_acc = markov_correct / n_predictions
    marginal_acc = marginal_correct / n_predictions

    # Bootstrap CI for difference
    diff_bootstrap = []
    all_pairs = [(seq[i], seq[i+1]) for seq in animal_sequences.values() for i in range(len(seq)-1)]

    for _ in range(n_bootstrap):
        # Resample pairs
        indices = np.random.choice(len(all_pairs), size=len(all_pairs), replace=True)
        boot_pairs = [all_pairs[i] for i in indices]

        # Recompute transition matrix
        P_boot = np.zeros((4, 4)) + 0.01
        for current, next_state in boot_pairs:
            P_boot[current, next_state] += 1
        P_boot = P_boot / P_boot.sum(axis=1, keepdims=True)

        # Marginal from bootstrap
        boot_marginal = Counter([p[0] for p in boot_pairs])
        boot_most_common = boot_marginal.most_common(1)[0][0]

        # Accuracy
        markov_c = sum(1 for c, n in boot_pairs if np.argmax(P_boot[c]) == n)
        marginal_c = sum(1 for c, n in boot_pairs if boot_most_common == n)

        diff_bootstrap.append((markov_c - marginal_c) / len(boot_pairs))

    diff_bootstrap = np.array(diff_bootstrap)

    # P-value: is the improvement over marginal significant?
    observed_diff = markov_acc - marginal_acc
    p_value = np.mean(diff_bootstrap <= 0)  # Proportion where Markov is not better

    ci_lower = np.percentile(diff_bootstrap, 2.5)
    ci_upper = np.percentile(diff_bootstrap, 97.5)

    results = {
        'test': 'Predictive Null',
        'hypothesis': 'Markov prediction = Marginal prediction',
        'n_predictions': n_predictions,
        'markov_accuracy': float(markov_acc),
        'marginal_accuracy': float(marginal_acc),
        'improvement': float(observed_diff),
        'improvement_95ci': [float(ci_lower), float(ci_upper)],
        'p_value': float(p_value),
        'significant': p_value < 0.05 and ci_lower > 0
    }

    print(f"\n   Results:")
    print(f"   Markov Accuracy:   {markov_acc:.4f} ({markov_correct}/{n_predictions})")
    print(f"   Marginal Accuracy: {marginal_acc:.4f} ({marginal_correct}/{n_predictions})")
    print(f"   Improvement:       {observed_diff:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"   p-value:           {p_value:.4f}")

    if results['significant']:
        print(f"   *** SIGNIFICANT: State-based prediction beats marginal")

    return results


def plot_results(results: Dict, output_dir: str):
    """Generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Mapping null distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mapping_results = results['mapping_null']['metrics']

    for i, (metric, ax) in enumerate(zip(['nmi', 'info_retention', 'predictability'], axes)):
        data = mapping_results[metric]
        null_dist = np.array(results['mapping_null']['null_distributions'][metric])

        ax.hist(null_dist, bins=50, density=True, alpha=0.7, color='gray', label='Null (Random Mapping)')
        ax.axvline(data['observed'], color='red', linewidth=2, linestyle='--',
                   label=f"Observed = {data['observed']:.4f}")
        ax.axvline(data['null_mean'], color='blue', linewidth=1.5, linestyle=':',
                   label=f"Null Mean = {data['null_mean']:.4f}")

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f"{metric.replace('_', ' ').title()}\np = {data['p_value']:.4f}, d = {data['effect_size']:.2f}")
        ax.legend(fontsize=8)

    plt.suptitle('Test 1: Theoretical Mapping vs Random Mapping', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test1_mapping_null.png'), dpi=300)
    plt.close()

    # Plot 2: Summary
    fig, ax = plt.subplots(figsize=(10, 6))

    tests = ['Mapping\n(NMI)', 'Mapping\n(Info Ret.)', 'Mapping\n(Predict.)', 'Sequence', 'Predictive']
    p_values = [
        results['mapping_null']['metrics']['nmi']['p_value'],
        results['mapping_null']['metrics']['info_retention']['p_value'],
        results['mapping_null']['metrics']['predictability']['p_value'],
        results['sequence_null']['p_value'],
        results['predictive_null']['p_value']
    ]

    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    bars = ax.bar(tests, [-np.log10(max(p, 1e-10)) for p in p_values], color=colors, alpha=0.7)

    ax.axhline(-np.log10(0.05), color='black', linestyle='--', label='p = 0.05')
    ax.axhline(-np.log10(0.01), color='gray', linestyle=':', label='p = 0.01')

    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance of All Tests\n(Higher = More Significant)', fontweight='bold')
    ax.legend()

    # Add p-value labels
    for bar, p in zip(bars, p_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'p={p:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_significance.png'), dpi=300)
    plt.close()

    print(f"\nSaved visualizations to {output_dir}")


def main():
    """Run all proper null tests."""
    print("=" * 70)
    print("PROPER NULL MODEL TESTS")
    print("=" * 70)

    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(analysis_dir, 'null_model_tests')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading sequences...")
    cluster_sequences = load_sequences(analysis_dir)
    print(f"Loaded {len(cluster_sequences)} individuals")

    if not cluster_sequences:
        print("ERROR: No sequences found.")
        return

    # Run tests
    results = {}

    # Test 1: Mapping null
    results['mapping_null'] = test_mapping_null(
        cluster_sequences, THEORETICAL_MAPPING, n_permutations=10000
    )

    # Test 2: Sequence null
    results['sequence_null'] = test_sequence_null(
        cluster_sequences, THEORETICAL_MAPPING, n_permutations=1000
    )

    # Test 3: Predictive null
    results['predictive_null'] = test_predictive_null(
        cluster_sequences, THEORETICAL_MAPPING, n_bootstrap=1000
    )

    # Generate plots
    plot_results(results, output_dir)

    # Save results (without large distributions for JSON)
    save_results = {
        'mapping_null': {k: v for k, v in results['mapping_null'].items() if k != 'null_distributions'},
        'sequence_null': results['sequence_null'],
        'predictive_null': results['predictive_null']
    }

    results_path = os.path.join(output_dir, 'null_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nTest 1 - Mapping Null (Is theoretical mapping better than random?):")
    for metric in ['nmi', 'info_retention', 'predictability']:
        m = results['mapping_null']['metrics'][metric]
        sig = '***' if m['p_value'] < 0.001 else '**' if m['p_value'] < 0.01 else '*' if m['p_value'] < 0.05 else 'ns'
        print(f"   {metric}: p = {m['p_value']:.4f} ({sig})")

    print(f"\nTest 2 - Sequence Null (Are transitions structured?):")
    print(f"   p = {results['sequence_null']['p_value']:.4f}")

    print(f"\nTest 3 - Predictive Null (Does Markov beat marginal?):")
    print(f"   Improvement: {results['predictive_null']['improvement']:.4f}")
    print(f"   p = {results['predictive_null']['p_value']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
