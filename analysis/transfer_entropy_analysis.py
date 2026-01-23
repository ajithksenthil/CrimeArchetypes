#!/usr/bin/env python3
"""
Transfer Entropy Analysis for Criminal Behavioral Sequences

Measures information transfer and behavioral influence patterns across
criminals using the 4-Animal State Space framework.

Transfer entropy quantifies the directed information flow from one
process to another, revealing which behavioral patterns "influence"
or predict others.
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

ANIMAL_NAMES = {0: "Seeking", 1: "Directing", 2: "Conferring", 3: "Revising"}


# =============================================================================
# TRANSFER ENTROPY COMPUTATION
# =============================================================================

def compute_entropy(sequence: List[int], n_states: int = 4) -> float:
    """Compute Shannon entropy of a sequence."""
    counts = np.bincount(sequence, minlength=n_states)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))


def compute_conditional_entropy(
    target: List[int],
    source: List[int],
    lag: int = 1,
    n_states: int = 4
) -> float:
    """
    Compute H(Y_t | Y_{t-1}, X_{t-lag}).

    This is the conditional entropy of the target given its own past
    and the source's past.
    """
    if len(target) <= lag or len(source) <= lag:
        return 0.0

    # Build joint counts: (Y_{t-1}, X_{t-lag}) -> Y_t
    joint_counts = defaultdict(lambda: np.zeros(n_states))

    for t in range(max(1, lag), min(len(target), len(source))):
        y_prev = target[t - 1]
        x_lag = source[t - lag] if t >= lag else source[0]
        y_curr = target[t]

        joint_counts[(y_prev, x_lag)][y_curr] += 1

    # Compute conditional entropy
    total = sum(counts.sum() for counts in joint_counts.values())
    if total == 0:
        return 0.0

    cond_entropy = 0.0
    for (y_prev, x_lag), counts in joint_counts.items():
        p_condition = counts.sum() / total
        if p_condition > 0:
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            cond_entropy -= p_condition * np.sum(probs * np.log2(probs))

    return cond_entropy


def compute_transfer_entropy(
    source: List[int],
    target: List[int],
    lag: int = 1,
    n_states: int = 4
) -> float:
    """
    Compute transfer entropy from source to target.

    TE(X -> Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

    This measures how much knowing X's past reduces uncertainty about Y's future,
    beyond what Y's own past tells us.
    """
    if len(target) < 2 or len(source) < lag + 1:
        return 0.0

    # H(Y_t | Y_{t-1}) - entropy of target given only its own past
    transition_counts = np.zeros((n_states, n_states))
    for t in range(1, len(target)):
        transition_counts[target[t-1], target[t]] += 1

    h_given_self = 0.0
    for i in range(n_states):
        row_sum = transition_counts[i].sum()
        if row_sum > 0:
            probs = transition_counts[i] / row_sum
            probs = probs[probs > 0]
            p_state = row_sum / transition_counts.sum()
            h_given_self -= p_state * np.sum(probs * np.log2(probs))

    # H(Y_t | Y_{t-1}, X_{t-lag})
    h_given_both = compute_conditional_entropy(target, source, lag, n_states)

    # Transfer entropy
    te = h_given_self - h_given_both
    return max(0, te)  # TE should be non-negative


def compute_normalized_transfer_entropy(
    source: List[int],
    target: List[int],
    lag: int = 1,
    n_states: int = 4
) -> float:
    """
    Compute normalized transfer entropy (0 to 1 scale).

    Normalized by H(Y_t | Y_{t-1}) so result is in [0, 1].
    """
    te = compute_transfer_entropy(source, target, lag, n_states)

    # Compute H(Y_t | Y_{t-1}) for normalization
    transition_counts = np.zeros((n_states, n_states))
    for t in range(1, len(target)):
        transition_counts[target[t-1], target[t]] += 1

    h_given_self = 0.0
    total = transition_counts.sum()
    if total > 0:
        for i in range(n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                probs = transition_counts[i] / row_sum
                probs = probs[probs > 0]
                p_state = row_sum / total
                h_given_self -= p_state * np.sum(probs * np.log2(probs))

    if h_given_self > 0:
        return te / h_given_self
    return 0.0


def significance_test_te(
    source: List[int],
    target: List[int],
    lag: int = 1,
    n_permutations: int = 1000,
    n_states: int = 4
) -> Tuple[float, float]:
    """
    Test significance of transfer entropy using permutation test.

    Returns (observed_te, p_value).
    """
    observed_te = compute_transfer_entropy(source, target, lag, n_states)

    # Permutation test: shuffle source to break temporal relationship
    null_distribution = []
    for _ in range(n_permutations):
        shuffled_source = np.random.permutation(source).tolist()
        null_te = compute_transfer_entropy(shuffled_source, target, lag, n_states)
        null_distribution.append(null_te)

    # p-value: proportion of null values >= observed
    p_value = np.mean([nte >= observed_te for nte in null_distribution])

    return observed_te, p_value


# =============================================================================
# CROSS-CRIMINAL ANALYSIS
# =============================================================================

def compute_archetypal_influence(
    sequences: Dict[str, List[int]],
    n_states: int = 4
) -> Dict:
    """
    Compute how each behavioral archetype influences transitions across all criminals.

    This aggregates sequences to find universal influence patterns.
    """
    # Aggregate all sequences
    all_events = []
    for seq in sequences.values():
        all_events.extend(seq)

    # For each state, compute its "influence" on subsequent states
    influence_matrix = np.zeros((n_states, n_states))
    state_counts = np.zeros(n_states)

    for seq in sequences.values():
        for t in range(len(seq) - 1):
            curr_state = seq[t]
            next_state = seq[t + 1]
            influence_matrix[curr_state, next_state] += 1
            state_counts[curr_state] += 1

    # Normalize to get transition probabilities
    for i in range(n_states):
        if state_counts[i] > 0:
            influence_matrix[i] /= state_counts[i]

    # Compute "influence strength" for each state
    # = how much it pulls the system toward itself (attractor strength)
    attractor_strength = {}
    for i in range(n_states):
        # Self-loop probability (persistence)
        persistence = influence_matrix[i, i]
        # Probability of transitioning TO this state from others
        attractiveness = np.mean([influence_matrix[j, i] for j in range(n_states) if j != i])
        attractor_strength[ANIMAL_NAMES[i]] = {
            "persistence": float(persistence),
            "attractiveness": float(attractiveness),
            "combined": float(persistence * 0.5 + attractiveness * 0.5)
        }

    return {
        "influence_matrix": influence_matrix.tolist(),
        "attractor_strength": attractor_strength,
        "state_counts": {ANIMAL_NAMES[i]: int(state_counts[i]) for i in range(n_states)}
    }


def compute_criminal_similarity(
    sequences: Dict[str, List[int]],
    n_states: int = 4
) -> np.ndarray:
    """
    Compute pairwise similarity between criminals based on their behavioral patterns.

    Uses Jensen-Shannon divergence of transition distributions.
    """
    criminals = list(sequences.keys())
    n = len(criminals)
    similarity_matrix = np.zeros((n, n))

    # Compute transition distribution for each criminal
    def get_transition_dist(seq):
        counts = np.zeros((n_states, n_states))
        for t in range(len(seq) - 1):
            counts[seq[t], seq[t+1]] += 1
        # Flatten and normalize
        flat = counts.flatten()
        if flat.sum() > 0:
            flat = flat / flat.sum()
        else:
            flat = np.ones(n_states * n_states) / (n_states * n_states)
        return flat

    transition_dists = {c: get_transition_dist(seq) for c, seq in sequences.items()}

    # Compute pairwise JS divergence
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        eps = 1e-10
        kl_pm = np.sum(p * np.log2((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log2((q + eps) / (m + eps)))
        return 0.5 * (kl_pm + kl_qm)

    for i, c1 in enumerate(criminals):
        for j, c2 in enumerate(criminals):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                jsd = js_divergence(transition_dists[c1], transition_dists[c2])
                sim = 1 - np.sqrt(jsd)  # Convert to similarity
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

    return similarity_matrix, criminals


def identify_behavioral_clusters(
    similarity_matrix: np.ndarray,
    criminals: List[str],
    n_clusters: int = 3
) -> Dict:
    """
    Cluster criminals based on behavioral similarity.
    """
    from sklearn.cluster import AgglomerativeClustering

    # Use similarity as affinity
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)

    clusters = defaultdict(list)
    for criminal, label in zip(criminals, labels):
        clusters[f"Cluster_{label}"].append(criminal)

    return dict(clusters)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_transfer_entropy_matrix(
    te_matrix: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "Transfer Entropy Matrix"
):
    """Plot transfer entropy as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(te_matrix, cmap='YlOrRd', vmin=0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')
    ax.set_title(title)

    # Add values
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if te_matrix[i, j] > te_matrix.max() * 0.5 else 'black'
            ax.text(j, i, f'{te_matrix[i, j]:.3f}', ha='center', va='center',
                   color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Transfer Entropy (bits)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    criminals: List[str],
    output_path: str
):
    """Plot criminal similarity matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Cluster for better visualization
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    distance = 1 - similarity_matrix
    np.fill_diagonal(distance, 0)
    condensed = squareform(distance)
    Z = linkage(condensed, method='average')

    # Reorder matrix
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)
    reordered = similarity_matrix[order][:, order]
    reordered_names = [criminals[i] for i in order]

    im = ax.imshow(reordered, cmap='RdYlBu', vmin=0, vmax=1)

    ax.set_xticks(range(len(criminals)))
    ax.set_yticks(range(len(criminals)))
    ax.set_xticklabels(reordered_names, rotation=90, fontsize=6)
    ax.set_yticklabels(reordered_names, fontsize=6)
    ax.set_title('Criminal Behavioral Similarity\n(Based on Transition Patterns)')

    plt.colorbar(im, ax=ax, label='Similarity')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attractor_analysis(
    attractor_data: Dict,
    output_path: str
):
    """Plot attractor strength analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    states = list(attractor_data.keys())
    persistence = [attractor_data[s]['persistence'] for s in states]
    attractiveness = [attractor_data[s]['attractiveness'] for s in states]

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    # Panel A: Persistence
    ax = axes[0]
    bars = ax.bar(states, persistence, color=colors, edgecolor='black')
    ax.set_ylabel('Self-Loop Probability')
    ax.set_title('A. State Persistence', fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, persistence):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', fontsize=9)

    # Panel B: Attractiveness
    ax = axes[1]
    bars = ax.bar(states, attractiveness, color=colors, edgecolor='black')
    ax.set_ylabel('Mean Transition Probability (from other states)')
    ax.set_title('B. State Attractiveness', fontweight='bold')
    ax.set_ylim(0, 0.8)
    for bar, val in zip(bars, attractiveness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_classified_events(results_dir: str) -> Dict[str, List[int]]:
    """Load pre-classified events from previous study."""
    results_dir = Path(results_dir)

    # Find the most recent comparison directory
    comparison_dirs = sorted(results_dir.glob("four_animal_comparison_*"))
    if not comparison_dirs:
        raise FileNotFoundError("No classification results found. Run embedding comparison first.")

    latest_dir = comparison_dirs[-1]
    labels_file = latest_dir / "animal_labels.json"

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    with open(labels_file, 'r') as f:
        data = json.load(f)

    # We need to reconstruct sequences - load from original data
    logger.info("Loading original data to reconstruct sequences...")

    data_dir = Path("/Users/ajithsenthil/Desktop/CrimeArchetypes/mnt/data/csv")
    sequences = {}
    event_idx = 0

    for filepath in sorted(data_dir.glob("Type1_*.csv")):
        name = filepath.stem[6:]  # Remove "Type1_"
        seq = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)

                for row in reader:
                    if len(row) >= 3 and row[2].strip():
                        if event_idx < len(data['labels']):
                            seq.append(data['labels'][event_idx])
                        event_idx += 1
        except:
            continue

        if len(seq) >= 3:
            sequences[name] = seq

    return sequences


def run_transfer_entropy_analysis(results_dir: str, output_dir: str) -> Dict:
    """Run complete transfer entropy analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(output_dir) / f"transfer_entropy_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TRANSFER ENTROPY ANALYSIS")
    logger.info("=" * 70)

    # Load classified sequences
    logger.info("\n[1] Loading classified sequences...")
    sequences = load_classified_events(results_dir)
    logger.info(f"Loaded {len(sequences)} individuals")

    # Compute archetypal influence
    logger.info("\n[2] Computing archetypal influence patterns...")
    influence = compute_archetypal_influence(sequences)

    print("\nAttractor Strength Analysis:")
    print("-" * 50)
    for state, metrics in influence['attractor_strength'].items():
        print(f"  {state}:")
        print(f"    Persistence: {metrics['persistence']:.3f}")
        print(f"    Attractiveness: {metrics['attractiveness']:.3f}")

    # Compute state-to-state transfer entropy
    logger.info("\n[3] Computing state-level transfer entropy...")

    # Aggregate by state type
    state_sequences = {i: [] for i in range(4)}
    for seq in sequences.values():
        for state in seq:
            state_sequences[state].append(state)

    # Compute TE between state types (conceptual - based on transition patterns)
    te_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                # Use aggregated sequence to compute TE
                te_matrix[i, j] = influence['influence_matrix'][i][j]

    # Compute criminal similarity
    logger.info("\n[4] Computing criminal similarity matrix...")
    similarity_matrix, criminals = compute_criminal_similarity(sequences)

    # Identify clusters
    logger.info("\n[5] Identifying behavioral clusters...")
    clusters = identify_behavioral_clusters(similarity_matrix, criminals, n_clusters=3)

    print("\nBehavioral Clusters:")
    print("-" * 50)
    for cluster, members in clusters.items():
        print(f"  {cluster}: {len(members)} individuals")
        print(f"    Examples: {', '.join(members[:3])}")

    # Generate visualizations
    logger.info("\n[6] Generating visualizations...")

    plot_attractor_analysis(
        influence['attractor_strength'],
        study_dir / 'attractor_analysis.png'
    )

    plot_similarity_heatmap(
        similarity_matrix,
        criminals,
        study_dir / 'criminal_similarity.png'
    )

    plot_transfer_entropy_matrix(
        np.array(influence['influence_matrix']),
        [ANIMAL_NAMES[i] for i in range(4)],
        study_dir / 'influence_matrix.png',
        title='State Influence Matrix (Transition Probabilities)'
    )

    # Compile results
    results = {
        "n_individuals": len(sequences),
        "archetypal_influence": influence,
        "behavioral_clusters": clusters,
        "similarity_statistics": {
            "mean": float(np.mean(similarity_matrix[np.triu_indices(len(criminals), k=1)])),
            "std": float(np.std(similarity_matrix[np.triu_indices(len(criminals), k=1)])),
            "min": float(np.min(similarity_matrix[np.triu_indices(len(criminals), k=1)])),
            "max": float(np.max(similarity_matrix[np.triu_indices(len(criminals), k=1)]))
        }
    }

    # Save results
    with open(study_dir / 'transfer_entropy_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {study_dir}")

    return results


if __name__ == "__main__":
    results_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"
    output_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"

    run_transfer_entropy_analysis(results_dir, output_dir)
