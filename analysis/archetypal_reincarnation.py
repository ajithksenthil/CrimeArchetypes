#!/usr/bin/env python3
"""
Archetypal Reincarnation Analysis

Detects cross-criminal archetypal influence by computing pairwise transfer entropy
between life histories. High transfer entropy from Criminal A to Criminal B suggests
that A's archetypal pattern has "reincarnated" or manifested in B's life trajectory.

This analysis reveals:
- Source archetypes: Criminals whose patterns recur in many others
- Sink archetypes: Criminals who embody patterns from many predecessors
- Archetypal lineages: Chains of influence showing recurring patterns
- Archetypal clusters: Groups sharing the same behavioral "soul"

Based on the Computational Psychodynamics 4-Animal State Space framework.
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations, permutations
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

ANIMAL_NAMES = {0: "Seeking", 1: "Directing", 2: "Conferring", 3: "Revising"}
ANIMAL_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']


# =============================================================================
# CROSS-CRIMINAL TRANSFER ENTROPY
# =============================================================================

def compute_transfer_entropy(
    source_seq: List[int],
    target_seq: List[int],
    lag: int = 1,
    n_states: int = 4,
    history_length: int = 1
) -> float:
    """
    Compute transfer entropy from source sequence to target sequence.

    TE(Source → Target) measures how much knowing Source's past reduces
    uncertainty about Target's future, beyond what Target's own past tells us.

    For cross-criminal analysis, we align sequences by normalized position
    (life phase) rather than absolute time.

    Args:
        source_seq: Source criminal's behavioral sequence
        target_seq: Target criminal's behavioral sequence
        lag: How far back in source to look
        n_states: Number of behavioral states
        history_length: How much of target's own history to condition on

    Returns:
        Transfer entropy in bits (non-negative)
    """
    # Normalize sequences to same length for comparison
    # This aligns "life phases" rather than absolute time
    min_len = min(len(source_seq), len(target_seq))
    if min_len < lag + history_length + 1:
        return 0.0

    # Resample sequences to common length
    source_normalized = resample_sequence(source_seq, min_len)
    target_normalized = resample_sequence(target_seq, min_len)

    # Count joint occurrences: (target_history, source_lag) -> target_next
    joint_counts = defaultdict(lambda: np.zeros(n_states))
    target_only_counts = defaultdict(lambda: np.zeros(n_states))

    for t in range(max(history_length, lag), min_len):
        # Target's history
        target_history = tuple(target_normalized[t-history_length:t])
        # Source's lagged value
        source_lag = source_normalized[t - lag]
        # Target's next state
        target_next = target_normalized[t]

        # Count for TE computation
        joint_counts[(target_history, source_lag)][target_next] += 1
        target_only_counts[target_history][target_next] += 1

    # Compute conditional entropies
    # H(Target_t | Target_history, Source_lag)
    h_joint = 0.0
    total_joint = sum(counts.sum() for counts in joint_counts.values())
    if total_joint == 0:
        return 0.0

    for key, counts in joint_counts.items():
        p_condition = counts.sum() / total_joint
        if p_condition > 0:
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h_joint -= p_condition * np.sum(probs * np.log2(probs))

    # H(Target_t | Target_history)
    h_target = 0.0
    total_target = sum(counts.sum() for counts in target_only_counts.values())
    if total_target == 0:
        return 0.0

    for key, counts in target_only_counts.items():
        p_condition = counts.sum() / total_target
        if p_condition > 0:
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h_target -= p_condition * np.sum(probs * np.log2(probs))

    # Transfer entropy = H(Target | Target_history) - H(Target | Target_history, Source)
    te = h_target - h_joint
    return max(0.0, te)  # TE is non-negative


def resample_sequence(seq: List[int], target_len: int) -> List[int]:
    """Resample sequence to target length using nearest neighbor interpolation."""
    if len(seq) == target_len:
        return seq

    indices = np.linspace(0, len(seq) - 1, target_len)
    return [seq[int(round(i))] for i in indices]


def compute_symbolic_transfer_entropy(
    source_seq: List[int],
    target_seq: List[int],
    n_states: int = 4
) -> float:
    """
    Alternative TE computation based on transition pattern similarity.

    Measures how well source's transition patterns predict target's transitions.
    """
    if len(source_seq) < 3 or len(target_seq) < 3:
        return 0.0

    # Compute transition distributions
    def get_transition_dist(seq):
        counts = np.zeros((n_states, n_states))
        for i in range(len(seq) - 1):
            counts[seq[i], seq[i+1]] += 1
        # Flatten and normalize
        flat = counts.flatten()
        if flat.sum() > 0:
            return flat / flat.sum()
        return np.ones(n_states * n_states) / (n_states * n_states)

    source_dist = get_transition_dist(source_seq)
    target_dist = get_transition_dist(target_seq)

    # KL divergence from target to source (how well source "explains" target)
    eps = 1e-10
    kl = np.sum(target_dist * np.log2((target_dist + eps) / (source_dist + eps)))

    # Convert to similarity (bounded transfer entropy proxy)
    # Lower KL = higher similarity = higher "reincarnation"
    max_kl = np.log2(n_states * n_states)  # Maximum possible KL
    te_proxy = max(0, 1 - kl / max_kl)

    return float(te_proxy)


def significance_test_cross_te(
    source_seq: List[int],
    target_seq: List[int],
    observed_te: float,
    n_permutations: int = 500,
    n_states: int = 4
) -> float:
    """
    Test significance of cross-criminal TE using permutation test.

    Null hypothesis: No directed influence from source to target.
    """
    null_distribution = []

    for _ in range(n_permutations):
        # Shuffle source sequence to break any relationship
        shuffled_source = list(np.random.permutation(source_seq))
        null_te = compute_transfer_entropy(shuffled_source, target_seq, n_states=n_states)
        null_distribution.append(null_te)

    # p-value: proportion of null values >= observed
    p_value = np.mean([nte >= observed_te for nte in null_distribution])

    return p_value


# =============================================================================
# PAIRWISE ANALYSIS
# =============================================================================

def compute_reincarnation_matrix(
    sequences: Dict[str, List[int]],
    method: str = 'transfer_entropy',
    n_states: int = 4
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise "reincarnation" scores between all criminals.

    Returns matrix where M[i,j] = strength of archetypal influence from i to j.
    """
    criminals = sorted(sequences.keys())
    n = len(criminals)
    matrix = np.zeros((n, n))

    logger.info(f"Computing {n}x{n} reincarnation matrix ({n*(n-1)} pairs)...")

    total_pairs = n * (n - 1)
    computed = 0

    for i, source_name in enumerate(criminals):
        for j, target_name in enumerate(criminals):
            if i == j:
                continue  # No self-influence

            source_seq = sequences[source_name]
            target_seq = sequences[target_name]

            if method == 'transfer_entropy':
                score = compute_transfer_entropy(source_seq, target_seq, n_states=n_states)
            elif method == 'symbolic':
                score = compute_symbolic_transfer_entropy(source_seq, target_seq, n_states=n_states)
            else:
                score = compute_transfer_entropy(source_seq, target_seq, n_states=n_states)

            matrix[i, j] = score
            computed += 1

            if computed % 100 == 0:
                logger.info(f"  Computed {computed}/{total_pairs} pairs...")

    return matrix, criminals


def identify_archetypal_roles(
    matrix: np.ndarray,
    criminals: List[str],
    threshold_percentile: float = 75
) -> Dict:
    """
    Identify archetypal roles based on influence patterns.

    - Sources: High outgoing influence (their pattern recurs in many others)
    - Sinks: High incoming influence (they embody many archetypal patterns)
    - Hubs: High both ways (central archetypal nodes)
    """
    n = len(criminals)

    # Compute influence scores
    outgoing = matrix.sum(axis=1)  # Row sums = total influence exerted
    incoming = matrix.sum(axis=0)  # Column sums = total influence received

    # Normalize
    outgoing_norm = outgoing / outgoing.max() if outgoing.max() > 0 else outgoing
    incoming_norm = incoming / incoming.max() if incoming.max() > 0 else incoming

    # Thresholds
    out_threshold = np.percentile(outgoing, threshold_percentile)
    in_threshold = np.percentile(incoming, threshold_percentile)

    roles = {
        'sources': [],      # High outgoing, pattern "originators"
        'sinks': [],        # High incoming, pattern "receivers"
        'hubs': [],         # High both, central archetypes
        'isolated': []      # Low both, unique patterns
    }

    for i, name in enumerate(criminals):
        is_source = outgoing[i] >= out_threshold
        is_sink = incoming[i] >= in_threshold

        if is_source and is_sink:
            roles['hubs'].append({
                'name': name,
                'outgoing': float(outgoing[i]),
                'incoming': float(incoming[i])
            })
        elif is_source:
            roles['sources'].append({
                'name': name,
                'outgoing': float(outgoing[i]),
                'incoming': float(incoming[i])
            })
        elif is_sink:
            roles['sinks'].append({
                'name': name,
                'outgoing': float(outgoing[i]),
                'incoming': float(incoming[i])
            })
        else:
            roles['isolated'].append({
                'name': name,
                'outgoing': float(outgoing[i]),
                'incoming': float(incoming[i])
            })

    # Sort by influence
    for role in roles:
        if role == 'sources':
            roles[role].sort(key=lambda x: x['outgoing'], reverse=True)
        elif role == 'sinks':
            roles[role].sort(key=lambda x: x['incoming'], reverse=True)
        elif role == 'hubs':
            roles[role].sort(key=lambda x: x['outgoing'] + x['incoming'], reverse=True)

    return roles


def find_archetypal_lineages(
    matrix: np.ndarray,
    criminals: List[str],
    threshold_percentile: float = 90
) -> List[List[str]]:
    """
    Find chains of archetypal influence (lineages).

    A lineage is a path A → B → C where each link has high transfer entropy.
    """
    threshold = np.percentile(matrix[matrix > 0], threshold_percentile)

    # Build directed graph of strong connections
    G = nx.DiGraph()
    G.add_nodes_from(criminals)

    for i, source in enumerate(criminals):
        for j, target in enumerate(criminals):
            if i != j and matrix[i, j] >= threshold:
                G.add_edge(source, target, weight=matrix[i, j])

    # Find all simple paths (lineages)
    lineages = []

    # Start from nodes with high outgoing but low incoming (sources)
    outgoing = matrix.sum(axis=1)
    incoming = matrix.sum(axis=0)

    source_indices = np.where((outgoing > np.median(outgoing)) &
                               (incoming < np.median(incoming)))[0]

    for source_idx in source_indices:
        source = criminals[source_idx]
        # Find paths from this source
        for target in criminals:
            if target != source:
                try:
                    paths = list(nx.all_simple_paths(G, source, target, cutoff=4))
                    for path in paths:
                        if len(path) >= 3:  # At least 3 nodes in lineage
                            lineages.append(path)
                except nx.NetworkXError:
                    continue

    # Remove duplicates and sort by length
    unique_lineages = []
    seen = set()
    for lineage in sorted(lineages, key=len, reverse=True):
        key = tuple(lineage)
        if key not in seen:
            seen.add(key)
            unique_lineages.append(lineage)

    return unique_lineages[:20]  # Top 20 lineages


def find_archetypal_clusters(
    matrix: np.ndarray,
    criminals: List[str],
    n_clusters: int = 4
) -> Dict[int, List[str]]:
    """
    Cluster criminals by archetypal similarity.

    Criminals in the same cluster share the same "archetypal soul".
    """
    # Create symmetric similarity matrix
    similarity = (matrix + matrix.T) / 2

    # Convert to distance
    max_sim = similarity.max()
    distance = max_sim - similarity if max_sim > 0 else similarity
    np.fill_diagonal(distance, 0)

    # Hierarchical clustering
    condensed = squareform(distance)
    Z = linkage(condensed, method='average')

    # Cut tree to get clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    clusters = defaultdict(list)
    for name, label in zip(criminals, labels):
        clusters[int(label)].append(name)

    return dict(clusters)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_reincarnation_matrix(
    matrix: np.ndarray,
    criminals: List[str],
    output_path: str
):
    """Plot the reincarnation (transfer entropy) matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Shorten names for display
    short_names = [c.split('_')[0][:12] for c in criminals]

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(criminals)))
    ax.set_yticks(range(len(criminals)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('Target (Influenced)', fontsize=11)
    ax.set_ylabel('Source (Influencer)', fontsize=11)
    ax.set_title('Archetypal Reincarnation Matrix\n(Transfer Entropy: Source → Target)',
                fontweight='bold', fontsize=12)

    plt.colorbar(im, ax=ax, label='Transfer Entropy (bits)', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_influence_network(
    matrix: np.ndarray,
    criminals: List[str],
    roles: Dict,
    output_path: str,
    threshold_percentile: float = 85
):
    """Plot directed network of archetypal influence."""
    threshold = np.percentile(matrix[matrix > 0], threshold_percentile)

    # Build graph
    G = nx.DiGraph()

    for i, source in enumerate(criminals):
        for j, target in enumerate(criminals):
            if i != j and matrix[i, j] >= threshold:
                G.add_edge(source, target, weight=matrix[i, j])

    if len(G.edges()) == 0:
        logger.warning("No edges above threshold for network plot")
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Color nodes by role
    node_colors = []
    node_sizes = []

    source_names = {r['name'] for r in roles['sources']}
    sink_names = {r['name'] for r in roles['sinks']}
    hub_names = {r['name'] for r in roles['hubs']}

    for node in G.nodes():
        if node in hub_names:
            node_colors.append('#e74c3c')  # Red for hubs
            node_sizes.append(800)
        elif node in source_names:
            node_colors.append('#f39c12')  # Orange for sources
            node_sizes.append(600)
        elif node in sink_names:
            node_colors.append('#3498db')  # Blue for sinks
            node_sizes.append(600)
        else:
            node_colors.append('#95a5a6')  # Gray for others
            node_sizes.append(300)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, ax=ax)

    # Edge weights for width
    edges = G.edges(data=True)
    weights = [e[2]['weight'] * 5 for e in edges]

    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5,
                          width=weights, arrows=True, arrowsize=15,
                          connectionstyle='arc3,rad=0.1', ax=ax)

    # Labels
    labels = {n: n.split('_')[0][:10] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Hubs (Central Archetypes)'),
        mpatches.Patch(color='#f39c12', label='Sources (Pattern Originators)'),
        mpatches.Patch(color='#3498db', label='Sinks (Pattern Receivers)'),
        mpatches.Patch(color='#95a5a6', label='Others')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_title('Archetypal Influence Network\n(Arrows show direction of "reincarnation")',
                fontweight='bold', fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_archetypal_roles(
    matrix: np.ndarray,
    criminals: List[str],
    output_path: str
):
    """Plot source vs sink influence for each criminal."""
    fig, ax = plt.subplots(figsize=(10, 8))

    outgoing = matrix.sum(axis=1)  # Influence exerted
    incoming = matrix.sum(axis=0)  # Influence received

    # Normalize
    outgoing_norm = outgoing / outgoing.max() if outgoing.max() > 0 else outgoing
    incoming_norm = incoming / incoming.max() if incoming.max() > 0 else incoming

    # Scatter plot
    scatter = ax.scatter(outgoing_norm, incoming_norm, s=100, alpha=0.7,
                        c=outgoing_norm + incoming_norm, cmap='RdYlBu_r')

    # Add labels for extreme points
    for i, name in enumerate(criminals):
        if outgoing_norm[i] > 0.7 or incoming_norm[i] > 0.7:
            ax.annotate(name.split('_')[0][:10], (outgoing_norm[i], incoming_norm[i]),
                       fontsize=8, alpha=0.8)

    # Quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax.text(0.75, 0.75, 'HUBS\n(Central Archetypes)', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.25, 'SOURCES\n(Pattern Originators)', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.25, 0.75, 'SINKS\n(Pattern Receivers)', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.25, 0.25, 'ISOLATED\n(Unique Patterns)', ha='center', fontsize=10, alpha=0.7)

    ax.set_xlabel('Outgoing Influence (Normalized)\n"How much this archetype reincarnates in others"')
    ax.set_ylabel('Incoming Influence (Normalized)\n"How much other archetypes reincarnated here"')
    ax.set_title('Archetypal Roles: Sources vs Sinks',
                fontweight='bold', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.colorbar(scatter, ax=ax, label='Total Influence')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_lineages(
    lineages: List[List[str]],
    matrix: np.ndarray,
    criminals: List[str],
    output_path: str
):
    """Plot top archetypal lineages."""
    if not lineages:
        logger.warning("No lineages to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    criminal_idx = {c: i for i, c in enumerate(criminals)}

    # Plot top 10 lineages
    y_positions = np.arange(min(10, len(lineages)))

    for y, lineage in enumerate(lineages[:10]):
        x_positions = np.arange(len(lineage))

        # Draw nodes
        ax.scatter(x_positions, [y] * len(lineage), s=200, zorder=10,
                  c=[ANIMAL_COLORS[1]], alpha=0.8)

        # Draw edges with weights
        for i in range(len(lineage) - 1):
            source_idx = criminal_idx[lineage[i]]
            target_idx = criminal_idx[lineage[i+1]]
            weight = matrix[source_idx, target_idx]

            ax.annotate('', xy=(x_positions[i+1]-0.15, y),
                       xytext=(x_positions[i]+0.15, y),
                       arrowprops=dict(arrowstyle='->', color='gray',
                                      lw=1 + weight*3))

            # Weight label
            ax.text((x_positions[i] + x_positions[i+1])/2, y + 0.15,
                   f'{weight:.3f}', ha='center', fontsize=7, alpha=0.7)

        # Node labels
        for i, name in enumerate(lineage):
            ax.text(x_positions[i], y - 0.3, name.split('_')[0][:8],
                   ha='center', fontsize=7, rotation=45)

    ax.set_ylim(-0.8, min(10, len(lineages)) - 0.2)
    ax.set_xlim(-0.5, max(len(l) for l in lineages[:10]) - 0.5)
    ax.set_ylabel('Lineage Rank')
    ax.set_xlabel('Position in Lineage')
    ax.set_title('Top Archetypal Lineages\n(Chains of "Reincarnation")',
                fontweight='bold', fontsize=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'#{i+1}' for i in range(min(10, len(lineages)))])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_classified_sequences(results_dir: str) -> Dict[str, List[int]]:
    """Load pre-classified sequences."""
    results_dir = Path(results_dir)

    comparison_dirs = sorted(results_dir.glob("four_animal_comparison_*"))
    if not comparison_dirs:
        raise FileNotFoundError("No classification results found.")

    latest_dir = comparison_dirs[-1]
    labels_file = latest_dir / "animal_labels.json"

    with open(labels_file, 'r') as f:
        data = json.load(f)

    # Reconstruct sequences per criminal
    sequences = defaultdict(list)

    for i, (label, criminal) in enumerate(zip(data['labels'], data['event_to_criminal'])):
        sequences[criminal].append(label)

    # Filter to sequences with enough events
    sequences = {c: seq for c, seq in sequences.items() if len(seq) >= 5}

    return dict(sequences)


def run_reincarnation_analysis(results_dir: str, output_dir: str) -> Dict:
    """Run complete archetypal reincarnation analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(output_dir) / f"archetypal_reincarnation_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ARCHETYPAL REINCARNATION ANALYSIS")
    logger.info("Cross-Criminal Transfer Entropy Study")
    logger.info("=" * 70)

    # Load sequences
    logger.info("\n[1] Loading classified sequences...")
    sequences = load_classified_sequences(results_dir)
    logger.info(f"Loaded {len(sequences)} individuals")

    # Compute reincarnation matrix
    logger.info("\n[2] Computing pairwise transfer entropy...")
    matrix, criminals = compute_reincarnation_matrix(sequences, method='transfer_entropy')

    # Also compute symbolic TE for comparison
    logger.info("\n[3] Computing symbolic transfer entropy (pattern similarity)...")
    symbolic_matrix, _ = compute_reincarnation_matrix(sequences, method='symbolic')

    # Identify roles
    logger.info("\n[4] Identifying archetypal roles...")
    roles = identify_archetypal_roles(matrix, criminals)

    print("\n" + "=" * 60)
    print("ARCHETYPAL ROLES")
    print("=" * 60)

    print(f"\nSOURCES (Pattern Originators): {len(roles['sources'])}")
    for r in roles['sources'][:5]:
        print(f"  - {r['name']}: outgoing={r['outgoing']:.4f}")

    print(f"\nSINKS (Pattern Receivers): {len(roles['sinks'])}")
    for r in roles['sinks'][:5]:
        print(f"  - {r['name']}: incoming={r['incoming']:.4f}")

    print(f"\nHUBS (Central Archetypes): {len(roles['hubs'])}")
    for r in roles['hubs'][:5]:
        print(f"  - {r['name']}: total={r['outgoing']+r['incoming']:.4f}")

    # Find lineages
    logger.info("\n[5] Finding archetypal lineages...")
    lineages = find_archetypal_lineages(matrix, criminals)

    print(f"\nTOP ARCHETYPAL LINEAGES: {len(lineages)}")
    for i, lineage in enumerate(lineages[:5]):
        print(f"  {i+1}. {' → '.join([l.split('_')[0][:10] for l in lineage])}")

    # Find clusters
    logger.info("\n[6] Clustering by archetypal similarity...")
    clusters = find_archetypal_clusters(matrix, criminals, n_clusters=4)

    print(f"\nARCHETYPAL CLUSTERS (Shared 'Souls'):")
    for cluster_id, members in sorted(clusters.items()):
        print(f"  Cluster {cluster_id}: {len(members)} members")
        print(f"    {', '.join([m.split('_')[0][:10] for m in members[:5]])}")

    # Generate visualizations
    logger.info("\n[7] Generating visualizations...")

    plot_reincarnation_matrix(matrix, criminals, study_dir / 'reincarnation_matrix.png')
    plot_influence_network(matrix, criminals, roles, study_dir / 'influence_network.png')
    plot_archetypal_roles(matrix, criminals, study_dir / 'archetypal_roles.png')

    if lineages:
        plot_lineages(lineages, matrix, criminals, study_dir / 'archetypal_lineages.png')

    # Compile results
    results = {
        'n_individuals': len(sequences),
        'matrix_stats': {
            'mean': float(matrix.mean()),
            'std': float(matrix.std()),
            'max': float(matrix.max()),
            'nonzero_fraction': float((matrix > 0).sum() / (matrix.size - len(criminals)))
        },
        'roles': {
            'sources': roles['sources'],
            'sinks': roles['sinks'],
            'hubs': roles['hubs'],
            'n_isolated': len(roles['isolated'])
        },
        'lineages': lineages[:20],
        'clusters': {str(k): v for k, v in clusters.items()}
    }

    # Save results
    with open(study_dir / 'reincarnation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save matrices
    np.save(study_dir / 'reincarnation_matrix.npy', matrix)
    np.save(study_dir / 'symbolic_matrix.npy', symbolic_matrix)

    # Save criminal order
    with open(study_dir / 'criminal_order.json', 'w') as f:
        json.dump(criminals, f, indent=2)

    logger.info(f"\nResults saved to: {study_dir}")

    return results


if __name__ == "__main__":
    results_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"
    output_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"

    run_reincarnation_analysis(results_dir, output_dir)
