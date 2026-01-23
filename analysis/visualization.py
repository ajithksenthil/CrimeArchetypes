"""
Visualization Module for Crime Archetype Analysis

Provides visualization functions for both K-Cluster and 4-Animal approaches,
including:
    - Transition matrix heatmaps
    - State transition diagrams (network graphs)
    - Stationary distribution plots
    - Entropy comparisons
    - Trait proxy radar charts
    - Time-block drift visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# COLOR SCHEMES
# =============================================================================

# 4-Animal colors
ANIMAL_COLORS = {
    'Seeking': '#2ecc71',     # Green - self-exploration
    'Directing': '#e74c3c',   # Red - other-exploitation
    'Conferring': '#3498db',  # Blue - other-exploration
    'Revising': '#f39c12'     # Orange - self-exploitation
}

# K-Cluster color palette (for up to 20 clusters)
CLUSTER_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


# =============================================================================
# TRANSITION MATRIX VISUALIZATIONS
# =============================================================================

def plot_transition_matrix(
    kernel: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Transition Matrix",
    cmap: str = "Blues",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    annotate: bool = True
):
    """
    Plot transition matrix as a heatmap.

    Args:
        kernel: Transition probability matrix
        labels: State labels
        title: Plot title
        cmap: Colormap name
        output_path: Path to save figure (if None, displays)
        figsize: Figure size
        annotate: Whether to show probability values
    """
    n = kernel.shape[0]
    labels = labels or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(kernel, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability')

    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title(title)

    # Annotate cells with values
    if annotate and n <= 10:
        for i in range(n):
            for j in range(n):
                value = kernel[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=color, fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved transition matrix plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_transition_diagram(
    kernel: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "State Transition Diagram",
    output_path: Optional[str] = None,
    threshold: float = 0.05,
    figsize: Tuple[int, int] = (10, 8),
    colors: Optional[Dict[str, str]] = None
):
    """
    Plot state transition diagram as a directed graph.

    Args:
        kernel: Transition probability matrix
        labels: State labels
        title: Plot title
        output_path: Path to save figure
        threshold: Minimum probability to show edge
        figsize: Figure size
        colors: Dict mapping label -> color (optional)
    """
    n = kernel.shape[0]
    labels = labels or [str(i) for i in range(n)]

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for i, label in enumerate(labels):
        G.add_node(label)

    # Add edges with weights
    for i in range(n):
        for j in range(n):
            if kernel[i, j] > threshold:
                G.add_edge(labels[i], labels[j], weight=kernel[i, j])

    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node colors
    if colors:
        node_colors = [colors.get(label, '#cccccc') for label in G.nodes()]
    else:
        node_colors = ['lightblue'] * len(G.nodes())

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=2000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    # Draw edges with varying widths
    edges = G.edges(data=True)
    edge_widths = [d['weight'] * 5 for _, _, d in edges]
    edge_colors = [d['weight'] for _, _, d in edges]

    edges_drawn = nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Reds,
        alpha=0.7,
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1"
    )

    # Add edge labels for significant transitions
    edge_labels = {(u, v): f'{d["weight"]:.2f}'
                   for u, v, d in edges if d['weight'] > 0.15}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)

    ax.set_title(title)
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved transition diagram to {output_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# STATIONARY DISTRIBUTION VISUALIZATIONS
# =============================================================================

def plot_stationary_distribution(
    stationary: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Stationary Distribution",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None
):
    """
    Plot stationary distribution as a bar chart.

    Args:
        stationary: Stationary distribution vector
        labels: State labels
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        colors: Bar colors
    """
    n = len(stationary)
    labels = labels or [str(i) for i in range(n)]

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(n), stationary, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.set_ylim(0, max(stationary) * 1.2)

    # Add value labels on bars
    for bar, val in zip(bars, stationary):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved stationary distribution plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_stationary_comparison(
    distributions: Dict[str, np.ndarray],
    labels: List[str],
    title: str = "Stationary Distribution Comparison",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compare stationary distributions from multiple sources.

    Args:
        distributions: Dict mapping source name -> stationary distribution
        labels: State labels
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    n_states = len(labels)
    n_sources = len(distributions)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_states)
    width = 0.8 / n_sources

    colors = plt.cm.Set2(np.linspace(0, 1, n_sources))

    for i, (source_name, dist) in enumerate(distributions.items()):
        offset = (i - n_sources/2 + 0.5) * width
        ax.bar(x + offset, dist, width, label=source_name, color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {output_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# TRAIT PROXY VISUALIZATIONS
# =============================================================================

def plot_trait_radar(
    traits: Dict[str, float],
    title: str = "Trait Profile",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Plot Big-5 trait proxies as a radar chart.

    Args:
        traits: Dict mapping trait name -> score (0-1)
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    categories = list(traits.keys())
    values = list(traits.values())

    # Close the polygon
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Plot data
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.plot(angles, values, color='teal', linewidth=2, marker='o')

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)

    # Set radial limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved trait radar plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_trait_comparison(
    trait_data: Dict[str, Dict[str, float]],
    title: str = "Trait Comparison",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compare trait profiles across multiple individuals.

    Args:
        trait_data: Dict mapping individual name -> traits dict
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    individuals = list(trait_data.keys())
    if not individuals:
        return

    trait_names = list(trait_data[individuals[0]].keys())
    n_individuals = len(individuals)
    n_traits = len(trait_names)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_traits)
    width = 0.8 / n_individuals

    colors = plt.cm.tab10(np.linspace(0, 1, n_individuals))

    for i, individual in enumerate(individuals):
        offset = (i - n_individuals/2 + 0.5) * width
        values = [trait_data[individual].get(t, 0) for t in trait_names]
        ax.bar(x + offset, values, width, label=individual, color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in trait_names])
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved trait comparison plot to {output_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# ENTROPY AND METRICS VISUALIZATIONS
# =============================================================================

def plot_entropy_comparison(
    entropy_data: Dict[str, float],
    max_entropy_data: Optional[Dict[str, float]] = None,
    title: str = "Entropy Rate Comparison",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Compare entropy rates across approaches or individuals.

    Args:
        entropy_data: Dict mapping name -> entropy rate
        max_entropy_data: Optional dict of maximum entropies for normalization
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    names = list(entropy_data.keys())
    entropies = list(entropy_data.values())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x, entropies, width, label='Actual Entropy', color='steelblue')

    if max_entropy_data:
        max_entropies = [max_entropy_data.get(n, e*1.5) for n, e in zip(names, entropies)]
        bars2 = ax.bar(x + width, max_entropies, width, label='Max Entropy',
                      color='lightsteelblue', alpha=0.7)

    ax.set_xticks(x + width/2 if max_entropy_data else x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Entropy Rate (bits)')
    ax.set_title(title)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved entropy comparison plot to {output_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# TIME-BLOCK DRIFT VISUALIZATIONS
# =============================================================================

def plot_temporal_drift(
    drift_data: Dict,
    title: str = "Temporal Dynamics",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot temporal drift in transition dynamics.

    Args:
        drift_data: Output from TimeBlockAnalysis.analyze_temporal_dynamics()
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    blocks = drift_data.get("blocks", [])
    metrics = drift_data.get("drift_metrics", {})

    if not blocks:
        logger.warning("No block data to visualize")
        return

    n_blocks = len(blocks)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Entropy rate over time
    ax1 = axes[0, 0]
    entropies = [b["entropy_rate"] for b in blocks]
    ax1.plot(range(n_blocks), entropies, 'o-', linewidth=2, markersize=8, color='teal')
    ax1.set_xlabel('Time Block')
    ax1.set_ylabel('Entropy Rate (bits)')
    ax1.set_title('Entropy Rate Over Time')
    ax1.set_xticks(range(n_blocks))
    ax1.set_xticklabels([f'Block {i+1}' for i in range(n_blocks)])

    # Plot 2: Kernel drift
    ax2 = axes[0, 1]
    if "kernel_drift" in metrics:
        drifts = metrics["kernel_drift"]
        ax2.bar(range(len(drifts)), drifts, color='coral', alpha=0.8)
        ax2.set_xlabel('Transition')
        ax2.set_ylabel('Frobenius Norm')
        ax2.set_title('Kernel Drift Between Blocks')
        ax2.set_xticks(range(len(drifts)))
        ax2.set_xticklabels([f'{i+1}→{i+2}' for i in range(len(drifts))])

    # Plot 3: Stationary distributions over time
    ax3 = axes[1, 0]
    n_states = len(blocks[0]["stationary"])
    x = np.arange(n_states)
    width = 0.8 / n_blocks

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_blocks))

    for i, block in enumerate(blocks):
        offset = (i - n_blocks/2 + 0.5) * width
        ax3.bar(x + offset, block["stationary"], width,
               label=f'Block {i+1}', color=colors[i], alpha=0.8)

    ax3.set_xlabel('State')
    ax3.set_ylabel('Probability')
    ax3.set_title('Stationary Distribution by Block')
    ax3.legend()

    # Plot 4: Summary metrics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    Temporal Analysis Summary
    {'='*30}

    Number of Blocks: {n_blocks}

    Entropy Range: {min(entropies):.3f} - {max(entropies):.3f}
    Entropy Trend: {'Increasing' if entropies[-1] > entropies[0] else 'Decreasing'}

    Total Kernel Drift: {metrics.get('total_kernel_drift', 'N/A'):.4f}
    """

    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved temporal drift plot to {output_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# COMPREHENSIVE COMPARISON PLOT
# =============================================================================

def plot_full_comparison(
    cluster_results: Dict,
    animal_results: Dict,
    output_dir: str
):
    """
    Generate full suite of comparison visualizations.

    Args:
        cluster_results: K-Cluster analysis results
        animal_results: 4-Animal analysis results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Side-by-side transition matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_kernel = np.array(cluster_results["aggregate_kernel"])
    a_kernel = np.array(animal_results["aggregate"]["kernel"])

    # K-Cluster
    im1 = axes[0].imshow(k_kernel, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f'K-Cluster (K={k_kernel.shape[0]})', fontsize=12)
    axes[0].set_xlabel('To Cluster')
    axes[0].set_ylabel('From Cluster')
    plt.colorbar(im1, ax=axes[0])

    # 4-Animal
    animal_labels = ['Seeking', 'Directing', 'Conferring', 'Revising']
    im2 = axes[1].imshow(a_kernel, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('4-Animal State Space', fontsize=12)
    axes[1].set_xlabel('To Animal')
    axes[1].set_ylabel('From Animal')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(animal_labels, rotation=45)
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(animal_labels)
    plt.colorbar(im2, ax=axes[1])

    # Annotate 4-Animal matrix
    for i in range(4):
        for j in range(4):
            color = 'white' if a_kernel[i,j] > 0.5 else 'black'
            axes[1].text(j, i, f'{a_kernel[i,j]:.2f}', ha='center', va='center',
                        color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_kernels.png'), dpi=150)
    plt.close()

    # 2. Stationary distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_stat = cluster_results["aggregate_stationary"]
    axes[0].bar(range(len(k_stat)), k_stat, color='steelblue', alpha=0.8)
    axes[0].set_title('K-Cluster Stationary Distribution')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Probability')

    a_stat = list(animal_results["aggregate"]["stationary_distribution"].values())
    colors = [ANIMAL_COLORS[a] for a in animal_labels]
    axes[1].bar(range(4), a_stat, color=colors, alpha=0.8)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(animal_labels, rotation=45)
    axes[1].set_title('4-Animal Stationary Distribution')
    axes[1].set_ylabel('Probability')

    for i, v in enumerate(a_stat):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_stationary.png'), dpi=150)
    plt.close()

    # 3. 4-Animal transition diagram
    plot_transition_diagram(
        a_kernel,
        labels=animal_labels,
        title='4-Animal State Transition Diagram',
        output_path=os.path.join(output_dir, 'animal_transition_diagram.png'),
        colors=ANIMAL_COLORS
    )

    # 4. Trait radar (if available)
    if "trait_proxies" in animal_results["aggregate"]:
        plot_trait_radar(
            animal_results["aggregate"]["trait_proxies"],
            title='Aggregate Trait Profile (4-Animal)',
            output_path=os.path.join(output_dir, 'aggregate_trait_radar.png')
        )

    logger.info(f"All comparison visualizations saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Visualization Module for Crime Archetype Analysis")
    print("="*50)

    # Create example data
    np.random.seed(42)

    # Example transition matrix (4 states)
    kernel = np.random.dirichlet([1, 1, 1, 1], size=4)

    labels = ['Seeking', 'Directing', 'Conferring', 'Revising']

    # Plot examples
    plot_transition_matrix(kernel, labels, title="Example Transition Matrix")

    plot_transition_diagram(kernel, labels, title="Example State Diagram",
                           colors=ANIMAL_COLORS)

    stationary = np.array([0.25, 0.30, 0.20, 0.25])
    plot_stationary_distribution(stationary, labels,
                                 colors=[ANIMAL_COLORS[l] for l in labels])

    traits = {
        'openness': 0.7,
        'conscientiousness': 0.4,
        'neuroticism': 0.6,
        'extraversion': 0.5,
        'agreeableness': 0.3
    }
    plot_trait_radar(traits, title="Example Trait Profile")
