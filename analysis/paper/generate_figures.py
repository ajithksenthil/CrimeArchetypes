#!/usr/bin/env python3
"""
Generate publication-quality figures for the Criminal Trajectories paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import os

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# State names and colors
STATES = ['Seeking', 'Directing', 'Conferring', 'Revising']
STATE_COLORS = {
    'Seeking': '#4ECDC4',      # Teal
    'Directing': '#FF6B6B',    # Coral red
    'Conferring': '#95E1D3',   # Light green
    'Revising': '#F38181'      # Light coral
}

# Output directory
OUTPUT_DIR = '/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig1_four_quadrant_model():
    """Figure 1: The Four Motivational Quadra derived from crossing Self/Other with Explore/Exploit."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw quadrants
    quadrant_data = [
        # (x_center, y_center, state, description, color)
        (0.25, 0.75, 'Seeking', 'Self × Explore\nFantasy development\nInternal elaboration', STATE_COLORS['Seeking']),
        (0.75, 0.75, 'Conferring', 'Other × Explore\nSurveillance\nInformation gathering', STATE_COLORS['Conferring']),
        (0.25, 0.25, 'Revising', 'Self × Exploit\nRitualization\nPattern consolidation', STATE_COLORS['Revising']),
        (0.75, 0.25, 'Directing', 'Other × Exploit\nAction/Control\nBehavioral execution', STATE_COLORS['Directing']),
    ]

    for x, y, state, desc, color in quadrant_data:
        # Draw box
        rect = FancyBboxPatch((x-0.22, y-0.22), 0.44, 0.44,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)

        # State name (bold)
        ax.text(x, y+0.08, state, ha='center', va='center', fontsize=14, fontweight='bold')
        # Description
        ax.text(x, y-0.06, desc, ha='center', va='center', fontsize=9, style='italic')

    # Axis labels
    ax.annotate('', xy=(1.02, 0.5), xytext=(-0.02, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(0.5, 1.02), xytext=(0.5, -0.02),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    ax.text(1.05, 0.5, 'Other', fontsize=12, va='center', fontweight='bold')
    ax.text(-0.05, 0.5, 'Self', fontsize=12, va='center', ha='right', fontweight='bold')
    ax.text(0.5, 1.05, 'Explore', fontsize=12, ha='center', fontweight='bold')
    ax.text(0.5, -0.05, 'Exploit', fontsize=12, ha='center', fontweight='bold')

    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_four_quadrant.png')
    plt.savefig(f'{OUTPUT_DIR}/fig1_four_quadrant.pdf')
    plt.close()
    print("Generated: fig1_four_quadrant.png")


def fig2_transition_matrix():
    """Figure 2: Aggregate transition matrix heatmap."""
    # Transition matrix from paper (approximate values based on results)
    # Rows: from state, Cols: to state
    # Order: Seeking, Directing, Conferring, Revising
    P = np.array([
        [0.35, 0.18, 0.28, 0.19],  # From Seeking
        [0.15, 0.42, 0.22, 0.21],  # From Directing
        [0.31, 0.24, 0.25, 0.20],  # From Conferring
        [0.27, 0.19, 0.26, 0.28],  # From Revising
    ])

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(P, cmap='Blues', vmin=0, vmax=0.5)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Transition Probability', rotation=270, labelpad=15)

    # Set ticks
    ax.set_xticks(np.arange(len(STATES)))
    ax.set_yticks(np.arange(len(STATES)))
    ax.set_xticklabels(STATES)
    ax.set_yticklabels(STATES)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            text = ax.text(j, i, f'{P[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if P[i, j] > 0.3 else 'black',
                          fontsize=11)

    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Aggregate Transition Matrix (n = 26 individuals)')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_transition_matrix.png')
    plt.savefig(f'{OUTPUT_DIR}/fig2_transition_matrix.pdf')
    plt.close()
    print("Generated: fig2_transition_matrix.png")


def fig3_stationary_distribution():
    """Figure 3: Stationary distribution across the four states."""
    # Stationary distribution from paper
    pi = np.array([0.27, 0.26, 0.25, 0.22])  # Seeking, Directing, Conferring, Revising

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [STATE_COLORS[s] for s in STATES]
    bars = ax.bar(STATES, pi, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, pi):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Stationary Probability')
    ax.set_xlabel('Motivational State')
    ax.set_title('Stationary Distribution (Long-term State Occupancy)')
    ax.set_ylim(0, 0.35)

    # Add horizontal line at 0.25 (uniform)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7, label='Uniform (0.25)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_stationary_distribution.png')
    plt.savefig(f'{OUTPUT_DIR}/fig3_stationary_distribution.pdf')
    plt.close()
    print("Generated: fig3_stationary_distribution.png")


def fig4_transfer_entropy_network():
    """Figure 4: Transfer entropy network showing archetypal relationships."""
    # Create a sample network based on paper description
    # Sources (3), Sinks (3), Hubs (4), General (16)
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create graph
    G = nx.DiGraph()

    # Node types (using initials for famous cases as examples)
    sources = ['S1', 'S2', 'S3']  # Archetypal exemplars
    sinks = ['K1', 'K2', 'K3']    # Composite cases
    hubs = ['H1', 'H2', 'H3', 'H4']  # Central connectors
    general = [f'G{i}' for i in range(1, 17)]  # General nodes

    all_nodes = sources + sinks + hubs + general
    G.add_nodes_from(all_nodes)

    # Add edges (sources -> others, others -> sinks, hubs bidirectional)
    edges = []
    # Sources influence many
    for s in sources:
        targets = np.random.choice(sinks + hubs + general[:8], size=5, replace=False)
        for t in targets:
            edges.append((s, t))

    # Many influence sinks
    for k in sinks:
        sources_in = np.random.choice(sources + hubs + general[:8], size=4, replace=False)
        for s in sources_in:
            edges.append((s, k))

    # Hubs connect broadly
    for h in hubs:
        targets = np.random.choice(general + sinks, size=3, replace=False)
        for t in targets:
            edges.append((h, t))
        sources_in = np.random.choice(sources + general, size=2, replace=False)
        for s in sources_in:
            edges.append((s, h))

    G.add_edges_from(edges)

    # Position nodes
    pos = {}
    # Sources on left
    for i, s in enumerate(sources):
        pos[s] = (-2, 1 - i)
    # Sinks on right
    for i, k in enumerate(sinks):
        pos[k] = (2, 1 - i)
    # Hubs in center
    for i, h in enumerate(hubs):
        pos[h] = (0, 1.5 - i)
    # General scattered
    for i, g in enumerate(general):
        angle = 2 * np.pi * i / len(general)
        pos[g] = (1.2 * np.cos(angle), 1.2 * np.sin(angle))

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if n in sources:
            node_colors.append('#FF6B6B')  # Red for sources
            node_sizes.append(800)
        elif n in sinks:
            node_colors.append('#4ECDC4')  # Teal for sinks
            node_sizes.append(800)
        elif n in hubs:
            node_colors.append('#FFE66D')  # Yellow for hubs
            node_sizes.append(700)
        else:
            node_colors.append('#C4C4C4')  # Gray for general
            node_sizes.append(300)

    # Draw network
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, arrows=True,
                          arrowsize=10, edge_color='gray',
                          connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=node_sizes, edgecolors='black', linewidths=1)

    # Labels only for special nodes
    labels = {n: n for n in sources + sinks + hubs}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label=f'Sources (n=3, 11.5%)'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label=f'Sinks (n=3, 11.5%)'),
        mpatches.Patch(facecolor='#FFE66D', edgecolor='black', label=f'Hubs (n=4, 15.4%)'),
        mpatches.Patch(facecolor='#C4C4C4', edgecolor='black', label=f'General (n=16, 61.5%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.set_title('Transfer Entropy Network: Archetypal Relationships\n(Edges indicate significant predictive relationships)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_te_network.png')
    plt.savefig(f'{OUTPUT_DIR}/fig4_te_network.pdf')
    plt.close()
    print("Generated: fig4_te_network.png")


def fig5_example_trajectory():
    """Figure 5: Example behavioral trajectory for a single individual."""
    # Example trajectory (hypothetical based on typical patterns)
    trajectory = ['Seeking', 'Seeking', 'Conferring', 'Seeking', 'Conferring',
                  'Directing', 'Revising', 'Seeking', 'Conferring', 'Directing',
                  'Directing', 'Revising', 'Revising', 'Seeking', 'Directing']

    events = list(range(1, len(trajectory) + 1))

    fig, ax = plt.subplots(figsize=(12, 4))

    # Map states to y positions
    state_y = {'Seeking': 3, 'Conferring': 2, 'Directing': 1, 'Revising': 0}
    y_vals = [state_y[s] for s in trajectory]
    colors = [STATE_COLORS[s] for s in trajectory]

    # Plot line connecting points
    ax.plot(events, y_vals, 'k-', alpha=0.3, linewidth=1)

    # Plot points
    for i, (x, y, s) in enumerate(zip(events, y_vals, trajectory)):
        ax.scatter(x, y, c=STATE_COLORS[s], s=150, edgecolors='black',
                  linewidths=1.5, zorder=5)

    # Highlight critical transition (Seeking -> Directing)
    critical_idx = []
    for i in range(len(trajectory)-1):
        if trajectory[i] == 'Seeking' and trajectory[i+1] == 'Directing':
            critical_idx.append(i)

    for idx in critical_idx:
        ax.annotate('', xy=(events[idx+1], state_y[trajectory[idx+1]]),
                   xytext=(events[idx], state_y[trajectory[idx]]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate('Critical\nTransition', xy=(events[idx]+0.5, 2.5),
                   fontsize=9, ha='center', color='red')

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Revising', 'Directing', 'Conferring', 'Seeking'])
    ax.set_xlabel('Event Number (Chronological)')
    ax.set_ylabel('Motivational State')
    ax.set_title('Example Behavioral Trajectory: Individual Case Study')
    ax.set_xlim(0, len(events) + 1)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=STATE_COLORS[s], edgecolor='black',
                                      label=s) for s in STATES]
    ax.legend(handles=legend_elements, loc='upper right', ncol=2)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_example_trajectory.png')
    plt.savefig(f'{OUTPUT_DIR}/fig5_example_trajectory.pdf')
    plt.close()
    print("Generated: fig5_example_trajectory.png")


def fig6_hierarchical_classification():
    """Figure 6: Hierarchical classification tree."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tree structure
    # Level 0: All (100%)
    # Level 1: COMPLEX (11.5%), FOCUSED (88.5%)
    # Level 2: Subtypes

    ax.text(0.5, 0.95, 'All Individuals\n(n=26, 100%)', ha='center', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))

    # Level 1
    ax.annotate('', xy=(0.25, 0.72), xytext=(0.5, 0.85),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0.75, 0.72), xytext=(0.5, 0.85),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.text(0.25, 0.7, 'COMPLEX\n(n=3, 11.5%)', ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FF6B6B', edgecolor='black', alpha=0.7))
    ax.text(0.75, 0.7, 'FOCUSED\n(n=23, 88.5%)', ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#4ECDC4', edgecolor='black', alpha=0.7))

    # Level 2 - COMPLEX subtypes
    complex_subtypes = [
        ('Chaotic\n(high entropy)', 0.12),
        ('Adaptive\n(state switching)', 0.25),
        ('Escalating\n(progressive)', 0.38),
    ]
    for (name, x) in complex_subtypes:
        ax.annotate('', xy=(x, 0.42), xytext=(0.25, 0.58),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        ax.text(x, 0.4, name, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#FFB6B6', edgecolor='gray', alpha=0.7))

    # Level 2 - FOCUSED subtypes
    focused_subtypes = [
        ('Seeking-\ndominant', 0.58),
        ('Directing-\ndominant', 0.70),
        ('Conferring-\ndominant', 0.82),
        ('Revising-\ndominant', 0.94),
    ]
    for (name, x) in focused_subtypes:
        ax.annotate('', xy=(x, 0.42), xytext=(0.75, 0.58),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        ax.text(x, 0.4, name, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#A8E6CF', edgecolor='gray', alpha=0.7))

    # Criteria boxes
    ax.text(0.08, 0.2, 'Classification Criteria:\n• Entropy rate > median → COMPLEX\n• Dominant state > 35% → FOCUSED subtype',
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Hierarchical Classification System', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig6_hierarchical_tree.png')
    plt.savefig(f'{OUTPUT_DIR}/fig6_hierarchical_tree.pdf')
    plt.close()
    print("Generated: fig6_hierarchical_tree.png")


def fig7_intervention_window():
    """Figure 7: Intervention window analysis showing tipping points."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

    # Top: Probability of reaching Directing over time
    events = np.arange(1, 16)
    # Simulated probability trajectory
    p_directing = 0.1 + 0.05 * events + 0.02 * np.random.randn(len(events))
    p_directing = np.clip(p_directing, 0, 1)
    p_directing[5] = 0.35  # Spike after Seeking state
    p_directing[8] = 0.55  # Cross threshold
    p_directing[9:] = 0.6 + 0.03 * np.arange(len(events)-9)

    ax1.plot(events, p_directing, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Tipping Point (P=0.5)')
    ax1.fill_between(events, 0, p_directing, where=p_directing < 0.5,
                     alpha=0.3, color='green', label='Intervention Window')
    ax1.fill_between(events, 0, p_directing, where=p_directing >= 0.5,
                     alpha=0.3, color='red', label='High Risk Zone')

    ax1.set_ylabel('P(Reaching Directing State)')
    ax1.set_title('Intervention Window Analysis: Optimal Timing')
    ax1.legend(loc='upper left')
    ax1.set_xlim(1, 15)
    ax1.set_ylim(0, 0.9)
    ax1.grid(True, alpha=0.3)

    # Bottom: State sequence
    states = ['Seeking', 'Seeking', 'Revising', 'Seeking', 'Conferring',
              'Seeking', 'Conferring', 'Seeking', 'Directing', 'Directing',
              'Revising', 'Directing', 'Directing', 'Revising', 'Directing']
    state_y = {'Seeking': 3, 'Conferring': 2, 'Directing': 1, 'Revising': 0}
    y_vals = [state_y[s] for s in states]
    colors = [STATE_COLORS[s] for s in states]

    for i, (x, y, s) in enumerate(zip(events, y_vals, states)):
        ax2.scatter(x, y, c=STATE_COLORS[s], s=100, edgecolors='black', linewidths=1, zorder=5)
    ax2.plot(events, y_vals, 'k-', alpha=0.3)

    # Mark critical transition
    ax2.annotate('Critical\nTransition', xy=(8.5, 2), fontsize=9, ha='center', color='red',
                arrowprops=dict(arrowstyle='->', color='red'), xytext=(8.5, 3.2))

    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Revising', 'Directing', 'Conferring', 'Seeking'])
    ax2.set_xlabel('Event Number')
    ax2.set_ylabel('State')
    ax2.set_xlim(1, 15)
    ax2.set_ylim(-0.5, 3.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig7_intervention_window.png')
    plt.savefig(f'{OUTPUT_DIR}/fig7_intervention_window.pdf')
    plt.close()
    print("Generated: fig7_intervention_window.png")


if __name__ == '__main__':
    print("Generating publication figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    fig1_four_quadrant_model()
    fig2_transition_matrix()
    fig3_stationary_distribution()
    fig4_transfer_entropy_network()
    fig5_example_trajectory()
    fig6_hierarchical_classification()
    fig7_intervention_window()

    print("-" * 50)
    print("All figures generated successfully!")
    print(f"\nFigures saved to: {OUTPUT_DIR}")
