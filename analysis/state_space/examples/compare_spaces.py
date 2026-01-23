#!/usr/bin/env python
"""
Example: Comparing 4-Animal States vs Archetypal Clusters

This script demonstrates the full workflow for comparing two state spaces
using the state_space framework.

Usage:
    python -m state_space.examples.compare_spaces
"""

import sys
import json
from pathlib import Path
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_space import (
    StateSpaceRegistry,
    StateSpaceAnalytics,
    animal_state_space,
    archetypal_cluster_space,
    compare_state_spaces,
    create_cluster_to_animal_mapping,
    js_divergence,
    mutual_information,
    normalized_mutual_information
)


def load_sample_data():
    """
    Load sample classification data.

    In a real scenario, this would load from:
    - LLM classification results (for 4-Animal states)
    - Cluster assignments (for Archetypal clusters)
    """
    # Try to load actual data if available
    base_dir = Path(__file__).parent.parent.parent

    # Load LLM animal classifications (detailed per-event)
    llm_dir = base_dir / "empirical_study"
    llm_detailed = None
    for study_dir in sorted(llm_dir.glob("llm_animal_study_*"), reverse=True):
        detailed_file = study_dir / "detailed_classifications.json"
        if detailed_file.exists():
            with open(detailed_file) as f:
                llm_detailed = json.load(f)
            break

    # Load cluster assignments
    clusters_file = base_dir / "clusters.json"
    cluster_data = None
    if clusters_file.exists():
        with open(clusters_file) as f:
            cluster_data = json.load(f)

    return llm_detailed, cluster_data


def create_synthetic_data(n_events: int = 100):
    """Create synthetic data for demonstration."""
    np.random.seed(42)

    # Animal state distribution (matching empirical results)
    animal_probs = [0.16, 0.52, 0.19, 0.13]  # Seeking, Directing, Conferring, Revising
    animal_states = ['Seeking', 'Directing', 'Conferring', 'Revising']

    animal_labels = np.random.choice(animal_states, size=n_events, p=animal_probs)

    # Cluster distribution
    cluster_probs = np.ones(10) / 10  # Uniform for demo
    cluster_labels = np.random.choice([str(i) for i in range(10)], size=n_events, p=cluster_probs)

    return list(animal_labels), list(cluster_labels)


def main():
    print("=" * 60)
    print("State Space Comparison: 4-Animal vs Archetypal Clusters")
    print("=" * 60)

    # List available spaces
    print("\n1. Available State Spaces:")
    print("-" * 40)
    for name in StateSpaceRegistry.list_all():
        space = StateSpaceRegistry.get(name)
        print(f"   • {space.metadata.display_name} ({space.n_states} states)")

    # Get our two spaces
    animal = StateSpaceRegistry.get('animal_states')
    clusters = StateSpaceRegistry.get('archetypal_clusters')

    # Show space details
    print("\n2. State Space Details:")
    print("-" * 40)
    print(f"\n   {animal.metadata.display_name}:")
    print(f"   Type: {animal.metadata.type.value}")
    print(f"   States: {animal.state_names}")
    print(f"   Dimensions: {animal.metadata.dimensions}")

    print(f"\n   {clusters.metadata.display_name}:")
    print(f"   Type: {clusters.metadata.type.value}")
    print(f"   States: {clusters.state_names}")

    # Create mapping
    print("\n3. Cluster to Animal Mapping:")
    print("-" * 40)
    mapping = create_cluster_to_animal_mapping()

    for cluster_id in range(10):
        m = mapping.mappings.get(str(cluster_id))
        if m:
            dist = m.get_target_distribution()
            primary = max(dist, key=dist.get)
            print(f"   Cluster {cluster_id} -> {primary} ({dist[primary]:.0%})")

    # Load or create data
    print("\n4. Loading Data:")
    print("-" * 40)
    llm_detailed, cluster_data = load_sample_data()

    if llm_detailed and len(llm_detailed) > 0:
        print("   Found LLM classification results!")
        # Each entry has 'event', 'criminal', 'state', 'confidence', 'reasoning'
        animal_labels = [c.get('state') for c in llm_detailed if c.get('state')]
        print(f"   Loaded {len(animal_labels)} animal state labels")
    else:
        print("   No LLM results found, using synthetic data")
        animal_labels, _ = create_synthetic_data(100)

    # For clusters, use synthetic for now (would need cluster assignments per event)
    _, cluster_labels = create_synthetic_data(len(animal_labels))

    # Ensure same length
    n = min(len(animal_labels), len(cluster_labels))
    animal_labels = animal_labels[:n]
    cluster_labels = cluster_labels[:n]

    print(f"   Total events to compare: {n}")

    # Run comparison
    print("\n5. State Space Comparison Metrics:")
    print("-" * 40)

    result = compare_state_spaces(
        animal_labels,
        cluster_labels,
        animal.state_names,
        clusters.state_names,
        "4-Animal States",
        "Archetypal Clusters"
    )

    print(f"\n   Events compared: {result.n_events}")
    print("\n   Information-Theoretic Metrics:")
    for metric, value in result.metrics.items():
        print(f"   • {metric}: {value:.4f}")

    # Distribution comparison
    print("\n6. State Distributions:")
    print("-" * 40)

    print("\n   4-Animal State Distribution:")
    for state, pct in result.details['distribution_a'].items():
        bar = "█" * int(pct * 40)
        print(f"   {state:12s} {bar} {pct*100:.1f}%")

    # Map clusters to animal states and compare
    print("\n7. Mapped Comparison:")
    print("-" * 40)

    # Map cluster labels to animal states using the mapping
    mapped_animal_from_clusters = []
    for cluster_label in cluster_labels:
        target_dist = mapping.map_state(cluster_label)
        if target_dist:
            # Use primary target
            primary = max(target_dist, key=target_dist.get)
            mapped_animal_from_clusters.append(primary)
        else:
            mapped_animal_from_clusters.append('Directing')  # Default

    # Compare direct animal labels vs mapped from clusters
    from collections import Counter
    direct_counts = Counter(animal_labels)
    mapped_counts = Counter(mapped_animal_from_clusters)

    print("\n   State     | Direct LLM | Cluster-Mapped")
    print("   " + "-" * 45)
    for state in animal.state_names:
        direct_pct = direct_counts.get(state, 0) / n * 100
        mapped_pct = mapped_counts.get(state, 0) / n * 100
        print(f"   {state:12s} | {direct_pct:6.1f}%    | {mapped_pct:6.1f}%")

    # Summary
    print("\n8. Summary:")
    print("-" * 40)
    print(f"""
   The 4-Animal State Space and Archetypal Clusters represent
   complementary approaches:

   • 4-Animal: Theory-driven, based on psychological dimensions
     (Self/Other × Explore/Exploit). Best for transition analysis
     and individual profiling.

   • Clusters: Data-driven, captures semantic/contextual patterns.
     Best for pattern discovery and case-based reasoning.

   Mutual Information: {result.metrics.get('mutual_information', 0):.3f} bits
   - Indicates how much one space predicts the other

   The mapping allows translation between frameworks, enabling:
   - Cross-validation of classifications
   - Rich context (clusters) with theoretical grounding (animal)
   - Flexible analysis depending on research question
""")

    print("=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
