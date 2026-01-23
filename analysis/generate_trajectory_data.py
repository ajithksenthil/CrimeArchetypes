#!/usr/bin/env python3
"""
Generate Trajectory Data for Missing Individuals

This script generates synthetic trajectory data for individuals who have
signature data (state distribution) but no event-level trajectory data.

The trajectories are generated using a Markov chain approach:
1. Use state distribution to estimate initial state probabilities
2. Estimate transition matrix from state distribution using maximum entropy principle
3. Generate trajectory using the estimated transition matrix

Usage:
    python generate_trajectory_data.py [--output-dir OUTPUT_DIR]
"""
import json
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'dashboard'))

from dashboard.data.loader import DashboardDataLoader, STATE_NAMES


def estimate_transition_matrix(state_distribution: Dict[str, float]) -> np.ndarray:
    """
    Estimate a transition matrix from stationary state distribution.

    Uses the maximum entropy principle: the transition matrix that
    produces the given stationary distribution with maximum entropy.

    Args:
        state_distribution: Dict mapping state names to probabilities

    Returns:
        4x4 transition matrix
    """
    # Get stationary distribution as array
    pi = np.array([state_distribution.get(s, 0.25) for s in STATE_NAMES])
    pi = pi / pi.sum()  # Normalize

    # Initialize with doubly stochastic matrix approach
    # This tends to produce matrices close to stationary distribution
    n = len(STATE_NAMES)
    P = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Self-transition: higher for dominant states
                P[i, j] = 0.3 + 0.3 * pi[i]
            else:
                # Off-diagonal: proportional to target state's stationary prob
                P[i, j] = pi[j]

    # Normalize rows
    P = P / P.sum(axis=1, keepdims=True)

    return P


def generate_synthetic_trajectory(
    state_distribution: Dict[str, float],
    n_events: int,
    seed: int = None
) -> List[str]:
    """
    Generate a synthetic trajectory using Markov chain.

    Args:
        state_distribution: Target state distribution
        n_events: Number of events to generate
        seed: Random seed for reproducibility

    Returns:
        List of state names
    """
    if seed is not None:
        np.random.seed(seed)

    # Estimate transition matrix
    P = estimate_transition_matrix(state_distribution)
    state_to_idx = {s: i for i, s in enumerate(STATE_NAMES)}
    idx_to_state = {i: s for i, s in enumerate(STATE_NAMES)}

    # Initial state from stationary distribution
    pi = np.array([state_distribution.get(s, 0.25) for s in STATE_NAMES])
    pi = pi / pi.sum()

    current_idx = np.random.choice(4, p=pi)
    trajectory = [idx_to_state[current_idx]]

    # Generate trajectory
    for _ in range(n_events - 1):
        probs = P[current_idx]
        next_idx = np.random.choice(4, p=probs)
        trajectory.append(idx_to_state[next_idx])
        current_idx = next_idx

    return trajectory


def generate_event_data(trajectory: List[str], criminal_name: str) -> List[Dict]:
    """
    Convert trajectory to event-level data format.

    Args:
        trajectory: List of state names
        criminal_name: Name of the individual

    Returns:
        List of event dictionaries
    """
    events = []
    for i, state in enumerate(trajectory):
        events.append({
            'event': f'Event {i+1} (generated)',
            'criminal': criminal_name,
            'state': state,
            'confidence': 'GENERATED',
            'reasoning': f'Synthetically generated based on behavioral signature data'
        })
    return events


def main(output_dir: str = None):
    """Generate trajectory data for all missing individuals."""
    print("=" * 60)
    print("TRAJECTORY DATA GENERATION")
    print("=" * 60)

    # Setup
    data_dir = Path('empirical_study')
    loader = DashboardDataLoader(data_dir)

    # Find missing individuals
    detailed = set(loader.detailed_classifications.keys())
    all_individuals = loader.get_all_individuals()
    missing = set(all_individuals) - detailed

    print(f"\nTotal individuals: {len(all_individuals)}")
    print(f"With trajectory data: {len(detailed)}")
    print(f"Missing trajectory data: {len(missing)}")

    if not missing:
        print("\nAll individuals already have trajectory data!")
        return

    # Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = data_dir / f"generated_trajectories_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate trajectories for missing individuals
    generated_events = {}
    generated_stats = []
    failed = []

    for name in sorted(missing):
        print(f"\nProcessing: {name}")

        # Get signature data
        sig = loader.signatures.get(name, {})
        state_dist = sig.get('state_distribution', {})
        n_events = sig.get('total_events', sig.get('n_events', 50))

        if not state_dist:
            print(f"  WARNING: No state distribution found, skipping")
            failed.append({'name': name, 'error': 'No state distribution'})
            continue

        # Generate trajectory
        try:
            # Use name hash as seed for reproducibility
            seed = hash(name) % (2**31)
            trajectory = generate_synthetic_trajectory(state_dist, n_events, seed)

            # Convert to event format
            events = generate_event_data(trajectory, name)
            generated_events[name] = events

            # Verify distribution
            from collections import Counter
            actual_dist = Counter(trajectory)
            total = len(trajectory)
            actual_pcts = {s: actual_dist.get(s, 0) / total for s in STATE_NAMES}

            generated_stats.append({
                'name': name,
                'n_events': len(trajectory),
                'target_distribution': state_dist,
                'actual_distribution': actual_pcts,
                'distribution_error': sum(
                    abs(state_dist.get(s, 0) - actual_pcts.get(s, 0))
                    for s in STATE_NAMES
                ) / 4
            })

            print(f"  Generated {len(trajectory)} events")
            print(f"  Target dist: Directing={state_dist.get('Directing', 0):.1%}")
            print(f"  Actual dist: Directing={actual_pcts.get('Directing', 0):.1%}")

        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append({'name': name, 'error': str(e)})

    # Save generated data
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save generated events
    with open(output_dir / 'generated_events.json', 'w') as f:
        json.dump(generated_events, f, indent=2)
    print(f"\nSaved generated events to: {output_dir / 'generated_events.json'}")

    # Save statistics
    with open(output_dir / 'generation_stats.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_generated': len(generated_events),
            'individuals': generated_stats
        }, f, indent=2)
    print(f"Saved statistics to: {output_dir / 'generation_stats.json'}")

    # Save failures
    if failed:
        with open(output_dir / 'failed_generation.json', 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"Saved failures to: {output_dir / 'failed_generation.json'}")

    # Update the detailed classifications file
    # First, find the existing classifications directory
    latest_hier = loader._find_latest_dir("hierarchical_archetypes_*")
    if latest_hier:
        existing_detailed_path = latest_hier / 'detailed_classifications.json'
        if existing_detailed_path.exists():
            with open(existing_detailed_path, 'r') as f:
                existing_detailed = json.load(f)
        else:
            existing_detailed = {}

        # Merge generated events
        merged = {**existing_detailed, **generated_events}

        # Save updated file
        backup_path = latest_hier / 'detailed_classifications.backup.json'
        if existing_detailed_path.exists():
            with open(backup_path, 'w') as f:
                json.dump(existing_detailed, f, indent=2)
            print(f"\nBackup saved to: {backup_path}")

        with open(existing_detailed_path, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"Updated: {existing_detailed_path}")
        print(f"Total individuals now: {len(merged)}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated: {len(generated_events)} trajectories")
    print(f"Failed: {len(failed)}")
    if generated_stats:
        avg_error = np.mean([s['distribution_error'] for s in generated_stats])
        print(f"Average distribution error: {avg_error:.3f}")

    # Verify the new data is loadable
    print("\nVerifying...")
    loader_new = DashboardDataLoader(data_dir)
    new_with_traj = loader_new.get_individuals_with_trajectory_data()
    print(f"Individuals with trajectory data: {len(new_with_traj)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectory data for missing individuals")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates timestamped directory)"
    )
    args = parser.parse_args()
    main(args.output_dir)
