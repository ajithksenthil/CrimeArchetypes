#!/usr/bin/env python
"""
State Space Comparison CLI

Command-line interface for running state space comparisons and analytics.

Usage:
    python -m state_space.cli compare --space1 animal --space2 cluster --data-dir ./data
    python -m state_space.cli list-spaces
    python -m state_space.cli analyze --space animal --data-dir ./data --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .core import StateSpaceRegistry
from .analytics import StateSpaceAnalytics, quick_compare
from .metrics import compare_state_spaces
from .definitions import (
    animal_state_space,
    archetypal_cluster_space,
    create_cluster_to_animal_mapping
)


def cmd_list_spaces(args):
    """List all registered state spaces."""
    print("\nRegistered State Spaces:")
    print("=" * 50)

    for name in StateSpaceRegistry.list_all():
        space = StateSpaceRegistry.get(name)
        meta = space.metadata
        print(f"\n  {meta.display_name}")
        print(f"  Name: {meta.name}")
        print(f"  Type: {meta.type.value}")
        print(f"  States: {space.n_states}")
        print(f"  State names: {', '.join(space.state_names)}")


def cmd_describe_space(args):
    """Describe a specific state space."""
    space = StateSpaceRegistry.get(args.space)
    if space is None:
        print(f"Error: Unknown space '{args.space}'")
        print(f"Available: {StateSpaceRegistry.list_all()}")
        return 1

    meta = space.metadata
    print(f"\n{meta.display_name}")
    print("=" * 50)
    print(f"Internal name: {meta.name}")
    print(f"Type: {meta.type.value}")
    print(f"Description: {meta.description}")
    print(f"Version: {meta.version}")
    print(f"Source: {meta.source}")

    print(f"\nStates ({space.n_states}):")
    for name, state in space.states.items():
        print(f"\n  {name} ({state.color})")
        print(f"    {state.description}")
        if state.dimensions:
            print(f"    Dimensions: {state.dimensions}")
        if state.keywords:
            print(f"    Keywords: {', '.join(state.keywords[:5])}...")
        if state.examples:
            print(f"    Examples: {state.examples[0]}")


def cmd_compare(args):
    """Compare two state spaces."""
    space1 = StateSpaceRegistry.get(args.space1)
    space2 = StateSpaceRegistry.get(args.space2)

    if space1 is None or space2 is None:
        print(f"Error: Unknown space(s)")
        print(f"Available: {StateSpaceRegistry.list_all()}")
        return 1

    # Load data if provided
    if args.data_dir:
        data_dir = Path(args.data_dir)
        # Try to load classifications
        # This is a placeholder - actual loading depends on data format
        print(f"Loading data from {data_dir}...")

    # Show mapping if available
    if args.space1 == 'archetypal_clusters' and args.space2 == 'animal_states':
        mapping = create_cluster_to_animal_mapping()
        print("\nCluster to Animal State Mapping:")
        print("-" * 40)
        for source, state_mapping in mapping.mappings.items():
            target_dist = state_mapping.get_target_distribution()
            targets = ", ".join(f"{k}:{v:.0%}" for k, v in target_dist.items())
            print(f"  Cluster {source} -> {targets}")

    # Basic space comparison info
    print(f"\nComparison: {space1.metadata.display_name} vs {space2.metadata.display_name}")
    print("-" * 50)
    print(f"Space 1: {space1.n_states} states ({space1.metadata.type.value})")
    print(f"Space 2: {space2.n_states} states ({space2.metadata.type.value})")

    if space1.n_states != space2.n_states:
        print("\nNote: Different state counts - direct comparison requires mapping.")


def cmd_run_analytics(args):
    """Run full analytics pipeline."""
    print("State Space Analytics Pipeline")
    print("=" * 50)

    # Initialize analytics
    analytics = StateSpaceAnalytics()

    # Add spaces
    for space_name in args.spaces.split(','):
        space = StateSpaceRegistry.get(space_name.strip())
        if space is None:
            print(f"Warning: Unknown space '{space_name}', skipping")
            continue

        print(f"Adding space: {space.metadata.display_name}")
        # Would need actual classification data here
        # analytics.add_space(space_name, space, classifications)

    # Run pipeline
    if len(analytics.spaces) >= 2:
        print("\nRunning comparison pipeline...")
        # results = analytics.run_pipeline()
        # if args.output:
        #     results.save(args.output)
        print("Note: Add classification data to run full pipeline")
    else:
        print("Need at least 2 spaces with data to run comparisons")


def cmd_create_mapping(args):
    """Create and display a mapping between spaces."""
    if args.source == 'archetypal_clusters' and args.target == 'animal_states':
        mapping = create_cluster_to_animal_mapping()
        print(f"\nMapping: {mapping.source} -> {mapping.target}")
        print(f"Type: {mapping.mapping_type.value}")
        print(f"Version: {mapping.version}")
        print(f"\nDescription: {mapping.description}")
        print(f"\nMethodology: {mapping.methodology}")

        print("\nMappings:")
        for source, m in mapping.mappings.items():
            dist = m.get_target_distribution()
            print(f"  {source}: {dist}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(mapping.to_dict(), f, indent=2)
            print(f"\nSaved to {args.output}")
    else:
        print(f"No predefined mapping from {args.source} to {args.target}")


def main():
    parser = argparse.ArgumentParser(
        description="State Space Comparison CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m state_space.cli list-spaces
    python -m state_space.cli describe --space animal_states
    python -m state_space.cli compare --space1 archetypal_clusters --space2 animal_states
    python -m state_space.cli create-mapping --source archetypal_clusters --target animal_states
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # list-spaces command
    list_parser = subparsers.add_parser('list-spaces', help='List registered state spaces')

    # describe command
    describe_parser = subparsers.add_parser('describe', help='Describe a state space')
    describe_parser.add_argument('--space', '-s', required=True, help='State space name')

    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two state spaces')
    compare_parser.add_argument('--space1', '-s1', required=True, help='First state space')
    compare_parser.add_argument('--space2', '-s2', required=True, help='Second state space')
    compare_parser.add_argument('--data-dir', '-d', help='Directory with classification data')
    compare_parser.add_argument('--output', '-o', help='Output file for results')

    # analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run analytics pipeline')
    analyze_parser.add_argument('--spaces', '-s', required=True,
                                help='Comma-separated list of spaces')
    analyze_parser.add_argument('--data-dir', '-d', help='Directory with classification data')
    analyze_parser.add_argument('--output', '-o', help='Output file for results')

    # create-mapping command
    mapping_parser = subparsers.add_parser('create-mapping', help='Create mapping between spaces')
    mapping_parser.add_argument('--source', '-s', required=True, help='Source state space')
    mapping_parser.add_argument('--target', '-t', required=True, help='Target state space')
    mapping_parser.add_argument('--output', '-o', help='Output file for mapping JSON')

    args = parser.parse_args()

    if args.command == 'list-spaces':
        return cmd_list_spaces(args)
    elif args.command == 'describe':
        return cmd_describe_space(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'analyze':
        return cmd_run_analytics(args)
    elif args.command == 'create-mapping':
        return cmd_create_mapping(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
