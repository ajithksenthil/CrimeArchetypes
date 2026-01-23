"""
State Space Analytics Pipeline

Provides downstream analytics capabilities:
- Cross-space transition analysis
- Behavioral signature comparison across spaces
- Temporal pattern analysis in different representations
- Individual profiling with multiple state spaces
- Aggregation and summary statistics

Usage:
    from state_space.analytics import StateSpaceAnalytics

    analytics = StateSpaceAnalytics()
    analytics.add_space('animal', animal_space, animal_classifications)
    analytics.add_space('cluster', cluster_space, cluster_classifications)

    results = analytics.run_pipeline()
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict

from .core import StateSpace, StateSpaceRegistry
from .mappings import StateSpaceMapping, MappingRegistry
from .metrics import (
    compare_state_spaces,
    compare_transition_matrices,
    js_divergence,
    mutual_information,
    normalized_mutual_information,
    entropy
)


@dataclass
class ClassificationData:
    """Container for a state space's classification data."""
    space_name: str
    classifications: List[Dict]  # List of {event_id, state, confidence, ...}
    sequences: Dict[str, List[str]] = field(default_factory=dict)  # individual_id -> [states]

    @property
    def labels(self) -> List[str]:
        """Extract just the state labels."""
        return [c.get('state') for c in self.classifications]

    @property
    def n_events(self) -> int:
        return len(self.classifications)

    def get_distribution(self, states: List[str]) -> np.ndarray:
        """Get distribution over given states."""
        counts = Counter(self.labels)
        return np.array([counts.get(s, 0) for s in states])


@dataclass
class AnalyticsResult:
    """Container for analytics pipeline results."""
    spaces_analyzed: List[str]
    n_events: int
    comparisons: Dict[Tuple[str, str], Dict]
    individual_profiles: Dict[str, Dict]
    aggregate_stats: Dict
    temporal_patterns: Dict

    def to_dict(self) -> Dict:
        return {
            'spaces_analyzed': self.spaces_analyzed,
            'n_events': self.n_events,
            'comparisons': {
                f"{k[0]}_vs_{k[1]}": v for k, v in self.comparisons.items()
            },
            'individual_profiles': self.individual_profiles,
            'aggregate_stats': self.aggregate_stats,
            'temporal_patterns': self.temporal_patterns
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class StateSpaceAnalytics:
    """
    Analytics pipeline for comparing and analyzing multiple state spaces.

    Provides methods for:
    - Pairwise state space comparisons
    - Individual behavioral profiling across spaces
    - Temporal pattern extraction
    - Aggregate statistics computation
    """

    def __init__(self):
        self.spaces: Dict[str, StateSpace] = {}
        self.data: Dict[str, ClassificationData] = {}
        self._mappings: Dict[Tuple[str, str], StateSpaceMapping] = {}

    def add_space(
        self,
        name: str,
        space: StateSpace,
        classifications: List[Dict],
        sequences: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        Add a state space with its classifications.

        Args:
            name: Identifier for this space
            space: StateSpace instance
            classifications: List of classification dicts with 'state' field
            sequences: Optional dict mapping individual IDs to state sequences
        """
        self.spaces[name] = space
        self.data[name] = ClassificationData(
            space_name=name,
            classifications=classifications,
            sequences=sequences or {}
        )

    def add_mapping(
        self,
        source: str,
        target: str,
        mapping: StateSpaceMapping
    ) -> None:
        """Add a mapping between two spaces."""
        self._mappings[(source, target)] = mapping

    def get_mapping(self, source: str, target: str) -> Optional[StateSpaceMapping]:
        """Get mapping between two spaces."""
        return self._mappings.get((source, target))

    # =========================================================================
    # PAIRWISE COMPARISONS
    # =========================================================================

    def compare_pair(
        self,
        space_a: str,
        space_b: str
    ) -> Dict:
        """
        Compare two state spaces.

        Returns:
            Dict with comparison metrics and details
        """
        if space_a not in self.spaces or space_b not in self.spaces:
            raise ValueError(f"Unknown space: {space_a} or {space_b}")

        data_a = self.data[space_a]
        data_b = self.data[space_b]

        # Ensure same events (by index matching)
        n = min(data_a.n_events, data_b.n_events)
        labels_a = data_a.labels[:n]
        labels_b = data_b.labels[:n]

        states_a = self.spaces[space_a].state_names
        states_b = self.spaces[space_b].state_names

        # Get mapping if available
        mapping_dict = None
        mapping = self.get_mapping(space_a, space_b)
        if mapping:
            mapping_dict = {
                s: mapping.map_state(s)
                for s in states_a
            }
            # Simplify to primary target only
            mapping_dict = {
                s: max(targets, key=targets.get) if targets else s
                for s, targets in mapping_dict.items()
            }

        result = compare_state_spaces(
            labels_a, labels_b,
            states_a, states_b,
            space_a, space_b,
            mapping_dict
        )

        return result.to_dict()

    def compare_all_pairs(self) -> Dict[Tuple[str, str], Dict]:
        """Compare all pairs of state spaces."""
        comparisons = {}
        space_names = list(self.spaces.keys())

        for i, space_a in enumerate(space_names):
            for space_b in space_names[i + 1:]:
                comparisons[(space_a, space_b)] = self.compare_pair(space_a, space_b)

        return comparisons

    # =========================================================================
    # TRANSITION ANALYSIS
    # =========================================================================

    def compute_transition_matrix(
        self,
        space_name: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute transition matrix for a state space from sequences.

        Args:
            space_name: Name of the state space
            normalize: Whether to normalize rows to probabilities

        Returns:
            Transition matrix (n_states x n_states)
        """
        if space_name not in self.spaces:
            raise ValueError(f"Unknown space: {space_name}")

        space = self.spaces[space_name]
        data = self.data[space_name]

        sequences = list(data.sequences.values())
        return space.compute_transition_matrix(sequences, normalize)

    def compare_transitions(
        self,
        space_a: str,
        space_b: str
    ) -> Dict:
        """
        Compare transition dynamics between two state spaces.

        Requires same number of states or a mapping between them.
        """
        matrix_a = self.compute_transition_matrix(space_a)
        matrix_b = self.compute_transition_matrix(space_b)

        states_a = self.spaces[space_a].state_names
        states_b = self.spaces[space_b].state_names

        if matrix_a.shape == matrix_b.shape:
            return compare_transition_matrices(
                matrix_a, matrix_b,
                states_a, states_b,
                space_a, space_b
            )
        else:
            # Need to map one to the other
            mapping = self.get_mapping(space_a, space_b)
            if mapping:
                # Project matrix_a to space_b dimensions
                mapping_matrix = mapping.get_mapping_matrix(
                    self.spaces[space_a],
                    self.spaces[space_b]
                )
                # Transform: P_mapped = M^T @ P_a @ M
                matrix_a_mapped = mapping_matrix.T @ matrix_a @ mapping_matrix

                # Normalize rows
                row_sums = matrix_a_mapped.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                matrix_a_mapped = matrix_a_mapped / row_sums

                return compare_transition_matrices(
                    matrix_a_mapped, matrix_b,
                    states_b, states_b,
                    f"{space_a}_mapped", space_b
                )

            return {'error': 'Cannot compare: different dimensions and no mapping'}

    # =========================================================================
    # INDIVIDUAL PROFILING
    # =========================================================================

    def profile_individual(
        self,
        individual_id: str
    ) -> Dict:
        """
        Create multi-space profile for an individual.

        Returns profile with state distributions and signatures in each space.
        """
        profile = {
            'individual_id': individual_id,
            'spaces': {}
        }

        for space_name, data in self.data.items():
            space = self.spaces[space_name]

            # Get individual's events
            individual_events = [
                c for c in data.classifications
                if c.get('individual_id') == individual_id
            ]

            if not individual_events:
                continue

            # State distribution
            states = [e.get('state') for e in individual_events]
            state_counts = Counter(states)
            total = len(states)

            state_dist = {
                s: state_counts.get(s, 0) / total
                for s in space.state_names
            }

            # Sequence if available
            sequence = data.sequences.get(individual_id, [])

            # Compute individual transition matrix
            if sequence:
                trans_matrix = space.compute_transition_matrix([sequence])
                entropy_rate = space.compute_entropy_rate(trans_matrix)
            else:
                trans_matrix = None
                entropy_rate = None

            # Dominant state
            dominant = max(state_counts, key=state_counts.get) if state_counts else None

            profile['spaces'][space_name] = {
                'n_events': len(individual_events),
                'state_distribution': state_dist,
                'dominant_state': dominant,
                'sequence_length': len(sequence),
                'entropy_rate': entropy_rate,
                'avg_confidence': np.mean([
                    e.get('confidence', 1.0) for e in individual_events
                ])
            }

        return profile

    def profile_all_individuals(self) -> Dict[str, Dict]:
        """Create profiles for all individuals."""
        # Collect all individual IDs from all spaces
        all_ids = set()
        for data in self.data.values():
            for c in data.classifications:
                if 'individual_id' in c:
                    all_ids.add(c['individual_id'])

        return {
            ind_id: self.profile_individual(ind_id)
            for ind_id in all_ids
        }

    # =========================================================================
    # TEMPORAL PATTERNS
    # =========================================================================

    def extract_temporal_patterns(
        self,
        space_name: str,
        n_phases: int = 3
    ) -> Dict:
        """
        Extract temporal patterns by dividing sequences into phases.

        Args:
            space_name: State space to analyze
            n_phases: Number of temporal phases (e.g., early/middle/late)

        Returns:
            Dict with phase-wise distributions and transitions
        """
        space = self.spaces[space_name]
        data = self.data[space_name]

        phase_distributions = {f"phase_{i}": Counter() for i in range(n_phases)}
        phase_transitions = {f"phase_{i}": Counter() for i in range(n_phases)}

        for ind_id, sequence in data.sequences.items():
            if len(sequence) < n_phases:
                continue

            # Divide sequence into phases
            phase_size = len(sequence) // n_phases
            for phase_idx in range(n_phases):
                start = phase_idx * phase_size
                end = start + phase_size if phase_idx < n_phases - 1 else len(sequence)

                phase_seq = sequence[start:end]

                # Count states in this phase
                phase_distributions[f"phase_{phase_idx}"].update(phase_seq)

                # Count transitions in this phase
                for i in range(len(phase_seq) - 1):
                    transition = (phase_seq[i], phase_seq[i + 1])
                    phase_transitions[f"phase_{phase_idx}"][transition] += 1

        # Normalize to proportions
        result = {
            'n_phases': n_phases,
            'phase_distributions': {},
            'phase_top_transitions': {}
        }

        for phase in phase_distributions:
            total = sum(phase_distributions[phase].values())
            if total > 0:
                result['phase_distributions'][phase] = {
                    state: count / total
                    for state, count in phase_distributions[phase].items()
                }

                # Top 5 transitions
                top_trans = phase_transitions[phase].most_common(5)
                result['phase_top_transitions'][phase] = [
                    {'from': t[0][0], 'to': t[0][1], 'count': t[1]}
                    for t in top_trans
                ]

        return result

    def compute_phase_divergence(
        self,
        space_name: str,
        n_phases: int = 3
    ) -> Dict:
        """
        Compute how much distributions diverge between phases.

        Returns JS divergence between consecutive phases.
        """
        patterns = self.extract_temporal_patterns(space_name, n_phases)
        space = self.spaces[space_name]
        states = space.state_names

        divergences = []
        for i in range(n_phases - 1):
            phase_a = f"phase_{i}"
            phase_b = f"phase_{i + 1}"

            dist_a = patterns['phase_distributions'].get(phase_a, {})
            dist_b = patterns['phase_distributions'].get(phase_b, {})

            vec_a = np.array([dist_a.get(s, 0) for s in states])
            vec_b = np.array([dist_b.get(s, 0) for s in states])

            if vec_a.sum() > 0 and vec_b.sum() > 0:
                divergences.append({
                    'phases': f"{phase_a}_to_{phase_b}",
                    'js_divergence': js_divergence(vec_a, vec_b)
                })

        return {'phase_divergences': divergences}

    # =========================================================================
    # AGGREGATE STATISTICS
    # =========================================================================

    def compute_aggregate_stats(self) -> Dict:
        """Compute aggregate statistics across all spaces."""
        stats = {}

        for space_name, space in self.spaces.items():
            data = self.data[space_name]

            # Distribution
            dist = data.get_distribution(space.state_names)
            dist_normalized = dist / dist.sum() if dist.sum() > 0 else dist

            # Effective number of states (from entropy)
            ent = entropy(dist_normalized)
            effective_states = 2 ** ent

            # Dominant state
            dominant_idx = np.argmax(dist)
            dominant_state = space.state_names[dominant_idx]
            dominant_pct = dist_normalized[dominant_idx] * 100

            # Confidence summary
            confidences = [
                c.get('confidence', 1.0) for c in data.classifications
            ]

            stats[space_name] = {
                'n_states': space.n_states,
                'n_events': data.n_events,
                'n_individuals': len(data.sequences),
                'distribution': {
                    s: float(dist_normalized[i])
                    for i, s in enumerate(space.state_names)
                },
                'entropy': float(ent),
                'effective_states': float(effective_states),
                'dominant_state': dominant_state,
                'dominant_pct': float(dominant_pct),
                'mean_confidence': float(np.mean(confidences)),
                'median_confidence': float(np.median(confidences))
            }

        return stats

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================

    def run_pipeline(
        self,
        include_individuals: bool = True,
        include_temporal: bool = True,
        temporal_phases: int = 3
    ) -> AnalyticsResult:
        """
        Run the full analytics pipeline.

        Args:
            include_individuals: Whether to profile all individuals
            include_temporal: Whether to extract temporal patterns
            temporal_phases: Number of phases for temporal analysis

        Returns:
            AnalyticsResult with all computed analytics
        """
        # Total events (use max across spaces as reference)
        n_events = max(
            data.n_events for data in self.data.values()
        ) if self.data else 0

        # Pairwise comparisons
        comparisons = self.compare_all_pairs()

        # Individual profiles
        individual_profiles = {}
        if include_individuals:
            individual_profiles = self.profile_all_individuals()

        # Aggregate stats
        aggregate_stats = self.compute_aggregate_stats()

        # Temporal patterns
        temporal_patterns = {}
        if include_temporal:
            for space_name in self.spaces:
                if self.data[space_name].sequences:
                    temporal_patterns[space_name] = {
                        'phase_analysis': self.extract_temporal_patterns(
                            space_name, temporal_phases
                        ),
                        'phase_divergence': self.compute_phase_divergence(
                            space_name, temporal_phases
                        )
                    }

        return AnalyticsResult(
            spaces_analyzed=list(self.spaces.keys()),
            n_events=n_events,
            comparisons=comparisons,
            individual_profiles=individual_profiles,
            aggregate_stats=aggregate_stats,
            temporal_patterns=temporal_patterns
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_compare(
    labels_a: List[str],
    labels_b: List[str],
    states_a: List[str],
    states_b: List[str],
    name_a: str = "Space A",
    name_b: str = "Space B"
) -> None:
    """
    Quick comparison utility - prints summary to console.

    Args:
        labels_a: Classifications from space A
        labels_b: Classifications from space B
        states_a: State names for space A
        states_b: State names for space B
    """
    result = compare_state_spaces(
        labels_a, labels_b,
        states_a, states_b,
        name_a, name_b
    )

    print(result.summary())
    print("\nDistributions:")
    print(f"  {name_a}: {result.details['distribution_a']}")
    print(f"  {name_b}: {result.details['distribution_b']}")
