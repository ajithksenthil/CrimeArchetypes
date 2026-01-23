"""
Data Loader for Dashboard

Loads and provides access to all analysis outputs from empirical_study directory.
Includes methods for individual trajectory and transition matrix loading
for intervention analysis.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
from collections import Counter


# State names for the 4-animal model
STATE_NAMES = ['Seeking', 'Directing', 'Conferring', 'Revising']


class DashboardDataLoader:
    """Load and provide access to all dashboard data."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._classifications = None
        self._signatures = None
        self._profiles = None
        self._reincarnation = None
        self._te_matrix = None
        self._criminal_order = None
        self._clusters = None
        self._label_to_theme = None
        self._llm_results = None
        self._llm_detailed = None
        self._detailed_classifications = None  # For event-level data

    def _find_latest_dir(self, pattern: str) -> Optional[Path]:
        """Find the most recent results directory matching pattern."""
        dirs = sorted(self.data_dir.glob(pattern))
        return dirs[-1] if dirs else None

    @property
    def classifications(self) -> Dict:
        """Load hierarchical classification results."""
        if self._classifications is None:
            latest = self._find_latest_dir("hierarchical_archetypes_*")
            if latest:
                with open(latest / "classification_results.json") as f:
                    self._classifications = json.load(f)
            else:
                self._classifications = {'summary': {}, 'classifications': []}
        return self._classifications

    @property
    def signatures(self) -> Dict:
        """Load behavioral signatures."""
        if self._signatures is None:
            latest = self._find_latest_dir("archetypal_interpretation_*")
            if latest:
                with open(latest / "interpretation_results.json") as f:
                    data = json.load(f)
                    self._signatures = data.get('signatures', {})
            else:
                self._signatures = {}
        return self._signatures

    @property
    def profiles(self) -> Dict:
        """Load archetype profiles."""
        if self._profiles is None:
            latest = self._find_latest_dir("archetypal_interpretation_*")
            if latest:
                with open(latest / "interpretation_results.json") as f:
                    data = json.load(f)
                    self._profiles = data.get('profiles', {})
            else:
                self._profiles = {}
        return self._profiles

    @property
    def reincarnation(self) -> Dict:
        """Load reincarnation analysis results."""
        if self._reincarnation is None:
            latest = self._find_latest_dir("archetypal_reincarnation_*")
            if latest:
                with open(latest / "reincarnation_results.json") as f:
                    self._reincarnation = json.load(f)
            else:
                self._reincarnation = {'roles': {}, 'lineages': [], 'network_metrics': {}}
        return self._reincarnation

    @property
    def te_matrix(self) -> Optional[np.ndarray]:
        """Load transfer entropy matrix."""
        if self._te_matrix is None:
            latest = self._find_latest_dir("archetypal_reincarnation_*")
            if latest:
                matrix_path = latest / "reincarnation_matrix.npy"
                if matrix_path.exists():
                    self._te_matrix = np.load(matrix_path)
        return self._te_matrix

    @property
    def criminal_order(self) -> List[str]:
        """Load criminal order for matrix indexing."""
        if self._criminal_order is None:
            latest = self._find_latest_dir("archetypal_reincarnation_*")
            if latest:
                order_path = latest / "criminal_order.json"
                if order_path.exists():
                    with open(order_path) as f:
                        self._criminal_order = json.load(f)
                else:
                    # Fall back to getting names from classifications
                    self._criminal_order = [c['name'] for c in self.classifications.get('classifications', [])]
            else:
                self._criminal_order = []
        return self._criminal_order

    @property
    def clusters(self) -> List[Dict]:
        """Load life event cluster data with LLM-generated archetypal themes."""
        if self._clusters is None:
            # Try multiple locations
            cluster_paths = [
                self.data_dir / "clusters.json",
                self.data_dir / "analysis_output" / "clusters.json",
                self.data_dir.parent / "clusters.json"
            ]

            for path in cluster_paths:
                if path.exists():
                    with open(path) as f:
                        self._clusters = json.load(f)
                    break
            else:
                self._clusters = []
        return self._clusters

    @property
    def llm_results(self) -> Dict:
        """Load LLM 4-Animal classification results."""
        if self._llm_results is None:
            latest = self._find_latest_dir("llm_animal_study_*")
            if latest:
                results_path = latest / "llm_results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        self._llm_results = json.load(f)
            if self._llm_results is None:
                self._llm_results = {}
        return self._llm_results

    @property
    def llm_detailed(self) -> List[Dict]:
        """Load detailed LLM classifications per event."""
        if self._llm_detailed is None:
            latest = self._find_latest_dir("llm_animal_study_*")
            if latest:
                detailed_path = latest / "detailed_classifications.json"
                if detailed_path.exists():
                    with open(detailed_path) as f:
                        self._llm_detailed = json.load(f)
            if self._llm_detailed is None:
                self._llm_detailed = []
        return self._llm_detailed

    @property
    def detailed_classifications(self) -> Dict[str, List[Dict]]:
        """
        Load detailed event classifications organized by individual.

        Returns:
            Dict mapping criminal name to list of event dicts
        """
        if self._detailed_classifications is None:
            self._detailed_classifications = {}
            # Get from llm_detailed which has per-event classifications
            for event in self.llm_detailed:
                criminal = event.get('criminal', '')
                if criminal:
                    if criminal not in self._detailed_classifications:
                        self._detailed_classifications[criminal] = []
                    self._detailed_classifications[criminal].append(event)
        return self._detailed_classifications

    def load_individual_trajectory(self, criminal_name: str) -> List[str]:
        """
        Load the state sequence (trajectory) for an individual.

        Args:
            criminal_name: Name of the individual

        Returns:
            List of state names in chronological order

        Raises:
            ValueError: If no data found for the individual
        """
        events = self.detailed_classifications.get(criminal_name, [])
        if not events:
            raise ValueError(f"No trajectory data found for {criminal_name}")

        # Extract state sequence in order (events are already chronological)
        trajectory = [e.get('state') for e in events if e.get('state')]
        return trajectory

    def load_individual_transition_matrix(self, criminal_name: str) -> np.ndarray:
        """
        Load or compute the transition matrix for an individual.

        First tries to get from signature data, falls back to computing
        from trajectory.

        Args:
            criminal_name: Name of the individual

        Returns:
            4x4 numpy array representing transition probabilities
        """
        # Try to get from signatures (has pre-computed transition matrix)
        signature = self.signatures.get(criminal_name, {})
        if 'transition_matrix' in signature:
            tm = signature['transition_matrix']
            # Convert from dict format to numpy array
            matrix = np.zeros((4, 4))
            for i, from_state in enumerate(STATE_NAMES):
                for j, to_state in enumerate(STATE_NAMES):
                    if from_state in tm and to_state in tm[from_state]:
                        matrix[i, j] = tm[from_state][to_state]
            # Normalize rows to ensure they sum to 1
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            matrix = matrix / row_sums
            return matrix

        # Fall back to computing from trajectory
        trajectory = self.load_individual_trajectory(criminal_name)
        return self._compute_transition_matrix(trajectory)

    def _compute_transition_matrix(self, trajectory: List[str]) -> np.ndarray:
        """
        Compute transition matrix from a state sequence.

        Args:
            trajectory: List of state names

        Returns:
            4x4 transition probability matrix
        """
        # Count transitions
        counts = np.zeros((4, 4))
        state_to_idx = {s: i for i, s in enumerate(STATE_NAMES)}

        for i in range(len(trajectory) - 1):
            from_state = trajectory[i]
            to_state = trajectory[i + 1]
            if from_state in state_to_idx and to_state in state_to_idx:
                counts[state_to_idx[from_state], state_to_idx[to_state]] += 1

        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = counts / row_sums

        return matrix

    def get_individual_events(self, criminal_name: str) -> List[Dict]:
        """
        Get detailed events for an individual with state classifications.

        Args:
            criminal_name: Name of the individual

        Returns:
            List of event dicts with keys: event, state, confidence, reasoning
        """
        events = self.detailed_classifications.get(criminal_name, [])
        if not events:
            raise ValueError(f"No event data found for {criminal_name}")
        return events

    def get_trajectory_statistics(self, criminal_name: str) -> Dict:
        """
        Get trajectory statistics for an individual.

        Args:
            criminal_name: Name of the individual

        Returns:
            Dict with trajectory statistics
        """
        trajectory = self.load_individual_trajectory(criminal_name)
        events = self.get_individual_events(criminal_name)

        # State distribution
        state_counts = Counter(trajectory)
        total = len(trajectory)
        state_distribution = {
            state: state_counts.get(state, 0) / total if total > 0 else 0
            for state in STATE_NAMES
        }

        # Dominant state
        dominant_state = max(state_distribution.items(), key=lambda x: x[1])

        # Transition counts
        transition_counts = Counter()
        for i in range(len(trajectory) - 1):
            transition_counts[(trajectory[i], trajectory[i + 1])] += 1

        # Critical transitions
        critical_transitions = []
        for trans, count in transition_counts.most_common():
            if trans[1] == 'Directing' and trans[0] != 'Directing':
                critical_transitions.append({
                    'transition': f"{trans[0]} → {trans[1]}",
                    'count': count
                })

        # Confidence distribution
        confidence_counts = Counter(e.get('confidence', 'UNKNOWN') for e in events)

        return {
            'n_events': len(trajectory),
            'state_distribution': state_distribution,
            'dominant_state': dominant_state[0],
            'dominant_state_pct': dominant_state[1],
            'critical_transitions': critical_transitions,
            'confidence_distribution': dict(confidence_counts),
            'trajectory': trajectory
        }

    def get_llm_stats(self) -> Dict:
        """Get LLM classification statistics for display."""
        results = self.llm_results
        if not results:
            return None

        return {
            'model': results.get('model', 'Unknown'),
            'n_events': results.get('n_events', 0),
            'n_individuals': results.get('n_individuals', 0),
            'state_distribution': results.get('state_distribution', {}),
            'confidence_distribution': results.get('confidence_distribution', {}),
            'entropy_rate': results.get('entropy_rate', 0),
            'effective_states': results.get('effective_states', 0),
            'transition_matrix': results.get('transition_matrix', []),
            'stationary_distribution': results.get('stationary_distribution', []),
            'cv_accuracy': results.get('cv_accuracy_mean', 0),
            'cv_std': results.get('cv_accuracy_std', 0)
        }

    @property
    def label_to_theme(self) -> Dict[int, str]:
        """Load mapping from cluster labels to theme names."""
        if self._label_to_theme is None:
            import pickle
            import re
            theme_paths = [
                self.data_dir / "label_to_theme.pkl",
                self.data_dir.parent / "label_to_theme.pkl"
            ]

            for path in theme_paths:
                if path.exists():
                    with open(path, 'rb') as f:
                        self._label_to_theme = pickle.load(f)
                    break
            else:
                # Build from clusters if available
                self._label_to_theme = {}
                for cluster in self.clusters:
                    cid = cluster.get('cluster_id', 0)
                    theme = cluster.get('archetypal_theme', f'Cluster {cid}')

                    # Extract short theme name
                    # Look for quoted theme names first
                    quoted = re.search(r'"([^"]+)"', theme)
                    if quoted:
                        short_theme = quoted.group(1)
                    else:
                        # Try to extract from "Theme:" or before first period/colon
                        parts = theme.split(':')
                        if len(parts) > 1 and len(parts[0]) < 60:
                            short_theme = parts[0].strip()
                        else:
                            short_theme = theme.split('.')[0].strip()

                    # Truncate if still too long
                    if len(short_theme) > 40:
                        short_theme = short_theme[:37] + '...'

                    self._label_to_theme[cid] = short_theme
        return self._label_to_theme

    def get_cluster_stats(self) -> Dict:
        """Get cluster-level statistics."""
        clusters = self.clusters
        if not clusters:
            return {'total_clusters': 0, 'total_events': 0, 'clusters': []}

        total_events = sum(c.get('size', 0) for c in clusters)

        cluster_info = []
        for c in clusters:
            cluster_info.append({
                'id': c.get('cluster_id', 0),
                'size': c.get('size', 0),
                'percentage': c.get('size', 0) / total_events * 100 if total_events > 0 else 0,
                'theme': self.label_to_theme.get(c.get('cluster_id', 0), 'Unknown'),
                'full_theme': c.get('archetypal_theme', ''),
                'representative_samples': c.get('representative_samples', [])
            })

        return {
            'total_clusters': len(clusters),
            'total_events': total_events,
            'clusters': sorted(cluster_info, key=lambda x: x['size'], reverse=True)
        }

    def get_individual_data(self, name: str) -> Dict:
        """Get all data for a specific individual."""
        # Get classification
        classification = next(
            (c for c in self.classifications.get('classifications', []) if c['name'] == name),
            None
        )

        # Get signature
        signature = self.signatures.get(name, {})

        # Find network role
        roles = self.reincarnation.get('roles', {})
        role = 'GENERAL'
        for role_name in ['hubs', 'sources', 'sinks']:
            role_list = roles.get(role_name, [])
            if any(r.get('name') == name for r in role_list):
                role = role_name.upper().rstrip('S')
                break

        # Get influence metrics
        influence_metrics = {}
        if self.te_matrix is not None and name in self.criminal_order:
            idx = self.criminal_order.index(name)
            influence_metrics = {
                'outgoing': float(np.sum(self.te_matrix[idx, :])),
                'incoming': float(np.sum(self.te_matrix[:, idx])),
                'top_influenced': self._get_top_influenced(idx, n=5),
                'top_influenced_by': self._get_top_influenced_by(idx, n=5)
            }

        # Find lineages containing this individual
        lineages = [
            l for l in self.reincarnation.get('lineages', [])
            if name in l
        ]

        return {
            'name': name,
            'classification': classification,
            'signature': signature,
            'network_role': role,
            'influence_metrics': influence_metrics,
            'lineages': lineages
        }

    def _get_top_influenced(self, idx: int, n: int = 5) -> List[Dict]:
        """Get top n individuals influenced by given index."""
        if self.te_matrix is None:
            return []

        row = self.te_matrix[idx, :]
        top_indices = np.argsort(row)[::-1][:n]

        return [
            {'name': self.criminal_order[i], 'te': float(row[i])}
            for i in top_indices if i != idx and row[i] > 0
        ]

    def _get_top_influenced_by(self, idx: int, n: int = 5) -> List[Dict]:
        """Get top n individuals who influenced given index."""
        if self.te_matrix is None:
            return []

        col = self.te_matrix[:, idx]
        top_indices = np.argsort(col)[::-1][:n]

        return [
            {'name': self.criminal_order[i], 'te': float(col[i])}
            for i in top_indices if i != idx and col[i] > 0
        ]

    def get_all_individuals(self) -> List[str]:
        """Get list of all individual names."""
        return [c['name'] for c in self.classifications.get('classifications', [])]

    def get_individuals_with_trajectory_data(self) -> List[str]:
        """
        Get list of individuals who have trajectory data available.

        Returns:
            List of criminal names with event-level data
        """
        return list(self.detailed_classifications.keys())

    def get_individual_intervention_data(self, criminal_name: str) -> Dict:
        """
        Get all data needed for intervention analysis for an individual.

        Args:
            criminal_name: Name of the individual

        Returns:
            Dict containing trajectory, transition_matrix, events,
            classification, and signature
        """
        return {
            'name': criminal_name,
            'trajectory': self.load_individual_trajectory(criminal_name),
            'transition_matrix': self.load_individual_transition_matrix(criminal_name),
            'events': self.get_individual_events(criminal_name),
            'statistics': self.get_trajectory_statistics(criminal_name),
            'classification': self.get_individual_data(criminal_name)
        }

    def get_archetype_members(self, primary: str = None, subtype: str = None) -> List[Dict]:
        """Get members of a specific archetype."""
        classifications = self.classifications.get('classifications', [])

        if primary:
            classifications = [c for c in classifications if c.get('primary_type') == primary]

        if subtype:
            classifications = [c for c in classifications if c.get('subtype') == subtype]

        return classifications

    def get_population_stats(self) -> Dict:
        """Get population-level statistics."""
        summary = self.classifications.get('summary', {})

        # Calculate risk distribution
        risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'UNPREDICTABLE': 0}
        for c in self.classifications.get('classifications', []):
            subtype = c.get('subtype', '')
            if subtype in ['Pure Predator', 'Fantasy-Actor']:
                risk_counts['CRITICAL'] += 1
            elif subtype in ['Strong Escalator', 'Stalker-Striker', 'Standard']:
                risk_counts['HIGH'] += 1
            elif subtype in ['Chameleon', 'Multi-Modal']:
                risk_counts['UNPREDICTABLE'] += 1

        return {
            'total': summary.get('total', len(self.get_all_individuals())),
            'primary_distribution': summary.get('primary_distribution', {}),
            'subtype_distribution': summary.get('subtype_distribution', {}),
            'primary_percentages': summary.get('primary_percentages', {}),
            'risk_distribution': risk_counts,
            'roles': {
                'sources': len(self.reincarnation.get('roles', {}).get('sources', [])),
                'sinks': len(self.reincarnation.get('roles', {}).get('sinks', [])),
                'hubs': len(self.reincarnation.get('roles', {}).get('hubs', []))
            }
        }

    def get_comparison_data(self, name_a: str, name_b: str) -> Dict:
        """Get data for comparing two individuals."""
        return {
            'a': self.get_individual_data(name_a),
            'b': self.get_individual_data(name_b)
        }

    def get_archetype_comparison(self, type_a: str, type_b: str) -> Dict:
        """Get comparison data for two archetype types (primary or subtype)."""
        members_a = []
        members_b = []

        for c in self.classifications.get('classifications', []):
            if c.get('primary_type') == type_a or c.get('subtype') == type_a:
                members_a.append(c['name'])
            if c.get('primary_type') == type_b or c.get('subtype') == type_b:
                members_b.append(c['name'])

        # Aggregate signatures
        def aggregate_signatures(names):
            sigs = [self.signatures.get(n, {}) for n in names if n in self.signatures]
            if not sigs:
                return {}

            # Average state distributions
            avg_dist = {}
            for state in ['Seeking', 'Directing', 'Conferring', 'Revising']:
                values = [s.get('state_distribution', {}).get(state, 0) for s in sigs]
                avg_dist[state] = sum(values) / len(values) if values else 0

            # Average escalation
            esc_values = [s.get('escalation_score', 0) for s in sigs]
            avg_escalation = sum(esc_values) / len(esc_values) if esc_values else 0

            return {
                'state_distribution': avg_dist,
                'escalation_score': avg_escalation,
                'member_count': len(sigs)
            }

        return {
            'type_a': {
                'name': type_a,
                'members': members_a,
                'aggregate': aggregate_signatures(members_a)
            },
            'type_b': {
                'name': type_b,
                'members': members_b,
                'aggregate': aggregate_signatures(members_b)
            }
        }


@lru_cache(maxsize=1)
def get_data_loader(data_dir: str) -> DashboardDataLoader:
    """Get cached data loader instance."""
    return DashboardDataLoader(Path(data_dir))
