"""
Trajectory Analysis for Intervention Planning

Analyzes behavioral trajectories to identify:
- Critical transitions (precursors to harmful outcomes)
- Tipping points (points of no return)
- Intervention windows (optimal timing for interventions)
- Early warning signals (precursor patterns)
- Trajectory momentum (direction of behavioral change)

These analyses inform clinical decision-making about when and
how to intervene in behavioral trajectories.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from collections import Counter
from enum import Enum


class TransitionRisk(Enum):
    """Risk level of state transitions."""
    CRITICAL = "critical"      # Immediate risk (e.g., Seeking->Directing)
    HIGH = "high"              # Elevated risk
    MODERATE = "moderate"      # Moderate concern
    LOW = "low"                # Routine transition


# Critical transition definitions
CRITICAL_TRANSITIONS = {
    ('Seeking', 'Directing'): {
        'risk': TransitionRisk.CRITICAL,
        'name': 'Fantasy to Action',
        'description': 'Direct translation of fantasy into harmful behavior',
        'intervention_urgency': 1.0
    },
    ('Conferring', 'Directing'): {
        'risk': TransitionRisk.CRITICAL,
        'name': 'Surveillance to Strike',
        'description': 'Target selected, moving to action phase',
        'intervention_urgency': 0.95
    },
    ('Revising', 'Directing'): {
        'risk': TransitionRisk.HIGH,
        'name': 'Compulsion to Action',
        'description': 'Ritualistic/habitual pattern leading to action',
        'intervention_urgency': 0.8
    },
    ('Directing', 'Directing'): {
        'risk': TransitionRisk.HIGH,
        'name': 'Sustained Predation',
        'description': 'Continued harmful behavior',
        'intervention_urgency': 0.85
    },
    ('Seeking', 'Conferring'): {
        'risk': TransitionRisk.MODERATE,
        'name': 'Fantasy to Surveillance',
        'description': 'Beginning target selection process',
        'intervention_urgency': 0.6
    }
}


@dataclass
class CriticalTransition:
    """
    A detected critical transition in a trajectory.

    Attributes:
        index: Position in trajectory where transition occurred
        from_state: Source state
        to_state: Target state
        risk_level: Risk classification
        name: Transition name
        description: Description of the transition
        intervention_urgency: Urgency score (0-1)
        context: Surrounding trajectory context
    """
    index: int
    from_state: str
    to_state: str
    risk_level: TransitionRisk
    name: str
    description: str
    intervention_urgency: float
    context: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'from_state': self.from_state,
            'to_state': self.to_state,
            'risk_level': self.risk_level.value,
            'name': self.name,
            'description': self.description,
            'intervention_urgency': self.intervention_urgency,
            'context': self.context
        }


@dataclass
class TippingPoint:
    """
    A tipping point in trajectory where probability of harmful
    outcome crosses a threshold.

    The reversibility score indicates how difficult it is to
    change trajectory after this point.
    """
    index: int
    state: str
    p_harm_before: float       # P(harm) before this point
    p_harm_after: float        # P(harm) after this point
    p_harm_delta: float        # Change in P(harm)
    reversibility_score: float  # How reversible (0=irreversible, 1=fully reversible)
    contributing_factors: List[str] = field(default_factory=list)

    @property
    def is_point_of_no_return(self) -> bool:
        """Check if this is effectively a point of no return."""
        return self.reversibility_score < 0.2 and self.p_harm_after > 0.7

    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'state': self.state,
            'p_harm_before': self.p_harm_before,
            'p_harm_after': self.p_harm_after,
            'p_harm_delta': self.p_harm_delta,
            'reversibility_score': self.reversibility_score,
            'is_point_of_no_return': self.is_point_of_no_return,
            'contributing_factors': self.contributing_factors
        }


@dataclass
class InterventionWindow:
    """
    A time window where intervention is recommended.

    Attributes:
        start_index: Start of window in trajectory
        end_index: End of window
        state_at_window: Dominant state during window
        urgency: Urgency score (0-1)
        recommended_protocols: List of recommended intervention protocols
        expected_effectiveness: Expected effectiveness if intervened
        window_closing_probability: P(window closes before intervention possible)
        rationale: Explanation for why this window was identified
    """
    start_index: int
    end_index: int
    state_at_window: str
    urgency: float
    recommended_protocols: List[str]
    expected_effectiveness: float
    window_closing_probability: float
    rationale: str

    @property
    def window_duration(self) -> int:
        return self.end_index - self.start_index

    def to_dict(self) -> Dict:
        return {
            'start_index': self.start_index,
            'end_index': self.end_index,
            'window_duration': self.window_duration,
            'state_at_window': self.state_at_window,
            'urgency': self.urgency,
            'recommended_protocols': self.recommended_protocols,
            'expected_effectiveness': self.expected_effectiveness,
            'window_closing_probability': self.window_closing_probability,
            'rationale': self.rationale
        }


@dataclass
class EarlyWarning:
    """
    An early warning signal detected in trajectory.

    Early warnings are patterns that precede escalation to
    harmful states.
    """
    index: int
    signal_type: str
    description: str
    severity: float  # 0-1
    time_to_escalation_estimate: int  # Estimated events until escalation
    recommended_action: str

    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'signal_type': self.signal_type,
            'description': self.description,
            'severity': self.severity,
            'time_to_escalation_estimate': self.time_to_escalation_estimate,
            'recommended_action': self.recommended_action
        }


class TrajectoryAnalyzer:
    """
    Analyzes behavioral trajectories for intervention opportunities.

    Uses transition matrices, state sequences, and risk models to
    identify critical points where intervention would be most effective.
    """

    def __init__(
        self,
        state_names: List[str] = None,
        transition_matrix: np.ndarray = None
    ):
        """
        Initialize analyzer.

        Args:
            state_names: Names of states (default: 4-Animal states)
            transition_matrix: Population transition matrix for probability calculations
        """
        self.state_names = state_names or ['Seeking', 'Directing', 'Conferring', 'Revising']
        self.transition_matrix = transition_matrix
        self.n_states = len(self.state_names)
        self.state_to_idx = {s: i for i, s in enumerate(self.state_names)}

        # Precompute absorbing Markov chain metrics if transition matrix provided
        self._mfpt = None
        self._absorption_probs = None
        if transition_matrix is not None:
            self._compute_markov_metrics()

    def _compute_markov_metrics(self) -> None:
        """Compute mean first passage times and absorption probabilities."""
        # Mean First Passage Time
        try:
            n = self.n_states
            K = self.transition_matrix

            # Compute stationary distribution
            eigenvalues, eigenvectors = np.linalg.eig(K.T)
            idx = np.argmin(np.abs(eigenvalues - 1))
            pi = np.real(eigenvectors[:, idx])
            pi = pi / pi.sum()

            # MFPT using fundamental matrix
            I = np.eye(n)
            ones_pi = np.outer(np.ones(n), pi)
            Z = np.linalg.inv(I - K + ones_pi)
            D = np.diag(np.diag(Z))
            self._mfpt = (I - Z + np.ones((n, n)) @ D) / np.diag(Z).reshape(1, -1)

            # Absorption probability to Directing
            # Treat Directing as absorbing state
            directing_idx = self.state_to_idx.get('Directing', 1)
            Q = np.delete(np.delete(K, directing_idx, axis=0), directing_idx, axis=1)
            R = np.delete(K[:, directing_idx], directing_idx)

            # Fundamental matrix for absorbing chain
            N = np.linalg.inv(np.eye(n - 1) - Q)
            # Absorption probabilities
            self._absorption_probs = N @ R

        except Exception:
            self._mfpt = None
            self._absorption_probs = None

    def identify_critical_transitions(
        self,
        trajectory: List[str],
        context_window: int = 2
    ) -> List[CriticalTransition]:
        """
        Find critical transitions in a trajectory.

        Args:
            trajectory: Sequence of state names
            context_window: Number of states before/after to include as context

        Returns:
            List of CriticalTransition objects
        """
        critical = []

        for i in range(len(trajectory) - 1):
            from_state = trajectory[i]
            to_state = trajectory[i + 1]
            transition_key = (from_state, to_state)

            if transition_key in CRITICAL_TRANSITIONS:
                info = CRITICAL_TRANSITIONS[transition_key]

                # Get context
                start = max(0, i - context_window)
                end = min(len(trajectory), i + context_window + 2)
                context = trajectory[start:end]

                critical.append(CriticalTransition(
                    index=i,
                    from_state=from_state,
                    to_state=to_state,
                    risk_level=info['risk'],
                    name=info['name'],
                    description=info['description'],
                    intervention_urgency=info['intervention_urgency'],
                    context=context
                ))

        return critical

    def find_tipping_points(
        self,
        trajectory: List[str],
        threshold: float = 0.5,
        harm_state: str = 'Directing'
    ) -> List[TippingPoint]:
        """
        Find points where P(harm) crosses a threshold.

        Uses absorbing Markov chain analysis to compute probability
        of eventually reaching the harm state.

        Args:
            trajectory: State sequence
            threshold: P(harm) threshold to flag as tipping point
            harm_state: State considered harmful (default: Directing)

        Returns:
            List of TippingPoint objects
        """
        if self.transition_matrix is None:
            return []

        tipping_points = []
        harm_idx = self.state_to_idx.get(harm_state, 1)

        # Compute P(eventually reach harm | current state) for each position
        p_harms = []
        for i, state in enumerate(trajectory):
            state_idx = self.state_to_idx.get(state, 0)

            # Simple estimate: use MFPT and steady-state
            if self._mfpt is not None:
                # Shorter MFPT to harm = higher probability
                mfpt_to_harm = self._mfpt[state_idx, harm_idx]
                # Convert to probability (heuristic)
                p_harm = 1.0 / (1.0 + mfpt_to_harm / 10)
            else:
                # Fallback: base rate from transition matrix
                p_harm = self.transition_matrix[state_idx, harm_idx]

            # Adjust based on recent trajectory momentum
            if i > 0:
                recent_directing = sum(
                    1 for s in trajectory[max(0, i-3):i]
                    if s == harm_state
                ) / min(3, i)
                p_harm = 0.7 * p_harm + 0.3 * recent_directing

            p_harms.append(p_harm)

        # Find threshold crossings
        for i in range(1, len(p_harms)):
            if p_harms[i-1] < threshold <= p_harms[i]:
                # Crossed threshold upward
                reversibility = 1.0 - p_harms[i]  # Higher P(harm) = less reversible

                # Identify contributing factors
                factors = []
                if trajectory[i] == harm_state:
                    factors.append(f"Entered {harm_state} state")
                if i > 1 and trajectory[i-1] in ['Seeking', 'Conferring']:
                    factors.append(f"Critical transition from {trajectory[i-1]}")

                tipping_points.append(TippingPoint(
                    index=i,
                    state=trajectory[i],
                    p_harm_before=p_harms[i-1],
                    p_harm_after=p_harms[i],
                    p_harm_delta=p_harms[i] - p_harms[i-1],
                    reversibility_score=reversibility,
                    contributing_factors=factors
                ))

        return tipping_points

    def compute_intervention_windows(
        self,
        trajectory: List[str],
        protocols: Dict = None
    ) -> List[InterventionWindow]:
        """
        Identify windows where intervention is recommended.

        Windows are identified based on:
        - Current state (non-Directing states offer windows)
        - Upcoming transitions (before critical transitions)
        - Trajectory momentum
        - MFPT to harm state

        Args:
            trajectory: State sequence
            protocols: Available intervention protocols

        Returns:
            List of InterventionWindow objects
        """
        from .protocols import get_protocols_for_state  # Import here to avoid circular

        windows = []

        # Find stretches of non-Directing states
        i = 0
        while i < len(trajectory):
            if trajectory[i] != 'Directing':
                # Start of potential window
                start = i
                state = trajectory[i]

                # Find end of window (next Directing or end)
                while i < len(trajectory) and trajectory[i] != 'Directing':
                    i += 1
                end = i

                # Calculate window properties
                window_states = trajectory[start:end]
                dominant_state = Counter(window_states).most_common(1)[0][0]

                # Calculate urgency based on position and what follows
                urgency = 0.5  # Base urgency
                if end < len(trajectory):
                    # There's a Directing state after this window
                    urgency += 0.3
                if dominant_state == 'Conferring':
                    urgency += 0.1  # Surveillance often precedes action
                if dominant_state == 'Seeking':
                    # Check if there are critical transitions ahead
                    if end < len(trajectory) - 1:
                        next_transition = (trajectory[end-1], trajectory[end])
                        if next_transition in CRITICAL_TRANSITIONS:
                            urgency += 0.2

                urgency = min(1.0, urgency)

                # Get recommended protocols
                try:
                    applicable_protocols = get_protocols_for_state(dominant_state)
                    protocol_names = [p.name for p in applicable_protocols[:3]]
                except Exception:
                    protocol_names = []

                # Expected effectiveness (simplified estimate)
                expected_effectiveness = 0.3 + 0.2 * (1 - urgency)  # Earlier = more effective

                # Window closing probability
                window_duration = end - start
                window_closing_prob = max(0, 1 - window_duration / 10)

                # Rationale
                if dominant_state == 'Seeking':
                    rationale = "Fantasy/planning phase - intervention can redirect before action"
                elif dominant_state == 'Conferring':
                    rationale = "Surveillance phase - disruption can prevent target selection"
                elif dominant_state == 'Revising':
                    rationale = "Processing phase - opportunity to break harmful patterns"
                else:
                    rationale = "Non-harmful state presents intervention opportunity"

                windows.append(InterventionWindow(
                    start_index=start,
                    end_index=end,
                    state_at_window=dominant_state,
                    urgency=urgency,
                    recommended_protocols=protocol_names,
                    expected_effectiveness=expected_effectiveness,
                    window_closing_probability=window_closing_prob,
                    rationale=rationale
                ))
            else:
                i += 1

        return windows

    def trajectory_momentum(
        self,
        trajectory: List[str],
        window_size: int = 5,
        target_state: str = 'Directing'
    ) -> List[Tuple[int, float]]:
        """
        Compute trajectory momentum toward target state.

        Momentum is the rate of change in proportion of target state
        over a sliding window.

        Args:
            trajectory: State sequence
            window_size: Size of sliding window
            target_state: State to measure momentum toward

        Returns:
            List of (index, momentum) tuples
        """
        if len(trajectory) < window_size * 2:
            return []

        momentum_values = []

        for i in range(window_size, len(trajectory) - window_size + 1):
            # Previous window
            prev_window = trajectory[i - window_size:i]
            prev_prop = sum(1 for s in prev_window if s == target_state) / window_size

            # Current window
            curr_window = trajectory[i:i + window_size]
            curr_prop = sum(1 for s in curr_window if s == target_state) / window_size

            # Momentum = rate of change
            momentum = curr_prop - prev_prop
            momentum_values.append((i, momentum))

        return momentum_values

    def early_warning_signals(
        self,
        trajectory: List[str]
    ) -> List[EarlyWarning]:
        """
        Detect early warning signals in trajectory.

        Signals include:
        - Increasing Directing frequency
        - Seeking -> Conferring patterns (fantasy to surveillance)
        - Sustained Conferring (prolonged target selection)
        - Breaking of stable patterns

        Args:
            trajectory: State sequence

        Returns:
            List of EarlyWarning objects
        """
        warnings = []

        # Signal 1: Increasing Directing frequency
        momentum = self.trajectory_momentum(trajectory, window_size=5)
        for idx, mom in momentum:
            if mom > 0.2:  # Significant increase
                warnings.append(EarlyWarning(
                    index=idx,
                    signal_type='momentum_increase',
                    description=f"Directing frequency increasing (momentum: {mom:.2f})",
                    severity=min(1.0, mom * 2),
                    time_to_escalation_estimate=max(1, int(5 / (mom + 0.01))),
                    recommended_action="Increase monitoring frequency and consider intervention"
                ))

        # Signal 2: Seeking -> Conferring pattern (fantasy to surveillance)
        for i in range(len(trajectory) - 1):
            if trajectory[i] == 'Seeking' and trajectory[i+1] == 'Conferring':
                warnings.append(EarlyWarning(
                    index=i,
                    signal_type='fantasy_to_surveillance',
                    description="Transition from fantasy to surveillance behavior",
                    severity=0.6,
                    time_to_escalation_estimate=3,
                    recommended_action="Engage CBT for fantasy management; increase supervision"
                ))

        # Signal 3: Sustained Conferring (3+ consecutive)
        conferring_count = 0
        for i, state in enumerate(trajectory):
            if state == 'Conferring':
                conferring_count += 1
                if conferring_count >= 3:
                    warnings.append(EarlyWarning(
                        index=i,
                        signal_type='sustained_surveillance',
                        description=f"Prolonged surveillance phase ({conferring_count} consecutive)",
                        severity=min(1.0, conferring_count / 5),
                        time_to_escalation_estimate=2,
                        recommended_action="Geographic restriction; disrupt target access"
                    ))
            else:
                conferring_count = 0

        # Signal 4: Pattern destabilization
        if len(trajectory) >= 10:
            first_half = trajectory[:len(trajectory)//2]
            second_half = trajectory[len(trajectory)//2:]

            first_entropy = self._compute_sequence_entropy(first_half)
            second_entropy = self._compute_sequence_entropy(second_half)

            entropy_change = second_entropy - first_entropy
            if abs(entropy_change) > 0.3:
                warnings.append(EarlyWarning(
                    index=len(trajectory)//2,
                    signal_type='pattern_destabilization',
                    description=f"Behavioral pattern {'becoming chaotic' if entropy_change > 0 else 'becoming rigid'}",
                    severity=min(1.0, abs(entropy_change)),
                    time_to_escalation_estimate=5,
                    recommended_action="Review case formulation; adjust treatment approach"
                ))

        return sorted(warnings, key=lambda w: w.index)

    def _compute_sequence_entropy(self, sequence: List[str]) -> float:
        """Compute Shannon entropy of state sequence."""
        if not sequence:
            return 0.0

        counts = Counter(sequence)
        total = len(sequence)
        probs = [count / total for count in counts.values()]

        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        return entropy

    def comprehensive_analysis(
        self,
        trajectory: List[str],
        include_momentum: bool = True,
        include_warnings: bool = True
    ) -> Dict:
        """
        Run comprehensive trajectory analysis.

        Args:
            trajectory: State sequence
            include_momentum: Whether to compute momentum analysis
            include_warnings: Whether to detect early warnings

        Returns:
            Dict with all analysis results
        """
        results = {
            'trajectory_length': len(trajectory),
            'state_distribution': dict(Counter(trajectory)),
            'critical_transitions': [
                ct.to_dict() for ct in self.identify_critical_transitions(trajectory)
            ],
            'tipping_points': [
                tp.to_dict() for tp in self.find_tipping_points(trajectory)
            ],
            'intervention_windows': [
                iw.to_dict() for iw in self.compute_intervention_windows(trajectory)
            ]
        }

        if include_momentum:
            results['momentum'] = self.trajectory_momentum(trajectory)

        if include_warnings:
            results['early_warnings'] = [
                ew.to_dict() for ew in self.early_warning_signals(trajectory)
            ]

        # Summary statistics
        critical_count = len(results['critical_transitions'])
        tipping_count = len(results['tipping_points'])
        window_count = len(results['intervention_windows'])

        results['summary'] = {
            'n_critical_transitions': critical_count,
            'n_tipping_points': tipping_count,
            'n_intervention_windows': window_count,
            'highest_urgency_window': max(
                (w['urgency'] for w in results['intervention_windows']),
                default=0
            ),
            'overall_risk_level': self._compute_overall_risk(
                critical_count, tipping_count, trajectory
            )
        }

        return results

    def _compute_overall_risk(
        self,
        n_critical: int,
        n_tipping: int,
        trajectory: List[str]
    ) -> str:
        """Compute overall risk level from analysis results."""
        directing_prop = trajectory.count('Directing') / len(trajectory) if trajectory else 0

        if n_critical >= 3 or directing_prop > 0.6:
            return 'CRITICAL'
        elif n_critical >= 2 or n_tipping >= 2 or directing_prop > 0.4:
            return 'HIGH'
        elif n_critical >= 1 or n_tipping >= 1 or directing_prop > 0.2:
            return 'MODERATE'
        else:
            return 'LOW'


def analyze_trajectory(
    trajectory: List[str],
    transition_matrix: np.ndarray = None
) -> Dict:
    """
    Convenience function for trajectory analysis.

    Args:
        trajectory: State sequence
        transition_matrix: Optional population transition matrix

    Returns:
        Comprehensive analysis results
    """
    analyzer = TrajectoryAnalyzer(transition_matrix=transition_matrix)
    return analyzer.comprehensive_analysis(trajectory)
