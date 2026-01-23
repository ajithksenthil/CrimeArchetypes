"""
Intervention Optimization Framework

Finds optimal intervention timing and protocol selection using:
- Dynamic programming for multi-stage optimization
- Multi-objective optimization (effectiveness vs cost)
- Pareto frontier analysis for trade-off visualization
- Urgency scoring for prioritization

The optimizer integrates with:
- BehavioralSCM for causal effect estimation
- InterventionProtocol library for available interventions
- TrajectoryAnalyzer for risk assessment

References:
- Sutton & Barto (2018). Reinforcement Learning: An Introduction
- Marler & Arora (2004). Survey of multi-objective optimization methods
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from enum import Enum
from copy import deepcopy
import warnings

from .causal_model import BehavioralSCM, InterventionNode
from .protocols import (
    InterventionProtocol,
    INTERVENTION_PROTOCOLS,
    ProtocolCategory,
    get_protocols_for_state,
    get_protocols_for_transition
)


class OptimizationObjective(Enum):
    """Objectives for optimization."""
    HARM_REDUCTION = "harm_reduction"
    COST_EFFECTIVENESS = "cost_effectiveness"
    EARLIEST_INTERVENTION = "earliest_intervention"
    MINIMUM_INTENSITY = "minimum_intensity"


@dataclass
class InterventionRecommendation:
    """
    A single intervention recommendation.

    Attributes:
        time_index: When to intervene (position in trajectory)
        protocol: Recommended protocol
        expected_benefit: Expected harm reduction
        cost: Total estimated cost
        urgency: How urgent is this intervention (0-1)
        confidence: Confidence in this recommendation (0-1)
        rationale: Explanation for the recommendation
    """
    time_index: int
    protocol: InterventionProtocol
    expected_benefit: float
    cost: float
    urgency: float
    confidence: float
    rationale: str

    @property
    def cost_effectiveness(self) -> float:
        """Benefit per dollar spent."""
        if self.cost <= 0:
            return float('inf')
        return self.expected_benefit / self.cost

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'time_index': self.time_index,
            'protocol_name': self.protocol.name,
            'protocol_display_name': self.protocol.display_name,
            'expected_benefit': self.expected_benefit,
            'cost': self.cost,
            'urgency': self.urgency,
            'confidence': self.confidence,
            'cost_effectiveness': self.cost_effectiveness,
            'rationale': self.rationale
        }


@dataclass
class OptimizationResult:
    """
    Result of intervention optimization.

    Attributes:
        recommendations: Ordered list of intervention recommendations
        pareto_frontier: Pareto-optimal intervention combinations
        total_expected_benefit: Combined benefit of all recommendations
        total_cost: Combined cost of all recommendations
        optimization_details: Additional optimization metrics
    """
    recommendations: List[InterventionRecommendation]
    pareto_frontier: List[Tuple[float, float]]  # (cost, benefit) pairs
    total_expected_benefit: float
    total_cost: float
    optimization_details: Dict = field(default_factory=dict)

    def get_top_recommendations(self, n: int = 3) -> List[InterventionRecommendation]:
        """Get top N recommendations by expected benefit."""
        sorted_recs = sorted(
            self.recommendations,
            key=lambda r: r.expected_benefit,
            reverse=True
        )
        return sorted_recs[:n]

    def get_cost_constrained(self, max_cost: float) -> List[InterventionRecommendation]:
        """Get recommendations within cost constraint."""
        # Sort by cost-effectiveness and accumulate within budget
        sorted_recs = sorted(
            self.recommendations,
            key=lambda r: r.cost_effectiveness,
            reverse=True
        )

        selected = []
        remaining_budget = max_cost
        for rec in sorted_recs:
            if rec.cost <= remaining_budget:
                selected.append(rec)
                remaining_budget -= rec.cost

        return selected


@dataclass
class UrgencyScore:
    """
    Urgency assessment for intervention.

    Combines multiple risk factors into an overall urgency score.
    """
    overall: float  # Combined urgency (0-1)

    # Component scores
    state_risk: float  # Risk from current state
    trajectory_momentum: float  # Recent escalation pattern
    mfpt_urgency: float  # Based on mean first passage time
    transition_risk: float  # Risk of imminent harmful transition

    # Contributing factors
    factors: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        current_state: str,
        recent_trajectory: List[str],
        mfpt_to_directing: float,
        transition_probs: Dict[str, float] = None,
        weights: Dict[str, float] = None
    ) -> 'UrgencyScore':
        """
        Compute urgency score from multiple factors.

        Args:
            current_state: Current behavioral state
            recent_trajectory: Recent state sequence
            mfpt_to_directing: Mean first passage time to Directing
            transition_probs: P(next_state | current_state)
            weights: Custom weights for components
        """
        if weights is None:
            weights = {
                'state_risk': 0.25,
                'trajectory_momentum': 0.25,
                'mfpt_urgency': 0.25,
                'transition_risk': 0.25
            }

        # State risk (higher for states closer to harm)
        state_risks = {
            'Directing': 1.0,
            'Conferring': 0.6,
            'Seeking': 0.4,
            'Revising': 0.2
        }
        state_risk = state_risks.get(current_state, 0.3)

        # Trajectory momentum (escalation pattern)
        if len(recent_trajectory) >= 2:
            escalations = 0
            for i in range(len(recent_trajectory) - 1):
                risk_change = state_risks.get(recent_trajectory[i+1], 0) - \
                             state_risks.get(recent_trajectory[i], 0)
                if risk_change > 0:
                    escalations += 1
            trajectory_momentum = escalations / (len(recent_trajectory) - 1)
        else:
            trajectory_momentum = 0.5  # Unknown

        # MFPT urgency (lower MFPT = higher urgency)
        # Normalize: MFPT of 1 = urgency 1.0, MFPT of 10+ = urgency 0.1
        mfpt_urgency = max(0.1, min(1.0, 1.0 / max(1, mfpt_to_directing)))

        # Transition risk (probability of harmful transition)
        if transition_probs is not None:
            transition_risk = transition_probs.get('Directing', 0.0)
        else:
            # Default based on state
            default_directing_probs = {
                'Seeking': 0.15,
                'Conferring': 0.25,
                'Directing': 1.0,
                'Revising': 0.05
            }
            transition_risk = default_directing_probs.get(current_state, 0.1)

        # Combine with weights
        overall = (
            weights['state_risk'] * state_risk +
            weights['trajectory_momentum'] * trajectory_momentum +
            weights['mfpt_urgency'] * mfpt_urgency +
            weights['transition_risk'] * transition_risk
        )

        return cls(
            overall=min(1.0, overall),
            state_risk=state_risk,
            trajectory_momentum=trajectory_momentum,
            mfpt_urgency=mfpt_urgency,
            transition_risk=transition_risk,
            factors={
                'current_state': current_state,
                'trajectory_length': len(recent_trajectory),
                'mfpt_to_directing': mfpt_to_directing
            }
        )


class InterventionOptimizer:
    """
    Optimizer for intervention timing and selection.

    Uses the Structural Causal Model to evaluate intervention effects
    and find optimal timing/protocol combinations.

    Attributes:
        scm: Behavioral Structural Causal Model
        protocols: Available intervention protocols
        harm_state: State representing harmful outcome (default: Directing)
        discount_factor: Temporal discounting for future benefits
    """

    def __init__(
        self,
        scm: BehavioralSCM,
        protocols: Dict[str, InterventionProtocol] = None,
        harm_state: str = 'Directing',
        discount_factor: float = 0.95
    ):
        """
        Initialize optimizer.

        Args:
            scm: Behavioral SCM for causal effect estimation
            protocols: Available protocols (defaults to full library)
            harm_state: State representing harm (for benefit calculation)
            discount_factor: Discount for future benefits (0-1)
        """
        self.scm = scm
        self.protocols = protocols or INTERVENTION_PROTOCOLS
        self.harm_state = harm_state
        self.discount_factor = discount_factor

        # Cache for efficiency
        self._effect_cache: Dict[Tuple, float] = {}

    def compute_intervention_value(
        self,
        protocol: InterventionProtocol,
        current_state: str,
        time_remaining: int,
        n_simulations: int = 500
    ) -> float:
        """
        Compute expected value of an intervention.

        Value = Expected reduction in probability of reaching harm state.

        Args:
            protocol: Intervention protocol to evaluate
            current_state: Current behavioral state
            time_remaining: Steps until end of analysis window
            n_simulations: Monte Carlo simulations for estimation

        Returns:
            Expected harm reduction (0-1)
        """
        # Check cache
        cache_key = (protocol.name, current_state, time_remaining)
        if cache_key in self._effect_cache:
            return self._effect_cache[cache_key]

        # Add intervention to SCM
        intervention_name = f"opt_{protocol.name}"

        # Create copy of SCM to avoid modifying original
        scm_copy = deepcopy(self.scm)

        try:
            scm_copy.add_intervention(
                name=intervention_name,
                target_transitions=protocol.target_transitions,
                effects=protocol.effect_on_transitions,
                description=protocol.display_name
            )

            # Compute causal effect (outcome names are lowercase)
            outcome_name = f"reached_{self.harm_state.lower()}"
            effect = scm_copy.compute_causal_effect(
                intervention_name=intervention_name,
                outcome_name=outcome_name,
                initial_state=current_state,
                n_simulations=n_simulations
            )

            # Value is the reduction in harm probability (ATE should be negative)
            value = -effect.get('ate', 0)

            # Apply effectiveness estimate from protocol
            value = value * protocol.effectiveness_estimate

            # Temporal discounting
            value = value * (self.discount_factor ** time_remaining)

        except Exception as e:
            warnings.warn(f"Error computing intervention value: {e}")
            value = 0.0

        # Cache result
        self._effect_cache[cache_key] = value
        return value

    def find_optimal_timing(
        self,
        trajectory: List[str],
        available_protocols: List[str] = None,
        budget_constraint: float = None,
        max_interventions: int = 3
    ) -> OptimizationResult:
        """
        Find optimal intervention points in a trajectory.

        Uses dynamic programming to find the combination of
        interventions and timing that maximizes harm reduction.

        Args:
            trajectory: Observed state trajectory
            available_protocols: Protocols to consider (None = all)
            budget_constraint: Maximum total cost
            max_interventions: Maximum number of interventions

        Returns:
            OptimizationResult with recommendations and Pareto frontier
        """
        if available_protocols is None:
            available_protocols = list(self.protocols.keys())

        protocols = [self.protocols[p] for p in available_protocols if p in self.protocols]
        T = len(trajectory)

        recommendations = []

        # Evaluate each time point and protocol
        for t in range(T):
            state = trajectory[t]
            time_remaining = T - t - 1

            # Get protocols applicable to this state
            applicable = [
                p for p in protocols
                if state in p.target_states or not p.target_states
            ]

            for protocol in applicable:
                # Skip if contraindicated
                if state in protocol.contraindicated_states:
                    continue

                # Compute expected benefit
                benefit = self.compute_intervention_value(
                    protocol=protocol,
                    current_state=state,
                    time_remaining=time_remaining
                )

                # Compute cost
                cost = protocol.cost_per_week * protocol.duration_weeks

                # Compute urgency
                recent = trajectory[max(0, t-5):t+1]
                urgency = UrgencyScore.compute(
                    current_state=state,
                    recent_trajectory=recent,
                    mfpt_to_directing=max(1, time_remaining)  # Approximation
                )

                # Create recommendation
                rec = InterventionRecommendation(
                    time_index=t,
                    protocol=protocol,
                    expected_benefit=benefit,
                    cost=cost,
                    urgency=urgency.overall,
                    confidence=0.7,  # Base confidence
                    rationale=self._generate_rationale(
                        state=state,
                        protocol=protocol,
                        benefit=benefit,
                        urgency=urgency
                    )
                )

                recommendations.append(rec)

        # Apply constraints
        if budget_constraint is not None:
            recommendations = self._apply_budget_constraint(
                recommendations, budget_constraint
            )

        # Select top recommendations (avoiding redundancy)
        selected = self._select_top_recommendations(
            recommendations, max_interventions
        )

        # Compute Pareto frontier
        pareto_frontier = self._compute_pareto_frontier(recommendations)

        # Calculate totals
        total_benefit = sum(r.expected_benefit for r in selected)
        total_cost = sum(r.cost for r in selected)

        return OptimizationResult(
            recommendations=selected,
            pareto_frontier=pareto_frontier,
            total_expected_benefit=total_benefit,
            total_cost=total_cost,
            optimization_details={
                'trajectory_length': T,
                'protocols_evaluated': len(protocols),
                'total_recommendations': len(recommendations),
                'budget_constraint': budget_constraint,
                'max_interventions': max_interventions
            }
        )

    def optimize_protocol_selection(
        self,
        current_state: str,
        archetype: str = None,
        constraints: Dict = None
    ) -> List[InterventionRecommendation]:
        """
        Select best protocol for current state and context.

        Multi-objective optimization considering:
        - Effectiveness for this state
        - Match to archetype (if known)
        - Resource constraints
        - Intensity preferences

        Args:
            current_state: Current behavioral state
            archetype: Behavioral archetype (optional)
            constraints: Dict with 'max_cost', 'max_intensity', etc.

        Returns:
            Ranked list of protocol recommendations
        """
        constraints = constraints or {}
        max_cost = constraints.get('max_cost', float('inf'))
        max_intensity = constraints.get('max_intensity', 'intensive')

        intensity_order = ['low', 'medium', 'high', 'intensive']
        max_intensity_idx = intensity_order.index(max_intensity) if max_intensity in intensity_order else 3

        recommendations = []

        for protocol in self.protocols.values():
            # Check state applicability
            if current_state not in protocol.target_states and protocol.target_states:
                continue

            # Check contraindications
            if current_state in protocol.contraindicated_states:
                continue

            # Check cost constraint
            total_cost = protocol.cost_per_week * protocol.duration_weeks
            if total_cost > max_cost:
                continue

            # Check intensity constraint
            if protocol.default_intensity:
                intensity_name = protocol.default_intensity.value if hasattr(protocol.default_intensity, 'value') else protocol.default_intensity
                if intensity_name in intensity_order:
                    if intensity_order.index(intensity_name) > max_intensity_idx:
                        continue

            # Compute expected benefit
            benefit = protocol.effectiveness_estimate

            # Adjust for archetype if known
            archetype_multipliers = {
                'hunter': {'supervision': 1.2, 'therapeutic': 0.9},
                'predator': {'therapeutic': 1.3, 'pharmacological': 1.1},
                'opportunist': {'environmental': 1.2, 'supervision': 1.1},
                'ritualist': {'therapeutic': 1.4, 'combined': 1.2}
            }

            if archetype and archetype.lower() in archetype_multipliers:
                category = protocol.category.value if hasattr(protocol.category, 'value') else protocol.category
                multiplier = archetype_multipliers[archetype.lower()].get(category, 1.0)
                benefit = benefit * multiplier

            rec = InterventionRecommendation(
                time_index=0,  # Current time
                protocol=protocol,
                expected_benefit=benefit,
                cost=total_cost,
                urgency=0.5,  # Default
                confidence=min(0.9, protocol.evidence_level.value == 'A' and 0.9 or 0.7),
                rationale=f"Recommended for {current_state} state"
            )

            recommendations.append(rec)

        # Sort by benefit
        recommendations.sort(key=lambda r: r.expected_benefit, reverse=True)

        return recommendations

    def compute_intervention_frontier(
        self,
        trajectory: List[str],
        granularity: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Compute Pareto frontier of cost vs harm reduction.

        Returns combinations where you can't improve one objective
        without worsening the other.

        Args:
            trajectory: State trajectory
            granularity: Number of budget levels to evaluate

        Returns:
            List of (cost, benefit) Pareto-optimal points
        """
        # Get all possible single interventions
        all_costs = []
        for protocol in self.protocols.values():
            all_costs.append(protocol.cost_per_week * protocol.duration_weeks)

        if not all_costs:
            return [(0, 0)]

        max_cost = max(all_costs) * 3  # Allow up to 3 interventions
        budget_levels = np.linspace(0, max_cost, granularity)

        frontier_points = [(0, 0)]  # Zero budget = zero benefit

        for budget in budget_levels[1:]:
            result = self.find_optimal_timing(
                trajectory=trajectory,
                budget_constraint=budget,
                max_interventions=3
            )

            if result.recommendations:
                frontier_points.append((
                    result.total_cost,
                    result.total_expected_benefit
                ))

        # Filter to Pareto-optimal points
        pareto_points = []
        for cost, benefit in sorted(frontier_points):
            # Check if dominated by any existing point
            dominated = False
            for pc, pb in pareto_points:
                if pc <= cost and pb >= benefit and (pc < cost or pb > benefit):
                    dominated = True
                    break

            if not dominated:
                # Remove points that this one dominates
                pareto_points = [
                    (pc, pb) for pc, pb in pareto_points
                    if not (cost <= pc and benefit >= pb and (cost < pc or benefit > pb))
                ]
                pareto_points.append((cost, benefit))

        return sorted(pareto_points)

    def compute_urgency_score(
        self,
        current_state: str,
        recent_trajectory: List[str],
        mfpt_to_directing: float = None
    ) -> UrgencyScore:
        """
        Compute urgency score for intervention.

        Args:
            current_state: Current behavioral state
            recent_trajectory: Recent state sequence
            mfpt_to_directing: Mean first passage time to Directing

        Returns:
            UrgencyScore with component breakdown
        """
        if mfpt_to_directing is None:
            # Estimate from state
            default_mfpts = {
                'Seeking': 5.0,
                'Conferring': 3.0,
                'Directing': 0.0,
                'Revising': 8.0
            }
            mfpt_to_directing = default_mfpts.get(current_state, 5.0)

        return UrgencyScore.compute(
            current_state=current_state,
            recent_trajectory=recent_trajectory,
            mfpt_to_directing=mfpt_to_directing
        )

    def _generate_rationale(
        self,
        state: str,
        protocol: InterventionProtocol,
        benefit: float,
        urgency: UrgencyScore
    ) -> str:
        """Generate human-readable rationale for recommendation."""
        parts = []

        # State match
        if state in protocol.target_states:
            parts.append(f"{protocol.display_name} directly targets the {state} state")

        # Transition disruption
        for from_s, to_s in protocol.target_transitions:
            if from_s == state:
                parts.append(f"disrupts {from_s}→{to_s} transition")
                break

        # Urgency
        if urgency.overall > 0.7:
            parts.append("high urgency indicates immediate need")
        elif urgency.overall > 0.4:
            parts.append("moderate urgency")

        # Benefit
        if benefit > 0.3:
            parts.append(f"expected benefit is substantial ({benefit:.0%} harm reduction)")
        elif benefit > 0.1:
            parts.append(f"expected benefit is moderate ({benefit:.0%})")

        if not parts:
            parts.append("General intervention for current risk profile")

        return "; ".join(parts)

    def _apply_budget_constraint(
        self,
        recommendations: List[InterventionRecommendation],
        max_budget: float
    ) -> List[InterventionRecommendation]:
        """Filter recommendations to fit within budget."""
        # Sort by cost-effectiveness
        sorted_recs = sorted(
            recommendations,
            key=lambda r: r.cost_effectiveness,
            reverse=True
        )

        # Greedy selection within budget
        selected = []
        remaining = max_budget

        for rec in sorted_recs:
            if rec.cost <= remaining:
                selected.append(rec)
                remaining -= rec.cost

        return selected

    def _select_top_recommendations(
        self,
        recommendations: List[InterventionRecommendation],
        max_count: int
    ) -> List[InterventionRecommendation]:
        """Select top recommendations avoiding redundancy."""
        if not recommendations:
            return []

        # Sort by expected benefit
        sorted_recs = sorted(
            recommendations,
            key=lambda r: r.expected_benefit,
            reverse=True
        )

        selected = []
        used_times = set()
        used_protocols = set()

        for rec in sorted_recs:
            if len(selected) >= max_count:
                break

            # Avoid recommending same protocol at adjacent times
            time_ok = all(
                abs(rec.time_index - t) > 2
                for t in used_times
            )

            # Avoid duplicate protocols unless at very different times
            protocol_ok = (
                rec.protocol.name not in used_protocols or
                all(abs(rec.time_index - t) > 5 for t in used_times if rec.protocol.name in used_protocols)
            )

            if time_ok or protocol_ok:
                selected.append(rec)
                used_times.add(rec.time_index)
                used_protocols.add(rec.protocol.name)

        return selected

    def _compute_pareto_frontier(
        self,
        recommendations: List[InterventionRecommendation]
    ) -> List[Tuple[float, float]]:
        """Extract Pareto-optimal points from recommendations."""
        if not recommendations:
            return [(0, 0)]

        points = [(r.cost, r.expected_benefit) for r in recommendations]
        points.append((0, 0))  # Always include origin

        pareto = []
        for cost, benefit in sorted(set(points)):
            dominated = False
            for pc, pb in pareto:
                if pc <= cost and pb >= benefit and (pc < cost or pb > benefit):
                    dominated = True
                    break

            if not dominated:
                pareto = [
                    (pc, pb) for pc, pb in pareto
                    if not (cost <= pc and benefit >= pb and (cost < pc or benefit > pb))
                ]
                pareto.append((cost, benefit))

        return sorted(pareto)


def expected_harm_reduction(
    scm: BehavioralSCM,
    protocol: InterventionProtocol,
    time_point: int,
    trajectory: List[str],
    n_simulations: int = 500
) -> float:
    """
    Compute expected harm reduction from an intervention.

    E[Harm_baseline] - E[Harm_intervened]

    Args:
        scm: Behavioral SCM
        protocol: Intervention protocol
        time_point: When to intervene
        trajectory: Observed trajectory
        n_simulations: Simulations for estimation

    Returns:
        Expected reduction in harm probability
    """
    optimizer = InterventionOptimizer(scm)

    if time_point >= len(trajectory):
        return 0.0

    return optimizer.compute_intervention_value(
        protocol=protocol,
        current_state=trajectory[time_point],
        time_remaining=len(trajectory) - time_point - 1,
        n_simulations=n_simulations
    )


def intervention_urgency_score(
    current_state: str,
    mfpt_to_directing: float,
    escalation_score: float,
    recent_trajectory: List[str]
) -> float:
    """
    Compute intervention urgency combining multiple factors.

    Args:
        current_state: Current behavioral state
        mfpt_to_directing: Mean first passage time to Directing
        escalation_score: Recent escalation pattern (0-1)
        recent_trajectory: Recent state sequence

    Returns:
        Urgency score (0-1)
    """
    urgency = UrgencyScore.compute(
        current_state=current_state,
        recent_trajectory=recent_trajectory,
        mfpt_to_directing=mfpt_to_directing
    )

    # Incorporate external escalation score
    combined = 0.7 * urgency.overall + 0.3 * escalation_score

    return min(1.0, combined)


def quick_optimize(
    transition_matrix: np.ndarray,
    state_names: List[str],
    trajectory: List[str],
    max_budget: float = None,
    max_interventions: int = 3
) -> OptimizationResult:
    """
    Quick optimization helper function.

    Args:
        transition_matrix: State transition matrix
        state_names: Names of states
        trajectory: Observed trajectory
        max_budget: Budget constraint
        max_interventions: Max number of interventions

    Returns:
        OptimizationResult with recommendations
    """
    from .causal_model import BehavioralSCM

    scm = BehavioralSCM(transition_matrix, state_names)
    optimizer = InterventionOptimizer(scm)

    return optimizer.find_optimal_timing(
        trajectory=trajectory,
        budget_constraint=max_budget,
        max_interventions=max_interventions
    )
