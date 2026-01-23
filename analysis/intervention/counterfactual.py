"""
Counterfactual Simulation Engine

Implements Pearl's three-step counterfactual reasoning:
1. Abduction: Infer latent variables from observed trajectory
2. Action: Apply intervention (do-operator)
3. Prediction: Simulate forward under intervention

Key capabilities:
- "What if we had intervened at time t?"
- Probability of Necessity (PN): Would outcome not have occurred with intervention?
- Probability of Sufficiency (PS): Would intervention cause desired outcome?
- Trajectory branching analysis: All possible futures from a decision point

References:
- Pearl, J. (2009). Causality, Chapter 7
- Pearl, J., Glymour, M., & Jewell, N.P. (2016). Causal Inference in Statistics
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

from .causal_model import BehavioralSCM
from .protocols import InterventionProtocol, INTERVENTION_PROTOCOLS


@dataclass
class CounterfactualResult:
    """
    Result of a counterfactual query.

    Attributes:
        observed_trajectory: What actually happened
        counterfactual_trajectory: What would have happened with intervention
        intervention_applied: The intervention that was simulated
        intervention_time: When the intervention was applied

        observed_outcome: Did harm occur in observed trajectory?
        counterfactual_outcome_distribution: P(outcomes) under intervention

        harm_reduction_estimate: Expected reduction in harm
        confidence_interval: CI for harm reduction
        trajectory_divergence_point: Where trajectories diverge
    """
    observed_trajectory: List[str]
    counterfactual_trajectory: List[str]
    intervention_name: str
    intervention_time: int

    observed_outcome: bool
    counterfactual_outcome_distribution: Dict[str, float]

    harm_reduction_estimate: float
    confidence_interval: Tuple[float, float]
    trajectory_divergence_point: int

    # Additional metrics
    probability_of_necessity: float = 0.0
    probability_of_sufficiency: float = 0.0
    n_simulations: int = 0

    def to_dict(self) -> Dict:
        return {
            'observed_trajectory': self.observed_trajectory,
            'counterfactual_trajectory': self.counterfactual_trajectory,
            'intervention_name': self.intervention_name,
            'intervention_time': self.intervention_time,
            'observed_outcome': self.observed_outcome,
            'counterfactual_outcome_distribution': self.counterfactual_outcome_distribution,
            'harm_reduction_estimate': self.harm_reduction_estimate,
            'confidence_interval': self.confidence_interval,
            'trajectory_divergence_point': self.trajectory_divergence_point,
            'probability_of_necessity': self.probability_of_necessity,
            'probability_of_sufficiency': self.probability_of_sufficiency,
            'n_simulations': self.n_simulations
        }

    @property
    def would_intervention_have_helped(self) -> bool:
        """Did the counterfactual trajectory have better outcome?"""
        cf_harm_prob = self.counterfactual_outcome_distribution.get('harm', 0)
        return cf_harm_prob < 0.5 and self.observed_outcome

    @property
    def effect_size(self) -> str:
        """Categorize effect size of intervention."""
        if abs(self.harm_reduction_estimate) < 0.1:
            return 'negligible'
        elif abs(self.harm_reduction_estimate) < 0.3:
            return 'small'
        elif abs(self.harm_reduction_estimate) < 0.5:
            return 'medium'
        else:
            return 'large'


@dataclass
class BranchingResult:
    """
    Result of trajectory branching analysis.

    Shows all possible futures from a branch point under
    different intervention scenarios.
    """
    branch_point: int
    state_at_branch: str
    branches: Dict[str, CounterfactualResult]  # intervention_name -> result
    no_intervention_outcomes: Dict[str, float]  # outcome -> probability
    best_intervention: str
    best_harm_reduction: float

    def to_dict(self) -> Dict:
        return {
            'branch_point': self.branch_point,
            'state_at_branch': self.state_at_branch,
            'branches': {k: v.to_dict() for k, v in self.branches.items()},
            'no_intervention_outcomes': self.no_intervention_outcomes,
            'best_intervention': self.best_intervention,
            'best_harm_reduction': self.best_harm_reduction
        }


class CounterfactualEngine:
    """
    Engine for counterfactual simulation and analysis.

    Implements the three-step counterfactual reasoning:
    1. Abduction - infer latent state from observed trajectory
    2. Action - apply intervention via do-operator
    3. Prediction - simulate forward under intervention
    """

    def __init__(
        self,
        scm: BehavioralSCM,
        n_simulations: int = 1000,
        random_seed: int = None
    ):
        """
        Initialize counterfactual engine.

        Args:
            scm: Structural Causal Model
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.scm = scm
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def simulate_trajectory(
        self,
        initial_state: str,
        n_steps: int,
        intervention_name: str = None,
        intervention_time: int = 0
    ) -> List[str]:
        """
        Simulate a single trajectory, optionally with intervention.

        Args:
            initial_state: Starting state
            n_steps: Number of steps
            intervention_name: Optional intervention to apply
            intervention_time: When to apply intervention

        Returns:
            Simulated trajectory
        """
        # Apply intervention if specified
        if intervention_name:
            scm = self.scm.do(intervention_name, intervention_time)
        else:
            scm = self.scm
            scm.deactivate_all_interventions()

        return scm.sample_trajectory(initial_state, n_steps, self.rng)

    def _abduction(
        self,
        observed_trajectory: List[str]
    ) -> Dict[str, any]:
        """
        Step 1: Abduction - infer latent variables from observed trajectory.

        In our Markov model, the "latent" information is the sequence of
        random choices that led to the observed trajectory. We capture this
        as the empirical transition frequencies.

        Returns:
            Dict with inferred latent state
        """
        # Compute empirical transition counts from observed trajectory
        n = self.scm.n_states
        transition_counts = np.zeros((n, n))

        for i in range(len(observed_trajectory) - 1):
            from_idx = self.scm.state_to_idx[observed_trajectory[i]]
            to_idx = self.scm.state_to_idx[observed_trajectory[i + 1]]
            transition_counts[from_idx, to_idx] += 1

        # This individual's "type" is characterized by their transition pattern
        # relative to the population
        individual_deviation = transition_counts - (
            self.scm.transition_matrix * len(observed_trajectory)
        )

        return {
            'transition_counts': transition_counts,
            'individual_deviation': individual_deviation,
            'initial_state': observed_trajectory[0],
            'trajectory_length': len(observed_trajectory)
        }

    def _action(
        self,
        latent_state: Dict,
        intervention_name: str,
        intervention_time: int
    ) -> BehavioralSCM:
        """
        Step 2: Action - apply do-operator with intervention.

        Returns modified SCM with intervention active.
        """
        return self.scm.do(intervention_name, intervention_time)

    def _prediction(
        self,
        modified_scm: BehavioralSCM,
        latent_state: Dict,
        n_simulations: int = None
    ) -> List[List[str]]:
        """
        Step 3: Prediction - simulate forward under intervention.

        Args:
            modified_scm: SCM with intervention applied
            latent_state: Inferred latent state from abduction
            n_simulations: Number of simulations

        Returns:
            List of counterfactual trajectories
        """
        if n_simulations is None:
            n_simulations = self.n_simulations

        initial_state = latent_state['initial_state']
        n_steps = latent_state['trajectory_length']

        trajectories = []
        for _ in range(n_simulations):
            traj = modified_scm.sample_trajectory(initial_state, n_steps, self.rng)
            trajectories.append(traj)

        return trajectories

    def counterfactual_query(
        self,
        observed_trajectory: List[str],
        intervention_name: str,
        intervention_time: int = 0,
        outcome_fn: callable = None
    ) -> CounterfactualResult:
        """
        Answer: "What would have happened if we intervened at time t?"

        Implements the three-step counterfactual:
        1. Abduction: Infer individual's latent state
        2. Action: Apply intervention
        3. Prediction: Simulate counterfactual trajectories

        Args:
            observed_trajectory: What actually happened
            intervention_name: Intervention to test
            intervention_time: When to apply intervention
            outcome_fn: Function(trajectory) -> bool for outcome

        Returns:
            CounterfactualResult with analysis
        """
        if intervention_name not in self.scm.interventions:
            # Try to load from protocol library
            if intervention_name in INTERVENTION_PROTOCOLS:
                protocol = INTERVENTION_PROTOCOLS[intervention_name]
                self.scm.add_intervention(
                    name=intervention_name,
                    target_transitions=protocol.target_transitions,
                    effects=protocol.effect_on_transitions,
                    description=protocol.description
                )
            else:
                raise ValueError(f"Unknown intervention: {intervention_name}")

        if outcome_fn is None:
            # Default: did Directing state occur?
            outcome_fn = lambda traj: 'Directing' in traj

        # Step 1: Abduction
        latent_state = self._abduction(observed_trajectory)

        # Step 2: Action
        modified_scm = self._action(latent_state, intervention_name, intervention_time)

        # Step 3: Prediction
        cf_trajectories = self._prediction(modified_scm, latent_state)

        # Evaluate outcomes
        observed_outcome = outcome_fn(observed_trajectory)

        cf_outcomes = [outcome_fn(traj) for traj in cf_trajectories]
        cf_harm_rate = sum(cf_outcomes) / len(cf_outcomes)

        # Compute harm reduction
        harm_reduction = (1.0 if observed_outcome else 0.0) - cf_harm_rate

        # Bootstrap confidence interval
        bootstrap_reductions = []
        for _ in range(100):
            sample_idx = self.rng.choice(len(cf_outcomes), size=len(cf_outcomes), replace=True)
            sample_harm_rate = sum(cf_outcomes[i] for i in sample_idx) / len(sample_idx)
            bootstrap_reductions.append((1.0 if observed_outcome else 0.0) - sample_harm_rate)

        ci_lower = np.percentile(bootstrap_reductions, 2.5)
        ci_upper = np.percentile(bootstrap_reductions, 97.5)

        # Find divergence point
        # Use most common counterfactual trajectory
        from collections import Counter
        cf_traj_strs = [tuple(t) for t in cf_trajectories]
        most_common_cf = list(Counter(cf_traj_strs).most_common(1)[0][0])

        divergence_point = 0
        for i in range(min(len(observed_trajectory), len(most_common_cf))):
            if observed_trajectory[i] != most_common_cf[i]:
                divergence_point = i
                break
            divergence_point = i + 1

        # Compute PN and PS
        pn = self.compute_probability_of_necessity(
            observed_trajectory, intervention_name, outcome_fn, intervention_time
        )
        ps = self.compute_probability_of_sufficiency(
            observed_trajectory, intervention_name, outcome_fn, intervention_time
        )

        return CounterfactualResult(
            observed_trajectory=observed_trajectory,
            counterfactual_trajectory=most_common_cf,
            intervention_name=intervention_name,
            intervention_time=intervention_time,
            observed_outcome=observed_outcome,
            counterfactual_outcome_distribution={
                'harm': cf_harm_rate,
                'no_harm': 1 - cf_harm_rate
            },
            harm_reduction_estimate=harm_reduction,
            confidence_interval=(ci_lower, ci_upper),
            trajectory_divergence_point=divergence_point,
            probability_of_necessity=pn,
            probability_of_sufficiency=ps,
            n_simulations=self.n_simulations
        )

    def compute_probability_of_necessity(
        self,
        observed_trajectory: List[str],
        intervention_name: str,
        outcome_fn: callable = None,
        intervention_time: int = 0
    ) -> float:
        """
        Compute Probability of Necessity (PN).

        PN = P(outcome_cf = False | outcome_observed = True, intervention)

        "Given that harm occurred, what's the probability that the
        intervention would have prevented it?"

        Args:
            observed_trajectory: Observed trajectory
            intervention_name: Intervention to test
            outcome_fn: Outcome function
            intervention_time: When to intervene

        Returns:
            Probability of necessity
        """
        if outcome_fn is None:
            outcome_fn = lambda traj: 'Directing' in traj

        # PN only makes sense if outcome actually occurred
        if not outcome_fn(observed_trajectory):
            return 0.0  # Not applicable

        # Simulate counterfactuals
        latent = self._abduction(observed_trajectory)
        modified_scm = self._action(latent, intervention_name, intervention_time)
        cf_trajectories = self._prediction(modified_scm, latent)

        # Count how often harm was prevented
        prevented = sum(1 for traj in cf_trajectories if not outcome_fn(traj))
        pn = prevented / len(cf_trajectories)

        return pn

    def compute_probability_of_sufficiency(
        self,
        observed_trajectory: List[str],
        intervention_name: str,
        outcome_fn: callable = None,
        intervention_time: int = 0
    ) -> float:
        """
        Compute Probability of Sufficiency (PS).

        PS = P(outcome_cf = True | outcome_observed = False, no_intervention)

        "Given that harm didn't occur (without intervention), what's the
        probability it would have occurred without the protective factor?"

        This is less directly applicable to our context but included
        for completeness.

        Args:
            observed_trajectory: Observed trajectory (without intervention)
            intervention_name: The intervention that was present
            outcome_fn: Outcome function
            intervention_time: When intervention was active

        Returns:
            Probability of sufficiency
        """
        if outcome_fn is None:
            outcome_fn = lambda traj: 'Directing' in traj

        # PS assumes intervention was present and outcome didn't occur
        # We simulate what would happen WITHOUT the intervention

        if outcome_fn(observed_trajectory):
            return 1.0  # Outcome occurred anyway

        # Simulate without intervention
        latent = self._abduction(observed_trajectory)
        self.scm.deactivate_all_interventions()
        no_intervention_trajectories = self._prediction(self.scm, latent)

        # Count how often harm would have occurred
        would_harm = sum(1 for traj in no_intervention_trajectories if outcome_fn(traj))
        ps = would_harm / len(no_intervention_trajectories)

        return ps

    def trajectory_branching_analysis(
        self,
        observed_trajectory: List[str],
        branch_point: int,
        interventions: List[str] = None,
        outcome_fn: callable = None
    ) -> BranchingResult:
        """
        Analyze all possible intervention branches from a decision point.

        Args:
            observed_trajectory: Observed trajectory
            branch_point: Index where branching occurs
            interventions: List of interventions to test (default: all)
            outcome_fn: Outcome function

        Returns:
            BranchingResult with all branches analyzed
        """
        if interventions is None:
            interventions = list(INTERVENTION_PROTOCOLS.keys())

        if outcome_fn is None:
            outcome_fn = lambda traj: 'Directing' in traj

        # Ensure interventions are registered
        for intervention_name in interventions:
            if intervention_name not in self.scm.interventions:
                if intervention_name in INTERVENTION_PROTOCOLS:
                    protocol = INTERVENTION_PROTOCOLS[intervention_name]
                    self.scm.add_intervention(
                        name=intervention_name,
                        target_transitions=protocol.target_transitions,
                        effects=protocol.effect_on_transitions,
                        description=protocol.description
                    )

        state_at_branch = observed_trajectory[branch_point] if branch_point < len(observed_trajectory) else observed_trajectory[-1]

        # Run counterfactual for each intervention
        branches = {}
        for intervention_name in interventions:
            try:
                result = self.counterfactual_query(
                    observed_trajectory,
                    intervention_name,
                    intervention_time=branch_point,
                    outcome_fn=outcome_fn
                )
                branches[intervention_name] = result
            except Exception as e:
                # Skip interventions that fail
                continue

        # Compute no-intervention baseline
        self.scm.deactivate_all_interventions()
        latent = self._abduction(observed_trajectory)
        baseline_trajectories = self._prediction(self.scm, latent)
        baseline_harm_rate = sum(
            1 for traj in baseline_trajectories if outcome_fn(traj)
        ) / len(baseline_trajectories)

        no_intervention_outcomes = {
            'harm': baseline_harm_rate,
            'no_harm': 1 - baseline_harm_rate
        }

        # Find best intervention
        if branches:
            best_intervention = max(
                branches.keys(),
                key=lambda k: branches[k].harm_reduction_estimate
            )
            best_harm_reduction = branches[best_intervention].harm_reduction_estimate
        else:
            best_intervention = None
            best_harm_reduction = 0.0

        return BranchingResult(
            branch_point=branch_point,
            state_at_branch=state_at_branch,
            branches=branches,
            no_intervention_outcomes=no_intervention_outcomes,
            best_intervention=best_intervention,
            best_harm_reduction=best_harm_reduction
        )

    def find_optimal_intervention_point(
        self,
        observed_trajectory: List[str],
        intervention_name: str,
        outcome_fn: callable = None,
        search_range: Tuple[int, int] = None
    ) -> Tuple[int, CounterfactualResult]:
        """
        Find the optimal time to apply an intervention.

        Tests the intervention at different time points and returns
        the timing that maximizes harm reduction.

        Args:
            observed_trajectory: Observed trajectory
            intervention_name: Intervention to optimize
            outcome_fn: Outcome function
            search_range: (start, end) range to search

        Returns:
            Tuple of (optimal_time, counterfactual_result)
        """
        if search_range is None:
            search_range = (0, len(observed_trajectory) - 1)

        best_time = search_range[0]
        best_result = None
        best_reduction = -float('inf')

        for t in range(search_range[0], min(search_range[1], len(observed_trajectory))):
            result = self.counterfactual_query(
                observed_trajectory,
                intervention_name,
                intervention_time=t,
                outcome_fn=outcome_fn
            )

            if result.harm_reduction_estimate > best_reduction:
                best_reduction = result.harm_reduction_estimate
                best_time = t
                best_result = result

        return best_time, best_result

    def compare_interventions(
        self,
        observed_trajectory: List[str],
        interventions: List[str],
        intervention_time: int = 0,
        outcome_fn: callable = None
    ) -> Dict[str, CounterfactualResult]:
        """
        Compare multiple interventions at the same time point.

        Args:
            observed_trajectory: Observed trajectory
            interventions: List of intervention names to compare
            intervention_time: When to apply interventions
            outcome_fn: Outcome function

        Returns:
            Dict mapping intervention name to result
        """
        results = {}

        for intervention_name in interventions:
            try:
                result = self.counterfactual_query(
                    observed_trajectory,
                    intervention_name,
                    intervention_time,
                    outcome_fn
                )
                results[intervention_name] = result
            except Exception as e:
                continue

        return results

    def retrospective_analysis(
        self,
        observed_trajectory: List[str],
        intervention_name: str = None,
        outcome_fn: callable = None
    ) -> Dict:
        """
        Comprehensive retrospective analysis of a trajectory.

        Identifies:
        - All possible intervention points
        - Best intervention at each point
        - Missed opportunities
        - Overall harm reduction potential

        Args:
            observed_trajectory: What actually happened
            intervention_name: Specific intervention (or test all)
            outcome_fn: Outcome function

        Returns:
            Dict with comprehensive retrospective analysis
        """
        if outcome_fn is None:
            outcome_fn = lambda traj: 'Directing' in traj

        observed_harm = outcome_fn(observed_trajectory)

        # Test intervention at each non-Directing state
        intervention_points = []
        for i, state in enumerate(observed_trajectory):
            if state != 'Directing':
                intervention_points.append({
                    'index': i,
                    'state': state
                })

        # Analyze each intervention point
        point_analyses = []
        interventions_to_test = [intervention_name] if intervention_name else list(INTERVENTION_PROTOCOLS.keys())[:5]

        for point in intervention_points[:10]:  # Limit to first 10 points
            branching = self.trajectory_branching_analysis(
                observed_trajectory,
                point['index'],
                interventions_to_test,
                outcome_fn
            )

            point_analyses.append({
                'index': point['index'],
                'state': point['state'],
                'best_intervention': branching.best_intervention,
                'best_harm_reduction': branching.best_harm_reduction,
                'all_results': {k: v.harm_reduction_estimate for k, v in branching.branches.items()}
            })

        # Find best overall intervention point
        if point_analyses:
            best_point = max(point_analyses, key=lambda p: p['best_harm_reduction'])
        else:
            best_point = None

        # Calculate missed opportunity cost
        if best_point and observed_harm:
            missed_opportunity_cost = best_point['best_harm_reduction']
        else:
            missed_opportunity_cost = 0.0

        return {
            'observed_trajectory': observed_trajectory,
            'observed_harm': observed_harm,
            'n_intervention_points': len(intervention_points),
            'point_analyses': point_analyses,
            'best_overall_point': best_point,
            'missed_opportunity_cost': missed_opportunity_cost,
            'could_have_prevented': missed_opportunity_cost > 0.5 if observed_harm else False
        }


def quick_counterfactual(
    trajectory: List[str],
    transition_matrix: np.ndarray,
    intervention_name: str,
    intervention_time: int = 0
) -> CounterfactualResult:
    """
    Quick counterfactual analysis without full engine setup.

    Args:
        trajectory: Observed trajectory
        transition_matrix: Population transition matrix
        intervention_name: Intervention to test
        intervention_time: When to intervene

    Returns:
        CounterfactualResult
    """
    scm = BehavioralSCM(
        transition_matrix,
        ['Seeking', 'Directing', 'Conferring', 'Revising'],
        n_time_steps=len(trajectory)
    )

    # Add intervention from protocol library
    if intervention_name in INTERVENTION_PROTOCOLS:
        protocol = INTERVENTION_PROTOCOLS[intervention_name]
        scm.add_intervention(
            name=intervention_name,
            target_transitions=protocol.target_transitions,
            effects=protocol.effect_on_transitions,
            description=protocol.description
        )

    engine = CounterfactualEngine(scm, n_simulations=500)
    return engine.counterfactual_query(trajectory, intervention_name, intervention_time)
