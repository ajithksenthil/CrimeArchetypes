"""
Structural Causal Model for Behavioral Trajectories

Represents criminal behavioral trajectories as causal graphs where:
- States are nodes with temporal ordering
- Transitions are directed edges with probabilities
- Interventions are external nodes that modify transition probabilities
- Outcomes are terminal nodes (e.g., "reached_Directing", "harm_occurred")

This enables rigorous causal inference using do-calculus to estimate
intervention effects and answer counterfactual queries.

References:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Pearl, J. (2018). The Book of Why
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any, Set
import numpy as np
import networkx as nx
from enum import Enum
from copy import deepcopy


class NodeType(Enum):
    """Types of nodes in the causal graph."""
    STATE = "state"              # Behavioral state (Seeking, Directing, etc.)
    INTERVENTION = "intervention"  # External intervention node
    OUTCOME = "outcome"          # Terminal outcome node
    CONFOUNDER = "confounder"    # Latent confounder


@dataclass
class CausalNode:
    """
    Node in the structural causal model.

    Attributes:
        name: Unique identifier
        node_type: Type of node (state, intervention, outcome)
        time_index: Temporal position (for state nodes)
        parents: List of parent node names
        structural_equation: Function computing node value from parents
        conditional_probs: P(node | parents) for discrete nodes
    """
    name: str
    node_type: NodeType
    time_index: Optional[int] = None
    parents: List[str] = field(default_factory=list)
    conditional_probs: Dict[Tuple, Dict[str, float]] = field(default_factory=dict)
    description: str = ""

    def get_probability(self, value: str, parent_values: Dict[str, str]) -> float:
        """Get P(node=value | parents=parent_values)."""
        if not self.conditional_probs:
            return 1.0 if not self.parents else 0.0

        # Create key from parent values in order
        parent_key = tuple(parent_values.get(p, None) for p in sorted(self.parents))

        if parent_key in self.conditional_probs:
            return self.conditional_probs[parent_key].get(value, 0.0)
        return 0.0


@dataclass
class InterventionNode(CausalNode):
    """
    Specialized node for interventions.

    Interventions modify transition probabilities when active.
    """
    target_transitions: List[Tuple[str, str]] = field(default_factory=list)
    effect_on_transitions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    is_active: bool = False
    activation_time: Optional[int] = None
    duration: int = 1  # How many time steps the intervention affects

    def get_modified_transition_prob(
        self,
        from_state: str,
        to_state: str,
        base_prob: float
    ) -> float:
        """Get modified transition probability when intervention is active."""
        if not self.is_active:
            return base_prob

        key = (from_state, to_state)
        if key in self.effect_on_transitions:
            delta = self.effect_on_transitions[key]
            # Apply multiplicative or additive effect
            if isinstance(delta, float) and -1 <= delta <= 1:
                # Additive delta (e.g., -0.3 means reduce by 30%)
                new_prob = max(0, min(1, base_prob + delta))
            else:
                new_prob = base_prob
            return new_prob
        return base_prob


@dataclass
class OutcomeNode(CausalNode):
    """
    Terminal outcome node (e.g., "harm_occurred").

    Outcomes are computed from the trajectory of states.
    """
    outcome_condition: Optional[callable] = None  # Function(trajectory) -> bool
    harm_weight: float = 1.0  # For aggregating harm across outcomes

    def evaluate(self, trajectory: List[str]) -> bool:
        """Evaluate whether outcome occurred given trajectory."""
        if self.outcome_condition:
            return self.outcome_condition(trajectory)
        # Default: check if "Directing" appeared in trajectory
        return "Directing" in trajectory


class BehavioralSCM:
    """
    Structural Causal Model for criminal behavioral trajectories.

    The SCM represents:
    - State nodes S_t for each time step
    - Transition edges P(S_{t+1} | S_t) from Markov chain
    - Intervention nodes that can modify transitions
    - Outcome nodes that depend on trajectories

    Key capabilities:
    - Build causal graph from transition matrix
    - Add intervention nodes with specified effects
    - Compute P(outcome | do(intervention)) using do-calculus
    - Generate counterfactual trajectories
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        state_names: List[str],
        n_time_steps: int = 20
    ):
        """
        Initialize SCM from Markov transition matrix.

        Args:
            transition_matrix: Row-stochastic matrix P[i,j] = P(j|i)
            state_names: Names of states (e.g., ['Seeking', 'Directing', ...])
            n_time_steps: Number of time steps to model
        """
        self.transition_matrix = transition_matrix.copy()
        self.state_names = state_names
        self.n_states = len(state_names)
        self.n_time_steps = n_time_steps

        self.state_to_idx = {s: i for i, s in enumerate(state_names)}
        self.idx_to_state = {i: s for i, s in enumerate(state_names)}

        # Graph representation
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.interventions: Dict[str, InterventionNode] = {}
        self.outcomes: Dict[str, OutcomeNode] = {}

        # Build the causal graph
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct causal graph from transition matrix."""
        # Add state nodes for each time step
        for t in range(self.n_time_steps):
            for state in self.state_names:
                node_name = f"{state}_t{t}"
                node = CausalNode(
                    name=node_name,
                    node_type=NodeType.STATE,
                    time_index=t,
                    description=f"State {state} at time {t}"
                )

                # Set parents (previous time step states)
                if t > 0:
                    node.parents = [f"{s}_t{t-1}" for s in self.state_names]

                    # Set conditional probabilities from transition matrix
                    for prev_state in self.state_names:
                        prev_idx = self.state_to_idx[prev_state]
                        curr_idx = self.state_to_idx[state]
                        prob = self.transition_matrix[prev_idx, curr_idx]

                        # Key is tuple of parent values
                        parent_key = tuple(
                            1 if s == prev_state else 0
                            for s in self.state_names
                        )
                        if parent_key not in node.conditional_probs:
                            node.conditional_probs[parent_key] = {}
                        node.conditional_probs[parent_key][state] = prob

                self.nodes[node_name] = node
                self.graph.add_node(node_name, **{'type': 'state', 'time': t})

        # Add edges between consecutive time steps
        for t in range(self.n_time_steps - 1):
            for from_state in self.state_names:
                from_node = f"{from_state}_t{t}"
                for to_state in self.state_names:
                    to_node = f"{to_state}_t{t+1}"
                    prob = self.transition_matrix[
                        self.state_to_idx[from_state],
                        self.state_to_idx[to_state]
                    ]
                    if prob > 0:
                        self.graph.add_edge(
                            from_node, to_node,
                            weight=prob,
                            transition=(from_state, to_state)
                        )

        # Add default outcome nodes
        self._add_default_outcomes()

    def _add_default_outcomes(self) -> None:
        """Add standard outcome nodes."""
        # Outcome: Ever reached Directing
        self.add_outcome(
            name="reached_directing",
            condition=lambda traj: "Directing" in traj,
            description="Whether Directing state was ever reached",
            harm_weight=1.0
        )

        # Outcome: Sustained Directing (3+ consecutive)
        def sustained_directing(traj):
            count = 0
            max_count = 0
            for s in traj:
                if s == "Directing":
                    count += 1
                    max_count = max(max_count, count)
                else:
                    count = 0
            return max_count >= 3

        self.add_outcome(
            name="sustained_directing",
            condition=sustained_directing,
            description="3+ consecutive Directing states",
            harm_weight=2.0
        )

        # Outcome: Escalation pattern
        def escalation_pattern(traj):
            if len(traj) < 6:
                return False
            n = len(traj)
            early = traj[:n//3]
            late = traj[2*n//3:]
            early_directing = sum(1 for s in early if s == "Directing") / len(early)
            late_directing = sum(1 for s in late if s == "Directing") / len(late)
            return late_directing - early_directing > 0.2

        self.add_outcome(
            name="escalation",
            condition=escalation_pattern,
            description="Significant escalation from early to late phase",
            harm_weight=1.5
        )

    def add_intervention(
        self,
        name: str,
        target_transitions: List[Tuple[str, str]],
        effects: Dict[Tuple[str, str], float],
        description: str = ""
    ) -> InterventionNode:
        """
        Add an intervention node to the SCM.

        Args:
            name: Intervention identifier
            target_transitions: List of (from_state, to_state) affected
            effects: Dict mapping transitions to probability deltas
            description: Human-readable description

        Returns:
            The created InterventionNode
        """
        node = InterventionNode(
            name=name,
            node_type=NodeType.INTERVENTION,
            target_transitions=target_transitions,
            effect_on_transitions=effects,
            description=description
        )

        self.interventions[name] = node
        self.graph.add_node(name, **{'type': 'intervention'})

        # Add edges from intervention to affected state transitions
        for from_state, to_state in target_transitions:
            for t in range(self.n_time_steps - 1):
                to_node = f"{to_state}_t{t+1}"
                self.graph.add_edge(name, to_node, intervention=True)

        return node

    def add_outcome(
        self,
        name: str,
        condition: callable,
        description: str = "",
        harm_weight: float = 1.0
    ) -> OutcomeNode:
        """
        Add an outcome node to the SCM.

        Args:
            name: Outcome identifier
            condition: Function(trajectory) -> bool
            description: Human-readable description
            harm_weight: Weight for harm aggregation

        Returns:
            The created OutcomeNode
        """
        node = OutcomeNode(
            name=name,
            node_type=NodeType.OUTCOME,
            outcome_condition=condition,
            description=description,
            harm_weight=harm_weight
        )

        self.outcomes[name] = node
        self.graph.add_node(name, **{'type': 'outcome'})

        # Outcomes depend on all state nodes
        for t in range(self.n_time_steps):
            for state in self.state_names:
                state_node = f"{state}_t{t}"
                self.graph.add_edge(state_node, name)

        return node

    def activate_intervention(
        self,
        intervention_name: str,
        activation_time: int,
        duration: int = None
    ) -> None:
        """Activate an intervention at a specific time."""
        if intervention_name not in self.interventions:
            raise ValueError(f"Unknown intervention: {intervention_name}")

        intervention = self.interventions[intervention_name]
        intervention.is_active = True
        intervention.activation_time = activation_time
        if duration:
            intervention.duration = duration

    def deactivate_intervention(self, intervention_name: str) -> None:
        """Deactivate an intervention."""
        if intervention_name in self.interventions:
            self.interventions[intervention_name].is_active = False

    def deactivate_all_interventions(self) -> None:
        """Deactivate all interventions."""
        for intervention in self.interventions.values():
            intervention.is_active = False

    def get_transition_probability(
        self,
        from_state: str,
        to_state: str,
        time_step: int
    ) -> float:
        """
        Get transition probability, accounting for active interventions.

        Args:
            from_state: Source state
            to_state: Target state
            time_step: Current time step

        Returns:
            Modified transition probability
        """
        base_prob = self.transition_matrix[
            self.state_to_idx[from_state],
            self.state_to_idx[to_state]
        ]

        # Apply active interventions
        for intervention in self.interventions.values():
            if intervention.is_active:
                if intervention.activation_time is not None:
                    # Check if intervention is active at this time
                    start = intervention.activation_time
                    end = start + intervention.duration
                    if start <= time_step < end:
                        base_prob = intervention.get_modified_transition_prob(
                            from_state, to_state, base_prob
                        )

        return base_prob

    def get_modified_transition_matrix(self, time_step: int) -> np.ndarray:
        """Get transition matrix modified by active interventions at time t."""
        matrix = np.zeros_like(self.transition_matrix)

        for i, from_state in enumerate(self.state_names):
            row_sum = 0
            for j, to_state in enumerate(self.state_names):
                prob = self.get_transition_probability(from_state, to_state, time_step)
                matrix[i, j] = prob
                row_sum += prob

            # Renormalize row to sum to 1
            if row_sum > 0:
                matrix[i] /= row_sum

        return matrix

    def sample_trajectory(
        self,
        initial_state: str,
        n_steps: int = None,
        rng: np.random.Generator = None
    ) -> List[str]:
        """
        Sample a trajectory from the (possibly intervened) SCM.

        Args:
            initial_state: Starting state
            n_steps: Number of steps (default: n_time_steps)
            rng: Random number generator

        Returns:
            List of state names
        """
        if n_steps is None:
            n_steps = self.n_time_steps
        if rng is None:
            rng = np.random.default_rng()

        trajectory = [initial_state]
        current_state = initial_state

        for t in range(n_steps - 1):
            # Get transition probabilities for this time step
            trans_matrix = self.get_modified_transition_matrix(t)
            probs = trans_matrix[self.state_to_idx[current_state]]

            # Sample next state
            next_idx = rng.choice(self.n_states, p=probs)
            next_state = self.idx_to_state[next_idx]

            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    def compute_outcome_probability(
        self,
        outcome_name: str,
        initial_state: str,
        n_simulations: int = 1000,
        rng: np.random.Generator = None
    ) -> Tuple[float, float]:
        """
        Estimate P(outcome) via Monte Carlo simulation.

        Args:
            outcome_name: Name of outcome to evaluate
            initial_state: Starting state for trajectories
            n_simulations: Number of MC samples
            rng: Random number generator

        Returns:
            Tuple of (probability estimate, standard error)
        """
        if outcome_name not in self.outcomes:
            raise ValueError(f"Unknown outcome: {outcome_name}")

        if rng is None:
            rng = np.random.default_rng()

        outcome = self.outcomes[outcome_name]
        successes = 0

        for _ in range(n_simulations):
            trajectory = self.sample_trajectory(initial_state, rng=rng)
            if outcome.evaluate(trajectory):
                successes += 1

        p_hat = successes / n_simulations
        se = np.sqrt(p_hat * (1 - p_hat) / n_simulations)

        return p_hat, se

    def compute_expected_harm(
        self,
        initial_state: str,
        n_simulations: int = 1000,
        rng: np.random.Generator = None
    ) -> Tuple[float, float]:
        """
        Compute expected harm (weighted sum of outcome probabilities).

        Returns:
            Tuple of (expected harm, standard error)
        """
        if rng is None:
            rng = np.random.default_rng()

        harms = []

        for _ in range(n_simulations):
            trajectory = self.sample_trajectory(initial_state, rng=rng)
            harm = sum(
                outcome.harm_weight * (1 if outcome.evaluate(trajectory) else 0)
                for outcome in self.outcomes.values()
            )
            harms.append(harm)

        return np.mean(harms), np.std(harms) / np.sqrt(n_simulations)

    def do(
        self,
        intervention_name: str,
        activation_time: int = 0,
        duration: int = None
    ) -> 'BehavioralSCM':
        """
        Apply do-operator: do(intervention).

        Returns a new SCM with the intervention activated.
        This implements Pearl's do-calculus by:
        1. Making a copy of the SCM
        2. Activating the specified intervention

        Args:
            intervention_name: Name of intervention to apply
            activation_time: When to activate (default: 0)
            duration: How long intervention is active

        Returns:
            New BehavioralSCM with intervention active
        """
        if intervention_name not in self.interventions:
            raise ValueError(f"Unknown intervention: {intervention_name}")

        # Create a deep copy
        new_scm = BehavioralSCM(
            self.transition_matrix.copy(),
            self.state_names.copy(),
            self.n_time_steps
        )

        # Copy interventions
        for name, intervention in self.interventions.items():
            new_scm.add_intervention(
                name=intervention.name,
                target_transitions=intervention.target_transitions.copy(),
                effects=intervention.effect_on_transitions.copy(),
                description=intervention.description
            )

        # Activate the specified intervention
        new_scm.activate_intervention(
            intervention_name,
            activation_time,
            duration or self.n_time_steps
        )

        return new_scm

    def compute_causal_effect(
        self,
        intervention_name: str,
        outcome_name: str,
        initial_state: str,
        activation_time: int = 0,
        n_simulations: int = 1000,
        rng: np.random.Generator = None
    ) -> Dict[str, float]:
        """
        Compute causal effect: P(outcome | do(intervention)) - P(outcome).

        This is the Average Treatment Effect (ATE) for the intervention.

        Args:
            intervention_name: Intervention to evaluate
            outcome_name: Outcome to measure
            initial_state: Starting state
            activation_time: When intervention activates
            n_simulations: MC samples per condition
            rng: Random number generator

        Returns:
            Dict with 'ate', 'p_treated', 'p_control', 'se', 'ci_lower', 'ci_upper'
        """
        if rng is None:
            rng = np.random.default_rng()

        # P(outcome | no intervention)
        self.deactivate_all_interventions()
        p_control, se_control = self.compute_outcome_probability(
            outcome_name, initial_state, n_simulations, rng
        )

        # P(outcome | do(intervention))
        scm_intervened = self.do(intervention_name, activation_time)
        p_treated, se_treated = scm_intervened.compute_outcome_probability(
            outcome_name, initial_state, n_simulations, rng
        )

        # ATE and confidence interval
        ate = p_treated - p_control
        se_ate = np.sqrt(se_control**2 + se_treated**2)
        ci_lower = ate - 1.96 * se_ate
        ci_upper = ate + 1.96 * se_ate

        return {
            'ate': ate,
            'p_treated': p_treated,
            'p_control': p_control,
            'se': se_ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': ci_lower > 0 or ci_upper < 0
        }

    def get_causal_graph(self) -> nx.DiGraph:
        """Return the causal graph for visualization."""
        return self.graph.copy()

    def get_intervention_summary(self) -> Dict[str, Dict]:
        """Get summary of all registered interventions."""
        return {
            name: {
                'description': i.description,
                'target_transitions': i.target_transitions,
                'effects': i.effect_on_transitions,
                'is_active': i.is_active
            }
            for name, i in self.interventions.items()
        }

    def to_dict(self) -> Dict:
        """Serialize SCM to dictionary."""
        return {
            'transition_matrix': self.transition_matrix.tolist(),
            'state_names': self.state_names,
            'n_time_steps': self.n_time_steps,
            'interventions': {
                name: {
                    'target_transitions': i.target_transitions,
                    'effects': {str(k): v for k, v in i.effect_on_transitions.items()},
                    'description': i.description
                }
                for name, i in self.interventions.items()
            },
            'outcomes': {
                name: {
                    'description': o.description,
                    'harm_weight': o.harm_weight
                }
                for name, o in self.outcomes.items()
            }
        }


def create_scm_from_individual(
    transition_matrix: np.ndarray,
    state_names: List[str] = None,
    n_time_steps: int = 20
) -> BehavioralSCM:
    """
    Factory function to create SCM from individual's transition matrix.

    Args:
        transition_matrix: Individual's transition matrix
        state_names: State names (default: 4-Animal states)
        n_time_steps: Number of time steps to model

    Returns:
        BehavioralSCM instance
    """
    if state_names is None:
        state_names = ['Seeking', 'Directing', 'Conferring', 'Revising']

    return BehavioralSCM(transition_matrix, state_names, n_time_steps)


def create_population_scm(
    individual_matrices: List[np.ndarray],
    state_names: List[str] = None,
    n_time_steps: int = 20,
    aggregation: str = 'mean'
) -> BehavioralSCM:
    """
    Create SCM from population of individuals.

    Args:
        individual_matrices: List of transition matrices
        state_names: State names
        n_time_steps: Number of time steps
        aggregation: How to combine ('mean', 'median')

    Returns:
        BehavioralSCM with aggregated transition matrix
    """
    if state_names is None:
        state_names = ['Seeking', 'Directing', 'Conferring', 'Revising']

    matrices = np.stack(individual_matrices)

    if aggregation == 'mean':
        agg_matrix = np.mean(matrices, axis=0)
    elif aggregation == 'median':
        agg_matrix = np.median(matrices, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Renormalize rows
    row_sums = agg_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    agg_matrix = agg_matrix / row_sums

    return BehavioralSCM(agg_matrix, state_names, n_time_steps)
