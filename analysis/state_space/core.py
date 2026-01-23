"""
State Space Framework - Core Classes

Provides the foundation for defining, comparing, and analyzing different
state space representations of behavioral data.

A StateSpace defines:
- A set of discrete states
- Metadata about each state (dimensions, descriptions, colors)
- Methods for classifying events into states
- Transition dynamics between states

Usage:
    from state_space import StateSpace, StateSpaceRegistry

    # Define a custom state space
    class MyStateSpace(StateSpace):
        name = "my_states"
        states = ["A", "B", "C"]
        ...

    # Register it
    StateSpaceRegistry.register(MyStateSpace)

    # Use it
    space = StateSpaceRegistry.get("my_states")
    classifications = space.classify(events)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
from enum import Enum


@dataclass
class StateDefinition:
    """Definition of a single state within a state space."""
    name: str
    description: str
    color: str = "#95a5a6"
    dimensions: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'color': self.color,
            'dimensions': self.dimensions,
            'keywords': self.keywords,
            'examples': self.examples
        }


class StateSpaceType(Enum):
    """Types of state spaces based on their derivation method."""
    THEORY_DRIVEN = "theory_driven"      # Based on psychological theory
    DATA_DRIVEN = "data_driven"          # Emergent from clustering/ML
    HYBRID = "hybrid"                    # Combination of both
    DOMAIN_SPECIFIC = "domain_specific"  # Based on domain expertise


@dataclass
class StateSpaceMetadata:
    """Metadata about a state space."""
    name: str
    display_name: str
    description: str
    type: StateSpaceType
    n_states: int
    version: str = "1.0"
    source: str = ""  # Paper, method, etc.
    dimensions: List[str] = field(default_factory=list)  # Underlying dimensions

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'type': self.type.value,
            'n_states': self.n_states,
            'version': self.version,
            'source': self.source,
            'dimensions': self.dimensions
        }


class StateSpace(ABC):
    """
    Abstract base class for all state space definitions.

    Subclasses must implement:
    - metadata: StateSpaceMetadata
    - states: Dict[str, StateDefinition]
    - classify(): Method to classify events into states
    """

    @property
    @abstractmethod
    def metadata(self) -> StateSpaceMetadata:
        """Return metadata about this state space."""
        pass

    @property
    @abstractmethod
    def states(self) -> Dict[str, StateDefinition]:
        """Return dictionary of state definitions."""
        pass

    @property
    def state_names(self) -> List[str]:
        """Return ordered list of state names."""
        return list(self.states.keys())

    @property
    def n_states(self) -> int:
        """Return number of states."""
        return len(self.states)

    @abstractmethod
    def classify(self, events: List[Dict]) -> List[Dict]:
        """
        Classify a list of events into states.

        Args:
            events: List of event dictionaries with at least 'text' field

        Returns:
            List of dictionaries with 'state', 'confidence', 'reasoning' fields
        """
        pass

    def get_state(self, name: str) -> Optional[StateDefinition]:
        """Get a state definition by name."""
        return self.states.get(name)

    def get_colors(self) -> Dict[str, str]:
        """Return state -> color mapping."""
        return {name: state.color for name, state in self.states.items()}

    def compute_distribution(self, classifications: List[Dict]) -> Dict[str, int]:
        """Compute state distribution from classifications."""
        dist = {name: 0 for name in self.state_names}
        for c in classifications:
            state = c.get('state')
            if state in dist:
                dist[state] += 1
        return dist

    def compute_transition_matrix(
        self,
        sequences: List[List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute transition matrix from state sequences.

        Args:
            sequences: List of state sequences (each sequence is list of state names)
            normalize: Whether to normalize rows to probabilities

        Returns:
            Transition matrix as numpy array (rows=from, cols=to)
        """
        n = self.n_states
        state_to_idx = {name: i for i, name in enumerate(self.state_names)}
        matrix = np.zeros((n, n))

        for seq in sequences:
            for i in range(len(seq) - 1):
                from_state = seq[i]
                to_state = seq[i + 1]
                if from_state in state_to_idx and to_state in state_to_idx:
                    matrix[state_to_idx[from_state], state_to_idx[to_state]] += 1

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            matrix = matrix / row_sums

        return matrix

    def compute_stationary_distribution(
        self,
        transition_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute stationary distribution from transition matrix."""
        # Find eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return stationary

    def compute_entropy_rate(self, transition_matrix: np.ndarray) -> float:
        """Compute entropy rate of the Markov chain."""
        stationary = self.compute_stationary_distribution(transition_matrix)

        entropy_rate = 0.0
        for i, pi in enumerate(stationary):
            if pi > 0:
                for j, p_ij in enumerate(transition_matrix[i]):
                    if p_ij > 0:
                        entropy_rate -= pi * p_ij * np.log2(p_ij)

        return entropy_rate

    def to_dict(self) -> Dict:
        """Serialize state space to dictionary."""
        return {
            'metadata': self.metadata.to_dict(),
            'states': {name: state.to_dict() for name, state in self.states.items()}
        }


class StateSpaceRegistry:
    """
    Registry for state space definitions.

    Allows registration and retrieval of state spaces by name.
    """
    _registry: Dict[str, StateSpace] = {}

    @classmethod
    def register(cls, state_space: StateSpace) -> None:
        """Register a state space."""
        name = state_space.metadata.name
        cls._registry[name] = state_space

    @classmethod
    def get(cls, name: str) -> Optional[StateSpace]:
        """Get a registered state space by name."""
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered state space names."""
        return list(cls._registry.keys())

    @classmethod
    def get_all(cls) -> Dict[str, StateSpace]:
        """Get all registered state spaces."""
        return cls._registry.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._registry.clear()


# Convenience function for creating simple state spaces
def create_simple_state_space(
    name: str,
    display_name: str,
    states: List[Tuple[str, str, str]],  # (name, description, color)
    space_type: StateSpaceType = StateSpaceType.DOMAIN_SPECIFIC,
    description: str = "",
    classifier: Optional[Callable] = None
) -> StateSpace:
    """
    Factory function to create simple state spaces without subclassing.

    Args:
        name: Internal name
        display_name: Human-readable name
        states: List of (state_name, description, color) tuples
        space_type: Type of state space
        description: Description of the state space
        classifier: Optional classification function

    Returns:
        StateSpace instance
    """
    state_defs = {
        s[0]: StateDefinition(name=s[0], description=s[1], color=s[2])
        for s in states
    }

    meta = StateSpaceMetadata(
        name=name,
        display_name=display_name,
        description=description,
        type=space_type,
        n_states=len(states)
    )

    class SimpleStateSpace(StateSpace):
        @property
        def metadata(self) -> StateSpaceMetadata:
            return meta

        @property
        def states(self) -> Dict[str, StateDefinition]:
            return state_defs

        def classify(self, events: List[Dict]) -> List[Dict]:
            if classifier:
                return classifier(events, self)
            raise NotImplementedError("No classifier provided for this state space")

    return SimpleStateSpace()
