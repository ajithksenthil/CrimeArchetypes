"""
State Space Mappings

Provides systematic mapping between different state spaces, enabling:
- Translation of classifications between frameworks
- Comparison of different categorization schemes
- Hierarchical state space relationships

Mapping Types:
- One-to-One: Each source state maps to exactly one target state
- One-to-Many: Each source state can map to multiple target states (with weights)
- Many-to-One: Multiple source states collapse to one target state
- Probabilistic: Source states map to target states with probabilities
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum

from .core import StateSpace, StateSpaceRegistry


class MappingType(Enum):
    """Types of mappings between state spaces."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    PROBABILISTIC = "probabilistic"


@dataclass
class StateMapping:
    """
    Defines how a single source state maps to target state(s).

    For deterministic mappings:
        primary: Main target state
        secondary: Optional secondary target state
        weight: Proportion going to primary (rest goes to secondary)

    For probabilistic mappings:
        probabilities: Dict of target_state -> probability
    """
    source_state: str
    primary: str
    secondary: Optional[str] = None
    weight: float = 1.0  # Proportion to primary (0-1)
    probabilities: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 1.0

    def get_target_distribution(self) -> Dict[str, float]:
        """Get distribution over target states."""
        if self.probabilities:
            return self.probabilities

        dist = {self.primary: self.weight}
        if self.secondary and self.weight < 1.0:
            dist[self.secondary] = 1.0 - self.weight
        return dist

    def to_dict(self) -> Dict:
        return {
            'source_state': self.source_state,
            'primary': self.primary,
            'secondary': self.secondary,
            'weight': self.weight,
            'probabilities': self.probabilities,
            'rationale': self.rationale,
            'confidence': self.confidence
        }


@dataclass
class StateSpaceMapping:
    """
    Complete mapping between two state spaces.

    Attributes:
        source: Source state space name
        target: Target state space name
        mappings: Dict of source_state -> StateMapping
        mapping_type: Type of mapping
        bidirectional: Whether mapping works both ways
        version: Version string for tracking changes
    """
    source: str
    target: str
    mappings: Dict[str, StateMapping]
    mapping_type: MappingType = MappingType.ONE_TO_MANY
    bidirectional: bool = False
    version: str = "1.0"
    description: str = ""
    methodology: str = ""

    def map_state(self, source_state: str) -> Dict[str, float]:
        """
        Map a single source state to target state distribution.

        Returns:
            Dict of target_state -> proportion
        """
        mapping = self.mappings.get(source_state)
        if mapping is None:
            return {}
        return mapping.get_target_distribution()

    def map_distribution(
        self,
        source_distribution: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Map a distribution over source states to target states.

        Args:
            source_distribution: Dict of source_state -> count

        Returns:
            Dict of target_state -> expected count
        """
        target_dist: Dict[str, float] = {}

        for source_state, count in source_distribution.items():
            target_probs = self.map_state(source_state)
            for target_state, prob in target_probs.items():
                target_dist[target_state] = target_dist.get(target_state, 0) + count * prob

        return target_dist

    def map_classifications(
        self,
        classifications: List[Dict],
        source_field: str = 'state'
    ) -> List[Dict]:
        """
        Map a list of classified events from source to target space.

        Args:
            classifications: List of classification dicts with source state
            source_field: Field name containing source state

        Returns:
            List of dicts with target state assignments
        """
        mapped = []
        for c in classifications:
            source_state = c.get(source_field)
            target_probs = self.map_state(source_state)

            if not target_probs:
                # No mapping found, preserve original
                mapped.append({
                    **c,
                    'mapped_state': None,
                    'mapping_confidence': 0.0
                })
                continue

            # Get primary target (highest probability)
            primary_target = max(target_probs, key=target_probs.get)
            mapping = self.mappings.get(source_state)

            mapped.append({
                **c,
                'mapped_state': primary_target,
                'mapped_distribution': target_probs,
                'mapping_confidence': mapping.confidence if mapping else 1.0,
                'mapping_rationale': mapping.rationale if mapping else ""
            })

        return mapped

    def get_mapping_matrix(
        self,
        source_space: StateSpace,
        target_space: StateSpace
    ) -> np.ndarray:
        """
        Get mapping as a matrix (source_states x target_states).

        Each row sums to 1 (or 0 if no mapping exists).
        """
        n_source = source_space.n_states
        n_target = target_space.n_states

        source_idx = {name: i for i, name in enumerate(source_space.state_names)}
        target_idx = {name: i for i, name in enumerate(target_space.state_names)}

        matrix = np.zeros((n_source, n_target))

        for source_state, mapping in self.mappings.items():
            if source_state not in source_idx:
                continue

            target_dist = mapping.get_target_distribution()
            for target_state, prob in target_dist.items():
                if target_state in target_idx:
                    matrix[source_idx[source_state], target_idx[target_state]] = prob

        return matrix

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'mapping_type': self.mapping_type.value,
            'bidirectional': self.bidirectional,
            'version': self.version,
            'description': self.description,
            'methodology': self.methodology,
            'mappings': {k: v.to_dict() for k, v in self.mappings.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StateSpaceMapping':
        """Create mapping from dictionary."""
        mappings = {
            k: StateMapping(**v) for k, v in data.get('mappings', {}).items()
        }
        return cls(
            source=data['source'],
            target=data['target'],
            mappings=mappings,
            mapping_type=MappingType(data.get('mapping_type', 'one_to_many')),
            bidirectional=data.get('bidirectional', False),
            version=data.get('version', '1.0'),
            description=data.get('description', ''),
            methodology=data.get('methodology', '')
        )


class MappingRegistry:
    """Registry for state space mappings."""
    _registry: Dict[Tuple[str, str], StateSpaceMapping] = {}

    @classmethod
    def register(cls, mapping: StateSpaceMapping) -> None:
        """Register a mapping."""
        key = (mapping.source, mapping.target)
        cls._registry[key] = mapping

        # If bidirectional, also register reverse
        if mapping.bidirectional:
            reverse_key = (mapping.target, mapping.source)
            # Note: Reversing requires different logic, simplified here
            cls._registry[reverse_key] = mapping

    @classmethod
    def get(cls, source: str, target: str) -> Optional[StateSpaceMapping]:
        """Get a mapping by source and target names."""
        return cls._registry.get((source, target))

    @classmethod
    def list_all(cls) -> List[Tuple[str, str]]:
        """List all registered mapping pairs."""
        return list(cls._registry.keys())

    @classmethod
    def get_mappings_from(cls, source: str) -> List[StateSpaceMapping]:
        """Get all mappings from a given source state space."""
        return [m for (s, t), m in cls._registry.items() if s == source]

    @classmethod
    def get_mappings_to(cls, target: str) -> List[StateSpaceMapping]:
        """Get all mappings to a given target state space."""
        return [m for (s, t), m in cls._registry.items() if t == target]


def create_mapping_from_table(
    source: str,
    target: str,
    mapping_table: List[Dict],
    description: str = "",
    methodology: str = ""
) -> StateSpaceMapping:
    """
    Create a mapping from a simple table format.

    Args:
        source: Source state space name
        target: Target state space name
        mapping_table: List of dicts with keys:
            - source: Source state name
            - primary: Primary target state
            - secondary: (optional) Secondary target state
            - weight: (optional) Weight for primary (default 1.0)
            - rationale: (optional) Explanation

    Returns:
        StateSpaceMapping instance
    """
    mappings = {}
    for row in mapping_table:
        mappings[row['source']] = StateMapping(
            source_state=row['source'],
            primary=row['primary'],
            secondary=row.get('secondary'),
            weight=row.get('weight', 1.0),
            rationale=row.get('rationale', '')
        )

    return StateSpaceMapping(
        source=source,
        target=target,
        mappings=mappings,
        description=description,
        methodology=methodology
    )
