"""
State Space Framework for Behavioral Analysis

A systematic framework for defining, comparing, and analyzing different
state space representations of behavioral data.

Modules:
- core: Base classes (StateSpace, StateDefinition, StateSpaceRegistry)
- mappings: Mappings between state spaces (StateSpaceMapping, MappingRegistry)
- metrics: Comparison metrics (divergence, agreement, information-theoretic)
- analytics: Downstream analytics pipeline
- definitions: Predefined state spaces (4-Animal, Archetypal Clusters)

Quick Start:
    from state_space import (
        StateSpaceRegistry,
        StateSpaceAnalytics,
        animal_state_space,
        archetypal_cluster_space
    )

    # Get predefined state spaces
    animal = StateSpaceRegistry.get('animal_states')
    clusters = StateSpaceRegistry.get('archetypal_clusters')

    # Run comparison analytics
    analytics = StateSpaceAnalytics()
    analytics.add_space('animal', animal, animal_classifications)
    analytics.add_space('cluster', clusters, cluster_classifications)

    results = analytics.run_pipeline()
    results.save('comparison_results.json')
"""

# Core classes
from .core import (
    StateSpace,
    StateDefinition,
    StateSpaceMetadata,
    StateSpaceType,
    StateSpaceRegistry,
    create_simple_state_space
)

# Mapping classes
from .mappings import (
    StateMapping,
    StateSpaceMapping,
    MappingType,
    MappingRegistry,
    create_mapping_from_table
)

# Metrics
from .metrics import (
    ComparisonResult,
    compare_state_spaces,
    compare_transition_matrices,
    kl_divergence,
    js_divergence,
    earth_movers_distance,
    distribution_overlap,
    compute_confusion_matrix,
    agreement_rate,
    cohens_kappa,
    weighted_kappa,
    entropy,
    joint_entropy,
    mutual_information,
    normalized_mutual_information,
    variation_of_information,
    mapping_coverage,
    mapping_ambiguity,
    mapping_confidence_score
)

# Analytics
from .analytics import (
    StateSpaceAnalytics,
    ClassificationData,
    AnalyticsResult,
    quick_compare
)

# Predefined state spaces (auto-registered on import)
from .definitions import (
    # 4-Animal State Space
    AnimalStateSpace,
    animal_state_space,
    ANIMAL_COLORS,
    CLASSIFICATION_PROMPT,
    get_classification_prompt,

    # Archetypal Clusters
    ArchetypalClusterSpace,
    archetypal_cluster_space,
    create_archetypal_cluster_space,
    CLUSTER_COLORS,
    DEFAULT_CLUSTER_THEMES,
    CLUSTER_TO_ANIMAL,
    get_cluster_to_animal_mapping,
    create_cluster_to_animal_mapping
)


__version__ = "1.0.0"

__all__ = [
    # Core
    'StateSpace',
    'StateDefinition',
    'StateSpaceMetadata',
    'StateSpaceType',
    'StateSpaceRegistry',
    'create_simple_state_space',

    # Mappings
    'StateMapping',
    'StateSpaceMapping',
    'MappingType',
    'MappingRegistry',
    'create_mapping_from_table',

    # Metrics
    'ComparisonResult',
    'compare_state_spaces',
    'compare_transition_matrices',
    'kl_divergence',
    'js_divergence',
    'earth_movers_distance',
    'distribution_overlap',
    'compute_confusion_matrix',
    'agreement_rate',
    'cohens_kappa',
    'weighted_kappa',
    'entropy',
    'joint_entropy',
    'mutual_information',
    'normalized_mutual_information',
    'variation_of_information',
    'mapping_coverage',
    'mapping_ambiguity',
    'mapping_confidence_score',

    # Analytics
    'StateSpaceAnalytics',
    'ClassificationData',
    'AnalyticsResult',
    'quick_compare',

    # 4-Animal State Space
    'AnimalStateSpace',
    'animal_state_space',
    'ANIMAL_COLORS',
    'CLASSIFICATION_PROMPT',
    'get_classification_prompt',

    # Archetypal Clusters
    'ArchetypalClusterSpace',
    'archetypal_cluster_space',
    'create_archetypal_cluster_space',
    'CLUSTER_COLORS',
    'DEFAULT_CLUSTER_THEMES',
    'CLUSTER_TO_ANIMAL',
    'get_cluster_to_animal_mapping',
    'create_cluster_to_animal_mapping'
]


def list_registered_spaces() -> list:
    """List all registered state spaces."""
    return StateSpaceRegistry.list_all()


def get_space(name: str) -> StateSpace:
    """Get a registered state space by name."""
    space = StateSpaceRegistry.get(name)
    if space is None:
        raise KeyError(f"Unknown state space: {name}. Available: {list_registered_spaces()}")
    return space
