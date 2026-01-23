"""
Predefined State Space Definitions

Available state spaces:
- animal_states: 4-Animal State Space (theory-driven)
- archetypal_clusters: 10 Archetypal Event Clusters (data-driven)
"""

from .animal_states import (
    AnimalStateSpace,
    animal_state_space,
    ANIMAL_COLORS,
    CLASSIFICATION_PROMPT,
    get_classification_prompt
)

from .archetypal_clusters import (
    ArchetypalClusterSpace,
    archetypal_cluster_space,
    create_archetypal_cluster_space,
    CLUSTER_COLORS,
    DEFAULT_CLUSTER_THEMES,
    CLUSTER_TO_ANIMAL,
    get_cluster_to_animal_mapping,
    create_cluster_to_animal_mapping
)

__all__ = [
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
