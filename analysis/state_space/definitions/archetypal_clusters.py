"""
10 Archetypal Event Clusters State Space Definition

Data-driven state space emergent from semantic clustering:
1. Events embedded using SentenceTransformer
2. K-Means clustering groups semantically similar events
3. LLM labels each cluster with an archetypal theme

This captures contextual meaning rather than abstract psychological dimensions.
"""
from typing import Dict, List, Optional
import json
from pathlib import Path
from ..core import (
    StateSpace,
    StateDefinition,
    StateSpaceMetadata,
    StateSpaceType,
    StateSpaceRegistry
)


# Default cluster themes (can be overridden by loading from JSON)
DEFAULT_CLUSTER_THEMES = {
    0: "Predatory/Geographical Stalking",
    1: "Sexually Motivated Serial Killer",
    2: "Escalating Criminal Behavior",
    3: "Sadistic Duo/Killer Couple",
    4: "Power and Control",
    5: "Authority Obsession",
    6: "Legal Judgment/Sentencing",
    7: "Search for Identity",
    8: "Domestic Violence/Discord",
    9: "Angel of Death/Mercy Killer"
}

# Cluster colors for visualization (distinct from animal colors)
CLUSTER_COLORS = {
    0: '#1abc9c',   # Turquoise
    1: '#e74c3c',   # Red
    2: '#f39c12',   # Orange
    3: '#9b59b6',   # Purple
    4: '#34495e',   # Dark gray
    5: '#3498db',   # Blue
    6: '#95a5a6',   # Gray
    7: '#2ecc71',   # Green
    8: '#e67e22',   # Dark orange
    9: '#1e3a5f'    # Navy
}

# Mapping from clusters to 4-Animal states
CLUSTER_TO_ANIMAL = {
    0: {'primary': 'Conferring', 'secondary': 'Directing', 'weight': 0.7},
    1: {'primary': 'Directing', 'secondary': 'Seeking', 'weight': 0.7},
    2: {'primary': 'Directing', 'secondary': 'Revising', 'weight': 0.7},
    3: {'primary': 'Directing', 'secondary': 'Conferring', 'weight': 0.7},
    4: {'primary': 'Directing', 'secondary': None, 'weight': 1.0},
    5: {'primary': 'Directing', 'secondary': 'Seeking', 'weight': 0.7},
    6: {'primary': 'Revising', 'secondary': None, 'weight': 1.0},
    7: {'primary': 'Seeking', 'secondary': 'Revising', 'weight': 0.7},
    8: {'primary': 'Directing', 'secondary': 'Revising', 'weight': 0.7},
    9: {'primary': 'Directing', 'secondary': 'Conferring', 'weight': 0.7}
}


class ArchetypalClusterSpace(StateSpace):
    """
    10 Archetypal Event Clusters from semantic embedding + K-Means.

    Each cluster represents a thematically coherent group of criminal
    life events, labeled by an LLM with an archetypal theme.
    """

    def __init__(
        self,
        cluster_themes: Optional[Dict[int, str]] = None,
        cluster_descriptions: Optional[Dict[int, str]] = None
    ):
        """
        Initialize with optional custom cluster themes.

        Args:
            cluster_themes: Dict mapping cluster ID to theme name
            cluster_descriptions: Dict mapping cluster ID to description
        """
        self._themes = cluster_themes or DEFAULT_CLUSTER_THEMES
        self._descriptions = cluster_descriptions or {}
        self._build_states()

    def _build_states(self):
        """Build state definitions from themes."""
        self._states = {}
        for cluster_id in range(10):
            theme = self._themes.get(cluster_id, f"Cluster {cluster_id}")
            description = self._descriptions.get(
                cluster_id,
                f"Events semantically grouped as '{theme}'"
            )

            # State name is the cluster ID as string
            state_name = str(cluster_id)

            self._states[state_name] = StateDefinition(
                name=state_name,
                description=description,
                color=CLUSTER_COLORS.get(cluster_id, '#95a5a6'),
                dimensions={
                    'theme': theme,
                    'cluster_id': cluster_id,
                    'primary_animal': CLUSTER_TO_ANIMAL.get(cluster_id, {}).get('primary'),
                    'secondary_animal': CLUSTER_TO_ANIMAL.get(cluster_id, {}).get('secondary')
                },
                keywords=[],  # Populated from cluster analysis
                examples=[]
            )

        self._metadata = StateSpaceMetadata(
            name="archetypal_clusters",
            display_name="10 Archetypal Event Clusters",
            description=(
                "Data-driven event clusters from semantic embedding and K-Means. "
                "Captures contextual meaning and thematic patterns."
            ),
            type=StateSpaceType.DATA_DRIVEN,
            n_states=10,
            version="1.0",
            source="SentenceTransformer + K-Means + LLM labeling",
            dimensions=['semantic_similarity']
        )

    @property
    def metadata(self) -> StateSpaceMetadata:
        return self._metadata

    @property
    def states(self) -> Dict[str, StateDefinition]:
        return self._states

    def get_theme(self, cluster_id: int) -> str:
        """Get theme name for a cluster."""
        return self._themes.get(cluster_id, f"Cluster {cluster_id}")

    def get_animal_mapping(self, cluster_id: int) -> Dict:
        """Get the 4-Animal state mapping for a cluster."""
        return CLUSTER_TO_ANIMAL.get(cluster_id, {})

    def classify(self, events: List[Dict]) -> List[Dict]:
        """
        Classify events into clusters.

        Note: This requires the embedding model and trained K-Means.
        For actual classification, use the full clustering pipeline.

        Args:
            events: List of event dicts with 'text' field

        Returns:
            List of classification results with cluster assignments
        """
        # Placeholder - actual classification requires:
        # 1. SentenceTransformer embedding
        # 2. K-Means centroid assignment
        raise NotImplementedError(
            "Cluster classification requires embedding model. "
            "Use the full clustering pipeline in serial-killer-life-events-clustering.py"
        )

    def update_from_clustering_results(
        self,
        clusters_path: str,
        prototypes_path: Optional[str] = None
    ) -> None:
        """
        Update state definitions from clustering results.

        Args:
            clusters_path: Path to clusters.json with cluster info
            prototypes_path: Optional path to prototypes.pkl
        """
        clusters_file = Path(clusters_path)
        if clusters_file.exists():
            with open(clusters_file) as f:
                cluster_data = json.load(f)

            for cluster_id, info in cluster_data.items():
                cid = int(cluster_id)
                state_name = str(cid)

                if state_name in self._states:
                    # Update theme
                    if 'theme' in info:
                        self._themes[cid] = info['theme']
                        self._states[state_name].dimensions['theme'] = info['theme']

                    # Update description
                    if 'summary' in info:
                        self._states[state_name].description = info['summary']

                    # Add examples
                    if 'sample_events' in info:
                        self._states[state_name].examples = info['sample_events'][:5]


def create_archetypal_cluster_space(
    clusters_json_path: Optional[str] = None
) -> ArchetypalClusterSpace:
    """
    Factory function to create an ArchetypalClusterSpace.

    Args:
        clusters_json_path: Optional path to clusters.json

    Returns:
        ArchetypalClusterSpace instance
    """
    space = ArchetypalClusterSpace()

    if clusters_json_path:
        space.update_from_clustering_results(clusters_json_path)

    return space


# ============================================================================
# MAPPING TO 4-ANIMAL STATES
# ============================================================================

def get_cluster_to_animal_mapping():
    """
    Get the mapping from archetypal clusters to 4-Animal states.

    Returns a list of dicts suitable for create_mapping_from_table().
    """
    from ..mappings import StateMapping, StateSpaceMapping, MappingType

    mapping_table = []
    for cluster_id, mapping in CLUSTER_TO_ANIMAL.items():
        mapping_table.append({
            'source': str(cluster_id),
            'primary': mapping['primary'],
            'secondary': mapping.get('secondary'),
            'weight': mapping.get('weight', 1.0),
            'rationale': f"Cluster {cluster_id} ({DEFAULT_CLUSTER_THEMES.get(cluster_id, 'Unknown')}) -> {mapping['primary']}"
        })

    return mapping_table


def create_cluster_to_animal_mapping() -> 'StateSpaceMapping':
    """
    Create a StateSpaceMapping from archetypal clusters to 4-Animal states.
    """
    from ..mappings import create_mapping_from_table

    return create_mapping_from_table(
        source="archetypal_clusters",
        target="animal_states",
        mapping_table=get_cluster_to_animal_mapping(),
        description="Maps 10 archetypal event clusters to 4-Animal behavioral states",
        methodology=(
            "Semantic analysis of cluster themes mapped to psychological dimensions. "
            "Primary state reflects dominant mode, secondary captures complexity."
        )
    )


# Create default instance
archetypal_cluster_space = ArchetypalClusterSpace()

# Register with global registry
StateSpaceRegistry.register(archetypal_cluster_space)
