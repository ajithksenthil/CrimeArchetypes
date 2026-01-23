"""
4-Animal State Space Definition

Theory-driven state space based on two psychological dimensions:
- Self / Other: Internal vs external focus
- Explore / Exploit: Discovery vs utilization

States:
- Seeking (Self + Explore): Internal fantasy, self-exploration
- Directing (Other + Exploit): Control, manipulation, exploitation
- Conferring (Other + Explore): Observation, stalking, target selection
- Revising (Self + Exploit): Rituals, habits, processing

This is the primary framework for Markov chain transition analysis.
"""
from typing import Dict, List
from ..core import (
    StateSpace,
    StateDefinition,
    StateSpaceMetadata,
    StateSpaceType,
    StateSpaceRegistry
)


# Color scheme for visualization
ANIMAL_COLORS = {
    'Seeking': '#2ecc71',     # Green
    'Directing': '#e74c3c',   # Red
    'Conferring': '#3498db',  # Blue
    'Revising': '#9b59b6'     # Purple
}


class AnimalStateSpace(StateSpace):
    """
    4-Animal State Space for behavioral classification.

    Based on the intersection of two dimensions:
    - Focus: Self (internal) vs Other (external)
    - Mode: Explore (discovery) vs Exploit (utilization)
    """

    _metadata = StateSpaceMetadata(
        name="animal_states",
        display_name="4-Animal State Space",
        description=(
            "Theory-driven behavioral classification based on "
            "Self/Other focus and Explore/Exploit mode dimensions"
        ),
        type=StateSpaceType.THEORY_DRIVEN,
        n_states=4,
        version="1.0",
        source="Behavioral dynamics framework",
        dimensions=['focus', 'mode']
    )

    _states = {
        'Seeking': StateDefinition(
            name='Seeking',
            description=(
                "Self + Explore: Internal fantasy, introspection, "
                "identity exploration, planning without action"
            ),
            color=ANIMAL_COLORS['Seeking'],
            dimensions={'focus': 'self', 'mode': 'explore'},
            keywords=[
                'fantasy', 'introspection', 'planning', 'imagination',
                'self-exploration', 'identity', 'thinking', 'contemplation'
            ],
            examples=[
                'Daydreaming about future crimes',
                'Developing internal fantasy world',
                'Exploring personal identity',
                'Planning without action'
            ]
        ),
        'Directing': StateDefinition(
            name='Directing',
            description=(
                "Other + Exploit: Active control, manipulation, "
                "exploitation of others, direct action on targets"
            ),
            color=ANIMAL_COLORS['Directing'],
            dimensions={'focus': 'other', 'mode': 'exploit'},
            keywords=[
                'control', 'manipulation', 'exploitation', 'domination',
                'violence', 'action', 'power', 'assault', 'abuse'
            ],
            examples=[
                'Committing violent acts',
                'Manipulating victims',
                'Exercising control over others',
                'Active exploitation'
            ]
        ),
        'Conferring': StateDefinition(
            name='Conferring',
            description=(
                "Other + Explore: Observation, stalking, target selection, "
                "information gathering about potential victims"
            ),
            color=ANIMAL_COLORS['Conferring'],
            dimensions={'focus': 'other', 'mode': 'explore'},
            keywords=[
                'observation', 'stalking', 'surveillance', 'watching',
                'scouting', 'target selection', 'gathering information'
            ],
            examples=[
                'Stalking potential victims',
                'Observing targets',
                'Gathering information about victims',
                'Surveillance activities'
            ]
        ),
        'Revising': StateDefinition(
            name='Revising',
            description=(
                "Self + Exploit: Processing, rituals, habits, "
                "internal consolidation of experiences"
            ),
            color=ANIMAL_COLORS['Revising'],
            dimensions={'focus': 'self', 'mode': 'exploit'},
            keywords=[
                'ritual', 'habit', 'processing', 'consolidation',
                'reflection', 'trophy keeping', 'routine', 'compulsion'
            ],
            examples=[
                'Performing post-crime rituals',
                'Keeping trophies',
                'Processing experiences internally',
                'Habitual behaviors'
            ]
        )
    }

    @property
    def metadata(self) -> StateSpaceMetadata:
        return self._metadata

    @property
    def states(self) -> Dict[str, StateDefinition]:
        return self._states

    def classify(self, events: List[Dict]) -> List[Dict]:
        """
        Classify events into 4-Animal states.

        This is a placeholder - actual classification requires:
        - LLM-based classification (see llm_animal_classifier.py)
        - Rule-based keyword matching
        - Or other classification methods

        Args:
            events: List of event dicts with 'text' field

        Returns:
            List of classification results
        """
        results = []
        for event in events:
            # Basic keyword-based classification (fallback)
            text = event.get('text', '').lower()

            # Score each state
            scores = {}
            for state_name, state_def in self._states.items():
                score = sum(
                    1 for kw in state_def.keywords
                    if kw.lower() in text
                )
                scores[state_name] = score

            # Get best match
            best_state = max(scores, key=scores.get)
            best_score = scores[best_state]

            # Confidence based on score
            total_score = sum(scores.values())
            confidence = best_score / total_score if total_score > 0 else 0.25

            results.append({
                'event_id': event.get('id'),
                'text': event.get('text'),
                'state': best_state,
                'confidence': confidence,
                'reasoning': f"Keyword match (score: {best_score})",
                'scores': scores
            })

        return results

    def get_dimension_quadrant(self, state: str) -> Dict[str, str]:
        """Get the dimension values for a state."""
        state_def = self._states.get(state)
        if state_def:
            return state_def.dimensions
        return {}

    @classmethod
    def get_state_by_dimensions(cls, focus: str, mode: str) -> str:
        """
        Get state name from dimension values.

        Args:
            focus: 'self' or 'other'
            mode: 'explore' or 'exploit'

        Returns:
            State name
        """
        mapping = {
            ('self', 'explore'): 'Seeking',
            ('other', 'exploit'): 'Directing',
            ('other', 'explore'): 'Conferring',
            ('self', 'exploit'): 'Revising'
        }
        return mapping.get((focus.lower(), mode.lower()))


# Create singleton instance
animal_state_space = AnimalStateSpace()

# Register with global registry
StateSpaceRegistry.register(animal_state_space)


# ============================================================================
# CLASSIFICATION PROMPT (for LLM-based classification)
# ============================================================================

CLASSIFICATION_PROMPT = """
You are classifying life events of criminals into behavioral states.

## The 4-Animal State Space

Based on two dimensions:
1. **Focus**: Self (internal) vs Other (external)
2. **Mode**: Explore (discovery) vs Exploit (utilization)

### States:

**SEEKING** (Self + Explore)
- Internal fantasy, introspection, identity exploration
- Thinking, planning, imagining without acting
- Examples: Fantasizing, daydreaming, self-reflection, developing internal world

**DIRECTING** (Other + Exploit)
- Active control, manipulation, exploitation of others
- Direct action on external targets
- Examples: Violence, manipulation, abuse, domination, exercising power

**CONFERRING** (Other + Explore)
- Observation, surveillance, information gathering about others
- Exploring potential targets without acting
- Examples: Stalking, watching, scouting, target selection

**REVISING** (Self + Exploit)
- Processing, rituals, habits, internal consolidation
- Using internal resources in established patterns
- Examples: Post-crime rituals, trophy keeping, habitual behaviors, compulsions

## Classification Task

For each event, determine:
1. The PRIMARY state that best describes the event
2. Your CONFIDENCE level (HIGH, MEDIUM, LOW)
3. Brief REASONING for your classification

Focus on the psychological mode, not the moral valence.
"""


def get_classification_prompt() -> str:
    """Get the LLM classification prompt."""
    return CLASSIFICATION_PROMPT
