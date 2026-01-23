#!/usr/bin/env python3
"""
Archetype Classifier - Practical Tools for Pattern Tracking

Provides utility functions for:
1. Classifying new life events into 4-Animal states
2. Computing behavioral signatures
3. Identifying archetypal roles
4. Detecting escalation patterns
5. Generating risk assessments

This is the practical implementation of the Archetypal Tracking Guide.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path

# Try to import sentence transformers, fall back to keyword-based if not available
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers not available. Using keyword-based classification.")


# =============================================================================
# CONSTANTS
# =============================================================================

ANIMAL_NAMES = {0: "Seeking", 1: "Directing", 2: "Conferring", 3: "Revising"}
ANIMAL_INDICES = {"Seeking": 0, "Directing": 1, "Conferring": 2, "Revising": 3}

# Keywords for rule-based classification (fallback)
STATE_KEYWORDS = {
    0: [  # SEEKING
        "questioned", "struggled", "confused", "identity", "fantasy", "obsessed",
        "introspect", "wonder", "felt", "internal", "thought", "imagined",
        "dream", "desire", "urge", "impulse", "conflict", "doubt", "fear"
    ],
    1: [  # DIRECTING
        "killed", "murdered", "attacked", "assaulted", "controlled", "manipulated",
        "abused", "tortured", "strangled", "stabbed", "raped", "abducted",
        "threatened", "dominated", "coerced", "victim", "violence", "harm"
    ],
    2: [  # CONFERRING
        "watched", "observed", "studied", "followed", "stalked", "monitored",
        "researched", "learned", "noticed", "tracked", "surveillance", "spy",
        "gather", "information", "target", "selected", "scouted"
    ],
    3: [  # REVISING
        "ritual", "routine", "habit", "pattern", "repeated", "always",
        "compuls", "collect", "trophy", "souven", "same", "method",
        "signature", "procedure", "systematic", "organized"
    ]
}

# Prototype sentences for embedding-based classification
PROTOTYPES = {
    0: [  # SEEKING
        "questioned his own identity and sense of self",
        "experienced internal conflict and psychological turmoil",
        "explored dark thoughts and fantasies privately",
        "struggled with feelings of inadequacy",
        "became introspective about violent urges"
    ],
    1: [  # DIRECTING
        "manipulated and controlled his victims",
        "physically assaulted and harmed others",
        "killed the victim after abduction",
        "exerted dominance over family members",
        "committed violent acts against strangers"
    ],
    2: [  # CONFERRING
        "observed potential victims from a distance",
        "studied crime documentaries and murder cases",
        "followed and stalked individuals",
        "gathered information about targets",
        "watched people in public places"
    ],
    3: [  # REVISING
        "developed ritualistic behaviors before crimes",
        "maintained strict daily routines",
        "performed the same ritual with each victim",
        "returned repeatedly to the same locations",
        "collected trophies from victims"
    ]
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClassifiedEvent:
    """A single classified life event."""
    text: str
    state: int
    state_name: str
    confidence: float
    method: str  # 'embedding' or 'keyword'


@dataclass
class BehavioralSignature:
    """Complete behavioral signature for an individual."""
    name: str
    n_events: int
    state_distribution: Dict[str, float]
    dominant_state: str
    dominant_state_pct: float
    transition_matrix: Dict[str, Dict[str, float]]
    top_bigrams: List[Tuple[List[str], int]]
    top_trigrams: List[Tuple[List[str], int]]
    escalation_score: float
    shows_escalation: bool
    phase_distributions: Dict[str, Dict[str, float]]
    archetype_role: str
    risk_level: str


@dataclass
class RiskAssessment:
    """Risk assessment based on behavioral pattern."""
    current_state: str
    recent_trajectory: List[str]
    escalation_detected: bool
    escalation_score: float
    critical_transitions: List[str]
    risk_level: str  # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    risk_factors: List[str]
    recommendations: List[str]


# =============================================================================
# CLASSIFIER
# =============================================================================

class ArchetypeClassifier:
    """
    Classifies life events and computes behavioral signatures.

    Usage:
        classifier = ArchetypeClassifier()

        # Classify single event
        result = classifier.classify_event("he killed his first victim")

        # Classify sequence and get signature
        events = ["felt internal conflict", "watched neighbors", "attacked victim"]
        signature = classifier.compute_signature("John Doe", events)

        # Get risk assessment
        risk = classifier.assess_risk(signature)
    """

    def __init__(self, use_embeddings: bool = True):
        """
        Initialize classifier.

        Args:
            use_embeddings: Use embedding-based classification if available
        """
        self.use_embeddings = use_embeddings and HAS_EMBEDDINGS
        self.model = None
        self.prototype_embeddings = None

        if self.use_embeddings:
            self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embedding model and prototypes."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.prototype_embeddings = {}
        for state, texts in PROTOTYPES.items():
            embeddings = self.model.encode(texts)
            self.prototype_embeddings[state] = embeddings.mean(axis=0)

    def classify_event(self, event: str) -> ClassifiedEvent:
        """
        Classify a single life event into a 4-Animal state.

        Args:
            event: Text description of the life event

        Returns:
            ClassifiedEvent with state, confidence, and method used
        """
        if self.use_embeddings:
            return self._classify_embedding(event)
        else:
            return self._classify_keyword(event)

    def _classify_embedding(self, event: str) -> ClassifiedEvent:
        """Classify using embedding similarity."""
        embedding = self.model.encode([event])[0]

        similarities = {}
        for state, proto_emb in self.prototype_embeddings.items():
            sim = np.dot(embedding, proto_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(proto_emb)
            )
            similarities[state] = sim

        best_state = max(similarities, key=similarities.get)
        confidence = similarities[best_state]

        return ClassifiedEvent(
            text=event,
            state=best_state,
            state_name=ANIMAL_NAMES[best_state],
            confidence=float(confidence),
            method='embedding'
        )

    def _classify_keyword(self, event: str) -> ClassifiedEvent:
        """Classify using keyword matching (fallback)."""
        event_lower = event.lower()

        scores = {state: 0 for state in range(4)}

        for state, keywords in STATE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in event_lower:
                    scores[state] += 1

        total = sum(scores.values())
        if total == 0:
            # Default to Directing if no keywords match
            best_state = 1
            confidence = 0.25
        else:
            best_state = max(scores, key=scores.get)
            confidence = scores[best_state] / total

        return ClassifiedEvent(
            text=event,
            state=best_state,
            state_name=ANIMAL_NAMES[best_state],
            confidence=float(confidence),
            method='keyword'
        )

    def classify_sequence(self, events: List[str]) -> List[ClassifiedEvent]:
        """Classify a sequence of events."""
        return [self.classify_event(e) for e in events]

    def compute_signature(
        self,
        name: str,
        events: List[str],
        classified: Optional[List[ClassifiedEvent]] = None
    ) -> BehavioralSignature:
        """
        Compute complete behavioral signature for an individual.

        Args:
            name: Name/identifier for the individual
            events: List of life event descriptions
            classified: Optional pre-classified events

        Returns:
            BehavioralSignature with all pattern metrics
        """
        if classified is None:
            classified = self.classify_sequence(events)

        sequence = [c.state for c in classified]
        n_events = len(sequence)

        # State distribution
        counts = np.bincount(sequence, minlength=4)
        distribution = counts / counts.sum()
        state_distribution = {ANIMAL_NAMES[i]: float(distribution[i]) for i in range(4)}

        dominant_idx = np.argmax(distribution)
        dominant_state = ANIMAL_NAMES[dominant_idx]
        dominant_pct = float(distribution[dominant_idx])

        # Transition matrix
        trans_counts = np.zeros((4, 4))
        for i in range(len(sequence) - 1):
            trans_counts[sequence[i], sequence[i+1]] += 1

        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        trans_matrix = trans_counts / row_sums

        transition_matrix = {
            ANIMAL_NAMES[i]: {ANIMAL_NAMES[j]: float(trans_matrix[i, j]) for j in range(4)}
            for i in range(4)
        }

        # N-grams
        bigrams = []
        for i in range(len(sequence) - 1):
            bigrams.append((ANIMAL_NAMES[sequence[i]], ANIMAL_NAMES[sequence[i+1]]))
        top_bigrams = [([list(b)[0], list(b)[1]], c) for b, c in Counter(bigrams).most_common(5)]

        trigrams = []
        for i in range(len(sequence) - 2):
            trigrams.append((ANIMAL_NAMES[sequence[i]], ANIMAL_NAMES[sequence[i+1]],
                           ANIMAL_NAMES[sequence[i+2]]))
        top_trigrams = [([t[0], t[1], t[2]], c) for t, c in Counter(trigrams).most_common(5)]

        # Phase analysis
        third = len(sequence) // 3
        phases = {
            'early': sequence[:third] if third > 0 else sequence[:1],
            'middle': sequence[third:2*third] if third > 0 else sequence[1:2],
            'late': sequence[2*third:] if third > 0 else sequence[2:]
        }

        phase_distributions = {}
        for phase_name, phase_seq in phases.items():
            if len(phase_seq) > 0:
                phase_counts = np.bincount(phase_seq, minlength=4)
                phase_dist = phase_counts / phase_counts.sum()
                phase_distributions[phase_name] = {
                    ANIMAL_NAMES[i]: float(phase_dist[i]) for i in range(4)
                }
            else:
                phase_distributions[phase_name] = {ANIMAL_NAMES[i]: 0.0 for i in range(4)}

        # Escalation
        early_directing = phase_distributions['early'].get('Directing', 0)
        late_directing = phase_distributions['late'].get('Directing', 0)
        escalation_score = late_directing - early_directing
        shows_escalation = escalation_score > 0.1

        # Archetype role
        archetype_role = self._determine_role(
            state_distribution, phase_distributions, escalation_score
        )

        # Risk level
        risk_level = self._determine_risk_level(
            state_distribution, escalation_score, sequence[-5:] if len(sequence) >= 5 else sequence
        )

        return BehavioralSignature(
            name=name,
            n_events=n_events,
            state_distribution=state_distribution,
            dominant_state=dominant_state,
            dominant_state_pct=dominant_pct,
            transition_matrix=transition_matrix,
            top_bigrams=top_bigrams,
            top_trigrams=top_trigrams,
            escalation_score=float(escalation_score),
            shows_escalation=shows_escalation,
            phase_distributions=phase_distributions,
            archetype_role=archetype_role,
            risk_level=risk_level
        )

    def _determine_role(
        self,
        distribution: Dict[str, float],
        phases: Dict[str, Dict[str, float]],
        escalation: float
    ) -> str:
        """Determine archetypal role based on pattern."""
        directing_pct = distribution.get('Directing', 0)
        early_seeking = phases.get('early', {}).get('Seeking', 0)
        late_directing = phases.get('late', {}).get('Directing', 0)

        # Check for HUB (complex, multi-modal)
        non_zero_states = sum(1 for v in distribution.values() if v > 0.1)
        if non_zero_states >= 3 and directing_pct < 0.6:
            return "HUB"

        # Check for SINK (Seeking → Directing trajectory)
        if early_seeking > 0.2 and late_directing > 0.5 and escalation > 0.15:
            return "SINK"

        # Check for SOURCE (consistent Directing)
        if directing_pct > 0.65 and escalation < 0.1:
            return "SOURCE"

        return "GENERAL"

    def _determine_risk_level(
        self,
        distribution: Dict[str, float],
        escalation: float,
        recent_states: List[int]
    ) -> str:
        """Determine current risk level."""
        directing_pct = distribution.get('Directing', 0)

        # Check recent states
        recent_directing = sum(1 for s in recent_states if s == 1) / len(recent_states) if recent_states else 0

        # Critical transitions
        has_critical_transition = False
        for i in range(len(recent_states) - 1):
            if recent_states[i] in [0, 2] and recent_states[i+1] == 1:  # Seeking/Conferring → Directing
                has_critical_transition = True
                break

        if recent_directing > 0.8 or has_critical_transition:
            return "CRITICAL"
        elif directing_pct > 0.7 or escalation > 0.2:
            return "HIGH"
        elif directing_pct > 0.5 or escalation > 0.1:
            return "MODERATE"
        else:
            return "LOW"

    def assess_risk(self, signature: BehavioralSignature) -> RiskAssessment:
        """
        Generate comprehensive risk assessment.

        Args:
            signature: BehavioralSignature for the individual

        Returns:
            RiskAssessment with risk factors and recommendations
        """
        # Get recent trajectory from top bigrams
        recent_trajectory = []
        if signature.top_bigrams:
            for bigram, count in signature.top_bigrams[:3]:
                recent_trajectory.extend(bigram)
        recent_trajectory = list(dict.fromkeys(recent_trajectory))[:5]  # Dedupe, keep order

        # Identify critical transitions
        critical_transitions = []
        for bigram, count in signature.top_bigrams:
            if bigram[0] in ['Seeking', 'Conferring'] and bigram[1] == 'Directing':
                critical_transitions.append(f"{bigram[0]} → {bigram[1]}")

        # Identify risk factors
        risk_factors = []

        if signature.dominant_state == 'Directing':
            risk_factors.append("Dominant state is Directing (exploitation)")

        if signature.dominant_state_pct > 0.7:
            risk_factors.append(f"High state concentration ({signature.dominant_state_pct:.0%} in {signature.dominant_state})")

        if signature.shows_escalation:
            risk_factors.append(f"Escalation detected (score: {signature.escalation_score:.2f})")

        if critical_transitions:
            risk_factors.append(f"Critical transitions present: {', '.join(critical_transitions)}")

        if signature.archetype_role == 'HUB':
            risk_factors.append("HUB archetype - complex, unpredictable pattern")

        if signature.archetype_role == 'SINK':
            risk_factors.append("SINK archetype - fantasy-to-action trajectory")

        # Generate recommendations
        recommendations = []

        if signature.risk_level == 'CRITICAL':
            recommendations.append("IMMEDIATE INTERVENTION REQUIRED")
            recommendations.append("Active Directing phase detected")

        if signature.shows_escalation:
            recommendations.append("Monitor for continued escalation")
            recommendations.append("Review recent Seeking phases for fantasy content")

        if 'Conferring' in recent_trajectory:
            recommendations.append("Surveillance behavior detected - check for target selection")

        if signature.archetype_role == 'HUB':
            recommendations.append("Complex pattern - expect unpredictable transitions")

        if not recommendations:
            recommendations.append("Continue monitoring")
            recommendations.append("Watch for transition to Directing state")

        return RiskAssessment(
            current_state=signature.dominant_state,
            recent_trajectory=recent_trajectory,
            escalation_detected=signature.shows_escalation,
            escalation_score=signature.escalation_score,
            critical_transitions=critical_transitions,
            risk_level=signature.risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations
        )

    def to_dict(self, obj) -> Dict:
        """Convert dataclass to dictionary."""
        return asdict(obj)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_classify(events: List[str], name: str = "Subject") -> Dict:
    """
    Quick classification and assessment of a life event sequence.

    Args:
        events: List of life event descriptions
        name: Name/identifier for the subject

    Returns:
        Dictionary with signature and risk assessment
    """
    classifier = ArchetypeClassifier()
    signature = classifier.compute_signature(name, events)
    risk = classifier.assess_risk(signature)

    return {
        'signature': classifier.to_dict(signature),
        'risk_assessment': classifier.to_dict(risk)
    }


def print_assessment(events: List[str], name: str = "Subject"):
    """Print formatted assessment to console."""
    result = quick_classify(events, name)
    sig = result['signature']
    risk = result['risk_assessment']

    print("\n" + "=" * 60)
    print(f"BEHAVIORAL ASSESSMENT: {name}")
    print("=" * 60)

    print(f"\nEvents analyzed: {sig['n_events']}")
    print(f"Archetype Role: {sig['archetype_role']}")
    print(f"Dominant State: {sig['dominant_state']} ({sig['dominant_state_pct']:.0%})")

    print("\nState Distribution:")
    for state, pct in sig['state_distribution'].items():
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"  {state:12s}: {bar} {pct:.0%}")

    print(f"\nEscalation: {'YES' if sig['shows_escalation'] else 'NO'} (score: {sig['escalation_score']:.2f})")

    print("\n" + "-" * 60)
    print(f"RISK LEVEL: {risk['risk_level']}")
    print("-" * 60)

    print("\nRisk Factors:")
    for factor in risk['risk_factors']:
        print(f"  • {factor}")

    print("\nRecommendations:")
    for rec in risk['recommendations']:
        print(f"  → {rec}")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Archetype Classifier - Example Usage")
    print("=" * 60)

    # Example life events
    example_events = [
        "struggled with feelings of inadequacy during childhood",
        "became increasingly isolated from peers",
        "developed obsessive thoughts about violence",
        "started watching neighbors from his window",
        "followed a woman home from work",
        "attacked his first victim",
        "developed a ritual around the attacks",
        "killed again using the same method"
    ]

    print("\nExample Events:")
    for i, e in enumerate(example_events, 1):
        print(f"  {i}. {e}")

    print_assessment(example_events, "Example Subject")
