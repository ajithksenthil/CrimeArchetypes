#!/usr/bin/env python3
"""
Archetypal Interpretation Analysis

Makes the discovered archetypes interpretable by:
1. Extracting behavioral signatures (state distributions, transitions, sequences)
2. Identifying characteristic life events and themes
3. Creating narrative profiles for each archetypal role
4. Building "archetype cards" with interpretable descriptions

This transforms abstract transfer entropy patterns into understandable
behavioral templates that can be tracked and studied.
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

ANIMAL_NAMES = {0: "Seeking", 1: "Directing", 2: "Conferring", 3: "Revising"}
ANIMAL_COLORS = {'Seeking': '#2ecc71', 'Directing': '#e74c3c',
                 'Conferring': '#3498db', 'Revising': '#9b59b6'}


def to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj

# Behavioral interpretations for state combinations
STATE_INTERPRETATIONS = {
    'Seeking': {
        'description': 'Self-exploration and introspection',
        'psychological': 'Internal conflict, identity struggles, fantasy development',
        'behavioral': 'Withdrawal, rumination, obsessive thinking',
        'risk_indicator': 'Building internal pressure, escalating fantasies'
    },
    'Directing': {
        'description': 'Control and exploitation of others',
        'psychological': 'Dominance needs, lack of empathy, predatory focus',
        'behavioral': 'Manipulation, violence, coercion, murder',
        'risk_indicator': 'Active harm, peak danger phase'
    },
    'Conferring': {
        'description': 'Observation and social learning',
        'psychological': 'Studying others, gathering information, stalking',
        'behavioral': 'Surveillance, research, target selection',
        'risk_indicator': 'Pre-offense planning, victim selection'
    },
    'Revising': {
        'description': 'Ritualistic and habitual patterns',
        'psychological': 'Compulsion, addiction, routine reinforcement',
        'behavioral': 'Signature behaviors, trophy collection, rituals',
        'risk_indicator': 'Established MO, predictable patterns'
    }
}

# Transition interpretations
TRANSITION_INTERPRETATIONS = {
    ('Seeking', 'Directing'): 'Fantasy-to-action escalation',
    ('Seeking', 'Conferring'): 'Internal focus shifting to external observation',
    ('Seeking', 'Revising'): 'Internalizing patterns into habits',
    ('Directing', 'Seeking'): 'Post-offense introspection/guilt',
    ('Directing', 'Conferring'): 'Returning to surveillance after action',
    ('Directing', 'Revising'): 'Consolidating methods into rituals',
    ('Conferring', 'Seeking'): 'External observation triggering internal fantasy',
    ('Conferring', 'Directing'): 'Observation leading to action',
    ('Conferring', 'Revising'): 'Learning becoming habit',
    ('Revising', 'Seeking'): 'Routine disruption causing introspection',
    ('Revising', 'Directing'): 'Ritual triggering offense',
    ('Revising', 'Conferring'): 'Habit-driven surveillance',
    ('Seeking', 'Seeking'): 'Prolonged internal struggle',
    ('Directing', 'Directing'): 'Sustained predatory activity',
    ('Conferring', 'Conferring'): 'Extended surveillance phase',
    ('Revising', 'Revising'): 'Entrenched ritualistic behavior'
}


# =============================================================================
# BEHAVIORAL SIGNATURE EXTRACTION
# =============================================================================

class BehavioralSignature:
    """Captures the behavioral fingerprint of a criminal."""

    def __init__(self, name: str, sequence: List[int], events: List[str]):
        self.name = name
        self.sequence = sequence
        self.events = events
        self.n_events = len(sequence)

        # Compute signature components
        self._compute_state_distribution()
        self._compute_transition_matrix()
        self._compute_dominant_patterns()
        self._compute_phase_analysis()

    def _compute_state_distribution(self):
        """Compute state distribution."""
        counts = np.bincount(self.sequence, minlength=4)
        self.state_distribution = counts / counts.sum()
        self.dominant_state = ANIMAL_NAMES[np.argmax(self.state_distribution)]
        self.dominant_state_pct = self.state_distribution.max()

    def _compute_transition_matrix(self):
        """Compute transition matrix."""
        counts = np.zeros((4, 4))
        for i in range(len(self.sequence) - 1):
            counts[self.sequence[i], self.sequence[i+1]] += 1

        # Normalize
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition_matrix = counts / row_sums

        # Find dominant transition
        max_idx = np.unravel_index(np.argmax(self.transition_matrix), (4, 4))
        self.dominant_transition = (ANIMAL_NAMES[max_idx[0]], ANIMAL_NAMES[max_idx[1]])
        self.dominant_transition_prob = self.transition_matrix[max_idx]

    def _compute_dominant_patterns(self):
        """Find most common 2-gram and 3-gram patterns."""
        # 2-grams
        bigrams = []
        for i in range(len(self.sequence) - 1):
            bigram = (ANIMAL_NAMES[self.sequence[i]], ANIMAL_NAMES[self.sequence[i+1]])
            bigrams.append(bigram)

        self.top_bigrams = Counter(bigrams).most_common(5)

        # 3-grams
        trigrams = []
        for i in range(len(self.sequence) - 2):
            trigram = (ANIMAL_NAMES[self.sequence[i]],
                      ANIMAL_NAMES[self.sequence[i+1]],
                      ANIMAL_NAMES[self.sequence[i+2]])
            trigrams.append(trigram)

        self.top_trigrams = Counter(trigrams).most_common(5)

    def _compute_phase_analysis(self):
        """Analyze early, middle, late phases of life history."""
        n = len(self.sequence)
        third = n // 3

        phases = {
            'early': self.sequence[:third] if third > 0 else self.sequence[:1],
            'middle': self.sequence[third:2*third] if third > 0 else self.sequence[1:2],
            'late': self.sequence[2*third:] if third > 0 else self.sequence[2:]
        }

        self.phase_distributions = {}
        for phase_name, phase_seq in phases.items():
            if len(phase_seq) > 0:
                counts = np.bincount(phase_seq, minlength=4)
                self.phase_distributions[phase_name] = counts / counts.sum()
            else:
                self.phase_distributions[phase_name] = np.zeros(4)

        # Detect escalation (increase in Directing over time)
        early_directing = self.phase_distributions['early'][1]
        late_directing = self.phase_distributions['late'][1]
        self.escalation_score = late_directing - early_directing
        self.shows_escalation = self.escalation_score > 0.1

    def get_characteristic_events(self, n_per_state: int = 3) -> Dict[str, List[str]]:
        """Get characteristic events for each state."""
        state_events = defaultdict(list)

        for i, (state, event) in enumerate(zip(self.sequence, self.events)):
            state_name = ANIMAL_NAMES[state]
            state_events[state_name].append(event)

        # Return top events for each state (by position - earlier = more formative)
        return {state: events[:n_per_state] for state, events in state_events.items()}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'n_events': self.n_events,
            'dominant_state': self.dominant_state,
            'dominant_state_pct': float(self.dominant_state_pct),
            'state_distribution': {ANIMAL_NAMES[i]: float(self.state_distribution[i])
                                   for i in range(4)},
            'dominant_transition': list(self.dominant_transition),
            'dominant_transition_prob': float(self.dominant_transition_prob),
            'top_bigrams': [(list(b), c) for b, c in self.top_bigrams],
            'top_trigrams': [(list(t), c) for t, c in self.top_trigrams],
            'escalation_score': float(self.escalation_score),
            'shows_escalation': self.shows_escalation,
            'phase_distributions': {
                phase: {ANIMAL_NAMES[i]: float(dist[i]) for i in range(4)}
                for phase, dist in self.phase_distributions.items()
            }
        }


# =============================================================================
# ARCHETYPE PROFILES
# =============================================================================

def create_archetype_profile(
    signatures: List[BehavioralSignature],
    role_name: str
) -> Dict:
    """
    Create an aggregate profile for a group of criminals sharing an archetypal role.
    """
    if not signatures:
        return {}

    # Aggregate state distributions
    all_distributions = np.array([s.state_distribution for s in signatures])
    mean_distribution = all_distributions.mean(axis=0)
    std_distribution = all_distributions.std(axis=0)

    # Aggregate transition matrices
    all_matrices = np.array([s.transition_matrix for s in signatures])
    mean_matrix = all_matrices.mean(axis=0)

    # Most common dominant states
    dominant_states = Counter([s.dominant_state for s in signatures])

    # Most common bigrams across all
    all_bigrams = []
    for s in signatures:
        all_bigrams.extend([b for b, c in s.top_bigrams])
    common_bigrams = Counter(all_bigrams).most_common(5)

    # Most common trigrams
    all_trigrams = []
    for s in signatures:
        all_trigrams.extend([t for t, c in s.top_trigrams])
    common_trigrams = Counter(all_trigrams).most_common(5)

    # Escalation analysis
    escalation_scores = [s.escalation_score for s in signatures]
    pct_escalating = sum(1 for s in signatures if s.shows_escalation) / len(signatures)

    # Phase patterns
    phase_patterns = {}
    for phase in ['early', 'middle', 'late']:
        phase_dists = np.array([s.phase_distributions[phase] for s in signatures])
        phase_patterns[phase] = {
            'mean': {ANIMAL_NAMES[i]: float(phase_dists[:, i].mean()) for i in range(4)},
            'dominant': ANIMAL_NAMES[np.argmax(phase_dists.mean(axis=0))]
        }

    # Generate narrative interpretation
    narrative = generate_archetype_narrative(
        role_name, mean_distribution, mean_matrix,
        common_bigrams, phase_patterns, pct_escalating
    )

    return {
        'role': role_name,
        'n_members': len(signatures),
        'members': [s.name for s in signatures],
        'state_distribution': {
            'mean': {ANIMAL_NAMES[i]: float(mean_distribution[i]) for i in range(4)},
            'std': {ANIMAL_NAMES[i]: float(std_distribution[i]) for i in range(4)}
        },
        'dominant_states': dict(dominant_states),
        'common_bigrams': [(list(b), c) for b, c in common_bigrams],
        'common_trigrams': [(list(t), c) for t, c in common_trigrams],
        'escalation': {
            'mean_score': float(np.mean(escalation_scores)),
            'pct_escalating': float(pct_escalating)
        },
        'phase_patterns': phase_patterns,
        'transition_matrix': {
            ANIMAL_NAMES[i]: {ANIMAL_NAMES[j]: float(mean_matrix[i, j]) for j in range(4)}
            for i in range(4)
        },
        'narrative': narrative
    }


def generate_archetype_narrative(
    role_name: str,
    mean_distribution: np.ndarray,
    mean_matrix: np.ndarray,
    common_bigrams: List,
    phase_patterns: Dict,
    pct_escalating: float
) -> Dict:
    """Generate human-readable narrative for an archetype."""

    # Dominant state interpretation
    dominant_idx = np.argmax(mean_distribution)
    dominant_state = ANIMAL_NAMES[dominant_idx]
    dominant_pct = mean_distribution[dominant_idx] * 100

    # Key transitions
    key_transitions = []
    for i in range(4):
        for j in range(4):
            if mean_matrix[i, j] > 0.3:  # Significant transition
                trans = (ANIMAL_NAMES[i], ANIMAL_NAMES[j])
                interp = TRANSITION_INTERPRETATIONS.get(trans, "Unknown pattern")
                key_transitions.append({
                    'from': trans[0],
                    'to': trans[1],
                    'probability': float(mean_matrix[i, j]),
                    'interpretation': interp
                })

    # Phase narrative
    early_dominant = phase_patterns['early']['dominant']
    late_dominant = phase_patterns['late']['dominant']

    if early_dominant != late_dominant:
        trajectory = f"Trajectory shifts from {early_dominant} to {late_dominant}"
    else:
        trajectory = f"Consistent {early_dominant} pattern throughout"

    # Behavioral signature description
    signature_parts = []
    if mean_distribution[1] > 0.5:  # High Directing
        signature_parts.append("dominated by control/exploitation behaviors")
    if mean_distribution[0] > 0.2:  # Notable Seeking
        signature_parts.append("significant introspective phases")
    if mean_distribution[2] > 0.15:  # Notable Conferring
        signature_parts.append("active surveillance/observation periods")
    if mean_distribution[3] > 0.1:  # Notable Revising
        signature_parts.append("ritualistic/compulsive elements")

    signature_desc = ", ".join(signature_parts) if signature_parts else "mixed behavioral pattern"

    # Escalation interpretation
    if pct_escalating > 0.7:
        escalation_desc = "Strong escalation pattern - Directing behavior increases significantly over time"
    elif pct_escalating > 0.4:
        escalation_desc = "Moderate escalation - some increase in Directing behavior"
    else:
        escalation_desc = "Stable or variable pattern - no consistent escalation"

    return {
        'summary': f"{role_name} archetype: {dominant_state} dominant ({dominant_pct:.1f}%), {signature_desc}",
        'dominant_behavior': STATE_INTERPRETATIONS[dominant_state],
        'key_transitions': key_transitions,
        'trajectory': trajectory,
        'escalation': escalation_desc,
        'risk_profile': STATE_INTERPRETATIONS[dominant_state]['risk_indicator']
    }


# =============================================================================
# ARCHETYPE CARDS
# =============================================================================

def create_archetype_card(profile: Dict, output_path: str):
    """Create a visual archetype card."""
    fig = plt.figure(figsize=(12, 10))

    # Title
    fig.suptitle(f"ARCHETYPE CARD: {profile['role'].upper()}",
                fontsize=16, fontweight='bold', y=0.98)

    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Panel 1: State Distribution (pie)
    ax1 = fig.add_subplot(gs[0, 0])
    states = list(profile['state_distribution']['mean'].keys())
    values = [profile['state_distribution']['mean'][s] for s in states]
    colors = [ANIMAL_COLORS[s] for s in states]

    wedges, texts, autotexts = ax1.pie(values, labels=states, colors=colors,
                                        autopct='%1.0f%%', startangle=90)
    ax1.set_title('State Distribution', fontweight='bold', fontsize=10)

    # Panel 2: Phase Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    phases = ['early', 'middle', 'late']
    x = np.arange(len(phases))
    width = 0.2

    for i, state in enumerate(states):
        values = [profile['phase_patterns'][p]['mean'][state] for p in phases]
        ax2.bar(x + i*width, values, width, label=state, color=ANIMAL_COLORS[state])

    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels(['Early Life', 'Middle', 'Late'])
    ax2.set_ylabel('Proportion')
    ax2.set_title('Behavioral Evolution', fontweight='bold', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 1)

    # Panel 3: Key Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    metrics_text = f"""
    MEMBERS: {profile['n_members']}

    ESCALATION:
    • {profile['escalation']['pct_escalating']*100:.0f}% show escalation
    • Mean score: {profile['escalation']['mean_score']:.2f}

    DOMINANT STATES:
    """
    for state, count in profile['dominant_states'].items():
        metrics_text += f"\n    • {state}: {count}"

    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')
    ax3.set_title('Key Metrics', fontweight='bold', fontsize=10)

    # Panel 4: Transition Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    matrix_data = np.array([[profile['transition_matrix'][s1][s2]
                            for s2 in states] for s1 in states])
    im = ax4.imshow(matrix_data, cmap='RdYlBu_r', vmin=0, vmax=0.8)
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels([s[:4] for s in states], fontsize=8)
    ax4.set_yticklabels([s[:4] for s in states], fontsize=8)

    for i in range(4):
        for j in range(4):
            color = 'white' if matrix_data[i,j] > 0.4 else 'black'
            ax4.text(j, i, f'{matrix_data[i,j]:.2f}', ha='center', va='center',
                    color=color, fontsize=8)

    ax4.set_title('Transition Matrix', fontweight='bold', fontsize=10)

    # Panel 5: Common Patterns
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    patterns_text = "COMMON SEQUENCES:\n\n"
    patterns_text += "2-grams:\n"
    for pattern, count in profile['common_bigrams'][:3]:
        patterns_text += f"  {pattern[0][:4]} → {pattern[1][:4]}: {count}x\n"

    patterns_text += "\n3-grams:\n"
    for pattern, count in profile['common_trigrams'][:3]:
        patterns_text += f"  {pattern[0][:4]} → {pattern[1][:4]} → {pattern[2][:4]}: {count}x\n"

    ax5.text(0.1, 0.9, patterns_text, transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')
    ax5.set_title('Common Patterns', fontweight='bold', fontsize=10)

    # Panel 6: Key Transitions Interpretation
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    trans_text = "KEY TRANSITIONS:\n\n"
    for trans in profile['narrative']['key_transitions'][:4]:
        trans_text += f"• {trans['from'][:4]} → {trans['to'][:4]} ({trans['probability']:.0%})\n"
        trans_text += f"  {trans['interpretation']}\n\n"

    ax6.text(0.05, 0.95, trans_text, transform=ax6.transAxes, fontsize=8,
            verticalalignment='top', wrap=True)
    ax6.set_title('Transition Interpretations', fontweight='bold', fontsize=10)

    # Panel 7-9: Narrative Summary (spanning bottom)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    narrative = profile['narrative']
    summary_text = f"""
    ARCHETYPE SUMMARY
    {'='*60}

    {narrative['summary']}

    BEHAVIORAL PROFILE:
    • Primary Mode: {narrative['dominant_behavior']['description']}
    • Psychological: {narrative['dominant_behavior']['psychological']}
    • Behavioral: {narrative['dominant_behavior']['behavioral']}

    TRAJECTORY: {narrative['trajectory']}

    ESCALATION: {narrative['escalation']}

    RISK INDICATOR: {narrative['risk_profile']}
    """

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='serif', wrap=True,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_data(results_dir: str) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
    """Load sequences and events."""
    results_dir = Path(results_dir)

    # Find latest comparison
    comparison_dirs = sorted(results_dir.glob("four_animal_comparison_*"))
    latest_dir = comparison_dirs[-1]

    with open(latest_dir / "animal_labels.json", 'r') as f:
        data = json.load(f)

    # Reconstruct sequences and events
    sequences = defaultdict(list)
    events = defaultdict(list)

    for label, criminal, event in zip(data['labels'], data['event_to_criminal'], data['events']):
        sequences[criminal].append(label)
        events[criminal].append(event)

    return dict(sequences), dict(events)


def load_reincarnation_results(results_dir: str) -> Dict:
    """Load reincarnation analysis results."""
    results_dir = Path(results_dir)

    reincarnation_dirs = sorted(results_dir.glob("archetypal_reincarnation_*"))
    if not reincarnation_dirs:
        return None

    latest_dir = reincarnation_dirs[-1]

    with open(latest_dir / "reincarnation_results.json", 'r') as f:
        return json.load(f)


def run_interpretation_analysis(results_dir: str, output_dir: str) -> Dict:
    """Run complete interpretation analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(output_dir) / f"archetypal_interpretation_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ARCHETYPAL INTERPRETATION ANALYSIS")
    logger.info("Making Patterns Trackable and Understandable")
    logger.info("=" * 70)

    # Load data
    logger.info("\n[1] Loading data...")
    sequences, events = load_data(results_dir)
    reincarnation = load_reincarnation_results(results_dir)

    logger.info(f"Loaded {len(sequences)} individuals")

    # Create behavioral signatures for all criminals
    logger.info("\n[2] Extracting behavioral signatures...")
    signatures = {}
    for name in sequences:
        if len(sequences[name]) >= 5:
            sig = BehavioralSignature(name, sequences[name], events[name])
            signatures[name] = sig

    logger.info(f"Created {len(signatures)} behavioral signatures")

    # Group by archetypal role if reincarnation results available
    if reincarnation:
        logger.info("\n[3] Creating archetype profiles by role...")

        roles = reincarnation['roles']
        profiles = {}

        # Sources
        source_names = [r['name'] for r in roles['sources']]
        source_sigs = [signatures[n] for n in source_names if n in signatures]
        profiles['SOURCES'] = create_archetype_profile(source_sigs, 'SOURCES')

        # Sinks
        sink_names = [r['name'] for r in roles['sinks']]
        sink_sigs = [signatures[n] for n in sink_names if n in signatures]
        profiles['SINKS'] = create_archetype_profile(sink_sigs, 'SINKS')

        # Hubs
        hub_names = [r['name'] for r in roles['hubs']]
        hub_sigs = [signatures[n] for n in hub_names if n in signatures]
        profiles['HUBS'] = create_archetype_profile(hub_sigs, 'HUBS')

        # All others
        role_names = set(source_names + sink_names + hub_names)
        other_sigs = [s for n, s in signatures.items() if n not in role_names]
        profiles['GENERAL'] = create_archetype_profile(other_sigs, 'GENERAL')

    else:
        # Create single aggregate profile
        profiles = {
            'ALL': create_archetype_profile(list(signatures.values()), 'ALL')
        }

    # Print summary
    print("\n" + "=" * 70)
    print("ARCHETYPE PROFILES")
    print("=" * 70)

    for role, profile in profiles.items():
        if not profile:
            continue

        print(f"\n{role} ({profile['n_members']} members)")
        print("-" * 50)
        print(f"Narrative: {profile['narrative']['summary']}")
        print(f"Trajectory: {profile['narrative']['trajectory']}")
        print(f"Escalation: {profile['narrative']['escalation']}")
        print(f"Risk: {profile['narrative']['risk_profile']}")

    # Generate archetype cards
    logger.info("\n[4] Generating archetype cards...")
    for role, profile in profiles.items():
        if profile:
            create_archetype_card(profile, study_dir / f'archetype_card_{role.lower()}.png')
            logger.info(f"  Created card for {role}")

    # Save individual signatures
    logger.info("\n[5] Saving detailed results...")

    all_signatures = {name: sig.to_dict() for name, sig in signatures.items()}

    # Get characteristic events for each person
    characteristic_events = {}
    for name, sig in signatures.items():
        characteristic_events[name] = sig.get_characteristic_events(n_per_state=5)

    results = {
        'profiles': profiles,
        'signatures': all_signatures,
        'characteristic_events': characteristic_events
    }

    # Convert to JSON-serializable format
    results_serializable = to_serializable(results)

    with open(study_dir / 'interpretation_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Create summary markdown
    create_summary_markdown(profiles, study_dir / 'ARCHETYPE_GUIDE.md')

    logger.info(f"\nResults saved to: {study_dir}")

    return results


def create_summary_markdown(profiles: Dict, output_path: Path):
    """Create a markdown guide to the archetypes."""

    md = """# Archetypal Pattern Guide

## Understanding the 4-Animal State Space

The behavioral patterns are categorized into four states based on two dimensions:
- **Self/Other**: Focus of attention
- **Explore/Exploit**: Behavioral mode

| State | Dimensions | Description | Risk Indicator |
|-------|------------|-------------|----------------|
| **Seeking** | Self + Explore | Introspection, fantasy | Building pressure |
| **Directing** | Other + Exploit | Control, violence | Active danger |
| **Conferring** | Other + Explore | Observation, stalking | Target selection |
| **Revising** | Self + Exploit | Rituals, habits | Predictable MO |

---

## Archetypal Roles

"""

    for role, profile in profiles.items():
        if not profile:
            continue

        md += f"""### {role}

**Members**: {profile['n_members']}

**Summary**: {profile['narrative']['summary']}

**State Distribution**:
"""
        for state, pct in profile['state_distribution']['mean'].items():
            md += f"- {state}: {pct*100:.1f}%\n"

        md += f"""
**Key Patterns**:
- Trajectory: {profile['narrative']['trajectory']}
- Escalation: {profile['narrative']['escalation']}
- Risk Profile: {profile['narrative']['risk_profile']}

**Common Sequences**:
"""
        for pattern, count in profile['common_bigrams'][:3]:
            interp = TRANSITION_INTERPRETATIONS.get(tuple(pattern), "")
            md += f"- {pattern[0]} → {pattern[1]} ({count}x) - {interp}\n"

        md += "\n---\n\n"

    md += """
## Tracking These Patterns

To use this framework for analysis:

1. **Classify events** using the 4-Animal State Space
2. **Compute behavioral signature** (state distribution, transitions)
3. **Identify role** (Source, Sink, Hub) based on influence patterns
4. **Track escalation** by comparing early vs late phase distributions
5. **Monitor key transitions** especially Seeking→Directing (fantasy-to-action)

## Warning Signs

High-risk transitions to monitor:
- **Seeking → Directing**: Fantasy escalating to action
- **Conferring → Directing**: Observation phase ending, action beginning
- **Persistent Directing**: Sustained predatory activity
- **Increasing Directing over time**: Clear escalation pattern
"""

    with open(output_path, 'w') as f:
        f.write(md)


if __name__ == "__main__":
    results_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"
    output_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"

    run_interpretation_analysis(results_dir, output_dir)
