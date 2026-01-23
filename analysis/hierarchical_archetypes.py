#!/usr/bin/env python3
"""
Hierarchical Archetype Classification System

A two-level archetype system that respects natural data structure:
- Level 1: Data-driven primary types (COMPLEX vs FOCUSED)
- Level 2: Theory-driven subtypes within each primary type

This approach combines empirical validation with theoretical interpretability.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# CONSTANTS
# =============================================================================

ANIMAL_NAMES = {0: "Seeking", 1: "Directing", 2: "Conferring", 3: "Revising"}
ANIMAL_COLORS = {
    'Seeking': '#2ecc71',
    'Directing': '#e74c3c',
    'Conferring': '#3498db',
    'Revising': '#9b59b6'
}

PRIMARY_COLORS = {
    'COMPLEX': '#8e44ad',  # Purple
    'FOCUSED': '#c0392b'   # Red
}

SUBTYPE_COLORS = {
    'Chameleon': '#9b59b6',
    'Multi-Modal': '#8e44ad',
    'Pure Predator': '#c0392b',
    'Strong Escalator': '#e74c3c',
    'Stalker-Striker': '#e67e22',
    'Fantasy-Actor': '#f39c12',
    'Standard': '#95a5a6'
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PrimaryType:
    """Level 1: Data-driven primary archetype type."""
    name: str
    description: str
    characteristics: Dict[str, float]  # Key metrics that define this type
    risk_level: str


@dataclass
class Subtype:
    """Level 2: Theory-driven subtype within a primary type."""
    name: str
    code: str
    parent: str  # "COMPLEX" or "FOCUSED"
    description: str
    criteria: Dict  # Quantitative criteria for matching
    psychology: str
    risk_profile: str


@dataclass
class HierarchicalClassification:
    """Complete hierarchical classification for an individual."""
    name: str
    primary_type: str
    subtype: str
    primary_confidence: float
    subtype_confidence: float
    signature: Dict

    def __str__(self):
        return f"{self.primary_type}/{self.subtype} ({self.primary_confidence:.0%}/{self.subtype_confidence:.0%})"

    def to_dict(self):
        return {
            'name': self.name,
            'primary_type': self.primary_type,
            'subtype': self.subtype,
            'primary_confidence': float(self.primary_confidence),
            'subtype_confidence': float(self.subtype_confidence),
            'full_label': f"{self.primary_type}/{self.subtype}"
        }


# =============================================================================
# PRIMARY TYPE DEFINITIONS
# =============================================================================

PRIMARY_TYPES = {
    'COMPLEX': PrimaryType(
        name="COMPLEX",
        description="Multi-modal behavioral pattern with unpredictable state transitions",
        characteristics={
            'directing_pct': 0.51,  # Lower than FOCUSED
            'directing_persistence': 0.38,  # Lower persistence
            'escalation': 0.0,  # Minimal escalation
            'n_active_states': 3.0  # Uses more states
        },
        risk_level="UNPREDICTABLE - Difficult to anticipate; utilizes multiple behavioral modes"
    ),
    'FOCUSED': PrimaryType(
        name="FOCUSED",
        description="Directing-dominant pattern with clear escalation trajectory",
        characteristics={
            'directing_pct': 0.72,  # High Directing
            'directing_persistence': 0.65,  # High persistence
            'escalation': 0.32,  # Strong escalation
            'n_active_states': 2.0  # More focused
        },
        risk_level="HIGH - Predictable escalation pattern; sustained exploitation behavior"
    )
}


# =============================================================================
# SUBTYPE DEFINITIONS
# =============================================================================

SUBTYPES = [
    # COMPLEX subtypes
    Subtype(
        name="Chameleon",
        code="CHA",
        parent="COMPLEX",
        description="Utilizes all four behavioral modes fluidly",
        criteria={
            'min_active_states': 3,
            'max_dominant_pct': 0.60
        },
        psychology="Highly adaptive predator. Can shift between fantasy, surveillance, "
                   "action, and ritual modes. Most unpredictable type.",
        risk_profile="UNPREDICTABLE - No consistent pattern to monitor. "
                     "Can switch modes rapidly without warning."
    ),
    Subtype(
        name="Multi-Modal",
        code="MUL",
        parent="COMPLEX",
        description="Multiple active states without full chameleon flexibility",
        criteria={
            'min_active_states': 2,
            'max_dominant_pct': 0.65
        },
        psychology="Less extreme than Chameleon but still unpredictable. "
                   "May have preferred modes but doesn't fully commit.",
        risk_profile="MODERATE-HIGH - Some pattern variability. "
                     "Harder to profile than FOCUSED types."
    ),

    # FOCUSED subtypes
    Subtype(
        name="Pure Predator",
        code="PRD",
        parent="FOCUSED",
        description="Overwhelming dominance of exploitation behavior (75%+)",
        criteria={
            'min_directing_pct': 0.75,
            'has_directing_self_loop': True
        },
        psychology="Pure exploitation focus. Minimal internal conflict or planning. "
                   "Direct, sustained predatory behavior with high persistence.",
        risk_profile="CRITICAL - Highest danger. Once in Directing mode, "
                     "76%+ probability of staying there."
    ),
    Subtype(
        name="Strong Escalator",
        code="ESC",
        parent="FOCUSED",
        description="Dramatic increase in Directing behavior over time",
        criteria={
            'min_escalation': 0.35,
            'shows_escalation': True
        },
        psychology="Clear trajectory from lower to higher Directing. "
                   "May start with more exploration/fantasy, ends in sustained action.",
        risk_profile="HIGH - Escalation is the key warning sign. "
                     "Early detection critical before full escalation."
    ),
    Subtype(
        name="Stalker-Striker",
        code="STK",
        parent="FOCUSED",
        description="Methodical observation followed by calculated action",
        criteria={
            'has_conferring_to_directing': True,
            'min_directing_pct': 0.40
        },
        psychology="Patient, methodical predator. Extended surveillance and "
                   "target selection before striking. Organized type.",
        risk_profile="HIGH - Surveillance phase offers detection opportunity. "
                     "Once target selected, action follows."
    ),
    Subtype(
        name="Fantasy-Actor",
        code="FAN",
        parent="FOCUSED",
        description="Direct leap from internal fantasy to violent action",
        criteria={
            'has_seeking_to_directing': True,
            'no_conferring_to_directing': True
        },
        psychology="Rapid fantasy-to-action without surveillance phase. "
                   "Impulsive translation of internal urges into violence.",
        risk_profile="CRITICAL - Fast escalation. Short window for intervention "
                     "between fantasy development and action."
    ),
    Subtype(
        name="Standard",
        code="STD",
        parent="FOCUSED",
        description="Typical Directing-dominant pattern without distinctive features",
        criteria={
            'min_directing_pct': 0.50
        },
        psychology="Standard criminal pattern. Directing dominant but without "
                   "extreme features of other subtypes.",
        risk_profile="HIGH - Typical danger profile. "
                     "Follows expected Directing-dominant trajectory."
    )
]


# =============================================================================
# CLASSIFIER
# =============================================================================

class HierarchicalArchetypeClassifier:
    """
    Two-level hierarchical archetype classifier.

    Level 1 uses data-driven clustering to separate COMPLEX from FOCUSED.
    Level 2 uses theory-driven criteria to assign subtypes.
    """

    def __init__(self):
        self.primary_types = PRIMARY_TYPES
        self.subtypes = {s.name: s for s in SUBTYPES}
        self.fitted = False
        self.cluster_centers_ = None
        self.complex_cluster_id_ = None

    def _extract_features(self, signature: Dict) -> np.ndarray:
        """Extract feature vector from a behavioral signature."""
        # State distribution (4 features)
        state_dist = [
            signature['state_distribution'].get(ANIMAL_NAMES[i], 0)
            for i in range(4)
        ]

        # Self-loops from bigrams (4 features)
        bigrams = {}
        for b in signature.get('top_bigrams', []):
            key = tuple(b[0]) if isinstance(b[0], list) else b[0]
            bigrams[key] = b[1]

        total_trans = sum(bigrams.values()) if bigrams else 1
        self_loops = [
            bigrams.get((ANIMAL_NAMES[i], ANIMAL_NAMES[i]), 0) / total_trans
            for i in range(4)
        ]

        # Escalation (1 feature)
        escalation = [signature.get('escalation_score', 0)]

        return np.array(state_dist + self_loops + escalation)

    def fit(self, signatures: List[Dict]):
        """
        Fit the Level 1 classifier using hierarchical clustering.

        This determines the COMPLEX vs FOCUSED boundary from data.
        """
        features = np.array([self._extract_features(s) for s in signatures])

        # Hierarchical clustering with k=2
        distances = pdist(features, metric='euclidean')
        Z = linkage(distances, method='ward')
        labels = fcluster(Z, 2, criterion='maxclust')

        # Determine which cluster is COMPLEX (lower Directing mean)
        c1_directing = features[labels == 1][:, 1].mean()
        c2_directing = features[labels == 2][:, 1].mean()

        if c1_directing < c2_directing:
            self.complex_cluster_id_ = 1
        else:
            self.complex_cluster_id_ = 2

        # Store cluster centers for future classification
        self.cluster_centers_ = {
            'COMPLEX': features[labels == self.complex_cluster_id_].mean(axis=0),
            'FOCUSED': features[labels != self.complex_cluster_id_].mean(axis=0)
        }

        self.fitted = True
        return self

    def _classify_primary(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify into primary type (COMPLEX or FOCUSED)."""
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        # Distance to each cluster center
        dist_complex = np.linalg.norm(features - self.cluster_centers_['COMPLEX'])
        dist_focused = np.linalg.norm(features - self.cluster_centers_['FOCUSED'])

        # Assign to closer cluster
        if dist_complex < dist_focused:
            primary = 'COMPLEX'
            # Confidence based on relative distance
            confidence = dist_focused / (dist_complex + dist_focused)
        else:
            primary = 'FOCUSED'
            confidence = dist_complex / (dist_complex + dist_focused)

        return primary, confidence

    def _classify_subtype(self, signature: Dict, primary: str) -> Tuple[str, float]:
        """Classify into subtype within the primary type."""
        # Get subtypes for this primary type
        candidate_subtypes = [s for s in SUBTYPES if s.parent == primary]

        best_subtype = None
        best_score = -1

        # Extract key metrics
        directing_pct = signature.get('dominant_state_pct', 0)
        if signature.get('dominant_state') != 'Directing':
            directing_pct = signature['state_distribution'].get('Directing', 0)

        escalation = signature.get('escalation_score', 0)
        shows_escalation = signature.get('shows_escalation', False)

        n_active = sum(1 for v in signature['state_distribution'].values() if v > 0.1)

        # Extract transitions
        bigrams = {}
        for b in signature.get('top_bigrams', []):
            key = tuple(b[0]) if isinstance(b[0], list) else b[0]
            bigrams[key] = b[1]

        has_seeking_to_directing = bigrams.get(('Seeking', 'Directing'), 0) > 0
        has_conferring_to_directing = bigrams.get(('Conferring', 'Directing'), 0) > 0
        has_directing_self_loop = bigrams.get(('Directing', 'Directing'), 0) > 0

        for subtype in candidate_subtypes:
            score = 0
            max_score = 0
            criteria = subtype.criteria

            # Check each criterion
            if 'min_active_states' in criteria:
                max_score += 1
                if n_active >= criteria['min_active_states']:
                    score += 1

            if 'max_dominant_pct' in criteria:
                max_score += 1
                if directing_pct <= criteria['max_dominant_pct']:
                    score += 1

            if 'min_directing_pct' in criteria:
                max_score += 1
                if directing_pct >= criteria['min_directing_pct']:
                    score += 1
                elif directing_pct >= criteria['min_directing_pct'] - 0.1:
                    score += 0.5

            if 'min_escalation' in criteria:
                max_score += 1
                if escalation >= criteria['min_escalation']:
                    score += 1
                elif escalation >= criteria['min_escalation'] - 0.1:
                    score += 0.5

            if 'shows_escalation' in criteria:
                max_score += 1
                if shows_escalation == criteria['shows_escalation']:
                    score += 1

            if 'has_conferring_to_directing' in criteria:
                max_score += 1
                if has_conferring_to_directing:
                    score += 1

            if 'has_seeking_to_directing' in criteria:
                max_score += 1
                if has_seeking_to_directing:
                    score += 1

            if 'no_conferring_to_directing' in criteria:
                max_score += 1
                if not has_conferring_to_directing:
                    score += 1

            if 'has_directing_self_loop' in criteria:
                max_score += 1
                if has_directing_self_loop:
                    score += 1

            # Calculate confidence
            confidence = score / max_score if max_score > 0 else 0

            if confidence > best_score:
                best_score = confidence
                best_subtype = subtype.name

        # Default to Standard/Multi-Modal if no good match
        if best_score < 0.5:
            if primary == 'COMPLEX':
                best_subtype = 'Multi-Modal'
            else:
                best_subtype = 'Standard'
            best_score = 0.5

        return best_subtype, best_score

    def classify(self, signature: Dict) -> HierarchicalClassification:
        """
        Classify a single signature into the hierarchical system.
        """
        features = self._extract_features(signature)

        # Level 1: Primary type
        primary, primary_conf = self._classify_primary(features)

        # Level 2: Subtype
        subtype, subtype_conf = self._classify_subtype(signature, primary)

        return HierarchicalClassification(
            name=signature['name'],
            primary_type=primary,
            subtype=subtype,
            primary_confidence=primary_conf,
            subtype_confidence=subtype_conf,
            signature=signature
        )

    def classify_all(self, signatures: List[Dict]) -> List[HierarchicalClassification]:
        """Classify all signatures."""
        return [self.classify(s) for s in signatures]

    def get_summary(self, classifications: List[HierarchicalClassification]) -> Dict:
        """Generate summary statistics for classifications."""
        primary_counts = defaultdict(int)
        subtype_counts = defaultdict(lambda: defaultdict(int))

        for c in classifications:
            primary_counts[c.primary_type] += 1
            subtype_counts[c.primary_type][c.subtype] += 1

        return {
            'total': len(classifications),
            'primary_distribution': dict(primary_counts),
            'subtype_distribution': {k: dict(v) for k, v in subtype_counts.items()},
            'primary_percentages': {
                k: v / len(classifications) * 100
                for k, v in primary_counts.items()
            }
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_hierarchical_classification(
    classifications: List[HierarchicalClassification],
    output_path: Path
):
    """Create visualization of hierarchical classification results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Primary type distribution (pie chart)
    ax = axes[0, 0]
    primary_counts = defaultdict(int)
    for c in classifications:
        primary_counts[c.primary_type] += 1

    colors = [PRIMARY_COLORS[k] for k in primary_counts.keys()]
    wedges, texts, autotexts = ax.pie(
        primary_counts.values(),
        labels=primary_counts.keys(),
        autopct='%1.0f%%',
        colors=colors,
        explode=[0.05] * len(primary_counts)
    )
    ax.set_title('Level 1: Primary Type Distribution', fontsize=12, fontweight='bold')

    # 2. Subtype distribution within FOCUSED (bar chart)
    ax = axes[0, 1]
    focused_subtypes = defaultdict(int)
    for c in classifications:
        if c.primary_type == 'FOCUSED':
            focused_subtypes[c.subtype] += 1

    if focused_subtypes:
        bars = ax.bar(
            range(len(focused_subtypes)),
            focused_subtypes.values(),
            color=[SUBTYPE_COLORS.get(k, '#95a5a6') for k in focused_subtypes.keys()]
        )
        ax.set_xticks(range(len(focused_subtypes)))
        ax.set_xticklabels(focused_subtypes.keys(), rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Level 2: FOCUSED Subtypes', fontsize=12, fontweight='bold')

        # Add count labels
        for bar, count in zip(bars, focused_subtypes.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')

    # 3. Scatter plot: Directing % vs Escalation, colored by primary type
    ax = axes[1, 0]
    for c in classifications:
        directing_pct = c.signature['state_distribution'].get('Directing', 0) * 100
        escalation = c.signature.get('escalation_score', 0)
        color = PRIMARY_COLORS[c.primary_type]
        ax.scatter(directing_pct, escalation, c=color, s=100, alpha=0.7,
                  edgecolors='white', linewidth=1)

    ax.set_xlabel('Directing %')
    ax.set_ylabel('Escalation Score')
    ax.set_title('Classification Space', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # Add legend
    legend_handles = [
        mpatches.Patch(color=PRIMARY_COLORS['COMPLEX'], label='COMPLEX'),
        mpatches.Patch(color=PRIMARY_COLORS['FOCUSED'], label='FOCUSED')
    ]
    ax.legend(handles=legend_handles, loc='upper left')

    # 4. Hierarchical tree visualization
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw tree structure
    # Root
    ax.text(5, 9.5, 'ARCHETYPES', ha='center', fontsize=14, fontweight='bold')

    # Level 1
    complex_count = sum(1 for c in classifications if c.primary_type == 'COMPLEX')
    focused_count = sum(1 for c in classifications if c.primary_type == 'FOCUSED')

    ax.annotate('', xy=(2.5, 7.5), xytext=(5, 9),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('', xy=(7.5, 7.5), xytext=(5, 9),
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.add_patch(plt.Rectangle((1, 6.5), 3, 1.5, facecolor=PRIMARY_COLORS['COMPLEX'],
                               edgecolor='black', alpha=0.8))
    ax.text(2.5, 7.25, f'COMPLEX\n({complex_count})', ha='center', va='center',
            color='white', fontweight='bold')

    ax.add_patch(plt.Rectangle((6, 6.5), 3, 1.5, facecolor=PRIMARY_COLORS['FOCUSED'],
                               edgecolor='black', alpha=0.8))
    ax.text(7.5, 7.25, f'FOCUSED\n({focused_count})', ha='center', va='center',
            color='white', fontweight='bold')

    # Level 2 - COMPLEX subtypes
    complex_subtypes = defaultdict(int)
    for c in classifications:
        if c.primary_type == 'COMPLEX':
            complex_subtypes[c.subtype] += 1

    y_pos = 4.5
    for i, (subtype, count) in enumerate(complex_subtypes.items()):
        x = 1.5 + i * 1.5
        ax.annotate('', xy=(x, y_pos + 0.5), xytext=(2.5, 6.5),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        ax.text(x, y_pos, f'{subtype}\n({count})', ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor=SUBTYPE_COLORS.get(subtype, '#ccc'), alpha=0.7))

    # Level 2 - FOCUSED subtypes
    focused_subtypes_dict = defaultdict(int)
    for c in classifications:
        if c.primary_type == 'FOCUSED':
            focused_subtypes_dict[c.subtype] += 1

    y_pos = 4.5
    for i, (subtype, count) in enumerate(sorted(focused_subtypes_dict.items(),
                                                 key=lambda x: -x[1])):
        x = 5.5 + i * 1.0
        if x < 10:
            ax.annotate('', xy=(x, y_pos + 0.5), xytext=(7.5, 6.5),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
            ax.text(x, y_pos, f'{subtype[:8]}\n({count})', ha='center', fontsize=7,
                   bbox=dict(boxstyle='round',
                            facecolor=SUBTYPE_COLORS.get(subtype, '#ccc'), alpha=0.7))

    ax.set_title('Hierarchical Structure', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_classification_report(
    classifications: List[HierarchicalClassification],
    output_path: Path
):
    """Create detailed markdown report of classifications."""

    md = """# Hierarchical Archetype Classification Report

## Overview

This report presents the results of the two-level hierarchical archetype classification system.

### Classification Approach

**Level 1 (Data-Driven):** Primary types determined by hierarchical clustering
- Validated by silhouette score analysis (k=2 optimal)
- COMPLEX vs FOCUSED distinction emerges naturally from data

**Level 2 (Theory-Driven):** Subtypes based on 4-Animal framework
- Refinements within each primary type
- Criteria based on state distributions, transitions, and escalation patterns

---

## Results Summary

"""

    # Summary statistics
    summary = defaultdict(lambda: defaultdict(list))
    for c in classifications:
        summary[c.primary_type][c.subtype].append(c.name)

    total = len(classifications)

    for primary in ['COMPLEX', 'FOCUSED']:
        if primary in summary:
            count = sum(len(v) for v in summary[primary].values())
            pct = count / total * 100

            md += f"\n### {primary} ({count} individuals, {pct:.0f}%)\n\n"
            md += f"*{PRIMARY_TYPES[primary].description}*\n\n"
            md += f"**Risk Level:** {PRIMARY_TYPES[primary].risk_level}\n\n"

            md += "| Subtype | Count | Members |\n"
            md += "|---------|-------|--------|\n"

            for subtype, members in sorted(summary[primary].items(), key=lambda x: -len(x[1])):
                member_str = ', '.join([m.split('_')[0] for m in members[:5]])
                if len(members) > 5:
                    member_str += f', ... (+{len(members)-5} more)'
                md += f"| {subtype} | {len(members)} | {member_str} |\n"

    # Detailed individual classifications
    md += "\n---\n\n## Individual Classifications\n\n"
    md += "| Name | Primary | Subtype | Primary Conf. | Subtype Conf. |\n"
    md += "|------|---------|---------|---------------|---------------|\n"

    for c in sorted(classifications, key=lambda x: (x.primary_type, x.subtype, x.name)):
        short_name = c.name.split('_')[0] + '_' + c.name.split('_')[1] if '_' in c.name else c.name
        md += f"| {short_name[:25]} | {c.primary_type} | {c.subtype} | {c.primary_confidence:.0%} | {c.subtype_confidence:.0%} |\n"

    # Methodology
    md += """
---

## Methodology

### Level 1: Primary Type Classification

Primary types are determined using Ward's hierarchical clustering on a 9-dimensional feature space:
- State distribution (4 features): % in each of the 4 Animal states
- State persistence (4 features): Self-loop probability for each state
- Escalation (1 feature): Change in Directing % from early to late phases

The optimal number of clusters (k=2) was determined by silhouette analysis.

### Level 2: Subtype Classification

Within each primary type, individuals are classified into subtypes based on:

**COMPLEX Subtypes:**
- **Chameleon**: ≥3 active states, <60% in any single state
- **Multi-Modal**: 2+ active states, default for COMPLEX without Chameleon criteria

**FOCUSED Subtypes:**
- **Pure Predator**: ≥75% Directing, high persistence
- **Strong Escalator**: Escalation score ≥0.35
- **Stalker-Striker**: Has Conferring→Directing transitions
- **Fantasy-Actor**: Has Seeking→Directing but NOT Conferring→Directing
- **Standard**: Meets FOCUSED criteria without distinctive subtype features

---

*Generated by Hierarchical Archetype Classification System*
"""

    with open(output_path, 'w') as f:
        f.write(md)

    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_hierarchical_classification(results_dir: str, output_dir: str) -> Dict:
    """Run complete hierarchical classification analysis."""
    from datetime import datetime

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = output_dir / f"hierarchical_archetypes_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HIERARCHICAL ARCHETYPE CLASSIFICATION")
    print("Two-Level System: Data-Driven + Theory-Driven")
    print("=" * 70)

    # Load signatures
    print("\n[1] Loading behavioral signatures...")
    interp_dirs = sorted(results_dir.glob("archetypal_interpretation_*"))
    if not interp_dirs:
        raise FileNotFoundError("No interpretation results found. Run archetypal_interpretation.py first.")

    latest_dir = interp_dirs[-1]
    with open(latest_dir / "interpretation_results.json", 'r') as f:
        data = json.load(f)

    signatures = list(data['signatures'].values())
    print(f"  Loaded {len(signatures)} signatures")

    # Initialize and fit classifier
    print("\n[2] Fitting Level 1 classifier (data-driven clustering)...")
    classifier = HierarchicalArchetypeClassifier()
    classifier.fit(signatures)
    print("  Fitted using hierarchical clustering (k=2)")

    # Classify all
    print("\n[3] Classifying all individuals...")
    classifications = classifier.classify_all(signatures)

    # Get summary
    summary = classifier.get_summary(classifications)

    # Print results
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    print(f"\nLevel 1 - Primary Types:")
    for primary, count in summary['primary_distribution'].items():
        pct = summary['primary_percentages'][primary]
        print(f"  {primary}: {count} ({pct:.0f}%)")

    print(f"\nLevel 2 - Subtypes:")
    for primary, subtypes in summary['subtype_distribution'].items():
        print(f"\n  {primary}:")
        for subtype, count in sorted(subtypes.items(), key=lambda x: -x[1]):
            print(f"    {subtype}: {count}")

    # Generate visualizations
    print("\n[4] Generating visualizations...")
    viz_path = visualize_hierarchical_classification(classifications, study_dir / 'classification_overview.png')
    print(f"  Saved: {viz_path}")

    # Generate report
    print("\n[5] Generating classification report...")
    report_path = create_classification_report(classifications, study_dir / 'CLASSIFICATION_REPORT.md')
    print(f"  Saved: {report_path}")

    # Save JSON results
    results = {
        'summary': summary,
        'classifications': [c.to_dict() for c in classifications],
        'primary_types': {k: {'description': v.description, 'risk_level': v.risk_level}
                         for k, v in PRIMARY_TYPES.items()},
        'subtypes': {s.name: {'code': s.code, 'parent': s.parent,
                              'description': s.description, 'risk_profile': s.risk_profile}
                    for s in SUBTYPES}
    }

    with open(study_dir / 'classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {study_dir}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    results_dir = "empirical_study"
    output_dir = "empirical_study"
    run_hierarchical_classification(results_dir, output_dir)
