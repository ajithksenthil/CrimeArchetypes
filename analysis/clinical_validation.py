#!/usr/bin/env python3
"""
Clinical Validation: Big-5 Trait Proxy Extraction and Validation

Derives Big-5 personality trait proxies from 4-Animal behavioral sequences
based on Computational Psychodynamics theory, and provides validation metrics.

Theoretical Mapping (from Computational Psychodynamics):
- Self/Other dimension → Introversion/Extraversion
- Explore/Exploit dimension → Openness (high explore) vs Conscientiousness (high exploit)
- Transition entropy → Neuroticism (high entropy = instability)
- Directing prevalence → Low Agreeableness (exploitation of others)

Note: These are behavioral PROXIES derived from life events, not clinical assessments.
Validation requires comparison with actual clinical data when available.
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

ANIMAL_NAMES = {0: "Seeking", 1: "Directing", 2: "Conferring", 3: "Revising"}


# =============================================================================
# BIG-5 TRAIT PROXY COMPUTATION
# =============================================================================

class Big5TraitExtractor:
    """
    Extract Big-5 personality trait proxies from 4-Animal behavioral sequences.

    Based on the Computational Psychodynamics mapping:
    - Openness: Preference for exploration (Seeking + Conferring)
    - Conscientiousness: Preference for exploitation/structure (Directing + Revising)
    - Extraversion: Other-orientation (Directing + Conferring)
    - Agreeableness: Inverse of exploitation of others (inverse of Directing)
    - Neuroticism: Behavioral instability (transition entropy)
    """

    def __init__(self, n_states: int = 4):
        self.n_states = n_states

    def compute_state_distribution(self, sequence: List[int]) -> np.ndarray:
        """Compute state distribution for a sequence."""
        counts = np.bincount(sequence, minlength=self.n_states)
        return counts / counts.sum()

    def compute_transition_matrix(self, sequence: List[int], alpha: float = 0.1) -> np.ndarray:
        """Compute Dirichlet-smoothed transition matrix."""
        counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(sequence) - 1):
            counts[sequence[i], sequence[i+1]] += 1

        smoothed = counts + alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return smoothed / row_sums

    def compute_entropy_rate(self, kernel: np.ndarray) -> float:
        """Compute entropy rate of transitions."""
        # Stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(kernel.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = eigenvectors[:, idx].real
        stationary = np.abs(stationary)
        stationary = stationary / stationary.sum()

        # Entropy rate
        eps = 1e-10
        log_kernel = np.log2(kernel + eps)
        conditional_entropy = -np.sum(kernel * log_kernel, axis=1)
        return float(np.sum(stationary * conditional_entropy))

    def extract_traits(self, sequence: List[int]) -> Dict[str, float]:
        """
        Extract Big-5 trait proxies from a behavioral sequence.

        Returns normalized scores in [0, 1] range.
        """
        if len(sequence) < 3:
            return {trait: 0.5 for trait in
                   ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']}

        dist = self.compute_state_distribution(sequence)
        kernel = self.compute_transition_matrix(sequence)
        entropy = self.compute_entropy_rate(kernel)

        # State indices: 0=Seeking, 1=Directing, 2=Conferring, 3=Revising
        seeking = dist[0]
        directing = dist[1]
        conferring = dist[2]
        revising = dist[3]

        # Openness: Exploration tendency (Seeking + Conferring)
        # High explore = high openness
        openness = (seeking + conferring)  # Already in [0, 1]

        # Conscientiousness: Structure/exploitation tendency (Revising primarily)
        # High Revising = high conscientiousness (self-regulation, routines)
        # Note: Directing is exploitation but of others, not self-discipline
        conscientiousness = revising + 0.3 * seeking  # Self-focused states
        conscientiousness = min(1.0, conscientiousness * 2)  # Scale up

        # Extraversion: Other-orientation (Directing + Conferring)
        # Both involve focus on others
        extraversion = (directing + conferring)

        # Agreeableness: Inverse of exploitation of others
        # High Directing = Low Agreeableness
        agreeableness = 1.0 - directing

        # Neuroticism: Behavioral instability (transition entropy)
        # High entropy = unpredictable, unstable behavior
        # Normalize: max entropy for 4 states = 2 bits
        neuroticism = entropy / 2.0

        return {
            'openness': float(np.clip(openness, 0, 1)),
            'conscientiousness': float(np.clip(conscientiousness, 0, 1)),
            'extraversion': float(np.clip(extraversion, 0, 1)),
            'agreeableness': float(np.clip(agreeableness, 0, 1)),
            'neuroticism': float(np.clip(neuroticism, 0, 1))
        }

    def extract_traits_detailed(self, sequence: List[int]) -> Dict:
        """Extract traits with additional diagnostic information."""
        basic_traits = self.extract_traits(sequence)

        dist = self.compute_state_distribution(sequence)
        kernel = self.compute_transition_matrix(sequence)

        # Additional metrics
        directing_persistence = kernel[1, 1]  # How sticky is Directing
        exploitation_index = dist[1] + dist[3]  # Total exploitation (self + other)
        exploration_index = dist[0] + dist[2]  # Total exploration

        return {
            **basic_traits,
            'state_distribution': {ANIMAL_NAMES[i]: float(dist[i]) for i in range(4)},
            'directing_persistence': float(directing_persistence),
            'exploitation_index': float(exploitation_index),
            'exploration_index': float(exploration_index),
            'self_focus': float(dist[0] + dist[3]),
            'other_focus': float(dist[1] + dist[2])
        }


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def compute_population_statistics(trait_profiles: List[Dict]) -> Dict:
    """Compute population-level statistics for trait proxies."""
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    stats_results = {}
    for trait in traits:
        values = [p[trait] for p in trait_profiles]
        stats_results[trait] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values))
        }

    return stats_results


def compute_trait_correlations(trait_profiles: List[Dict]) -> np.ndarray:
    """Compute correlation matrix between traits."""
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    data = np.array([[p[t] for t in traits] for p in trait_profiles])
    corr_matrix = np.corrcoef(data.T)

    return corr_matrix, traits


def compare_to_normative_data() -> Dict:
    """
    Provide normative Big-5 data for comparison.

    These are approximate population norms from psychological literature.
    Note: Criminal populations typically show specific deviations.
    """
    # Normative data (approximate means on 0-1 scale)
    # Based on Costa & McCrae (1992) and subsequent research
    normative = {
        'general_population': {
            'openness': 0.50,
            'conscientiousness': 0.50,
            'extraversion': 0.50,
            'agreeableness': 0.50,
            'neuroticism': 0.50
        },
        'antisocial_personality': {
            # Based on Miller & Lynam (2001), Lynam & Derefinko (2006)
            'openness': 0.45,
            'conscientiousness': 0.25,  # Low
            'extraversion': 0.55,
            'agreeableness': 0.20,  # Very low
            'neuroticism': 0.60  # Elevated
        },
        'psychopathy': {
            # Based on Decuyper et al. (2009)
            'openness': 0.50,
            'conscientiousness': 0.30,  # Low
            'extraversion': 0.55,
            'agreeableness': 0.15,  # Very low
            'neuroticism': 0.40  # Variable, often low (fearless)
        }
    }

    return normative


def validate_against_theory(sample_stats: Dict) -> Dict:
    """
    Validate extracted traits against theoretical expectations for criminal population.

    Expected patterns for serial killers based on criminological literature:
    - Low Agreeableness (exploitation, lack of empathy)
    - Low Conscientiousness (impulsivity, though some show high due to planning)
    - Variable Neuroticism (some anxious, some fearless)
    - Variable Extraversion (some charming, some reclusive)
    - Variable Openness
    """
    normative = compare_to_normative_data()

    validation_results = {}
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    theoretical_expectations = {
        'agreeableness': {
            'expected': 'low',
            'rationale': 'Exploitation of others (Directing) should yield low Agreeableness',
            'threshold': 0.35
        },
        'conscientiousness': {
            'expected': 'variable',
            'rationale': 'Some killers are impulsive, others highly organized',
            'threshold': None
        },
        'neuroticism': {
            'expected': 'elevated',
            'rationale': 'Behavioral instability should manifest as higher entropy',
            'threshold': 0.55
        },
        'extraversion': {
            'expected': 'elevated',
            'rationale': 'High Other-focus (Directing + Conferring)',
            'threshold': 0.55
        },
        'openness': {
            'expected': 'variable',
            'rationale': 'Balance of exploration vs exploitation varies',
            'threshold': None
        }
    }

    for trait in traits:
        sample_mean = sample_stats[trait]['mean']
        general_pop = normative['general_population'][trait]
        antisocial = normative['antisocial_personality'][trait]
        expectation = theoretical_expectations[trait]

        # Compare to norms
        deviation_from_norm = sample_mean - general_pop
        closer_to_antisocial = abs(sample_mean - antisocial) < abs(sample_mean - general_pop)

        # Check against threshold if specified
        meets_expectation = None
        if expectation['threshold'] is not None:
            if expectation['expected'] == 'low':
                meets_expectation = sample_mean < expectation['threshold']
            elif expectation['expected'] == 'elevated':
                meets_expectation = sample_mean > expectation['threshold']

        validation_results[trait] = {
            'sample_mean': sample_mean,
            'general_population_norm': general_pop,
            'antisocial_norm': antisocial,
            'deviation_from_norm': deviation_from_norm,
            'closer_to_antisocial': closer_to_antisocial,
            'theoretical_expectation': expectation['expected'],
            'rationale': expectation['rationale'],
            'meets_expectation': meets_expectation
        }

    return validation_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_trait_distributions(trait_profiles: List[Dict], output_path: str):
    """Plot distributions of Big-5 trait proxies."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    normative = compare_to_normative_data()

    for i, trait in enumerate(traits):
        ax = axes[i]
        values = [p[trait] for p in trait_profiles]

        ax.hist(values, bins=15, color=colors[i], edgecolor='black', alpha=0.7)

        # Add normative lines
        ax.axvline(x=normative['general_population'][trait], color='black',
                  linestyle='--', linewidth=2, label='General Pop.')
        ax.axvline(x=normative['antisocial_personality'][trait], color='red',
                  linestyle=':', linewidth=2, label='Antisocial')
        ax.axvline(x=np.mean(values), color='blue',
                  linestyle='-', linewidth=2, label=f'Sample Mean')

        ax.set_xlabel(trait.capitalize())
        ax.set_ylabel('Count')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)

    # Remove extra subplot
    axes[5].axis('off')

    fig.suptitle('Big-5 Trait Proxy Distributions\n(Derived from 4-Animal Behavioral Sequences)',
                fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_trait_radar(mean_traits: Dict, output_path: str):
    """Plot radar chart comparing sample to norms."""
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    normative = compare_to_normative_data()

    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    sample_values = [mean_traits[t] for t in traits] + [mean_traits[traits[0]]]
    general_values = [normative['general_population'][t] for t in traits] + [normative['general_population'][traits[0]]]
    antisocial_values = [normative['antisocial_personality'][t] for t in traits] + [normative['antisocial_personality'][traits[0]]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, general_values, 'o-', linewidth=2, label='General Population', color='gray')
    ax.fill(angles, general_values, alpha=0.1, color='gray')

    ax.plot(angles, antisocial_values, 's--', linewidth=2, label='Antisocial Norm', color='orange')
    ax.fill(angles, antisocial_values, alpha=0.1, color='orange')

    ax.plot(angles, sample_values, '^-', linewidth=3, label='Sample (Serial Killers)', color='red')
    ax.fill(angles, sample_values, alpha=0.2, color='red')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.capitalize() for t in traits], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Big-5 Trait Profile Comparison', fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_trait_correlations(corr_matrix: np.ndarray, traits: List[str], output_path: str):
    """Plot correlation matrix between traits."""
    fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               xticklabels=[t.capitalize() for t in traits],
               yticklabels=[t.capitalize() for t in traits],
               cmap='RdBu_r', vmin=-1, vmax=1, center=0, ax=ax)

    ax.set_title('Inter-Trait Correlations\n(Big-5 Proxies from 4-Animal)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_classified_sequences(results_dir: str) -> Dict[str, List[int]]:
    """Load pre-classified sequences."""
    results_dir = Path(results_dir)

    comparison_dirs = sorted(results_dir.glob("four_animal_comparison_*"))
    if not comparison_dirs:
        raise FileNotFoundError("No classification results found.")

    latest_dir = comparison_dirs[-1]
    labels_file = latest_dir / "animal_labels.json"

    with open(labels_file, 'r') as f:
        data = json.load(f)

    data_dir = Path("/Users/ajithsenthil/Desktop/CrimeArchetypes/mnt/data/csv")
    sequences = {}
    event_idx = 0

    for filepath in sorted(data_dir.glob("Type1_*.csv")):
        name = filepath.stem[6:]
        seq = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3 and row[2].strip():
                        if event_idx < len(data['labels']):
                            seq.append(data['labels'][event_idx])
                        event_idx += 1
        except:
            continue

        if len(seq) >= 5:
            sequences[name] = seq

    return sequences


def run_clinical_validation(results_dir: str, output_dir: str) -> Dict:
    """Run complete clinical validation analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(output_dir) / f"clinical_validation_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CLINICAL VALIDATION: BIG-5 TRAIT PROXY ANALYSIS")
    logger.info("=" * 70)

    # Load sequences
    logger.info("\n[1] Loading classified sequences...")
    sequences = load_classified_sequences(results_dir)
    logger.info(f"Loaded {len(sequences)} individuals")

    # Extract traits
    logger.info("\n[2] Extracting Big-5 trait proxies...")
    extractor = Big5TraitExtractor()

    trait_profiles = {}
    for name, seq in sequences.items():
        trait_profiles[name] = extractor.extract_traits_detailed(seq)

    # Population statistics
    logger.info("\n[3] Computing population statistics...")
    pop_stats = compute_population_statistics(list(trait_profiles.values()))

    print("\nBig-5 Trait Proxy Statistics:")
    print("-" * 60)
    print(f"{'Trait':<18} {'Mean':<8} {'Std':<8} {'Median':<8} {'Range':<12}")
    print("-" * 60)
    for trait, stats_dict in pop_stats.items():
        range_str = f"[{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]"
        print(f"{trait.capitalize():<18} {stats_dict['mean']:<8.3f} {stats_dict['std']:<8.3f} "
              f"{stats_dict['median']:<8.3f} {range_str:<12}")

    # Validate against theory
    logger.info("\n[4] Validating against theoretical expectations...")
    validation = validate_against_theory(pop_stats)

    print("\nTheoretical Validation:")
    print("-" * 60)
    for trait, v in validation.items():
        status = ""
        if v['meets_expectation'] is True:
            status = "CONFIRMED"
        elif v['meets_expectation'] is False:
            status = "NOT CONFIRMED"
        else:
            status = "N/A (variable expected)"

        print(f"\n{trait.capitalize()}:")
        print(f"  Sample mean: {v['sample_mean']:.3f}")
        print(f"  Expected: {v['theoretical_expectation']}")
        print(f"  Status: {status}")
        print(f"  Closer to antisocial norm: {v['closer_to_antisocial']}")

    # Compute correlations
    logger.info("\n[5] Computing trait correlations...")
    corr_matrix, traits = compute_trait_correlations(list(trait_profiles.values()))

    # Generate visualizations
    logger.info("\n[6] Generating visualizations...")

    plot_trait_distributions(list(trait_profiles.values()),
                            study_dir / 'trait_distributions.png')

    mean_traits = {t: pop_stats[t]['mean'] for t in pop_stats.keys()}
    plot_trait_radar(mean_traits, study_dir / 'trait_radar.png')

    plot_trait_correlations(corr_matrix, traits, study_dir / 'trait_correlations.png')

    # Compile results
    results = {
        'n_individuals': len(sequences),
        'population_statistics': pop_stats,
        'theoretical_validation': {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
            for k, v in validation.items()
        },
        'correlation_matrix': corr_matrix.tolist(),
        'individual_profiles': {
            name: {k: v for k, v in profile.items() if k != 'state_distribution'}
            for name, profile in list(trait_profiles.items())[:10]  # Sample
        }
    }

    # Save results
    with open(study_dir / 'clinical_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save full profiles
    with open(study_dir / 'all_trait_profiles.json', 'w') as f:
        json.dump(trait_profiles, f, indent=2)

    logger.info(f"\nResults saved to: {study_dir}")

    return results


if __name__ == "__main__":
    results_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"
    output_dir = "/Users/ajithsenthil/Desktop/CrimeArchetypes/analysis/empirical_study"

    run_clinical_validation(results_dir, output_dir)
