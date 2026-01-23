#!/usr/bin/env python3
"""
State Space Benchmarking Framework

A systematic, statistically rigorous framework for comparing different state space
representations of behavioral sequences.

Comparison Dimensions:
1. COMPRESSION - How much does the state space reduce complexity?
2. PRESERVATION - How much predictive structure is retained?
3. DISCRIMINABILITY - Can individual differences be distinguished?
4. STABILITY - How robust are the dynamics across subsamples?

Statistical Methods:
- Permutation tests with proper null models
- Bootstrap confidence intervals
- Cross-validation for generalization
- Information-theoretic model selection (AIC/BIC)
- Likelihood ratio tests for nested models

State Spaces Compared:
- K-Cluster (data-driven, variable K)
- 4-Animal (theory-driven, Computational Psychodynamics)
- Binary (action/non-action)
- Random baseline (null model)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from collections import Counter, defaultdict
from scipy import stats
from scipy.special import gammaln
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    v_measure_score
)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# STATE SPACE DEFINITIONS
# =============================================================================

class StateSpaceDefinition(ABC):
    """Abstract base class for state space definitions."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def n_states(self) -> int:
        pass

    @property
    @abstractmethod
    def state_names(self) -> List[str]:
        pass

    @abstractmethod
    def classify(self, event_text: str, cluster_label: int = None) -> int:
        """Classify an event into a state index."""
        pass

    @property
    def is_theory_driven(self) -> bool:
        return False


class ClusterStateSpace(StateSpaceDefinition):
    """Data-driven K-cluster state space."""

    def __init__(self, k: int, label_to_theme: Dict[int, str] = None):
        self._k = k
        self._label_to_theme = label_to_theme or {}

    @property
    def name(self) -> str:
        return f"{self._k}-Cluster"

    @property
    def n_states(self) -> int:
        return self._k

    @property
    def state_names(self) -> List[str]:
        return [f"C{i}" for i in range(self._k)]

    def classify(self, event_text: str, cluster_label: int = None) -> int:
        # Use pre-computed cluster label
        if cluster_label is not None and 0 <= cluster_label < self._k:
            return cluster_label
        return 0  # Default


class AnimalStateSpace(StateSpaceDefinition):
    """Theory-driven 4-Animal state space (Computational Psychodynamics)."""

    STATES = ['Seeking', 'Directing', 'Conferring', 'Revising']

    KEYWORDS = {
        'Seeking': ['fantasy', 'identity', 'internal', 'self', 'psychological',
                   'childhood', 'family', 'born', 'mother', 'father', 'abuse',
                   'struggled', 'depression', 'isolation', 'withdrawn', 'school'],
        'Directing': ['murder', 'kill', 'rape', 'assault', 'attack', 'victim',
                     'strangle', 'stab', 'shot', 'abduct', 'torture', 'body',
                     'death', 'violence', 'sexual assault', 'homicide', 'suffocate'],
        'Conferring': ['stalk', 'follow', 'watch', 'observe', 'surveillance',
                      'target', 'select', 'meet', 'approach', 'contact',
                      'relationship', 'dating', 'marriage', 'divorce', 'marry'],
        'Revising': ['arrest', 'prison', 'trial', 'convicted', 'sentence',
                    'parole', 'released', 'court', 'charge', 'police',
                    'routine', 'pattern', 'repeated', 'habit', 'ritual', 'jail']
    }

    # Mapping from 10-cluster to 4-animal (theory-driven)
    CLUSTER_MAPPING = {
        0: 2,  # Stalking → Conferring
        1: 1,  # Sexual murder → Directing
        2: 3,  # Escalating crime → Revising
        3: 1,  # Sadistic → Directing
        4: 1,  # Power/control → Directing
        5: 0,  # Workplace → Seeking
        6: 3,  # Legal → Revising
        7: 0,  # Identity → Seeking
        8: 2,  # Domestic → Conferring
        9: 1,  # Angel of death → Directing
    }

    @property
    def name(self) -> str:
        return "4-Animal"

    @property
    def n_states(self) -> int:
        return 4

    @property
    def state_names(self) -> List[str]:
        return self.STATES

    @property
    def is_theory_driven(self) -> bool:
        return True

    def classify(self, event_text: str, cluster_label: int = None) -> int:
        # If we have a cluster label, use the mapping
        if cluster_label is not None and cluster_label in self.CLUSTER_MAPPING:
            return self.CLUSTER_MAPPING[cluster_label]

        # Otherwise use keyword matching
        if not isinstance(event_text, str):
            return 0

        event_lower = event_text.lower()
        scores = [0, 0, 0, 0]

        for i, state in enumerate(self.STATES):
            for kw in self.KEYWORDS[state]:
                if kw in event_lower:
                    scores[i] += 1

        if max(scores) == 0:
            return 0  # Default to Seeking
        return np.argmax(scores)


class BinaryStateSpace(StateSpaceDefinition):
    """Binary state space: Harmful vs Non-harmful."""

    HARMFUL_KEYWORDS = ['murder', 'kill', 'rape', 'assault', 'attack', 'victim',
                        'strangle', 'stab', 'shot', 'abduct', 'torture', 'death',
                        'violence', 'homicide', 'suffocate', 'beat', 'harm']

    @property
    def name(self) -> str:
        return "2-Binary"

    @property
    def n_states(self) -> int:
        return 2

    @property
    def state_names(self) -> List[str]:
        return ['Non-Harmful', 'Harmful']

    def classify(self, event_text: str, cluster_label: int = None) -> int:
        if not isinstance(event_text, str):
            return 0
        event_lower = event_text.lower()
        for kw in self.HARMFUL_KEYWORDS:
            if kw in event_lower:
                return 1
        return 0


class RandomStateSpace(StateSpaceDefinition):
    """Random assignment baseline (null model)."""

    def __init__(self, k: int, seed: int = None):
        self._k = k
        self._rng = np.random.RandomState(seed)

    @property
    def name(self) -> str:
        return f"{self._k}-Random"

    @property
    def n_states(self) -> int:
        return self._k

    @property
    def state_names(self) -> List[str]:
        return [f"R{i}" for i in range(self._k)]

    def classify(self, event_text: str, cluster_label: int = None) -> int:
        return self._rng.randint(0, self._k)


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class MarkovMetrics:
    """Container for Markov chain metrics."""
    transition_matrix: np.ndarray
    stationary_distribution: np.ndarray
    entropy_rate: float
    mixing_time: float
    log_likelihood: float
    aic: float
    bic: float
    n_parameters: int
    n_observations: int


def compute_transition_matrix(sequence: List[int], n_states: int) -> np.ndarray:
    """Compute transition matrix from sequence."""
    counts = np.zeros((n_states, n_states))
    for i in range(len(sequence) - 1):
        if 0 <= sequence[i] < n_states and 0 <= sequence[i+1] < n_states:
            counts[sequence[i], sequence[i+1]] += 1

    # Add Laplace smoothing
    counts += 0.01
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute stationary distribution of transition matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def compute_entropy_rate(P: np.ndarray, pi: np.ndarray) -> float:
    """Compute entropy rate of Markov chain."""
    H = 0.0
    for i in range(len(pi)):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                H -= pi[i] * P[i, j] * np.log2(P[i, j])
    return H


def compute_mixing_time(P: np.ndarray, epsilon: float = 0.01) -> float:
    """Estimate mixing time (time to reach near-stationary)."""
    pi = compute_stationary_distribution(P)
    P_power = P.copy()

    for t in range(1, 1000):
        P_power = P_power @ P
        # Check if all rows are close to stationary
        max_diff = np.max(np.abs(P_power - pi))
        if max_diff < epsilon:
            return t
    return 1000  # Didn't converge


def compute_log_likelihood(sequences: List[List[int]], P: np.ndarray) -> float:
    """Compute log-likelihood of sequences under Markov model."""
    ll = 0.0
    for seq in sequences:
        for i in range(len(seq) - 1):
            s1, s2 = seq[i], seq[i+1]
            if 0 <= s1 < P.shape[0] and 0 <= s2 < P.shape[1]:
                if P[s1, s2] > 0:
                    ll += np.log(P[s1, s2])
                else:
                    ll += np.log(1e-10)  # Smoothing for zero probs
    return ll


def compute_markov_metrics(
    sequences: List[List[int]],
    n_states: int
) -> MarkovMetrics:
    """Compute all Markov chain metrics."""
    # Aggregate transition counts
    total_counts = np.zeros((n_states, n_states))
    n_transitions = 0

    for seq in sequences:
        for i in range(len(seq) - 1):
            if 0 <= seq[i] < n_states and 0 <= seq[i+1] < n_states:
                total_counts[seq[i], seq[i+1]] += 1
                n_transitions += 1

    # Compute transition matrix with Laplace smoothing
    smoothed = total_counts + 0.01
    P = smoothed / smoothed.sum(axis=1, keepdims=True)

    # Stationary distribution
    pi = compute_stationary_distribution(P)

    # Entropy rate
    H = compute_entropy_rate(P, pi)

    # Mixing time
    tau = compute_mixing_time(P)

    # Log-likelihood
    ll = compute_log_likelihood(sequences, P)

    # Model complexity
    n_params = n_states * (n_states - 1)  # Free parameters in transition matrix

    # AIC and BIC
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_transitions) - 2 * ll

    return MarkovMetrics(
        transition_matrix=P,
        stationary_distribution=pi,
        entropy_rate=H,
        mixing_time=tau,
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        n_parameters=n_params,
        n_observations=n_transitions
    )


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two state spaces."""
    nmi: float  # Normalized Mutual Information
    ari: float  # Adjusted Rand Index
    v_measure: float  # V-measure (harmonic mean of homogeneity and completeness)
    information_retention: float  # I(S1; S2) / H(S1)
    js_divergence_transitions: float  # JS divergence between transition matrices
    entropy_rate_ratio: float  # H(S2) / H(S1)


def compare_state_spaces(
    labels_1: List[int],
    labels_2: List[int],
    P1: np.ndarray,
    P2: np.ndarray
) -> ComparisonMetrics:
    """Compute comparison metrics between two state space labelings."""
    # Ensure same length
    n = min(len(labels_1), len(labels_2))
    l1, l2 = labels_1[:n], labels_2[:n]

    # Clustering comparison metrics
    nmi = normalized_mutual_info_score(l1, l2)
    ari = adjusted_rand_score(l1, l2)
    v_measure = v_measure_score(l1, l2)

    # Information retention
    mi = mutual_info_score(l1, l2) / np.log(2)  # Convert to bits
    h1 = stats.entropy(list(Counter(l1).values()), base=2)
    info_retention = mi / h1 if h1 > 0 else 0

    # Transition matrix comparison (if same size or can be compared)
    if P1.shape == P2.shape:
        # JS divergence between corresponding rows
        js_div = 0.0
        for i in range(P1.shape[0]):
            js_div += jensenshannon(P1[i] + 1e-10, P2[i] + 1e-10)
        js_div /= P1.shape[0]
    else:
        js_div = np.nan

    # Entropy rate ratio
    pi1 = compute_stationary_distribution(P1)
    pi2 = compute_stationary_distribution(P2)
    h1_rate = compute_entropy_rate(P1, pi1)
    h2_rate = compute_entropy_rate(P2, pi2)
    entropy_ratio = h2_rate / h1_rate if h1_rate > 0 else 0

    return ComparisonMetrics(
        nmi=nmi,
        ari=ari,
        v_measure=v_measure,
        information_retention=info_retention,
        js_divergence_transitions=js_div,
        entropy_rate_ratio=entropy_ratio
    )


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def permutation_test_information(
    labels_base: List[int],
    labels_target: List[int],
    n_states_target: int,
    n_permutations: int = 10000
) -> Tuple[float, float, float, np.ndarray]:
    """
    Permutation test for information retention.

    Null hypothesis: The target labeling retains no more information about
    the base labeling than a random assignment would.

    Returns: (observed, null_mean, p_value, null_distribution)
    """
    # Observed information retention
    mi_obs = mutual_info_score(labels_base, labels_target)
    h_base = stats.entropy(list(Counter(labels_base).values()))
    observed = mi_obs / h_base if h_base > 0 else 0

    # Null distribution: random permutation of target labels
    null_retentions = []
    for _ in range(n_permutations):
        # Random assignment preserving marginal distribution
        permuted = np.random.permutation(labels_target)
        mi_null = mutual_info_score(labels_base, permuted)
        null_retentions.append(mi_null / h_base if h_base > 0 else 0)

    null_retentions = np.array(null_retentions)
    null_mean = np.mean(null_retentions)

    # One-tailed p-value
    p_value = np.mean(null_retentions >= observed)

    return observed, null_mean, p_value, null_retentions


def bootstrap_ci_entropy_rate(
    sequences: List[List[int]],
    n_states: int,
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for entropy rate.

    Returns: (point_estimate, ci_lower, ci_upper)
    """
    # Point estimate
    metrics = compute_markov_metrics(sequences, n_states)
    point_estimate = metrics.entropy_rate

    # Bootstrap
    n_seq = len(sequences)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Resample sequences with replacement
        indices = np.random.choice(n_seq, size=n_seq, replace=True)
        boot_seqs = [sequences[i] for i in indices]
        boot_metrics = compute_markov_metrics(boot_seqs, n_states)
        bootstrap_estimates.append(boot_metrics.entropy_rate)

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Confidence interval
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def likelihood_ratio_test(
    ll_full: float,
    ll_reduced: float,
    df_full: int,
    df_reduced: int
) -> Tuple[float, float]:
    """
    Likelihood ratio test for nested models.

    Tests whether the full model (more states) significantly improves
    over the reduced model (fewer states).

    Returns: (chi2_statistic, p_value)
    """
    # LR statistic
    lr = 2 * (ll_full - ll_reduced)

    # Degrees of freedom
    df = df_full - df_reduced

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(lr, df)

    return lr, p_value


def cross_validate_predictive_power(
    sequences: Dict[str, List[int]],
    n_states: int,
    n_folds: int = 5
) -> Tuple[float, float]:
    """
    Cross-validate predictive log-likelihood.

    Returns: (mean_ll_per_transition, std_ll_per_transition)
    """
    individual_names = list(sequences.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_lls = []
    fold_n_trans = []

    for train_idx, test_idx in kf.split(individual_names):
        # Train sequences
        train_seqs = [sequences[individual_names[i]] for i in train_idx]

        # Test sequences
        test_seqs = [sequences[individual_names[i]] for i in test_idx]

        # Fit model on training data
        train_metrics = compute_markov_metrics(train_seqs, n_states)

        # Evaluate on test data
        test_ll = compute_log_likelihood(test_seqs, train_metrics.transition_matrix)
        n_transitions = sum(len(s) - 1 for s in test_seqs)

        fold_lls.append(test_ll)
        fold_n_trans.append(n_transitions)

    # Normalize by number of transitions
    ll_per_trans = [ll / n for ll, n in zip(fold_lls, fold_n_trans)]

    return np.mean(ll_per_trans), np.std(ll_per_trans)


# =============================================================================
# BENCHMARKING FRAMEWORK
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results for a single state space."""
    space_name: str
    n_states: int
    is_theory_driven: bool
    markov_metrics: MarkovMetrics
    entropy_rate_ci: Tuple[float, float, float]  # (point, lower, upper)
    cv_predictive_power: Tuple[float, float]  # (mean, std)


@dataclass
class ComparisonResult:
    """Results comparing two state spaces."""
    space_1: str
    space_2: str
    comparison_metrics: ComparisonMetrics
    permutation_test: Dict  # observed, null_mean, p_value
    likelihood_ratio_test: Optional[Dict]  # For nested models


class StateSpaceBenchmark:
    """
    Comprehensive benchmarking of state space representations.
    """

    def __init__(self, sequences: Dict[str, Dict], cluster_k: int = 10):
        """
        Args:
            sequences: Dict mapping individual name -> {'labels': [...], 'events': [...]}
            cluster_k: Number of clusters in the data-driven representation
        """
        self.sequences = sequences
        self.cluster_k = cluster_k

        # Extract cluster sequences
        self.cluster_sequences = {
            name: data['labels'] for name, data in sequences.items()
        }

        # Define state spaces to compare
        self.state_spaces = {
            '10-Cluster': ClusterStateSpace(10),
            '4-Animal': AnimalStateSpace(),
            '2-Binary': BinaryStateSpace(),
        }

        # Add random baselines
        for k in [2, 4, 10]:
            self.state_spaces[f'{k}-Random'] = RandomStateSpace(k, seed=42)

        self.results: Dict[str, BenchmarkResult] = {}
        self.comparisons: Dict[Tuple[str, str], ComparisonResult] = {}

    def classify_all(self, space: StateSpaceDefinition) -> Dict[str, List[int]]:
        """Classify all events using a state space."""
        classified = {}
        for name, data in self.sequences.items():
            seq = []
            for cluster_label, event_text in zip(data['labels'], data['events']):
                state = space.classify(event_text, cluster_label)
                seq.append(state)
            classified[name] = seq
        return classified

    def benchmark_space(self, space_name: str) -> BenchmarkResult:
        """Run full benchmark on a single state space."""
        space = self.state_spaces[space_name]

        # Classify events
        if space_name == '10-Cluster':
            classified = self.cluster_sequences
        else:
            classified = self.classify_all(space)

        # Convert to list of sequences
        seq_list = list(classified.values())

        # Markov metrics
        markov = compute_markov_metrics(seq_list, space.n_states)

        # Bootstrap CI for entropy rate
        entropy_ci = bootstrap_ci_entropy_rate(seq_list, space.n_states, n_bootstrap=500)

        # Cross-validated predictive power
        cv_power = cross_validate_predictive_power(classified, space.n_states)

        result = BenchmarkResult(
            space_name=space_name,
            n_states=space.n_states,
            is_theory_driven=space.is_theory_driven,
            markov_metrics=markov,
            entropy_rate_ci=entropy_ci,
            cv_predictive_power=cv_power
        )

        self.results[space_name] = result
        return result

    def compare_spaces(self, space_1: str, space_2: str) -> ComparisonResult:
        """Compare two state spaces."""
        s1 = self.state_spaces[space_1]
        s2 = self.state_spaces[space_2]

        # Get classifications
        if space_1 == '10-Cluster':
            labels_1 = [l for seq in self.cluster_sequences.values() for l in seq]
        else:
            classified_1 = self.classify_all(s1)
            labels_1 = [l for seq in classified_1.values() for l in seq]

        if space_2 == '10-Cluster':
            labels_2 = [l for seq in self.cluster_sequences.values() for l in seq]
        else:
            classified_2 = self.classify_all(s2)
            labels_2 = [l for seq in classified_2.values() for l in seq]

        # Get transition matrices
        P1 = self.results[space_1].markov_metrics.transition_matrix
        P2 = self.results[space_2].markov_metrics.transition_matrix

        # Comparison metrics
        comp_metrics = compare_state_spaces(labels_1, labels_2, P1, P2)

        # Permutation test
        obs, null_mean, p_val, _ = permutation_test_information(
            labels_1, labels_2, s2.n_states, n_permutations=5000
        )
        perm_result = {
            'observed': obs,
            'null_mean': null_mean,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

        # Likelihood ratio test (if comparable)
        lr_result = None
        if s1.n_states > s2.n_states:
            ll_full = self.results[space_1].markov_metrics.log_likelihood
            ll_reduced = self.results[space_2].markov_metrics.log_likelihood
            df_full = self.results[space_1].markov_metrics.n_parameters
            df_reduced = self.results[space_2].markov_metrics.n_parameters

            lr_stat, lr_p = likelihood_ratio_test(ll_full, ll_reduced, df_full, df_reduced)
            lr_result = {
                'chi2': lr_stat,
                'p_value': lr_p,
                'df': df_full - df_reduced,
                'significant': lr_p < 0.05
            }

        result = ComparisonResult(
            space_1=space_1,
            space_2=space_2,
            comparison_metrics=comp_metrics,
            permutation_test=perm_result,
            likelihood_ratio_test=lr_result
        )

        self.comparisons[(space_1, space_2)] = result
        return result

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmarking suite."""
        print("=" * 70)
        print("STATE SPACE BENCHMARKING")
        print("=" * 70)

        # Benchmark each space
        print("\n[1/3] Benchmarking individual state spaces...")
        for space_name in self.state_spaces:
            print(f"   Processing {space_name}...")
            self.benchmark_space(space_name)

        # Pairwise comparisons (non-random spaces)
        print("\n[2/3] Computing pairwise comparisons...")
        primary_spaces = ['10-Cluster', '4-Animal', '2-Binary']
        for i, s1 in enumerate(primary_spaces):
            for s2 in primary_spaces[i+1:]:
                print(f"   Comparing {s1} vs {s2}...")
                self.compare_spaces(s1, s2)

        # Compare to random baselines
        print("\n[3/3] Comparing to random baselines...")
        for space in ['4-Animal', '2-Binary']:
            k = self.state_spaces[space].n_states
            random_name = f'{k}-Random'
            print(f"   Comparing {space} vs {random_name}...")
            self.compare_spaces(space, random_name)

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate comprehensive report."""
        report = {
            'summary': {},
            'individual_results': {},
            'comparisons': {},
            'rankings': {}
        }

        # Individual results
        for name, result in self.results.items():
            report['individual_results'][name] = {
                'n_states': result.n_states,
                'is_theory_driven': result.is_theory_driven,
                'entropy_rate': float(result.entropy_rate_ci[0]),
                'entropy_rate_95ci': [float(result.entropy_rate_ci[1]),
                                       float(result.entropy_rate_ci[2])],
                'log_likelihood': float(result.markov_metrics.log_likelihood),
                'aic': float(result.markov_metrics.aic),
                'bic': float(result.markov_metrics.bic),
                'mixing_time': float(result.markov_metrics.mixing_time),
                'cv_predictive_ll': float(result.cv_predictive_power[0]),
                'cv_predictive_ll_std': float(result.cv_predictive_power[1]),
            }

        # Comparisons
        for (s1, s2), result in self.comparisons.items():
            key = f"{s1}_vs_{s2}"
            report['comparisons'][key] = {
                'nmi': float(result.comparison_metrics.nmi),
                'ari': float(result.comparison_metrics.ari),
                'v_measure': float(result.comparison_metrics.v_measure),
                'information_retention': float(result.comparison_metrics.information_retention),
                'permutation_test': {
                    'observed': float(result.permutation_test['observed']),
                    'null_mean': float(result.permutation_test['null_mean']),
                    'p_value': float(result.permutation_test['p_value']),
                    'significant': bool(result.permutation_test['significant'])
                }
            }
            if result.likelihood_ratio_test:
                report['comparisons'][key]['likelihood_ratio_test'] = {
                    'chi2': float(result.likelihood_ratio_test['chi2']),
                    'p_value': float(result.likelihood_ratio_test['p_value']),
                    'significant': bool(result.likelihood_ratio_test['significant'])
                }

        # Rankings by different criteria
        non_random = [n for n in self.results if 'Random' not in n]

        # By AIC (lower is better)
        by_aic = sorted(non_random, key=lambda x: self.results[x].markov_metrics.aic)
        report['rankings']['by_aic'] = by_aic

        # By BIC (lower is better)
        by_bic = sorted(non_random, key=lambda x: self.results[x].markov_metrics.bic)
        report['rankings']['by_bic'] = by_bic

        # By cross-validated predictive power (higher is better)
        by_cv = sorted(non_random, key=lambda x: self.results[x].cv_predictive_power[0], reverse=True)
        report['rankings']['by_cv_predictive'] = by_cv

        # By parsimony (fewer states is better, given similar performance)
        by_parsimony = sorted(non_random, key=lambda x: self.results[x].n_states)
        report['rankings']['by_parsimony'] = by_parsimony

        # Summary
        best_aic = by_aic[0]
        best_bic = by_bic[0]
        best_cv = by_cv[0]

        report['summary'] = {
            'n_events': sum(len(seq) for seq in self.cluster_sequences.values()),
            'n_individuals': len(self.cluster_sequences),
            'best_by_aic': best_aic,
            'best_by_bic': best_bic,
            'best_by_cv_predictive': best_cv,
            'recommendation': best_bic  # BIC balances fit and complexity
        }

        return report


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_benchmark_comparison(report: Dict, output_dir: str):
    """Generate comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Model selection criteria
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    spaces = [s for s in report['individual_results'] if 'Random' not in s]
    x = range(len(spaces))

    # AIC
    aics = [report['individual_results'][s]['aic'] for s in spaces]
    colors = ['#2ecc71' if s == report['rankings']['by_aic'][0] else '#95a5a6' for s in spaces]
    axes[0].bar(x, aics, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(spaces, rotation=45, ha='right')
    axes[0].set_ylabel('AIC (lower is better)')
    axes[0].set_title('Model Selection: AIC')

    # BIC
    bics = [report['individual_results'][s]['bic'] for s in spaces]
    colors = ['#3498db' if s == report['rankings']['by_bic'][0] else '#95a5a6' for s in spaces]
    axes[1].bar(x, bics, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(spaces, rotation=45, ha='right')
    axes[1].set_ylabel('BIC (lower is better)')
    axes[1].set_title('Model Selection: BIC')

    # CV Predictive LL
    cvs = [report['individual_results'][s]['cv_predictive_ll'] for s in spaces]
    cv_stds = [report['individual_results'][s]['cv_predictive_ll_std'] for s in spaces]
    colors = ['#e74c3c' if s == report['rankings']['by_cv_predictive'][0] else '#95a5a6' for s in spaces]
    axes[2].bar(x, cvs, yerr=cv_stds, color=colors, capsize=5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(spaces, rotation=45, ha='right')
    axes[2].set_ylabel('CV Log-Likelihood/transition')
    axes[2].set_title('Cross-Validated Predictive Power')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_selection.png'), dpi=300)
    plt.close()

    # 2. Entropy rate with CI
    fig, ax = plt.subplots(figsize=(10, 6))

    entropy_rates = [report['individual_results'][s]['entropy_rate'] for s in spaces]
    entropy_cis = [report['individual_results'][s]['entropy_rate_95ci'] for s in spaces]

    yerr = [[er - ci[0] for er, ci in zip(entropy_rates, entropy_cis)],
            [ci[1] - er for er, ci in zip(entropy_rates, entropy_cis)]]

    ax.bar(x, entropy_rates, yerr=yerr, capsize=5, color='#9b59b6', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(spaces, rotation=45, ha='right')
    ax.set_ylabel('Entropy Rate (bits)')
    ax.set_title('Markov Chain Entropy Rate with 95% Bootstrap CI')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_rate.png'), dpi=300)
    plt.close()

    # 3. Comparison heatmap
    comparison_keys = [k for k in report['comparisons'] if 'Random' not in k]
    if comparison_keys:
        fig, ax = plt.subplots(figsize=(10, 6))

        data = []
        labels = []
        for key in comparison_keys:
            comp = report['comparisons'][key]
            data.append([
                comp['nmi'],
                comp['ari'],
                comp['v_measure'],
                comp['information_retention']
            ])
            labels.append(key.replace('_vs_', ' vs '))

        data = np.array(data)

        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                    xticklabels=['NMI', 'ARI', 'V-measure', 'Info Retention'],
                    yticklabels=labels, ax=ax, vmin=0, vmax=1)
        ax.set_title('Pairwise State Space Comparisons')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_heatmap.png'), dpi=300)
        plt.close()

    print(f"Saved visualizations to {output_dir}")


def print_report(report: Dict):
    """Print formatted report to console."""
    print("\n" + "=" * 70)
    print("BENCHMARKING RESULTS")
    print("=" * 70)

    print(f"\nDataset: {report['summary']['n_individuals']} individuals, "
          f"{report['summary']['n_events']} events")

    print("\n" + "-" * 70)
    print("INDIVIDUAL STATE SPACE METRICS")
    print("-" * 70)
    print(f"{'Space':<15} {'K':>3} {'Entropy':>8} {'LL':>12} {'AIC':>10} {'BIC':>10} {'CV-LL':>10}")
    print("-" * 70)

    for name, res in report['individual_results'].items():
        if 'Random' not in name:
            print(f"{name:<15} {res['n_states']:>3} {res['entropy_rate']:>8.3f} "
                  f"{res['log_likelihood']:>12.1f} {res['aic']:>10.1f} "
                  f"{res['bic']:>10.1f} {res['cv_predictive_ll']:>10.4f}")

    print("\n" + "-" * 70)
    print("PAIRWISE COMPARISONS")
    print("-" * 70)
    print(f"{'Comparison':<25} {'NMI':>6} {'ARI':>6} {'Info%':>6} {'p-val':>8} {'Sig':>5}")
    print("-" * 70)

    for key, comp in report['comparisons'].items():
        if 'Random' not in key:
            sig = '*' if comp['permutation_test']['significant'] else ''
            print(f"{key.replace('_vs_', ' vs '):<25} "
                  f"{comp['nmi']:>6.3f} {comp['ari']:>6.3f} "
                  f"{comp['information_retention']*100:>5.1f}% "
                  f"{comp['permutation_test']['p_value']:>8.4f} {sig:>5}")

    print("\n" + "-" * 70)
    print("VS RANDOM BASELINES")
    print("-" * 70)

    for key, comp in report['comparisons'].items():
        if 'Random' in key:
            sig = '***' if comp['permutation_test']['p_value'] < 0.001 else \
                  '**' if comp['permutation_test']['p_value'] < 0.01 else \
                  '*' if comp['permutation_test']['p_value'] < 0.05 else ''
            print(f"{key.replace('_vs_', ' vs ')}: "
                  f"p = {comp['permutation_test']['p_value']:.4f} {sig}")

    print("\n" + "-" * 70)
    print("RANKINGS")
    print("-" * 70)
    print(f"Best by AIC:         {report['rankings']['by_aic'][0]}")
    print(f"Best by BIC:         {report['rankings']['by_bic'][0]}")
    print(f"Best by CV:          {report['rankings']['by_cv_predictive'][0]}")
    print(f"\nRECOMMENDATION: {report['summary']['recommendation']}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def load_sequences(analysis_dir: str) -> Dict[str, Dict]:
    """Load sequences from annotated CSVs."""
    sequences = {}
    annotated_files = glob.glob(os.path.join(analysis_dir, '*_annotated.csv'))

    for file_path in annotated_files:
        name = os.path.basename(file_path).replace('_annotated.csv', '')
        try:
            df = pd.read_csv(file_path)
            if 'PredictedLabel' in df.columns and 'Life Event' in df.columns:
                sequences[name] = {
                    'labels': df['PredictedLabel'].tolist(),
                    'events': df['Life Event'].fillna('').tolist()
                }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return sequences


def main():
    """Run the benchmarking framework."""
    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(analysis_dir, 'state_space_benchmark')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading sequences...")
    sequences = load_sequences(analysis_dir)
    print(f"Loaded {len(sequences)} individuals")

    if not sequences:
        print("ERROR: No sequences found. Run archetype_sequence_analysis.py first.")
        return

    # Run benchmark
    benchmark = StateSpaceBenchmark(sequences)
    report = benchmark.run_full_benchmark()

    # Print report
    print_report(report)

    # Generate visualizations
    plot_benchmark_comparison(report, output_dir)

    # Save report
    report_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
