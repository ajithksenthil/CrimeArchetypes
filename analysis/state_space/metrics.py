"""
State Space Comparison Metrics

Provides quantitative measures for comparing:
- Agreement between classification schemes
- Distribution divergence between state spaces
- Mapping quality and coverage
- Information-theoretic relationships

Metric Categories:
1. Distribution Metrics: KL/JS divergence, Earth Mover's Distance
2. Agreement Metrics: Cohen's Kappa, agreement rate, confusion matrix
3. Information Metrics: Mutual information, normalized MI, variation of info
4. Mapping Metrics: Coverage, ambiguity, confidence scores
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from collections import Counter


@dataclass
class ComparisonResult:
    """Result of comparing two state space classifications."""
    source_space: str
    target_space: str
    n_events: int
    metrics: Dict[str, float]
    details: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Comparison: {self.source_space} vs {self.target_space}",
            f"Events compared: {self.n_events}",
            "Metrics:"
        ]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            'source_space': self.source_space,
            'target_space': self.target_space,
            'n_events': self.n_events,
            'metrics': self.metrics,
            'details': self.details
        }


# ============================================================================
# DISTRIBUTION METRICS
# ============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(P || Q).

    Measures how much P diverges from Q (not symmetric).
    Lower values indicate more similar distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution (reference)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence in bits
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Add epsilon to avoid log(0)
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    return np.sum(p * np.log2(p / q))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence.

    Symmetric and bounded [0, 1]. Sqrt gives JS distance.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        JS divergence in [0, 1]
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return jensenshannon(p, q) ** 2  # scipy returns sqrt(JS)


def earth_movers_distance(
    p: np.ndarray,
    q: np.ndarray,
    distance_matrix: Optional[np.ndarray] = None
) -> float:
    """
    Compute Earth Mover's Distance (Wasserstein-1).

    For categorical distributions without explicit distance matrix,
    uses 1D Wasserstein on cumulative distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution
        distance_matrix: Optional pairwise distances between states

    Returns:
        EMD value
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    if distance_matrix is not None:
        # Full EMD with distance matrix (requires linear programming)
        # Simplified: use scipy's wasserstein for 1D case
        pass

    # 1D Wasserstein distance
    return stats.wasserstein_distance(
        np.arange(len(p)),
        np.arange(len(q)),
        p,
        q
    )


def distribution_overlap(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute overlap coefficient (Szymkiewicz-Simpson).

    Returns:
        Overlap in [0, 1], where 1 = identical distributions
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / p.sum()
    q = q / q.sum()

    return np.minimum(p, q).sum()


# ============================================================================
# AGREEMENT METRICS
# ============================================================================

def compute_confusion_matrix(
    labels_a: List[str],
    labels_b: List[str],
    states_a: List[str],
    states_b: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute confusion matrix between two labeling schemes.

    Args:
        labels_a: Classifications from scheme A
        labels_b: Classifications from scheme B
        states_a: Ordered state names for scheme A
        states_b: Ordered state names for scheme B

    Returns:
        Tuple of (confusion_matrix, row_labels, col_labels)
    """
    idx_a = {s: i for i, s in enumerate(states_a)}
    idx_b = {s: i for i, s in enumerate(states_b)}

    matrix = np.zeros((len(states_a), len(states_b)), dtype=int)

    for la, lb in zip(labels_a, labels_b):
        if la in idx_a and lb in idx_b:
            matrix[idx_a[la], idx_b[lb]] += 1

    return matrix, states_a, states_b


def agreement_rate(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Compute simple agreement rate (accuracy).

    Only meaningful when comparing same-size state spaces.

    Returns:
        Agreement rate in [0, 1]
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have same length")

    agreements = sum(a == b for a, b in zip(labels_a, labels_b))
    return agreements / len(labels_a)


def cohens_kappa(
    labels_a: List[str],
    labels_b: List[str],
    states: Optional[List[str]] = None
) -> float:
    """
    Compute Cohen's Kappa coefficient for inter-rater agreement.

    Accounts for chance agreement. Values:
    - < 0: Less than chance agreement
    - 0-0.20: Slight agreement
    - 0.21-0.40: Fair agreement
    - 0.41-0.60: Moderate agreement
    - 0.61-0.80: Substantial agreement
    - 0.81-1.00: Almost perfect agreement

    Returns:
        Kappa coefficient in [-1, 1]
    """
    if states is None:
        states = sorted(set(labels_a) | set(labels_b))

    n = len(labels_a)
    idx = {s: i for i, s in enumerate(states)}
    k = len(states)

    # Confusion matrix
    matrix = np.zeros((k, k))
    for la, lb in zip(labels_a, labels_b):
        if la in idx and lb in idx:
            matrix[idx[la], idx[lb]] += 1

    # Observed agreement
    p_o = np.trace(matrix) / n

    # Expected agreement by chance
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = np.sum(row_sums * col_sums)

    # Kappa
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def weighted_kappa(
    labels_a: List[str],
    labels_b: List[str],
    states: List[str],
    weights: str = 'linear'
) -> float:
    """
    Compute weighted Kappa (for ordinal categories).

    Args:
        labels_a: First set of labels
        labels_b: Second set of labels
        states: Ordered list of states
        weights: 'linear' or 'quadratic'

    Returns:
        Weighted Kappa coefficient
    """
    n = len(labels_a)
    k = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # Weight matrix
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == 'linear':
                W[i, j] = abs(i - j) / (k - 1)
            else:  # quadratic
                W[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    # Observed matrix
    O = np.zeros((k, k))
    for la, lb in zip(labels_a, labels_b):
        if la in idx and lb in idx:
            O[idx[la], idx[lb]] += 1
    O = O / n

    # Expected matrix
    row_marginals = O.sum(axis=1)
    col_marginals = O.sum(axis=0)
    E = np.outer(row_marginals, col_marginals)

    # Weighted kappa
    return 1 - (np.sum(W * O) / np.sum(W * E))


# ============================================================================
# INFORMATION-THEORETIC METRICS
# ============================================================================

def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Shannon entropy of a distribution.

    Args:
        p: Probability distribution
        epsilon: Small value to avoid log(0)

    Returns:
        Entropy in bits
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    p = np.clip(p, epsilon, 1)
    return -np.sum(p * np.log2(p))


def joint_entropy(
    labels_a: List[str],
    labels_b: List[str],
    epsilon: float = 1e-10
) -> float:
    """
    Compute joint entropy H(A, B).

    Returns:
        Joint entropy in bits
    """
    joint_counts = Counter(zip(labels_a, labels_b))
    n = len(labels_a)
    probs = np.array(list(joint_counts.values())) / n
    return entropy(probs, epsilon)


def mutual_information(
    labels_a: List[str],
    labels_b: List[str],
    epsilon: float = 1e-10
) -> float:
    """
    Compute mutual information I(A; B).

    Measures how much knowing one labeling reduces uncertainty about the other.

    Returns:
        Mutual information in bits
    """
    # Marginal entropies
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    n = len(labels_a)

    p_a = np.array(list(counts_a.values())) / n
    p_b = np.array(list(counts_b.values())) / n

    h_a = entropy(p_a, epsilon)
    h_b = entropy(p_b, epsilon)
    h_ab = joint_entropy(labels_a, labels_b, epsilon)

    return h_a + h_b - h_ab


def normalized_mutual_information(
    labels_a: List[str],
    labels_b: List[str],
    method: str = 'arithmetic'
) -> float:
    """
    Compute Normalized Mutual Information (NMI).

    Bounded in [0, 1] regardless of number of states.

    Args:
        labels_a: First labeling
        labels_b: Second labeling
        method: Normalization method ('arithmetic', 'geometric', 'min', 'max')

    Returns:
        NMI in [0, 1]
    """
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    n = len(labels_a)

    p_a = np.array(list(counts_a.values())) / n
    p_b = np.array(list(counts_b.values())) / n

    h_a = entropy(p_a)
    h_b = entropy(p_b)
    mi = mutual_information(labels_a, labels_b)

    if method == 'arithmetic':
        normalizer = (h_a + h_b) / 2
    elif method == 'geometric':
        normalizer = np.sqrt(h_a * h_b)
    elif method == 'min':
        normalizer = min(h_a, h_b)
    elif method == 'max':
        normalizer = max(h_a, h_b)
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalizer == 0:
        return 1.0  # Both are deterministic

    return mi / normalizer


def variation_of_information(
    labels_a: List[str],
    labels_b: List[str]
) -> float:
    """
    Compute Variation of Information (VI).

    A true metric (satisfies triangle inequality).
    VI = H(A|B) + H(B|A) = H(A) + H(B) - 2*I(A;B)

    Returns:
        VI in bits (0 = identical labelings)
    """
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    n = len(labels_a)

    p_a = np.array(list(counts_a.values())) / n
    p_b = np.array(list(counts_b.values())) / n

    h_a = entropy(p_a)
    h_b = entropy(p_b)
    mi = mutual_information(labels_a, labels_b)

    return h_a + h_b - 2 * mi


# ============================================================================
# MAPPING QUALITY METRICS
# ============================================================================

def mapping_coverage(
    mapping_dict: Dict[str, str],
    source_labels: List[str]
) -> float:
    """
    Compute what fraction of source labels have a mapping.

    Returns:
        Coverage in [0, 1]
    """
    covered = sum(1 for label in source_labels if label in mapping_dict)
    return covered / len(source_labels) if source_labels else 0.0


def mapping_ambiguity(
    mapping_dict: Dict[str, Dict[str, float]]
) -> float:
    """
    Compute average ambiguity of mappings.

    For probabilistic mappings, returns average entropy of target distributions.
    Lower = more deterministic mappings.

    Args:
        mapping_dict: Dict of source_state -> {target_state: probability}

    Returns:
        Average entropy of mappings
    """
    entropies = []
    for source, targets in mapping_dict.items():
        if targets:
            probs = np.array(list(targets.values()))
            entropies.append(entropy(probs))

    return np.mean(entropies) if entropies else 0.0


def mapping_confidence_score(
    confidences: List[float]
) -> Dict[str, float]:
    """
    Compute summary statistics of mapping confidences.

    Returns:
        Dict with mean, median, std, and percentage high confidence (>0.8)
    """
    confidences = np.array(confidences)
    return {
        'mean': float(np.mean(confidences)),
        'median': float(np.median(confidences)),
        'std': float(np.std(confidences)),
        'pct_high': float(np.mean(confidences > 0.8) * 100)
    }


# ============================================================================
# COMPREHENSIVE COMPARISON FUNCTION
# ============================================================================

def compare_state_spaces(
    labels_a: List[str],
    labels_b: List[str],
    states_a: List[str],
    states_b: List[str],
    name_a: str = "Space A",
    name_b: str = "Space B",
    mapping: Optional[Dict[str, str]] = None
) -> ComparisonResult:
    """
    Comprehensive comparison between two state space classifications.

    Args:
        labels_a: Classifications from state space A
        labels_b: Classifications from state space B
        states_a: State names for space A
        states_b: State names for space B
        name_a: Display name for space A
        name_b: Display name for space B
        mapping: Optional mapping from A states to B states for agreement metrics

    Returns:
        ComparisonResult with all computed metrics
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have same length")

    n = len(labels_a)

    # Compute distributions
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)

    dist_a = np.array([counts_a.get(s, 0) for s in states_a]) / n
    dist_b = np.array([counts_b.get(s, 0) for s in states_b]) / n

    metrics = {}
    details = {}

    # Distribution metrics (only if same number of states)
    if len(states_a) == len(states_b):
        metrics['js_divergence'] = js_divergence(dist_a, dist_b)
        metrics['distribution_overlap'] = distribution_overlap(dist_a, dist_b)

    # Information-theoretic metrics
    metrics['mutual_information'] = mutual_information(labels_a, labels_b)
    metrics['normalized_mi'] = normalized_mutual_information(labels_a, labels_b)
    metrics['variation_of_info'] = variation_of_information(labels_a, labels_b)

    # Entropy of each space
    metrics['entropy_a'] = entropy(dist_a)
    metrics['entropy_b'] = entropy(dist_b)

    # If mapping provided, compute agreement metrics
    if mapping:
        mapped_a = [mapping.get(la, la) for la in labels_a]
        if set(mapped_a).issubset(set(labels_b) | set(states_b)):
            metrics['agreement_rate'] = agreement_rate(mapped_a, labels_b)
            metrics['cohens_kappa'] = cohens_kappa(mapped_a, labels_b, states_b)

    # Confusion matrix
    conf_matrix, row_labels, col_labels = compute_confusion_matrix(
        labels_a, labels_b, states_a, states_b
    )
    details['confusion_matrix'] = conf_matrix.tolist()
    details['confusion_row_labels'] = row_labels
    details['confusion_col_labels'] = col_labels

    # Distribution details
    details['distribution_a'] = {s: float(dist_a[i]) for i, s in enumerate(states_a)}
    details['distribution_b'] = {s: float(dist_b[i]) for i, s in enumerate(states_b)}

    return ComparisonResult(
        source_space=name_a,
        target_space=name_b,
        n_events=n,
        metrics=metrics,
        details=details
    )


def compare_transition_matrices(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    states_a: List[str],
    states_b: List[str],
    name_a: str = "Space A",
    name_b: str = "Space B"
) -> Dict[str, float]:
    """
    Compare two transition matrices.

    Args:
        matrix_a: Transition matrix for space A
        matrix_b: Transition matrix for space B
        states_a: State names for space A
        states_b: State names for space B

    Returns:
        Dict of comparison metrics
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Transition matrices must have same shape for comparison")

    metrics = {}

    # Frobenius norm of difference
    metrics['frobenius_distance'] = float(np.linalg.norm(matrix_a - matrix_b, 'fro'))

    # Average row-wise JS divergence
    js_vals = []
    for i in range(matrix_a.shape[0]):
        row_a = matrix_a[i]
        row_b = matrix_b[i]
        if row_a.sum() > 0 and row_b.sum() > 0:
            js_vals.append(js_divergence(row_a, row_b))
    metrics['avg_row_js_divergence'] = float(np.mean(js_vals)) if js_vals else 0.0

    # Correlation of flattened matrices
    metrics['matrix_correlation'] = float(np.corrcoef(
        matrix_a.flatten(),
        matrix_b.flatten()
    )[0, 1])

    # Compare stationary distributions
    def get_stationary(M):
        eigenvalues, eigenvectors = np.linalg.eig(M.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        stat = np.real(eigenvectors[:, idx])
        return stat / stat.sum()

    try:
        stat_a = get_stationary(matrix_a)
        stat_b = get_stationary(matrix_b)
        metrics['stationary_js_divergence'] = float(js_divergence(stat_a, stat_b))
    except Exception:
        pass  # May fail if matrix is degenerate

    return metrics
