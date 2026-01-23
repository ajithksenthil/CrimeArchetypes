"""
Unified Markov Analysis Module

Provides consistent Markov chain analysis methods that work with both
the K-Cluster and 4-Animal state space representations.

Key Features:
    - Dirichlet-smoothed kernel estimation
    - Boltzmann transition probabilities
    - Stationary distribution computation
    - Entropy rate calculation
    - Mean first passage times
    - Transfer entropy (cross-individual influence)
    - Time-block analysis for temporal dynamics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarkovAnalysisResult:
    """Container for Markov analysis results."""
    kernel: np.ndarray
    stationary_distribution: np.ndarray
    entropy_rate: float
    mean_first_passage_time: np.ndarray
    transition_counts: np.ndarray
    n_states: int
    state_labels: Optional[List[str]] = None


class UnifiedMarkovAnalysis:
    """
    Unified Markov chain analysis for behavioral sequences.

    Works with any discrete state space representation (K-Cluster or 4-Animal).
    Implements methods from Computational Psychodynamics framework.
    """

    def __init__(self, n_states: int, alpha: float = 1.0, state_labels: Optional[List[str]] = None):
        """
        Initialize analysis.

        Args:
            n_states: Number of states in the Markov chain
            alpha: Dirichlet smoothing parameter (α ≥ 0)
                   α = 0: Maximum likelihood (no smoothing)
                   α = 1: Laplace smoothing
                   α > 1: Strong smoothing toward uniform
            state_labels: Optional names for each state
        """
        self.n_states = n_states
        self.alpha = alpha
        self.state_labels = state_labels or [str(i) for i in range(n_states)]

    def count_transitions(self, sequence: List[int]) -> np.ndarray:
        """
        Count transitions in a single sequence.

        Args:
            sequence: List of state indices (0 to n_states-1)

        Returns:
            n_states x n_states count matrix
        """
        counts = np.zeros((self.n_states, self.n_states))

        for i in range(len(sequence) - 1):
            from_state = sequence[i]
            to_state = sequence[i + 1]
            if 0 <= from_state < self.n_states and 0 <= to_state < self.n_states:
                counts[from_state, to_state] += 1

        return counts

    def aggregate_counts(self, sequences: Dict[str, List[int]]) -> np.ndarray:
        """
        Aggregate transition counts from multiple sequences.

        Args:
            sequences: Dict mapping ID -> sequence of states

        Returns:
            Aggregated count matrix
        """
        total_counts = np.zeros((self.n_states, self.n_states))

        for seq in sequences.values():
            total_counts += self.count_transitions(seq)

        return total_counts

    def dirichlet_kernel(self, counts: np.ndarray) -> np.ndarray:
        """
        Compute Dirichlet-smoothed transition kernel.

        K_ij = (N_ij + α) / Σ_k(N_ik + α)

        This is Eq. 7 from the Computational Psychodynamics paper.

        Args:
            counts: Raw transition count matrix

        Returns:
            Smoothed transition probability matrix
        """
        smoothed = counts + self.alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return smoothed / row_sums

    def boltzmann_kernel(self, kernel: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Apply Boltzmann rule to transition kernel.

        Pr(j|i) = exp(-E_ij/T) / Σ_k exp(-E_ik/T)

        where E_ij = -log(K_ij) is the "energy" of transition i→j.

        This is related to Eq. 8 from the paper.

        Args:
            kernel: Dirichlet-smoothed kernel
            temperature: T parameter (lower = more deterministic)

        Returns:
            Boltzmann-weighted transition matrix
        """
        eps = 1e-10
        energy = -np.log(kernel + eps)
        boltzmann = np.exp(-energy / temperature)
        row_sums = boltzmann.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return boltzmann / row_sums

    def stationary_distribution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Compute stationary distribution π where πK = π.

        Uses eigendecomposition to find the left eigenvector
        corresponding to eigenvalue 1.

        Args:
            kernel: Transition probability matrix

        Returns:
            Stationary distribution vector
        """
        eigenvalues, eigenvectors = np.linalg.eig(kernel.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = eigenvectors[:, idx].real
        stationary = np.abs(stationary)
        stationary /= stationary.sum()
        return stationary

    def entropy_rate(self, kernel: np.ndarray) -> float:
        """
        Compute entropy rate of the Markov chain.

        H(K) = -Σ_i π_i Σ_j K_ij log₂(K_ij)

        Lower entropy = more predictable behavior.

        Args:
            kernel: Transition probability matrix

        Returns:
            Entropy rate in bits
        """
        pi = self.stationary_distribution(kernel)
        eps = 1e-10
        log_kernel = np.log2(kernel + eps)
        conditional_entropy = -np.sum(kernel * log_kernel, axis=1)
        return float(np.sum(pi * conditional_entropy))

    def mean_first_passage_time(self, kernel: np.ndarray) -> np.ndarray:
        """
        Compute mean first passage times between states.

        MFPT[i,j] = expected number of steps to reach j from i.

        Uses the fundamental matrix approach.

        Args:
            kernel: Transition probability matrix

        Returns:
            n_states x n_states MFPT matrix
        """
        n = self.n_states

        try:
            pi = self.stationary_distribution(kernel)
            I = np.eye(n)
            ones_pi = np.outer(np.ones(n), pi)
            Z = np.linalg.inv(I - kernel + ones_pi)
            D = np.diag(np.diag(Z))
            mfpt = (I - Z + np.ones((n, n)) @ D) / np.diag(Z)
            return mfpt

        except np.linalg.LinAlgError:
            logger.warning("Could not compute MFPT - matrix may be singular")
            return np.full((n, n), np.inf)

    def mixing_time(self, kernel: np.ndarray, epsilon: float = 0.01) -> float:
        """
        Estimate mixing time of the Markov chain.

        Time for the chain to approach stationarity within epsilon
        in total variation distance.

        Args:
            kernel: Transition probability matrix
            epsilon: Convergence threshold

        Returns:
            Estimated mixing time (number of steps)
        """
        eigenvalues = np.linalg.eigvals(kernel)
        eigenvalues = np.abs(eigenvalues)
        eigenvalues.sort()

        # Second largest eigenvalue
        lambda_2 = eigenvalues[-2] if len(eigenvalues) > 1 else 0.5

        if lambda_2 >= 1.0:
            return np.inf

        # Mixing time bound: t_mix ≈ 1/(1-λ₂) * log(1/ε)
        spectral_gap = 1 - lambda_2
        return (1 / spectral_gap) * np.log(1 / epsilon)

    def analyze_sequence(self, sequence: List[int]) -> MarkovAnalysisResult:
        """
        Full analysis of a single sequence.

        Args:
            sequence: List of state indices

        Returns:
            MarkovAnalysisResult with all metrics
        """
        counts = self.count_transitions(sequence)
        kernel = self.dirichlet_kernel(counts)
        stationary = self.stationary_distribution(kernel)
        entropy = self.entropy_rate(kernel)
        mfpt = self.mean_first_passage_time(kernel)

        return MarkovAnalysisResult(
            kernel=kernel,
            stationary_distribution=stationary,
            entropy_rate=entropy,
            mean_first_passage_time=mfpt,
            transition_counts=counts,
            n_states=self.n_states,
            state_labels=self.state_labels
        )

    def analyze_multiple(self, sequences: Dict[str, List[int]]) -> Tuple[MarkovAnalysisResult, Dict[str, MarkovAnalysisResult]]:
        """
        Analyze multiple sequences and compute aggregate.

        Args:
            sequences: Dict mapping ID -> sequence

        Returns:
            (aggregate_result, individual_results_dict)
        """
        individual_results = {}

        for name, seq in sequences.items():
            individual_results[name] = self.analyze_sequence(seq)

        # Aggregate
        total_counts = self.aggregate_counts(sequences)
        agg_kernel = self.dirichlet_kernel(total_counts)
        agg_stationary = self.stationary_distribution(agg_kernel)
        agg_entropy = self.entropy_rate(agg_kernel)
        agg_mfpt = self.mean_first_passage_time(agg_kernel)

        aggregate = MarkovAnalysisResult(
            kernel=agg_kernel,
            stationary_distribution=agg_stationary,
            entropy_rate=agg_entropy,
            mean_first_passage_time=agg_mfpt,
            transition_counts=total_counts,
            n_states=self.n_states,
            state_labels=self.state_labels
        )

        return aggregate, individual_results


# =============================================================================
# TRANSFER ENTROPY (Cross-Sequence Influence)
# =============================================================================

class TransferEntropyAnalysis:
    """
    Transfer entropy analysis for detecting causal influence between sequences.

    TE(X→Y) measures how much information about Y's future is provided by X
    beyond what Y's past provides.
    """

    def __init__(self, n_states: int, lag: int = 1):
        """
        Args:
            n_states: Number of states
            lag: Time lag for prediction
        """
        self.n_states = n_states
        self.lag = lag

    def compute_transfer_entropy(
        self,
        source: List[int],
        target: List[int]
    ) -> float:
        """
        Compute transfer entropy from source to target.

        TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) log[ p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t) ]

        Args:
            source: Source sequence X
            target: Target sequence Y

        Returns:
            Transfer entropy in bits
        """
        min_len = min(len(source), len(target)) - self.lag
        if min_len < 10:
            return 0.0

        # Count joint distributions
        joint_yyy = defaultdict(int)  # p(y_{t+1}, y_t, x_t)
        joint_yy = defaultdict(int)   # p(y_{t+1}, y_t)
        joint_yx = defaultdict(int)   # p(y_t, x_t)
        marginal_y = defaultdict(int) # p(y_t)

        for t in range(min_len):
            y_next = target[t + self.lag]
            y_curr = target[t]
            x_curr = source[t]

            joint_yyy[(y_next, y_curr, x_curr)] += 1
            joint_yy[(y_next, y_curr)] += 1
            joint_yx[(y_curr, x_curr)] += 1
            marginal_y[y_curr] += 1

        total = min_len
        eps = 1e-10

        te = 0.0
        for (y_next, y_curr, x_curr), count in joint_yyy.items():
            p_yyy = count / total + eps
            p_yy = joint_yy[(y_next, y_curr)] / total + eps
            p_yx = joint_yx[(y_curr, x_curr)] / total + eps
            p_y = marginal_y[y_curr] / total + eps

            te += p_yyy * np.log2(p_yyy * p_y / (p_yx * p_yy))

        return max(0, te)

    def compute_influence_matrix(
        self,
        sequences: Dict[str, List[int]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise transfer entropy matrix.

        Args:
            sequences: Dict mapping ID -> sequence

        Returns:
            (influence_matrix, sequence_names)
        """
        names = list(sequences.keys())
        n = len(names)
        matrix = np.zeros((n, n))

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i != j:
                    te = self.compute_transfer_entropy(
                        sequences[name_i],
                        sequences[name_j]
                    )
                    matrix[i, j] = te

        return matrix, names


# =============================================================================
# TIME-BLOCK ANALYSIS
# =============================================================================

class TimeBlockAnalysis:
    """
    Analyze how transition dynamics change over time.

    Divides sequences into temporal blocks and computes
    separate kernels for each block to detect non-stationarity.
    """

    def __init__(self, n_states: int, n_blocks: int = 3, alpha: float = 1.0):
        """
        Args:
            n_states: Number of states
            n_blocks: Number of temporal blocks
            alpha: Dirichlet smoothing parameter
        """
        self.n_states = n_states
        self.n_blocks = n_blocks
        self.alpha = alpha
        self.markov = UnifiedMarkovAnalysis(n_states, alpha)

    def split_sequence(self, sequence: List[int]) -> List[List[int]]:
        """Split sequence into temporal blocks."""
        n = len(sequence)
        block_size = n // self.n_blocks

        blocks = []
        for i in range(self.n_blocks):
            start = i * block_size
            end = start + block_size if i < self.n_blocks - 1 else n
            blocks.append(sequence[start:end])

        return blocks

    def analyze_temporal_dynamics(
        self,
        sequences: Dict[str, List[int]]
    ) -> Dict:
        """
        Analyze how dynamics change across time blocks.

        Args:
            sequences: Dict mapping ID -> sequence

        Returns:
            Dict with block-wise kernels and drift metrics
        """
        results = {
            "blocks": [],
            "drift_metrics": {}
        }

        # Aggregate sequences by block
        block_sequences = [defaultdict(list) for _ in range(self.n_blocks)]

        for name, seq in sequences.items():
            blocks = self.split_sequence(seq)
            for i, block in enumerate(blocks):
                block_sequences[i][name] = block

        # Analyze each block
        block_kernels = []
        block_stationaries = []

        for i, block_seqs in enumerate(block_sequences):
            # Convert list-of-lists to single sequences for aggregate
            counts = np.zeros((self.n_states, self.n_states))
            for seq in block_seqs.values():
                counts += self.markov.count_transitions(seq)

            kernel = self.markov.dirichlet_kernel(counts)
            stationary = self.markov.stationary_distribution(kernel)
            entropy = self.markov.entropy_rate(kernel)

            results["blocks"].append({
                "block": i,
                "kernel": kernel.tolist(),
                "stationary": stationary.tolist(),
                "entropy_rate": entropy
            })

            block_kernels.append(kernel)
            block_stationaries.append(stationary)

        # Compute drift metrics
        if len(block_kernels) >= 2:
            # Kernel drift: Frobenius norm of difference
            kernel_drifts = []
            for i in range(len(block_kernels) - 1):
                drift = np.linalg.norm(block_kernels[i+1] - block_kernels[i], 'fro')
                kernel_drifts.append(float(drift))

            results["drift_metrics"]["kernel_drift"] = kernel_drifts
            results["drift_metrics"]["total_kernel_drift"] = float(
                np.linalg.norm(block_kernels[-1] - block_kernels[0], 'fro')
            )

            # Stationary drift: KL divergence
            def kl_divergence(p, q):
                eps = 1e-10
                return float(np.sum(p * np.log2((p + eps) / (q + eps))))

            stationary_drifts = []
            for i in range(len(block_stationaries) - 1):
                drift = kl_divergence(block_stationaries[i+1], block_stationaries[i])
                stationary_drifts.append(drift)

            results["drift_metrics"]["stationary_drift"] = stationary_drifts

        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_kernels(kernel1: np.ndarray, kernel2: np.ndarray) -> Dict[str, float]:
    """
    Compare two transition kernels.

    Args:
        kernel1, kernel2: Transition matrices to compare

    Returns:
        Dict with comparison metrics
    """
    # Frobenius norm difference
    frobenius = float(np.linalg.norm(kernel1 - kernel2, 'fro'))

    # KL divergence (row-wise, then averaged)
    eps = 1e-10
    kl_rows = []
    for i in range(kernel1.shape[0]):
        kl = np.sum(kernel1[i] * np.log2((kernel1[i] + eps) / (kernel2[i] + eps)))
        kl_rows.append(kl)
    kl_avg = float(np.mean(kl_rows))

    # Maximum absolute difference
    max_diff = float(np.max(np.abs(kernel1 - kernel2)))

    return {
        "frobenius_norm": frobenius,
        "kl_divergence_avg": kl_avg,
        "max_absolute_diff": max_diff
    }


def effective_number_of_states(stationary: np.ndarray) -> float:
    """
    Compute effective number of states (perplexity of stationary).

    A uniform distribution over K states has effective_states = K.
    Concentrated distributions have fewer effective states.

    Args:
        stationary: Stationary distribution

    Returns:
        Effective number of states
    """
    eps = 1e-10
    entropy = -np.sum(stationary * np.log2(stationary + eps))
    return float(2 ** entropy)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Unified Markov Analysis Module")
    print("="*50)

    # Create example sequences
    np.random.seed(42)
    n_states = 4  # 4-Animal state space

    # Generate synthetic sequences
    sequences = {
        "criminal_1": list(np.random.choice(n_states, size=50)),
        "criminal_2": list(np.random.choice(n_states, size=40)),
        "criminal_3": list(np.random.choice(n_states, size=60))
    }

    # Run analysis
    markov = UnifiedMarkovAnalysis(n_states, alpha=1.0, state_labels=['Seeking', 'Directing', 'Conferring', 'Revising'])
    aggregate, individuals = markov.analyze_multiple(sequences)

    print("\nAggregate Analysis:")
    print(f"  Entropy Rate: {aggregate.entropy_rate:.4f} bits")
    print(f"  Stationary Distribution: {aggregate.stationary_distribution}")

    print("\nTransfer Entropy Analysis:")
    te_analysis = TransferEntropyAnalysis(n_states)
    te_matrix, names = te_analysis.compute_influence_matrix(sequences)
    print(f"  Influence Matrix Shape: {te_matrix.shape}")

    print("\nTime-Block Analysis:")
    time_analysis = TimeBlockAnalysis(n_states, n_blocks=3)
    temporal = time_analysis.analyze_temporal_dynamics(sequences)
    print(f"  Kernel Drifts: {temporal['drift_metrics'].get('kernel_drift', 'N/A')}")
