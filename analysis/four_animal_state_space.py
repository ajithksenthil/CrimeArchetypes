"""
Four-Animal State Space Analysis for Criminal Life Events

This module implements the Computational Psychodynamics framework from
"Computational Psychodynamics" (Senthil, 2025) for analyzing serial killer
life events using a principled 4-state behavioral model.

The Four Animals:
    - SEEKING (Self + Explore): Introspective curiosity, self-discovery
    - DIRECTING (Other + Exploit): Manipulating/controlling others
    - CONFERRING (Other + Explore): Observing, learning from others
    - REVISING (Self + Exploit): Self-regulatory, habit-driven behavior

Key features:
    - LLM-based event classification into 4 behavioral states
    - Dirichlet-smoothed kernel estimation (Eq. 7 from paper)
    - Boltzmann rule for transition probabilities (Eq. 8 from paper)
    - Transfer entropy for causal influence analysis
    - Trait proxy extraction from kernel statistics
"""

import os
import csv
import json
import numpy as np
from enum import IntEnum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pickle
import logging

# For embeddings and similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For LLM classification
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# FOUR ANIMAL STATE DEFINITIONS
# =============================================================================

class Animal(IntEnum):
    """
    The four behavioral states from Computational Psychodynamics.

    Derived from the 2x2 crossing of:
        - Target: Self vs Other
        - Mode: Explore vs Exploit
    """
    SEEKING = 0     # Self + Explore: introspective curiosity
    DIRECTING = 1   # Other + Exploit: controlling/manipulating others
    CONFERRING = 2  # Other + Explore: observing, learning from others
    REVISING = 3    # Self + Exploit: self-regulatory, habitual behavior

ANIMAL_NAMES = {
    Animal.SEEKING: "Seeking",
    Animal.DIRECTING: "Directing",
    Animal.CONFERRING: "Conferring",
    Animal.REVISING: "Revising"
}

ANIMAL_DESCRIPTIONS = {
    Animal.SEEKING: "Self-exploration, introspection, curiosity about inner states",
    Animal.DIRECTING: "Controlling, manipulating, exploiting others for personal gain",
    Animal.CONFERRING: "Observing others, social learning, exploring relationships",
    Animal.REVISING: "Self-regulation, habit formation, reinforcing learned behaviors"
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ClassifiedEvent:
    """A life event with its classified Animal state."""
    text: str
    animal: Animal
    confidence: float
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


@dataclass
class CriminalSequence:
    """A sequence of classified events for one individual."""
    name: str
    events: List[ClassifiedEvent]
    transition_counts: Optional[np.ndarray] = None
    kernel: Optional[np.ndarray] = None


# =============================================================================
# LLM-BASED ANIMAL CLASSIFIER
# =============================================================================

class AnimalClassifier:
    """
    Classifies life events into the 4-Animal state space using LLM.

    Two classification modes:
        1. LLM-based: Uses GPT to classify based on behavioral semantics
        2. Prototype-based: Uses embeddings and learned prototypes
    """

    CLASSIFICATION_PROMPT = """You are an expert in behavioral psychology analyzing life events of individuals.

Classify the following life event into ONE of these four behavioral categories:

1. SEEKING (Self + Explore): Events involving self-discovery, introspection, exploring one's own thoughts/feelings, curiosity about oneself, internal struggles, identity formation.
   Examples: "questioned his identity", "became obsessed with morbid thoughts", "explored dark fantasies"

2. DIRECTING (Other + Exploit): Events involving controlling, manipulating, or exploiting others for personal gain. Includes acts of dominance, coercion, abuse.
   Examples: "manipulated his victims", "exerted control over family members", "exploited vulnerable individuals"

3. CONFERRING (Other + Explore): Events involving observing others, social learning, exploring relationships, watching/studying people. Passive engagement with the social world.
   Examples: "observed neighbors through windows", "studied crime documentaries", "watched potential victims"

4. REVISING (Self + Exploit): Events involving self-regulation, reinforcing habits, compulsive behaviors, addiction, ritualistic actions, maintaining established patterns.
   Examples: "developed ritualistic behaviors", "returned to same hunting grounds", "maintained strict daily routines"

Life Event: {event}

Respond with ONLY the category name (SEEKING, DIRECTING, CONFERRING, or REVISING) followed by a confidence score from 0-100.
Format: CATEGORY|CONFIDENCE

Example response: DIRECTING|85"""

    def __init__(self, model: str = "gpt-4o-mini", use_embeddings: bool = True):
        """
        Initialize the classifier.

        Args:
            model: OpenAI model to use for classification
            use_embeddings: Whether to also compute embeddings for events
        """
        self.model = model
        self.use_embeddings = use_embeddings

        if use_embeddings:
            logger.info("Loading SentenceTransformer for embeddings...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None

        # Prototype embeddings for each Animal (can be learned)
        self.prototypes: Dict[Animal, np.ndarray] = {}

        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment")

    def classify_event(self, event_text: str) -> ClassifiedEvent:
        """
        Classify a single event into an Animal state.

        Args:
            event_text: The life event text to classify

        Returns:
            ClassifiedEvent with animal state and confidence
        """
        # LLM-based classification
        animal, confidence = self._llm_classify(event_text)

        # Compute embedding if enabled
        embedding = None
        if self.use_embeddings and self.embedding_model:
            embedding = self.embedding_model.encode([event_text])[0]

        return ClassifiedEvent(
            text=event_text,
            animal=animal,
            confidence=confidence,
            embedding=embedding
        )

    def _llm_classify(self, event_text: str) -> Tuple[Animal, float]:
        """Use LLM to classify event into Animal state."""
        try:
            client = openai.OpenAI()
            prompt = self.CLASSIFICATION_PROMPT.format(event=event_text)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1  # Low temp for consistent classification
            )

            reply = response.choices[0].message.content.strip().upper()

            # Parse response
            if "|" in reply:
                parts = reply.split("|")
                animal_str = parts[0].strip()
                confidence = float(parts[1].strip()) / 100.0
            else:
                animal_str = reply.strip()
                confidence = 0.5

            # Map string to Animal enum
            animal_map = {
                "SEEKING": Animal.SEEKING,
                "DIRECTING": Animal.DIRECTING,
                "CONFERRING": Animal.CONFERRING,
                "REVISING": Animal.REVISING
            }

            animal = animal_map.get(animal_str, Animal.SEEKING)
            return animal, confidence

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return Animal.SEEKING, 0.0

    def classify_batch(self, events: List[str], show_progress: bool = True) -> List[ClassifiedEvent]:
        """Classify multiple events."""
        classified = []
        total = len(events)

        for i, event in enumerate(events):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Classifying event {i+1}/{total}...")

            classified.append(self.classify_event(event))

        return classified

    def learn_prototypes(self, classified_events: List[ClassifiedEvent]):
        """
        Learn prototype embeddings for each Animal from classified events.

        This enables hybrid classification using both LLM and embeddings.
        """
        if not self.use_embeddings:
            logger.warning("Embeddings disabled, cannot learn prototypes")
            return

        # Group embeddings by Animal
        animal_embeddings = defaultdict(list)
        for event in classified_events:
            if event.embedding is not None:
                animal_embeddings[event.animal].append(event.embedding)

        # Compute mean prototype for each Animal
        for animal in Animal:
            if animal_embeddings[animal]:
                self.prototypes[animal] = np.mean(animal_embeddings[animal], axis=0)
                logger.info(f"Learned prototype for {ANIMAL_NAMES[animal]} from {len(animal_embeddings[animal])} events")

    def save_prototypes(self, filepath: str):
        """Save learned prototypes to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.prototypes, f)
        logger.info(f"Prototypes saved to {filepath}")

    def load_prototypes(self, filepath: str):
        """Load prototypes from file."""
        with open(filepath, 'rb') as f:
            self.prototypes = pickle.load(f)
        logger.info(f"Prototypes loaded from {filepath}")


# =============================================================================
# MARKOV CHAIN ANALYSIS WITH DIRICHLET SMOOTHING
# =============================================================================

class FourAnimalMarkovAnalysis:
    """
    Markov chain analysis on the 4-Animal state space.

    Implements:
        - Dirichlet-smoothed kernel estimation (Eq. 7)
        - Boltzmann transition probabilities (Eq. 8)
        - Transfer entropy for causal influence
        - Trait proxy extraction
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize Markov analysis.

        Args:
            alpha: Dirichlet smoothing parameter (α in Eq. 7)
                   Higher α = more smoothing toward uniform
                   α = 1 gives Laplace smoothing
        """
        self.alpha = alpha
        self.n_states = 4  # Four Animals

    def compute_transition_counts(self, sequence: List[Animal]) -> np.ndarray:
        """
        Count transitions between Animal states.

        Args:
            sequence: List of Animal states in temporal order

        Returns:
            4x4 matrix of raw transition counts
        """
        counts = np.zeros((self.n_states, self.n_states))

        for i in range(len(sequence) - 1):
            from_state = int(sequence[i])
            to_state = int(sequence[i + 1])
            counts[from_state, to_state] += 1

        return counts

    def dirichlet_kernel(self, counts: np.ndarray) -> np.ndarray:
        """
        Compute Dirichlet-smoothed transition kernel (Eq. 7 from paper).

        K_ij(t) = (N_ij(t) + α) / Σ_k(N_ik(t) + α)

        Args:
            counts: Raw transition count matrix

        Returns:
            Smoothed probability kernel
        """
        # Add Dirichlet prior (smoothing)
        smoothed = counts + self.alpha

        # Normalize rows to get probabilities
        row_sums = smoothed.sum(axis=1, keepdims=True)

        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)

        kernel = smoothed / row_sums

        return kernel

    def boltzmann_kernel(self, kernel: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Apply Boltzmann rule to transition kernel (Eq. 8 from paper).

        Pr(Z_{t+1}=j|Z_t=i) = exp(-ΔΨ(i→j)/T) / Σ_k exp(-ΔΨ(i→k)/T)

        Here we interpret -log(K_ij) as the "energy" ΔΨ of transition.

        Args:
            kernel: Dirichlet-smoothed kernel
            temperature: T in Boltzmann distribution (lower = more deterministic)

        Returns:
            Boltzmann-weighted transition probabilities
        """
        # Interpret -log(probability) as energy
        # Avoid log(0) by using small epsilon
        eps = 1e-10
        energy = -np.log(kernel + eps)

        # Apply Boltzmann: exp(-E/T) / sum(exp(-E/T))
        boltzmann = np.exp(-energy / temperature)

        # Normalize
        row_sums = boltzmann.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)

        return boltzmann / row_sums

    def stationary_distribution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Compute stationary distribution π where πK = π.

        Args:
            kernel: Transition probability matrix

        Returns:
            Stationary distribution vector
        """
        eigenvalues, eigenvectors = np.linalg.eig(kernel.T)

        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = eigenvectors[:, idx].real

        # Normalize to probability distribution
        stationary = np.abs(stationary)
        stationary /= stationary.sum()

        return stationary

    def entropy_rate(self, kernel: np.ndarray) -> float:
        """
        Compute entropy rate of the Markov chain.

        H(K) = -Σ_i π_i Σ_j K_ij log(K_ij)

        Args:
            kernel: Transition probability matrix

        Returns:
            Entropy rate in bits
        """
        pi = self.stationary_distribution(kernel)

        # Avoid log(0)
        eps = 1e-10
        log_kernel = np.log2(kernel + eps)

        # Conditional entropy for each state
        conditional_entropy = -np.sum(kernel * log_kernel, axis=1)

        # Weight by stationary distribution
        h_rate = np.sum(pi * conditional_entropy)

        return h_rate

    def mean_first_passage_time(self, kernel: np.ndarray) -> np.ndarray:
        """
        Compute mean first passage times between states.

        MFPT[i,j] = expected steps to reach j starting from i

        Args:
            kernel: Transition probability matrix

        Returns:
            4x4 matrix of mean first passage times
        """
        n = self.n_states

        try:
            # Fundamental matrix approach
            pi = self.stationary_distribution(kernel)

            # Z = (I - K + ones * pi)^{-1}
            I = np.eye(n)
            ones_pi = np.outer(np.ones(n), pi)
            Z = np.linalg.inv(I - kernel + ones_pi)

            # MFPT formula
            D = np.diag(np.diag(Z))
            mfpt = (I - Z + np.ones((n, n)) @ D) / np.diag(Z)

            return mfpt

        except np.linalg.LinAlgError:
            logger.warning("Could not compute MFPT - matrix singular")
            return np.full((n, n), np.inf)

    def transfer_entropy(self, seq1: List[Animal], seq2: List[Animal], lag: int = 1) -> float:
        """
        Compute transfer entropy from seq1 to seq2.

        TE(X→Y) measures the amount of information that X provides about
        future Y beyond what past Y provides.

        Args:
            seq1: Source sequence
            seq2: Target sequence
            lag: Time lag for prediction

        Returns:
            Transfer entropy in bits
        """
        # Need same length sequences
        min_len = min(len(seq1), len(seq2)) - lag
        if min_len < 10:
            return 0.0

        # Count joint and marginal occurrences
        joint_yyy = defaultdict(int)  # p(y_{t+1}, y_t, x_t)
        joint_yy = defaultdict(int)   # p(y_{t+1}, y_t)
        joint_yyx = defaultdict(int)  # p(y_t, x_t)
        marginal_y = defaultdict(int) # p(y_t)

        for t in range(min_len):
            y_next = int(seq2[t + lag])
            y_curr = int(seq2[t])
            x_curr = int(seq1[t])

            joint_yyy[(y_next, y_curr, x_curr)] += 1
            joint_yy[(y_next, y_curr)] += 1
            joint_yyx[(y_curr, x_curr)] += 1
            marginal_y[y_curr] += 1

        # Normalize
        total = min_len

        # Compute transfer entropy
        te = 0.0
        eps = 1e-10

        for (y_next, y_curr, x_curr), count in joint_yyy.items():
            p_yyy = count / total + eps
            p_yy = joint_yy[(y_next, y_curr)] / total + eps
            p_yyx = joint_yyx[(y_curr, x_curr)] / total + eps
            p_y = marginal_y[y_curr] / total + eps

            # TE = Σ p(y',y,x) log( p(y'|y,x) / p(y'|y) )
            #    = Σ p(y',y,x) log( p(y',y,x) * p(y) / (p(y,x) * p(y',y)) )
            te += p_yyy * np.log2(p_yyy * p_y / (p_yyx * p_yy))

        return max(0, te)  # TE is non-negative


# =============================================================================
# TRAIT PROXY EXTRACTION
# =============================================================================

def extract_trait_proxies(kernel: np.ndarray, stationary: np.ndarray) -> Dict[str, float]:
    """
    Extract Big-5 trait proxies from kernel statistics.

    Based on Table 2 from Computational Psychodynamics paper:
        - Openness: H(K) - high entropy = more exploratory
        - Conscientiousness: π(Revising) - high = habitual/regulated
        - Neuroticism: Var(π) - uneven distribution = unstable
        - Extraversion: π(Directing) + π(Conferring) - other-focus
        - Agreeableness: π(Conferring) / π(Directing) - explore vs exploit others

    Args:
        kernel: Transition probability matrix
        stationary: Stationary distribution

    Returns:
        Dict with trait proxy scores (normalized 0-1)
    """
    analysis = FourAnimalMarkovAnalysis()

    # Entropy rate (Openness proxy)
    h_rate = analysis.entropy_rate(kernel)
    max_entropy = np.log2(4)  # Maximum for 4 states
    openness = h_rate / max_entropy

    # Stationary on Revising (Conscientiousness proxy)
    conscientiousness = stationary[Animal.REVISING]

    # Variance of stationary (Neuroticism proxy - inverted, high var = neurotic)
    neuroticism = np.std(stationary) * 2  # Scale to ~0-1
    neuroticism = min(1.0, neuroticism)

    # Other-focus (Extraversion proxy)
    extraversion = stationary[Animal.DIRECTING] + stationary[Animal.CONFERRING]

    # Conferring/Directing ratio (Agreeableness proxy)
    if stationary[Animal.DIRECTING] > 0.01:
        agree_ratio = stationary[Animal.CONFERRING] / stationary[Animal.DIRECTING]
        agreeableness = agree_ratio / (1 + agree_ratio)  # Sigmoid-like transform
    else:
        agreeableness = 0.9  # High agreeableness if minimal Directing

    return {
        "openness": float(openness),
        "conscientiousness": float(conscientiousness),
        "neuroticism": float(neuroticism),
        "extraversion": float(extraversion),
        "agreeableness": float(agreeableness)
    }


# =============================================================================
# DATA LOADING AND ANALYSIS PIPELINE
# =============================================================================

def load_criminal_events(directory: str, prefix: str = "Type1_") -> Dict[str, List[str]]:
    """
    Load life events from CSV files.

    Args:
        directory: Directory containing CSV files
        prefix: File prefix filter (e.g., "Type1_")

    Returns:
        Dict mapping criminal name -> list of events
    """
    criminals = {}

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".csv"):
            # Extract name from filename
            name = filename[len(prefix):-4]  # Remove prefix and .csv

            file_path = os.path.join(directory, filename)
            events = []

            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header

                for row in reader:
                    if len(row) >= 3 and row[2].strip():
                        events.append(row[2].strip())

            if events:
                criminals[name] = events
                logger.info(f"Loaded {len(events)} events for {name}")

    return criminals


def analyze_criminal_four_animal(
    events: List[str],
    classifier: AnimalClassifier,
    markov: FourAnimalMarkovAnalysis
) -> CriminalSequence:
    """
    Full 4-Animal analysis pipeline for one criminal.

    Args:
        events: List of life event texts
        classifier: AnimalClassifier instance
        markov: FourAnimalMarkovAnalysis instance

    Returns:
        CriminalSequence with classified events, kernel, and analysis
    """
    # Classify events
    classified = classifier.classify_batch(events, show_progress=False)

    # Extract Animal sequence
    animal_sequence = [e.animal for e in classified]

    # Compute transition counts and kernel
    counts = markov.compute_transition_counts(animal_sequence)
    kernel = markov.dirichlet_kernel(counts)

    return CriminalSequence(
        name="",  # To be filled by caller
        events=classified,
        transition_counts=counts,
        kernel=kernel
    )


def run_four_animal_analysis(
    data_directory: str,
    output_dir: str = "analysis_output",
    alpha: float = 1.0,
    prefix: str = "Type1_"
) -> Dict[str, any]:
    """
    Run complete 4-Animal state space analysis on criminal data.

    Args:
        data_directory: Directory containing CSV files
        output_dir: Directory for output files
        alpha: Dirichlet smoothing parameter
        prefix: File prefix filter

    Returns:
        Results dictionary with all analysis
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    classifier = AnimalClassifier(use_embeddings=True)
    markov = FourAnimalMarkovAnalysis(alpha=alpha)

    # Load data
    logger.info(f"Loading data from {data_directory}...")
    criminals = load_criminal_events(data_directory, prefix=prefix)

    if not criminals:
        logger.error("No criminal data loaded!")
        return {}

    # Analyze each criminal
    results = {
        "individuals": {},
        "aggregate": {
            "transition_counts": np.zeros((4, 4)),
            "n_events": 0,
            "n_individuals": 0
        }
    }

    all_classified_events = []

    for name, events in criminals.items():
        logger.info(f"Analyzing {name} ({len(events)} events)...")

        # Run analysis
        sequence = analyze_criminal_four_animal(events, classifier, markov)
        sequence.name = name

        # Store results
        animal_labels = [ANIMAL_NAMES[e.animal] for e in sequence.events]
        stationary = markov.stationary_distribution(sequence.kernel)
        traits = extract_trait_proxies(sequence.kernel, stationary)

        results["individuals"][name] = {
            "n_events": len(events),
            "animal_sequence": animal_labels,
            "transition_counts": sequence.transition_counts.tolist(),
            "kernel": sequence.kernel.tolist(),
            "stationary_distribution": {
                ANIMAL_NAMES[Animal(i)]: float(stationary[i])
                for i in range(4)
            },
            "entropy_rate": markov.entropy_rate(sequence.kernel),
            "trait_proxies": traits
        }

        # Aggregate
        results["aggregate"]["transition_counts"] += sequence.transition_counts
        results["aggregate"]["n_events"] += len(events)
        results["aggregate"]["n_individuals"] += 1

        all_classified_events.extend(sequence.events)

    # Compute aggregate kernel
    agg_counts = results["aggregate"]["transition_counts"]
    agg_kernel = markov.dirichlet_kernel(agg_counts)
    agg_stationary = markov.stationary_distribution(agg_kernel)
    agg_traits = extract_trait_proxies(agg_kernel, agg_stationary)

    results["aggregate"]["kernel"] = agg_kernel.tolist()
    results["aggregate"]["stationary_distribution"] = {
        ANIMAL_NAMES[Animal(i)]: float(agg_stationary[i])
        for i in range(4)
    }
    results["aggregate"]["entropy_rate"] = markov.entropy_rate(agg_kernel)
    results["aggregate"]["mean_first_passage_time"] = markov.mean_first_passage_time(agg_kernel).tolist()
    results["aggregate"]["trait_proxies"] = agg_traits
    results["aggregate"]["transition_counts"] = agg_counts.tolist()

    # Learn and save prototypes
    logger.info("Learning Animal prototypes from classified events...")
    classifier.learn_prototypes(all_classified_events)
    classifier.save_prototypes(os.path.join(output_dir, "animal_prototypes.pkl"))

    # Save results
    output_file = os.path.join(output_dir, "four_animal_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Save aggregate kernel as numpy
    np.save(os.path.join(output_dir, "aggregate_animal_kernel.npy"), agg_kernel)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="4-Animal State Space Analysis for Criminal Life Events"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="/Users/ajithsenthil/Desktop/CrimeArchetypes/mnt/data/csv",
        help="Directory containing CSV data files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet smoothing parameter"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Type1_",
        help="File prefix filter"
    )

    args = parser.parse_args()

    results = run_four_animal_analysis(
        data_directory=args.directory,
        output_dir=args.output,
        alpha=args.alpha,
        prefix=args.prefix
    )

    # Print summary
    if results:
        print("\n" + "="*60)
        print("4-ANIMAL STATE SPACE ANALYSIS SUMMARY")
        print("="*60)

        agg = results["aggregate"]
        print(f"\nAnalyzed {agg['n_individuals']} individuals, {agg['n_events']} total events")

        print("\nAggregate Stationary Distribution:")
        for animal, prob in agg["stationary_distribution"].items():
            print(f"  {animal}: {prob:.3f}")

        print(f"\nEntropy Rate: {agg['entropy_rate']:.3f} bits")

        print("\nTrait Proxies (Big-5):")
        for trait, value in agg["trait_proxies"].items():
            print(f"  {trait.capitalize()}: {value:.3f}")
