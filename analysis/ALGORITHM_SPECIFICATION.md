# Algorithm Specification: Computational Psychodynamics for Criminal Trajectory Analysis

## Document Overview

This document provides complete algorithmic specifications for the Computational Psychodynamics framework applied to criminal behavioral trajectory analysis. Each algorithm is specified with:
- **Input/Output**: Data types and structures
- **Pseudocode**: Step-by-step implementation
- **Complexity**: Time and space requirements
- **Parameters**: Configurable settings with defaults

---

## Table of Contents

1. [Data Structures](#1-data-structures)
2. [State Classification Pipeline](#2-state-classification-pipeline)
3. [Markov Chain Analysis](#3-markov-chain-analysis)
4. [Transfer Entropy Computation](#4-transfer-entropy-computation)
5. [Archetypal Role Assignment](#5-archetypal-role-assignment)
6. [Hierarchical Classification](#6-hierarchical-classification)
7. [Causal Modeling](#7-causal-modeling)
8. [Counterfactual Simulation](#8-counterfactual-simulation)
9. [State Space Validation](#9-state-space-validation)
10. [Optimal Mapping Search](#10-optimal-mapping-search)

---

## 1. Data Structures

### 1.1 Core Types

```
TYPE State = ENUM {Seeking, Directing, Conferring, Revising}
TYPE StateIndex = INTEGER in {0, 1, 2, 3}

TYPE Event = STRUCT {
    id: STRING
    individual_id: STRING
    timestamp: INTEGER  # Ordinal position in sequence
    description: STRING
    state: State
    confidence: FLOAT in [0, 1]
    embedding: VECTOR[384]  # Sentence transformer embedding
}

TYPE Individual = STRUCT {
    id: STRING
    events: LIST[Event]  # Ordered by timestamp
    sequence: LIST[StateIndex]  # Extracted state sequence
    transition_matrix: MATRIX[4, 4]
    stationary_distribution: VECTOR[4]
}

TYPE TransitionMatrix = MATRIX[4, 4] of FLOAT
    # K[i][j] = P(next_state = j | current_state = i)
    # Row-stochastic: sum(K[i]) = 1 for all i
```

### 1.2 Constants

```
STATES = ["Seeking", "Directing", "Conferring", "Revising"]
N_STATES = 4
STATE_TO_INDEX = {
    "Seeking": 0,
    "Directing": 1,
    "Conferring": 2,
    "Revising": 3
}
```

---

## 2. State Classification Pipeline

### 2.1 Lexical Imputation

**Purpose**: Generate paraphrases to handle lexical variation in event descriptions.

```
ALGORITHM LexicalImputation

INPUT:
    event_text: STRING          # Original event description
    n_paraphrases: INTEGER = 5  # Number of paraphrases to generate
    temperature: FLOAT = 0.7    # LLM sampling temperature

OUTPUT:
    paraphrases: LIST[STRING]   # List of paraphrased descriptions

PROCEDURE:
    1. prompt = CONSTRUCT_PROMPT(event_text)
       """
       Generate {n_paraphrases} paraphrases of the following event description.
       Each paraphrase should preserve the core meaning but vary the wording.

       Event: {event_text}

       Paraphrases:
       """

    2. response = LLM_GENERATE(prompt, temperature=temperature)

    3. paraphrases = PARSE_NUMBERED_LIST(response)

    4. RETURN [event_text] + paraphrases  # Include original

COMPLEXITY: O(1) LLM calls per event
```

### 2.2 Semantic Embedding

**Purpose**: Convert text to dense vector representations.

```
ALGORITHM SemanticEmbedding

INPUT:
    texts: LIST[STRING]         # Event descriptions (with paraphrases)
    model: STRING = "all-MiniLM-L6-v2"

OUTPUT:
    centroid: VECTOR[384]       # Centroid of all embeddings

PROCEDURE:
    1. embeddings = []

    2. FOR text IN texts:
           vec = SENTENCE_TRANSFORMER_ENCODE(text, model)
           embeddings.APPEND(vec)

    3. centroid = MEAN(embeddings, axis=0)

    4. centroid = centroid / NORM(centroid)  # L2 normalize

    5. RETURN centroid

COMPLEXITY: O(n * d) where n = number of texts, d = embedding dimension
```

### 2.3 LLM State Classification

**Purpose**: Classify events into the four-state motivational space.

```
ALGORITHM LLMStateClassification

INPUT:
    event_text: STRING
    context: STRING = ""        # Optional surrounding context

OUTPUT:
    state: State
    confidence: FLOAT
    reasoning: STRING

PROCEDURE:
    1. prompt = CONSTRUCT_CLASSIFICATION_PROMPT(event_text, context)
       """
       Classify the following life event into one of four motivational states:

       STATES:
       - Seeking (Self × Explore): Fantasy, introspection, internal urges,
         curiosity, planning without action
       - Directing (Other × Exploit): Control, manipulation, violence,
         domination, acting on others to achieve goals
       - Conferring (Other × Explore): Observation, surveillance, stalking,
         gathering information about potential targets
       - Revising (Self × Exploit): Rituals, compulsions, trophy collection,
         consolidating patterns, MO refinement

       EVENT: {event_text}

       Provide your reasoning step by step, then state your classification
       and confidence (0-1).

       Reasoning:
       Classification:
       Confidence:
       """

    2. response = LLM_GENERATE(prompt, temperature=0.0)  # Deterministic

    3. reasoning = EXTRACT_SECTION(response, "Reasoning")

    4. state_str = EXTRACT_SECTION(response, "Classification")
       state = PARSE_STATE(state_str)

    5. confidence = PARSE_FLOAT(EXTRACT_SECTION(response, "Confidence"))

    6. RETURN state, confidence, reasoning

COMPLEXITY: O(1) LLM calls per event
```

### 2.4 Keyword-Based Classification (Fallback)

**Purpose**: Fast heuristic classification when LLM unavailable.

```
ALGORITHM KeywordClassification

INPUT:
    event_text: STRING

OUTPUT:
    state: State

CONSTANTS:
    SEEKING_KEYWORDS = ["fantasy", "thought", "urge", "desire", "dream",
                        "imagine", "plan", "obsess", "internal"]
    DIRECTING_KEYWORDS = ["kill", "murder", "attack", "assault", "rape",
                          "strangle", "stab", "shoot", "control", "force"]
    CONFERRING_KEYWORDS = ["stalk", "follow", "watch", "observe", "surveil",
                           "track", "photograph", "learn", "study"]
    REVISING_KEYWORDS = ["ritual", "trophy", "collect", "pattern", "habit",
                         "repeat", "method", "routine", "dispose"]

PROCEDURE:
    1. text_lower = LOWERCASE(event_text)

    2. scores = {state: 0 for state in STATES}

    3. FOR word IN TOKENIZE(text_lower):
           IF word IN SEEKING_KEYWORDS: scores["Seeking"] += 1
           IF word IN DIRECTING_KEYWORDS: scores["Directing"] += 1
           IF word IN CONFERRING_KEYWORDS: scores["Conferring"] += 1
           IF word IN REVISING_KEYWORDS: scores["Revising"] += 1

    4. IF MAX(scores.values()) == 0:
           RETURN "Seeking"  # Default

    5. RETURN ARGMAX(scores)

COMPLEXITY: O(n) where n = number of words in event text
```

---

## 3. Markov Chain Analysis

### 3.1 Transition Matrix Estimation

**Purpose**: Estimate transition probabilities from observed sequences.

```
ALGORITHM EstimateTransitionMatrix

INPUT:
    sequence: LIST[StateIndex]  # State sequence for one individual
    smoothing: FLOAT = 0.0      # Laplace smoothing parameter

OUTPUT:
    K: TransitionMatrix         # 4×4 row-stochastic matrix

PROCEDURE:
    1. counts = ZEROS(N_STATES, N_STATES)

    2. FOR t IN RANGE(LEN(sequence) - 1):
           i = sequence[t]
           j = sequence[t + 1]
           counts[i][j] += 1

    3. # Add smoothing
       counts = counts + smoothing

    4. # Normalize rows
       K = ZEROS(N_STATES, N_STATES)
       FOR i IN RANGE(N_STATES):
           row_sum = SUM(counts[i])
           IF row_sum > 0:
               K[i] = counts[i] / row_sum
           ELSE:
               K[i] = [0.25, 0.25, 0.25, 0.25]  # Uniform if no data

    5. RETURN K

COMPLEXITY: O(T) where T = sequence length
```

### 3.2 Stationary Distribution

**Purpose**: Compute the long-run state distribution.

```
ALGORITHM StationaryDistribution

INPUT:
    K: TransitionMatrix

OUTPUT:
    pi: VECTOR[4]               # Stationary distribution

PROCEDURE:
    1. # Solve π = πK, i.e., π(K - I) = 0 with Σπ = 1

    2. A = TRANSPOSE(K) - IDENTITY(N_STATES)

    3. # Replace last row with constraint Σπ = 1
       A[N_STATES - 1] = [1, 1, 1, 1]

    4. b = ZEROS(N_STATES)
       b[N_STATES - 1] = 1

    5. pi = SOLVE_LINEAR_SYSTEM(A, b)

    6. # Ensure non-negative and normalized
       pi = MAX(pi, 0)
       pi = pi / SUM(pi)

    7. RETURN pi

COMPLEXITY: O(N_STATES³) = O(64) = O(1)
```

### 3.3 Mean First Passage Time

**Purpose**: Compute expected steps to reach each state from each starting state.

```
ALGORITHM MeanFirstPassageTime

INPUT:
    K: TransitionMatrix
    pi: VECTOR[4]               # Stationary distribution

OUTPUT:
    M: MATRIX[4, 4]             # M[i][j] = expected steps from i to j

PROCEDURE:
    1. # Using fundamental matrix approach

    2. Z = INVERSE(IDENTITY(N_STATES) - K + OUTER(ONES(N_STATES), pi))

    3. M = ZEROS(N_STATES, N_STATES)

    4. FOR i IN RANGE(N_STATES):
           FOR j IN RANGE(N_STATES):
               IF i == j:
                   M[i][j] = 0
               ELSE:
                   M[i][j] = (Z[j][j] - Z[i][j]) / pi[j]

    5. RETURN M

COMPLEXITY: O(N_STATES³) = O(1)
```

### 3.4 Entropy Rate

**Purpose**: Measure the randomness/complexity of the Markov chain.

```
ALGORITHM EntropyRate

INPUT:
    K: TransitionMatrix
    pi: VECTOR[4]               # Stationary distribution

OUTPUT:
    H: FLOAT                    # Entropy rate in bits

PROCEDURE:
    1. H = 0

    2. FOR i IN RANGE(N_STATES):
           FOR j IN RANGE(N_STATES):
               IF K[i][j] > 0 AND pi[i] > 0:
                   H -= pi[i] * K[i][j] * LOG2(K[i][j])

    3. RETURN H

COMPLEXITY: O(N_STATES²) = O(1)
```

---

## 4. Transfer Entropy Computation

### 4.1 Transfer Entropy Between Two Sequences

**Purpose**: Quantify directed predictive information flow.

```
ALGORITHM TransferEntropy

INPUT:
    X: LIST[StateIndex]         # Source sequence
    Y: LIST[StateIndex]         # Target sequence
    lag: INTEGER = 1            # Time lag

OUTPUT:
    TE: FLOAT                   # Transfer entropy X → Y in bits

PRECONDITION:
    LEN(X) == LEN(Y)

PROCEDURE:
    1. T = LEN(X)

    2. # Count joint occurrences
       # P(Y_{t+1}, Y_t, X_t)
       joint_counts = ZEROS(N_STATES, N_STATES, N_STATES)

       FOR t IN RANGE(T - lag):
           y_next = Y[t + lag]
           y_curr = Y[t]
           x_curr = X[t]
           joint_counts[y_next][y_curr][x_curr] += 1

    3. # Normalize to get joint probability
       total = SUM(joint_counts)
       IF total == 0:
           RETURN 0.0
       joint_prob = joint_counts / total

    4. # Compute marginals
       # P(Y_{t+1}, Y_t)
       p_y_next_y_curr = SUM(joint_prob, axis=2)

       # P(Y_t, X_t)
       p_y_curr_x_curr = SUM(joint_prob, axis=0)

       # P(Y_t)
       p_y_curr = SUM(p_y_next_y_curr, axis=0)

    5. # Compute transfer entropy
       TE = 0
       FOR y_next IN RANGE(N_STATES):
           FOR y_curr IN RANGE(N_STATES):
               FOR x_curr IN RANGE(N_STATES):
                   p_joint = joint_prob[y_next][y_curr][x_curr]
                   IF p_joint > 0:
                       # P(Y_{t+1} | Y_t, X_t)
                       p_cond_full = p_joint / p_y_curr_x_curr[y_curr][x_curr] \
                                     IF p_y_curr_x_curr[y_curr][x_curr] > 0 ELSE 0

                       # P(Y_{t+1} | Y_t)
                       p_cond_reduced = p_y_next_y_curr[y_next][y_curr] / p_y_curr[y_curr] \
                                        IF p_y_curr[y_curr] > 0 ELSE 0

                       IF p_cond_full > 0 AND p_cond_reduced > 0:
                           TE += p_joint * LOG2(p_cond_full / p_cond_reduced)

    6. RETURN MAX(TE, 0)  # TE is non-negative by definition

COMPLEXITY: O(T + N_STATES³)
```

### 4.2 Phase-Normalized Transfer Entropy

**Purpose**: Compare sequences of different lengths by normalizing to common phases.

```
ALGORITHM PhaseNormalizedTE

INPUT:
    X: LIST[StateIndex]         # Source sequence (length T_X)
    Y: LIST[StateIndex]         # Target sequence (length T_Y)
    n_phases: INTEGER = 50      # Number of normalized phases

OUTPUT:
    TE: FLOAT                   # Transfer entropy with phase alignment

PROCEDURE:
    1. # Resample both sequences to n_phases length
       X_resampled = RESAMPLE_SEQUENCE(X, n_phases)
       Y_resampled = RESAMPLE_SEQUENCE(Y, n_phases)

    2. TE = TransferEntropy(X_resampled, Y_resampled)

    3. RETURN TE

HELPER FUNCTION ResampleSequence(seq, target_length):
    1. source_length = LEN(seq)
    2. resampled = []
    3. FOR i IN RANGE(target_length):
           source_idx = FLOOR(i * source_length / target_length)
           resampled.APPEND(seq[source_idx])
    4. RETURN resampled

COMPLEXITY: O(n_phases + N_STATES³)
```

### 4.3 Pairwise TE Matrix

**Purpose**: Compute transfer entropy between all pairs of individuals.

```
ALGORITHM PairwiseTEMatrix

INPUT:
    individuals: LIST[Individual]
    n_phases: INTEGER = 50

OUTPUT:
    TE_matrix: MATRIX[N, N]     # TE_matrix[i][j] = TE(i → j)

PROCEDURE:
    1. N = LEN(individuals)
    2. TE_matrix = ZEROS(N, N)

    3. FOR i IN RANGE(N):
           FOR j IN RANGE(N):
               IF i != j:
                   X = individuals[i].sequence
                   Y = individuals[j].sequence
                   TE_matrix[i][j] = PhaseNormalizedTE(X, Y, n_phases)

    4. RETURN TE_matrix

COMPLEXITY: O(N² * (n_phases + N_STATES³))
```

---

## 5. Archetypal Role Assignment

### 5.1 Network Construction

**Purpose**: Build directed graph from TE matrix.

```
ALGORITHM ConstructTENetwork

INPUT:
    TE_matrix: MATRIX[N, N]
    threshold_percentile: FLOAT = 85

OUTPUT:
    adjacency: MATRIX[N, N]     # Binary adjacency matrix
    threshold: FLOAT            # Actual threshold value

PROCEDURE:
    1. # Flatten non-diagonal entries
       non_diag_values = []
       FOR i IN RANGE(N):
           FOR j IN RANGE(N):
               IF i != j AND TE_matrix[i][j] > 0:
                   non_diag_values.APPEND(TE_matrix[i][j])

    2. threshold = PERCENTILE(non_diag_values, threshold_percentile)

    3. adjacency = ZEROS(N, N)
       FOR i IN RANGE(N):
           FOR j IN RANGE(N):
               IF TE_matrix[i][j] >= threshold:
                   adjacency[i][j] = 1

    4. RETURN adjacency, threshold

COMPLEXITY: O(N²)
```

### 5.2 Role Classification

**Purpose**: Assign archetypal roles based on network position.

```
ALGORITHM AssignArchetypalRoles

INPUT:
    TE_matrix: MATRIX[N, N]
    adjacency: MATRIX[N, N]

OUTPUT:
    roles: LIST[STRING]         # Role for each individual

PROCEDURE:
    1. N = LEN(TE_matrix)

    2. # Compute incoming and outgoing TE sums
       outgoing = [SUM(TE_matrix[i]) for i in RANGE(N)]
       incoming = [SUM(TE_matrix[:, j]) for j in RANGE(N)]

    3. # Compute statistics
       out_mean, out_std = MEAN(outgoing), STD(outgoing)
       in_mean, in_std = MEAN(incoming), STD(incoming)

    4. roles = []
       FOR i IN RANGE(N):
           out_z = (outgoing[i] - out_mean) / out_std IF out_std > 0 ELSE 0
           in_z = (incoming[i] - in_mean) / in_std IF in_std > 0 ELSE 0

           IF out_z > 1.5 AND in_z < 0.5:
               roles.APPEND("Source")
           ELIF in_z > 1.5 AND out_z < 0.5:
               roles.APPEND("Sink")
           ELIF out_z > 1.0 AND in_z > 1.0:
               roles.APPEND("Hub")
           ELSE:
               roles.APPEND("General")

    5. RETURN roles

COMPLEXITY: O(N²)
```

### 5.3 Lineage Extraction

**Purpose**: Find chains of high-TE relationships.

```
ALGORITHM ExtractLineages

INPUT:
    adjacency: MATRIX[N, N]
    min_length: INTEGER = 3

OUTPUT:
    lineages: LIST[LIST[INTEGER]]  # Each lineage is a list of individual indices

PROCEDURE:
    1. # Build directed graph
       graph = {i: [] for i in RANGE(N)}
       FOR i IN RANGE(N):
           FOR j IN RANGE(N):
               IF adjacency[i][j] == 1:
                   graph[i].APPEND(j)

    2. # Find all paths using DFS
       lineages = []
       visited_paths = SET()

       FOR start IN RANGE(N):
           paths = DFS_ALL_PATHS(graph, start, visited_paths)
           FOR path IN paths:
               IF LEN(path) >= min_length:
                   lineages.APPEND(path)

    3. # Remove duplicates and subpaths
       lineages = FILTER_MAXIMAL_PATHS(lineages)

    4. RETURN SORT_BY_LENGTH(lineages, descending=True)

HELPER FUNCTION DFS_ALL_PATHS(graph, start, visited_paths):
    paths = []
    stack = [(start, [start])]

    WHILE stack NOT EMPTY:
        node, path = stack.POP()
        path_key = TUPLE(path)

        IF path_key IN visited_paths:
            CONTINUE
        visited_paths.ADD(path_key)

        neighbors = graph[node]
        IF LEN(neighbors) == 0 OR ALL(n IN path FOR n IN neighbors):
            paths.APPEND(path)
        ELSE:
            FOR neighbor IN neighbors:
                IF neighbor NOT IN path:
                    stack.APPEND((neighbor, path + [neighbor]))

    RETURN paths

COMPLEXITY: O(N! / (N-k)!) worst case, typically O(N² * avg_path_length)
```

---

## 6. Hierarchical Classification

### 6.1 Feature Extraction

**Purpose**: Extract features for clustering individuals.

```
ALGORITHM ExtractClassificationFeatures

INPUT:
    individual: Individual

OUTPUT:
    features: VECTOR[9]

PROCEDURE:
    1. seq = individual.sequence
       K = individual.transition_matrix
       T = LEN(seq)

    2. # State distribution (4 features)
       state_counts = [0, 0, 0, 0]
       FOR s IN seq:
           state_counts[s] += 1
       state_dist = [c / T for c in state_counts]

    3. # State persistence (4 features) - diagonal of K
       persistence = [K[i][i] for i in RANGE(N_STATES)]

    4. # Escalation (1 feature)
       midpoint = T // 2
       early_directing = SUM(1 for s in seq[:midpoint] if s == 1) / midpoint
       late_directing = SUM(1 for s in seq[midpoint:] if s == 1) / (T - midpoint)
       escalation = late_directing - early_directing

    5. features = state_dist + persistence + [escalation]

    6. RETURN features

COMPLEXITY: O(T)
```

### 6.2 Primary Type Clustering

**Purpose**: Cluster individuals into COMPLEX vs FOCUSED types.

```
ALGORITHM PrimaryTypeClustering

INPUT:
    individuals: LIST[Individual]
    n_clusters: INTEGER = 2

OUTPUT:
    labels: LIST[STRING]        # "COMPLEX" or "FOCUSED"

PROCEDURE:
    1. # Extract features
       features = [ExtractClassificationFeatures(ind) for ind in individuals]
       X = MATRIX(features)

    2. # Standardize
       X_scaled = STANDARDIZE(X)

    3. # Ward's hierarchical clustering
       linkage = WARD_LINKAGE(X_scaled)
       cluster_labels = CUT_DENDROGRAM(linkage, n_clusters)

    4. # Determine which cluster is COMPLEX vs FOCUSED
       cluster_0_directing = MEAN([features[i][1] for i where cluster_labels[i] == 0])
       cluster_1_directing = MEAN([features[i][1] for i where cluster_labels[i] == 1])

       IF cluster_0_directing < cluster_1_directing:
           complex_cluster = 0
       ELSE:
           complex_cluster = 1

    5. labels = []
       FOR label IN cluster_labels:
           IF label == complex_cluster:
               labels.APPEND("COMPLEX")
           ELSE:
               labels.APPEND("FOCUSED")

    6. RETURN labels

COMPLEXITY: O(N² log N) for hierarchical clustering
```

### 6.3 Subtype Assignment

**Purpose**: Assign theory-driven subtypes within primary types.

```
ALGORITHM SubtypeAssignment

INPUT:
    individual: Individual
    primary_type: STRING        # "COMPLEX" or "FOCUSED"

OUTPUT:
    subtype: STRING

PROCEDURE:
    1. features = ExtractClassificationFeatures(individual)
       state_dist = features[0:4]
       persistence = features[4:8]
       escalation = features[8]
       K = individual.transition_matrix

    2. IF primary_type == "COMPLEX":
           # Count active states (>10% of events)
           active_states = SUM(1 for p in state_dist if p > 0.10)
           max_proportion = MAX(state_dist)

           IF active_states >= 3 AND max_proportion < 0.60:
               RETURN "Chameleon"
           ELSE:
               RETURN "Multi-Modal"

    3. ELSE:  # FOCUSED
           directing_prop = state_dist[1]  # Index 1 = Directing

           # Check for Pure Predator
           IF directing_prop >= 0.75:
               RETURN "Pure Predator"

           # Check for Strong Escalator
           IF escalation >= 0.35:
               RETURN "Strong Escalator"

           # Check for Stalker-Striker (Conferring → Directing present)
           conf_to_dir = K[2][1]  # K[Conferring][Directing]
           IF conf_to_dir > 0.20:
               RETURN "Stalker-Striker"

           # Check for Fantasy-Actor (Seeking → Directing, no Conf → Dir)
           seek_to_dir = K[0][1]  # K[Seeking][Directing]
           IF seek_to_dir > 0.25 AND conf_to_dir <= 0.20:
               RETURN "Fantasy-Actor"

           RETURN "Standard"

COMPLEXITY: O(1)
```

---

## 7. Causal Modeling

### 7.1 Structural Causal Model Construction

**Purpose**: Build SCM from Markov transition structure.

```
ALGORITHM ConstructSCM

INPUT:
    K: TransitionMatrix
    T: INTEGER                  # Number of time steps

OUTPUT:
    scm: StructuralCausalModel

TYPE StructuralCausalModel = STRUCT {
    nodes: LIST[STRING]         # Node names
    edges: LIST[(STRING, STRING, FLOAT)]  # (from, to, weight)
    structural_equations: DICT[STRING → FUNCTION]
}

PROCEDURE:
    1. scm = NEW StructuralCausalModel()

    2. # Create nodes for each time step
       FOR t IN RANGE(T):
           scm.nodes.APPEND(f"S_{t}")

    3. # Add outcome node
       scm.nodes.APPEND("Harm")

    4. # Add edges from transition matrix
       FOR t IN RANGE(T - 1):
           FOR i IN RANGE(N_STATES):
               FOR j IN RANGE(N_STATES):
                   IF K[i][j] > 0:
                       scm.edges.APPEND((f"S_{t}", f"S_{t+1}", K[i][j]))

    5. # Add edges to outcome (Directing → Harm)
       FOR t IN RANGE(T):
           scm.edges.APPEND((f"S_{t}", "Harm", 1.0))

    6. # Define structural equations
       FOR t IN RANGE(1, T):
           scm.structural_equations[f"S_{t}"] = LAMBDA(parents):
               prev_state = parents[f"S_{t-1}"]
               RETURN SAMPLE_CATEGORICAL(K[prev_state])

       scm.structural_equations["Harm"] = LAMBDA(parents):
           RETURN ANY(parents[f"S_{t}"] == 1 FOR t IN RANGE(T))  # 1 = Directing

    7. RETURN scm

COMPLEXITY: O(T * N_STATES²)
```

### 7.2 Do-Operator

**Purpose**: Implement intervention via graph surgery.

```
ALGORITHM DoOperator

INPUT:
    scm: StructuralCausalModel
    intervention_node: STRING
    intervention_value: ANY

OUTPUT:
    scm_modified: StructuralCausalModel

PROCEDURE:
    1. scm_modified = DEEP_COPY(scm)

    2. # Remove all incoming edges to intervention node
       scm_modified.edges = [
           (u, v, w) FOR (u, v, w) IN scm_modified.edges
           IF v != intervention_node
       ]

    3. # Set structural equation to constant
       scm_modified.structural_equations[intervention_node] = LAMBDA(_):
           RETURN intervention_value

    4. RETURN scm_modified

COMPLEXITY: O(|E|) where E = number of edges
```

### 7.3 Intervention Effect Estimation

**Purpose**: Estimate causal effect of intervention on outcome.

```
ALGORITHM InterventionEffect

INPUT:
    scm: StructuralCausalModel
    intervention_node: STRING
    intervention_value: ANY
    outcome_node: STRING = "Harm"
    n_samples: INTEGER = 1000

OUTPUT:
    effect: FLOAT               # P(outcome | do(intervention))

PROCEDURE:
    1. # Apply do-operator
       scm_intervened = DoOperator(scm, intervention_node, intervention_value)

    2. # Monte Carlo estimation
       outcomes = []
       FOR i IN RANGE(n_samples):
           sample = FORWARD_SAMPLE(scm_intervened)
           outcomes.APPEND(sample[outcome_node])

    3. effect = MEAN(outcomes)

    4. RETURN effect

HELPER FUNCTION ForwardSample(scm):
    sample = {}
    FOR node IN TOPOLOGICAL_SORT(scm.nodes):
        parents = {p: sample[p] for (p, n, _) in scm.edges if n == node}
        sample[node] = scm.structural_equations[node](parents)
    RETURN sample

COMPLEXITY: O(n_samples * T * N_STATES)
```

---

## 8. Counterfactual Simulation

### 8.1 Three-Step Counterfactual

**Purpose**: Answer "What if we had intervened at time t?"

```
ALGORITHM CounterfactualSimulation

INPUT:
    observed_trajectory: LIST[StateIndex]
    intervention_time: INTEGER
    intervention_effect: DICT[Transition → FLOAT]  # Modification to K
    n_simulations: INTEGER = 1000

OUTPUT:
    counterfactual_outcomes: LIST[LIST[StateIndex]]
    harm_reduction: FLOAT

PROCEDURE:
    1. T = LEN(observed_trajectory)
       K_original = EstimateTransitionMatrix(observed_trajectory)

    2. # STEP 1: ABDUCTION
       # Infer that the individual follows K_original
       K_individual = K_original

    3. # STEP 2: ACTION
       # Modify transition matrix according to intervention
       K_intervened = COPY(K_individual)
       FOR (i, j), delta IN intervention_effect.items():
           K_intervened[i][j] = MAX(0, K_intervened[i][j] + delta)
       # Renormalize rows
       FOR i IN RANGE(N_STATES):
           row_sum = SUM(K_intervened[i])
           IF row_sum > 0:
               K_intervened[i] = K_intervened[i] / row_sum

    4. # STEP 3: PREDICTION
       counterfactual_outcomes = []
       FOR sim IN RANGE(n_simulations):
           trajectory = observed_trajectory[:intervention_time]
           current_state = trajectory[-1]

           FOR t IN RANGE(intervention_time, T):
               next_state = SAMPLE_CATEGORICAL(K_intervened[current_state])
               trajectory.APPEND(next_state)
               current_state = next_state

           counterfactual_outcomes.APPEND(trajectory)

    5. # Compute harm reduction
       observed_harm = COUNT_DIRECTING_STATES(observed_trajectory[intervention_time:])
       counterfactual_harms = [COUNT_DIRECTING_STATES(cf[intervention_time:])
                               for cf in counterfactual_outcomes]

       harm_reduction = (observed_harm - MEAN(counterfactual_harms)) / observed_harm \
                        IF observed_harm > 0 ELSE 0

    6. RETURN counterfactual_outcomes, harm_reduction

COMPLEXITY: O(n_simulations * T)
```

### 8.2 Optimal Intervention Timing

**Purpose**: Find the best time to intervene for maximum harm reduction.

```
ALGORITHM OptimalInterventionTiming

INPUT:
    trajectory: LIST[StateIndex]
    intervention_effect: DICT[Transition → FLOAT]
    n_simulations: INTEGER = 1000

OUTPUT:
    optimal_time: INTEGER
    expected_reduction: FLOAT

PROCEDURE:
    1. T = LEN(trajectory)
       best_time = 0
       best_reduction = 0

    2. FOR t IN RANGE(1, T - 1):
           _, reduction = CounterfactualSimulation(
               trajectory, t, intervention_effect, n_simulations
           )

           IF reduction > best_reduction:
               best_reduction = reduction
               best_time = t

    3. RETURN best_time, best_reduction

COMPLEXITY: O(T * n_simulations * T) = O(n_simulations * T²)
```

---

## 9. State Space Validation

### 9.1 Information Retention

**Purpose**: Measure how much information is preserved when mapping states.

```
ALGORITHM InformationRetention

INPUT:
    source_labels: LIST[INTEGER]   # Fine-grained labels (e.g., 10 clusters)
    target_labels: LIST[INTEGER]   # Coarse labels (e.g., 4 states)

OUTPUT:
    retention: FLOAT               # In [0, 1]

PROCEDURE:
    1. # Compute mutual information I(S; T)
       mi = MutualInformation(source_labels, target_labels)

    2. # Compute entropy H(S)
       source_counts = COUNTER(source_labels)
       h_source = Entropy(source_counts.values())

    3. retention = mi / h_source IF h_source > 0 ELSE 0

    4. RETURN retention

HELPER FUNCTION MutualInformation(X, Y):
    # I(X; Y) = H(X) + H(Y) - H(X, Y)
    h_x = Entropy(COUNTER(X).values())
    h_y = Entropy(COUNTER(Y).values())
    h_xy = JointEntropy(X, Y)
    RETURN h_x + h_y - h_xy

HELPER FUNCTION Entropy(counts):
    total = SUM(counts)
    probs = [c / total for c in counts if c > 0]
    RETURN -SUM(p * LOG2(p) for p in probs)

HELPER FUNCTION JointEntropy(X, Y):
    joint_counts = COUNTER(ZIP(X, Y))
    RETURN Entropy(joint_counts.values())

COMPLEXITY: O(n) where n = number of samples
```

### 9.2 Permutation Test for Mapping Quality

**Purpose**: Test if theoretical mapping is better than random.

```
ALGORITHM PermutationTestMapping

INPUT:
    source_labels: LIST[INTEGER]   # Cluster labels
    theoretical_mapping: DICT[INTEGER → INTEGER]  # Cluster → State
    n_permutations: INTEGER = 10000
    n_target_states: INTEGER = 4

OUTPUT:
    observed: FLOAT
    null_mean: FLOAT
    null_std: FLOAT
    p_value: FLOAT

PROCEDURE:
    1. # Compute observed information retention
       target_labels = [theoretical_mapping[s] for s in source_labels]
       observed = InformationRetention(source_labels, target_labels)

    2. # Generate null distribution
       source_states = UNIQUE(source_labels)
       null_distribution = []

       FOR i IN RANGE(n_permutations):
           # Random mapping
           random_mapping = {s: RANDOM_INT(0, n_target_states - 1)
                            for s in source_states}
           random_targets = [random_mapping[s] for s in source_labels]
           null_retention = InformationRetention(source_labels, random_targets)
           null_distribution.APPEND(null_retention)

    3. null_mean = MEAN(null_distribution)
       null_std = STD(null_distribution)

    4. # Compute p-value (one-tailed, greater)
       p_value = MEAN([1 if x >= observed else 0 for x in null_distribution])

    5. RETURN observed, null_mean, null_std, p_value

COMPLEXITY: O(n_permutations * n)
```

### 9.3 Sequence Structure Test

**Purpose**: Test if transitions are non-random (vs. shuffled baseline).

```
ALGORITHM SequenceStructureTest

INPUT:
    sequences: LIST[LIST[StateIndex]]
    n_permutations: INTEGER = 10000

OUTPUT:
    chi2_stat: FLOAT
    effect_size: FLOAT
    p_value: FLOAT

PROCEDURE:
    1. # Compute observed transition matrix
       all_transitions = []
       FOR seq IN sequences:
           FOR t IN RANGE(LEN(seq) - 1):
               all_transitions.APPEND((seq[t], seq[t + 1]))

       K_observed = TransitionCountMatrix(all_transitions)

    2. # Generate null distribution by shuffling within sequences
       null_stats = []
       FOR i IN RANGE(n_permutations):
           shuffled_transitions = []
           FOR seq IN sequences:
               shuffled_seq = SHUFFLE(seq)
               FOR t IN RANGE(LEN(shuffled_seq) - 1):
                   shuffled_transitions.APPEND((shuffled_seq[t], shuffled_seq[t + 1]))

           K_shuffled = TransitionCountMatrix(shuffled_transitions)
           chi2 = ChiSquareStatistic(K_observed, K_shuffled)
           null_stats.APPEND(chi2)

    3. # Observed vs. uniform baseline
       K_uniform = ExpectedUniformMatrix(K_observed)
       chi2_stat = ChiSquareStatistic(K_observed, K_uniform)

    4. # Effect size (Cohen's d)
       effect_size = (chi2_stat - MEAN(null_stats)) / STD(null_stats)

    5. # P-value
       p_value = MEAN([1 if x >= chi2_stat else 0 for x in null_stats])

    6. RETURN chi2_stat, effect_size, p_value

HELPER FUNCTION ChiSquareStatistic(observed, expected):
    chi2 = 0
    FOR i IN RANGE(N_STATES):
        FOR j IN RANGE(N_STATES):
            IF expected[i][j] > 0:
                chi2 += (observed[i][j] - expected[i][j])² / expected[i][j]
    RETURN chi2

COMPLEXITY: O(n_permutations * T_total)
```

### 9.4 Predictive Improvement Test

**Purpose**: Test if Markov prediction beats marginal prediction.

```
ALGORITHM PredictiveImprovementTest

INPUT:
    sequences: LIST[LIST[StateIndex]]
    n_folds: INTEGER = 5

OUTPUT:
    markov_accuracy: FLOAT
    marginal_accuracy: FLOAT
    improvement: FLOAT
    p_value: FLOAT

PROCEDURE:
    1. # Cross-validation
       markov_accuracies = []
       marginal_accuracies = []

       folds = SPLIT_INTO_FOLDS(sequences, n_folds)

       FOR fold_idx IN RANGE(n_folds):
           train_seqs = FLATTEN([folds[i] for i != fold_idx])
           test_seqs = folds[fold_idx]

           # Train Markov model
           K_train = EstimateTransitionMatrix(FLATTEN(train_seqs))

           # Train marginal model
           marginal_dist = COUNTER(FLATTEN(train_seqs))
           marginal_probs = [marginal_dist[i] / SUM(marginal_dist.values())
                            for i in RANGE(N_STATES)]

           # Test
           markov_correct = 0
           marginal_correct = 0
           total = 0

           FOR seq IN test_seqs:
               FOR t IN RANGE(LEN(seq) - 1):
                   true_next = seq[t + 1]

                   # Markov prediction
                   markov_pred = ARGMAX(K_train[seq[t]])
                   IF markov_pred == true_next:
                       markov_correct += 1

                   # Marginal prediction
                   marginal_pred = ARGMAX(marginal_probs)
                   IF marginal_pred == true_next:
                       marginal_correct += 1

                   total += 1

           markov_accuracies.APPEND(markov_correct / total)
           marginal_accuracies.APPEND(marginal_correct / total)

    2. markov_accuracy = MEAN(markov_accuracies)
       marginal_accuracy = MEAN(marginal_accuracies)
       improvement = markov_accuracy - marginal_accuracy

    3. # Paired t-test
       t_stat, p_value = PAIRED_T_TEST(markov_accuracies, marginal_accuracies)

    4. RETURN markov_accuracy, marginal_accuracy, improvement, p_value

COMPLEXITY: O(n_folds * T_total)
```

---

## 10. Optimal Mapping Search

### 10.1 Exhaustive Search

**Purpose**: Find mapping that maximizes information retention.

```
ALGORITHM ExhaustiveOptimalMapping

INPUT:
    source_labels: LIST[INTEGER]   # e.g., 10 clusters
    n_source_states: INTEGER       # e.g., 10
    n_target_states: INTEGER       # e.g., 4

OUTPUT:
    optimal_mapping: DICT[INTEGER → INTEGER]
    optimal_retention: FLOAT

PROCEDURE:
    1. # Total mappings = n_target^n_source
       total_mappings = n_target_states ** n_source_states

       IF total_mappings > 10_000_000:
           WARN("Large search space, consider greedy algorithm")

    2. best_mapping = None
       best_retention = -1

    3. # Iterate over all possible mappings
       FOR mapping_index IN RANGE(total_mappings):
           # Convert index to mapping
           mapping = {}
           temp = mapping_index
           FOR source_state IN RANGE(n_source_states):
               mapping[source_state] = temp % n_target_states
               temp = temp // n_target_states

           # Compute retention
           target_labels = [mapping[s] for s in source_labels]
           retention = InformationRetention(source_labels, target_labels)

           IF retention > best_retention:
               best_retention = retention
               best_mapping = COPY(mapping)

           # Progress reporting
           IF mapping_index % 100000 == 0:
               PRINT(f"Evaluated {mapping_index}/{total_mappings}")

    4. RETURN best_mapping, best_retention

COMPLEXITY: O(n_target^n_source * n_samples)
```

### 10.2 Greedy Optimization

**Purpose**: Fast approximation when exhaustive search is infeasible.

```
ALGORITHM GreedyOptimalMapping

INPUT:
    source_labels: LIST[INTEGER]
    n_source_states: INTEGER
    n_target_states: INTEGER
    max_iterations: INTEGER = 100

OUTPUT:
    mapping: DICT[INTEGER → INTEGER]
    retention: FLOAT

PROCEDURE:
    1. # Initialize with most frequent target for each source
       source_counts = COUNTER(source_labels)
       mapping = {s: 0 for s in RANGE(n_source_states)}

    2. # Initial retention
       target_labels = [mapping[s] for s in source_labels]
       current_retention = InformationRetention(source_labels, target_labels)

    3. # Hill climbing
       FOR iteration IN RANGE(max_iterations):
           improved = False

           FOR source_state IN RANGE(n_source_states):
               original_target = mapping[source_state]

               FOR new_target IN RANGE(n_target_states):
                   IF new_target == original_target:
                       CONTINUE

                   # Try new assignment
                   mapping[source_state] = new_target
                   target_labels = [mapping[s] for s in source_labels]
                   new_retention = InformationRetention(source_labels, target_labels)

                   IF new_retention > current_retention:
                       current_retention = new_retention
                       improved = True
                   ELSE:
                       mapping[source_state] = original_target

           IF NOT improved:
               BREAK

    4. RETURN mapping, current_retention

COMPLEXITY: O(max_iterations * n_source * n_target * n_samples)
```

### 10.3 Spectral Clustering Approach

**Purpose**: Use transition structure to group source states.

```
ALGORITHM SpectralOptimalMapping

INPUT:
    source_labels: LIST[INTEGER]
    sequences: LIST[LIST[INTEGER]]  # Sequences in source state space
    n_source_states: INTEGER
    n_target_states: INTEGER

OUTPUT:
    mapping: DICT[INTEGER → INTEGER]
    retention: FLOAT

PROCEDURE:
    1. # Build transition co-occurrence matrix
       cooccur = ZEROS(n_source_states, n_source_states)

       FOR seq IN sequences:
           FOR t IN RANGE(LEN(seq) - 1):
               s1, s2 = seq[t], seq[t + 1]
               cooccur[s1][s2] += 1
               cooccur[s2][s1] += 1  # Symmetric

    2. # Add self-loops for numerical stability
       cooccur = cooccur + IDENTITY(n_source_states)

    3. # Spectral clustering
       # Compute normalized Laplacian
       D = DIAG(SUM(cooccur, axis=1))
       D_inv_sqrt = DIAG(1 / SQRT(DIAG(D)))
       L_norm = IDENTITY(n_source_states) - D_inv_sqrt @ cooccur @ D_inv_sqrt

    4. # Eigendecomposition
       eigenvalues, eigenvectors = EIGEN(L_norm)

       # Take first n_target eigenvectors (smallest eigenvalues)
       indices = ARGSORT(eigenvalues)[:n_target_states]
       embedding = eigenvectors[:, indices]

    5. # K-means on eigenvector embedding
       cluster_labels = KMEANS(embedding, n_clusters=n_target_states)

    6. # Create mapping
       mapping = {s: cluster_labels[s] for s in RANGE(n_source_states)}

    7. # Compute retention
       target_labels = [mapping[s] for s in source_labels]
       retention = InformationRetention(source_labels, target_labels)

    8. RETURN mapping, retention

COMPLEXITY: O(n_source³) for eigendecomposition
```

---

## Appendix A: Parameter Defaults

| Algorithm | Parameter | Default | Description |
|-----------|-----------|---------|-------------|
| LexicalImputation | n_paraphrases | 5 | Number of paraphrases |
| LexicalImputation | temperature | 0.7 | LLM sampling temperature |
| TransferEntropy | lag | 1 | Time lag for TE computation |
| PhaseNormalizedTE | n_phases | 50 | Resampling length |
| ConstructTENetwork | threshold_percentile | 85 | Edge threshold |
| PrimaryTypeClustering | n_clusters | 2 | COMPLEX vs FOCUSED |
| CounterfactualSimulation | n_simulations | 1000 | Monte Carlo samples |
| PermutationTestMapping | n_permutations | 10000 | Null samples |
| PredictiveImprovementTest | n_folds | 5 | CV folds |
| GreedyOptimalMapping | max_iterations | 100 | Hill climbing iterations |

---

## Appendix B: Complexity Summary

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| EstimateTransitionMatrix | O(T) | O(N²) |
| StationaryDistribution | O(1) | O(N²) |
| TransferEntropy | O(T + N³) | O(N³) |
| PairwiseTEMatrix | O(M² · T) | O(M²) |
| ExtractLineages | O(M² · avg_path) | O(M²) |
| PrimaryTypeClustering | O(M² log M) | O(M²) |
| CounterfactualSimulation | O(n_sim · T) | O(n_sim · T) |
| ExhaustiveOptimalMapping | O(K^C · n) | O(C) |
| GreedyOptimalMapping | O(iter · C · K · n) | O(C) |

Where: T = sequence length, N = 4 states, M = number of individuals, C = source states, K = target states, n = total events.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01 | Initial specification |
