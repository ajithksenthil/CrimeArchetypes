# State Space Methodology for Computational Psychodynamics

## Overview

This document describes the methodology for constructing, comparing, and validating state space representations in behavioral sequence analysis. It addresses the fundamental question: **How do we partition the space of behavioral events into discrete states for Markov analysis?**

---

## 1. The State Space Problem

### 1.1 What is a State Space?

A **state space** $\mathcal{S} = \{s_1, s_2, ..., s_K\}$ is a partition of all possible behavioral events into $K$ discrete categories. Each event $e_t$ at time $t$ is assigned to exactly one state $s(e_t) \in \mathcal{S}$.

The choice of state space determines:
- **What transitions we can observe**: $s_t \rightarrow s_{t+1}$
- **What patterns emerge**: Transition probabilities $P(s_{t+1} | s_t)$
- **What we can predict**: Future behavioral trajectories

### 1.2 Two Approaches to State Space Construction

| Approach | Description | Advantages | Disadvantages |
|----------|-------------|------------|---------------|
| **Theory-Driven** | States derived from psychological/theoretical framework | Interpretable, grounded in domain knowledge | May not capture data structure |
| **Data-Driven** | States emerge from clustering/unsupervised learning | Captures empirical patterns | May lack interpretability |

**Computational Psychodynamics** advocates for a **hybrid approach**: Use theory-driven states that are validated against data-driven structure.

---

## 2. Theory-Driven State Spaces

### 2.1 The 4-Animal Framework

The Computational Psychodynamics framework defines four fundamental motivational states based on two dimensions:

| Dimension | Self-Focus | Other-Focus |
|-----------|------------|-------------|
| **Exploration (Epistemic)** | **Seeking** | **Conferring** |
| **Exploitation (Pragmatic)** | **Revising** | **Directing** |

**Mathematical Foundation**: This 2×2 structure corresponds to the Steiner system S(3,4,8), ensuring:
- Mutual exclusivity (each event belongs to exactly one state)
- Collective exhaustiveness (all events are covered)
- Minimal category count (4 is optimal for this dimensional structure)

### 2.2 Applying Theory-Driven States

To classify events into the 4-Animal framework:

```
For each event e:
    1. Determine Self vs Other focus
    2. Determine Exploration vs Exploitation orientation
    3. Assign to corresponding state
```

Classification can be done via:
- **Keyword matching**: Fast but shallow
- **LLM classification**: Semantic understanding with chain-of-thought
- **Human expert coding**: Gold standard but expensive

---

## 3. Data-Driven State Spaces

### 3.1 Clustering-Based Approach

Data-driven states emerge from clustering events in embedding space:

```
1. Embed events: e → v(e) ∈ ℝ^d using sentence transformers
2. Cluster embeddings: {v(e)} → K clusters via K-means, hierarchical, etc.
3. Each cluster = one state
```

### 3.2 Choosing the Number of Clusters

Methods for selecting $K$:
- **Elbow method**: Plot within-cluster variance vs K
- **Silhouette score**: Measure cluster cohesion/separation
- **Information criteria**: AIC/BIC for Markov model fit
- **Domain knowledge**: Match to theoretical expectations

### 3.3 Interpreting Clusters

After clustering, interpret each cluster by:
1. Examining representative samples (closest to centroid)
2. Using LLM to generate thematic labels
3. Mapping to theoretical constructs

---

## 4. Mapping Between State Spaces

### 4.1 The Mapping Problem

Given:
- Source state space $\mathcal{S} = \{s_1, ..., s_K\}$ (e.g., 10 clusters)
- Target state space $\mathcal{T} = \{t_1, ..., t_M\}$ (e.g., 4 animals)

Find mapping $\phi: \mathcal{S} \rightarrow \mathcal{T}$ that best preserves structure.

### 4.2 Types of Mappings

| Type | Definition | Use Case |
|------|------------|----------|
| **Deterministic** | Each source state maps to exactly one target | Simple, interpretable |
| **Probabilistic** | $P(t_j | s_i)$ for each source-target pair | Handles ambiguity |
| **Optimal** | Maximizes information retention | Best statistical properties |

### 4.3 Theoretical Mapping

Map source clusters to target states based on semantic/theoretical alignment:

```python
THEORETICAL_MAPPING = {
    0: 'Conferring',   # Stalking → surveillance
    1: 'Directing',    # Sexual murder → exploitation
    2: 'Revising',     # Escalating crime → habitual patterns
    ...
}
```

**Advantage**: Interpretable, theoretically grounded
**Limitation**: May not be statistically optimal

### 4.4 Optimal (Data-Driven) Mapping

Find mapping that **maximizes information retention**:

$$\phi^* = \arg\max_\phi I(S; \phi(S))$$

where $I(S; \phi(S))$ is the mutual information between source and mapped states.

---

## 5. Information Retention

### 5.1 Definition

**Information Retention** measures how much of the source state space's structure is preserved when mapping to a coarser target space.

$$\text{Information Retention} = \frac{I(S; T)}{H(S)}$$

where:
- $I(S; T)$ = Mutual information between source states $S$ and target states $T$
- $H(S)$ = Entropy of source state distribution

### 5.2 Interpretation

| Value | Meaning |
|-------|---------|
| 0% | Target states are independent of source (no structure preserved) |
| 50% | Half the source information is captured |
| 100% | Perfect preservation (only possible if $|T| \geq |S|$) |

### 5.3 Why It Matters

Information retention quantifies the **trade-off between parsimony and fidelity**:
- Fewer states → More interpretable but less information
- More states → More information but harder to interpret

A good mapping achieves **high retention with few states**.

### 5.4 Computing Information Retention

```python
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from collections import Counter

def information_retention(source_labels, target_labels):
    """
    Compute information retention when mapping source → target.

    Returns: float in [0, 1]
    """
    # Mutual information I(S; T)
    mi = mutual_info_score(source_labels, target_labels)

    # Entropy H(S)
    source_counts = list(Counter(source_labels).values())
    h_source = entropy(source_counts, base=np.e)  # Natural log

    # Retention ratio
    return mi / h_source if h_source > 0 else 0
```

---

## 6. Finding the Optimal Mapping

### 6.1 Algorithm

Given source labels $S$ and target state count $M$:

```python
def find_optimal_mapping(source_labels, n_target_states):
    """
    Find the mapping that maximizes information retention.

    Uses the fact that optimal mapping assigns each source state
    to the target state that maximizes conditional probability.
    """
    source_states = sorted(set(source_labels))
    n_source = len(source_states)

    # Compute co-occurrence matrix
    # P(target | source) should be maximized

    best_mapping = {}
    best_retention = 0

    # For small state spaces, try all possible mappings
    from itertools import product

    for mapping_tuple in product(range(n_target_states), repeat=n_source):
        mapping = {s: t for s, t in zip(source_states, mapping_tuple)}
        target_labels = [mapping[s] for s in source_labels]
        retention = information_retention(source_labels, target_labels)

        if retention > best_retention:
            best_retention = retention
            best_mapping = mapping

    return best_mapping, best_retention
```

### 6.2 Greedy Approximation (for large state spaces)

When exhaustive search is infeasible:

```python
def greedy_optimal_mapping(source_labels, n_target_states):
    """
    Greedy algorithm: Assign each source state to the target
    that maximizes marginal information gain.
    """
    from collections import Counter

    source_counts = Counter(source_labels)
    source_states = sorted(source_counts.keys())

    # Initialize: assign all to state 0
    mapping = {s: 0 for s in source_states}

    # Iterate: reassign each source to best target
    improved = True
    while improved:
        improved = False
        for source in source_states:
            best_target = mapping[source]
            best_retention = current_retention(source_labels, mapping)

            for target in range(n_target_states):
                mapping[source] = target
                retention = current_retention(source_labels, mapping)
                if retention > best_retention:
                    best_retention = retention
                    best_target = target
                    improved = True

            mapping[source] = best_target

    return mapping
```

### 6.3 Spectral Approach

For principled optimization, use spectral methods on the co-occurrence matrix:

```python
def spectral_optimal_mapping(source_labels, n_target_states):
    """
    Use spectral clustering on source state co-occurrence to find
    optimal grouping into target states.
    """
    from sklearn.cluster import SpectralClustering

    # Build co-occurrence matrix: how often do source states co-occur?
    source_states = sorted(set(source_labels))
    n_source = len(source_states)

    # Transition co-occurrence
    cooccur = np.zeros((n_source, n_source))
    for i in range(len(source_labels) - 1):
        s1 = source_states.index(source_labels[i])
        s2 = source_states.index(source_labels[i + 1])
        cooccur[s1, s2] += 1
        cooccur[s2, s1] += 1

    # Cluster source states based on transition similarity
    clustering = SpectralClustering(n_clusters=n_target_states,
                                     affinity='precomputed')
    target_assignments = clustering.fit_predict(cooccur + np.eye(n_source))

    mapping = {source_states[i]: target_assignments[i]
               for i in range(n_source)}

    return mapping
```

---

## 7. Comparing State Space Schemas

### 7.1 Comparison Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **NMI** | $\frac{I(S_1; S_2)}{\sqrt{H(S_1) H(S_2)}}$ | [0, 1] | Normalized agreement |
| **ARI** | Adjusted Rand Index | [-1, 1] | Chance-corrected agreement |
| **V-measure** | Harmonic mean of homogeneity & completeness | [0, 1] | Clustering quality |
| **Info Retention** | $\frac{I(S_1; S_2)}{H(S_1)}$ | [0, 1] | Structure preservation |

### 7.2 Statistical Testing

**Permutation Test for Mapping Quality**:

```python
def permutation_test_mapping(source_labels, target_labels, mapping,
                              n_permutations=10000):
    """
    Test H0: Observed mapping is no better than random mapping.
    """
    # Observed information retention
    observed = information_retention(source_labels, target_labels)

    # Null distribution: random mappings
    null_distribution = []
    source_states = sorted(set(source_labels))
    n_target = len(set(target_labels))

    for _ in range(n_permutations):
        # Random mapping
        random_mapping = {s: np.random.randint(0, n_target)
                         for s in source_states}
        random_targets = [random_mapping[s] for s in source_labels]
        null_distribution.append(
            information_retention(source_labels, random_targets)
        )

    # P-value
    p_value = np.mean(np.array(null_distribution) >= observed)

    return observed, np.mean(null_distribution), p_value
```

**Likelihood Ratio Test for Nested Models**:

```python
def likelihood_ratio_test(ll_full, ll_reduced, df_diff):
    """
    Test whether the full model (more states) significantly
    improves over the reduced model (fewer states).
    """
    lr_statistic = 2 * (ll_full - ll_reduced)
    p_value = 1 - stats.chi2.cdf(lr_statistic, df_diff)
    return lr_statistic, p_value
```

### 7.3 Model Selection Criteria

For choosing between state spaces:

| Criterion | Formula | Preference |
|-----------|---------|------------|
| **AIC** | $2k - 2\ln(L)$ | Lower is better |
| **BIC** | $k\ln(n) - 2\ln(L)$ | Lower is better (penalizes complexity more) |
| **Cross-validated LL** | Average test log-likelihood | Higher is better |

where $k$ = number of parameters, $L$ = likelihood, $n$ = sample size.

---

## 8. Validation Framework

### 8.1 Three Essential Tests

**Test 1: Mapping Null**
> Is the theoretical mapping better than random mappings?

```
H0: I(S; φ_theory(S)) = I(S; φ_random(S))
```

**Test 2: Sequence Null**
> Do observed transitions differ from shuffled sequences?

```
H0: P_observed = P_shuffled
```

**Test 3: Predictive Null**
> Does state-based prediction beat marginal prediction?

```
H0: P(s_{t+1} | s_t) = P(s_{t+1})
```

### 8.2 Interpreting Results

| Test 1 | Test 2 | Test 3 | Interpretation |
|--------|--------|--------|----------------|
| ✓ Sig | ✓ Sig | ✓ Sig | Strong validation: Theory-driven mapping is optimal AND captures structure |
| ✗ NS | ✓ Sig | ✓ Sig | Structure exists but theoretical mapping isn't uniquely optimal |
| ✓ Sig | ✗ NS | ✗ NS | Mapping is good but no temporal structure (sequences are random) |
| ✗ NS | ✗ NS | ✗ NS | Neither mapping nor states capture meaningful structure |

---

## 9. Practical Workflow

### Step 1: Data-Driven Exploration

```python
# 1. Embed events
embeddings = sentence_transformer.encode(events)

# 2. Cluster at multiple granularities
for k in [4, 6, 8, 10, 12]:
    clusters = KMeans(n_clusters=k).fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, clusters)
    print(f"K={k}: Silhouette={silhouette:.3f}")

# 3. Interpret clusters with LLM
for cluster_id in range(k):
    samples = get_representative_samples(cluster_id)
    theme = llm.generate_theme(samples)
```

### Step 2: Theory-Driven Classification

```python
# Classify events into theoretical states
for event in events:
    state = classify_into_4_animals(event)  # Using LLM or keywords
```

### Step 3: Derive Optimal Mapping

```python
# Find mapping that maximizes information retention
optimal_mapping = find_optimal_mapping(cluster_labels, n_target_states=4)
theoretical_mapping = PREDEFINED_THEORETICAL_MAPPING

# Compare
optimal_retention = information_retention(cluster_labels,
                                          apply_mapping(cluster_labels, optimal_mapping))
theoretical_retention = information_retention(cluster_labels,
                                              apply_mapping(cluster_labels, theoretical_mapping))
```

### Step 4: Statistical Validation

```python
# Test 1: Is theoretical mapping better than random?
obs, null_mean, p_value = permutation_test_mapping(
    cluster_labels, theoretical_targets, n_permutations=10000
)

# Test 2: Is there temporal structure?
sequence_p = test_sequence_null(sequences)

# Test 3: Does Markov beat marginal?
predictive_p = test_predictive_null(sequences)
```

### Step 5: Report Results

```
State Space Validation Results:
- Information Retention: 65% (theoretical), 68% (optimal)
- Mapping Null: p = 0.22 (theoretical not significantly better than random)
- Sequence Null: p < 0.0001 (significant temporal structure exists)
- Predictive Null: p < 0.0001 (Markov improves prediction by 12.6%)

Conclusion: The 4-state representation captures significant temporal
structure in behavioral sequences. While the specific cluster-to-state
mapping is not uniquely optimal, the theoretical framework provides
interpretable states grounded in psychological theory.
```

---

## 10. Recommendations

### When to Use Theory-Driven States

- When interpretability is paramount
- When you need to communicate with domain experts
- When prior literature uses the same framework
- When theoretical grounding matters for validity

### When to Use Data-Driven States

- When exploring unknown domains
- When maximizing predictive accuracy
- When the data may have structure not anticipated by theory
- When you need to discover new patterns

### Best Practice: Hybrid Approach

1. **Start with data-driven clustering** to discover empirical structure
2. **Compute optimal mapping** to theory-driven states
3. **Compare** optimal vs theoretical mappings
4. **Report both** information retention values
5. **Use theoretical states** for interpretation if retention is acceptable
6. **Refine theory** if optimal mapping suggests different groupings

---

## References

1. Senthil, A.K. (2026). Computational Psychodynamics: A Unified Framework for Behavioral Analysis.
2. Shannon, C.E. (1948). A Mathematical Theory of Communication.
3. Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory.
4. Vinh, N.X., Epps, J., & Bailey, J. (2010). Information Theoretic Measures for Clusterings Comparison.
