# Methods Documentation

## Empirical Validation of the 4-Animal State Space for Criminal Behavioral Sequences

**For publication use**

---

## 1. Dataset

### 1.1 Data Source
Life event data was collected from published biographical accounts and court records of 26 serial killers. Each individual's life history was coded as a sequence of discrete life events.

### 1.2 Data Structure
- **N = 1,246** total life events
- **n = 26** individuals
- Events per individual: Mean = 47.9, Range = [15, 89]
- Each event consists of a textual description of a life occurrence

### 1.3 Preprocessing
- Events with missing or empty descriptions were excluded
- Events were processed in chronological order within each individual
- No temporal normalization was applied (age-at-event preserved where available)

---

## 2. Theoretical Framework

### 2.1 The 4-Animal State Space
Following the Computational Psychodynamics framework (Senthil, 2025), behavioral states are categorized along two orthogonal dimensions:

1. **Self/Other**: Whether the behavior is directed toward the self or toward others
2. **Explore/Exploit**: Whether the behavior involves exploration of new possibilities or exploitation of established patterns

This yields four behavioral states:

| State | Dimensions | Behavioral Description |
|-------|------------|----------------------|
| **Seeking** | Self + Explore | Introspection, identity formation, internal conflict |
| **Directing** | Other + Exploit | Control, manipulation, violence toward others |
| **Conferring** | Other + Explore | Observation, social learning, studying others |
| **Revising** | Self + Exploit | Rituals, habit reinforcement, compulsive patterns |

### 2.2 Theoretical Predictions
Based on criminological theory, we expected:
1. Dominance of **Directing** states in criminal life events
2. High persistence (self-loop probability) for Directing states
3. Transitions converging toward Directing as an attractor state

---

## 3. Classification Methods

### 3.1 Theory-Driven: 4-Animal Embedding Classification

#### Prototype Definition
For each of the four animal states, we defined 8 prototype descriptions capturing the semantic essence of that behavioral mode (see Table S1 in Supplementary Materials).

#### Embedding Model
- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- Embedding dimension: 384
- Pre-trained on 1B+ sentence pairs

#### Classification Procedure
1. Compute embedding for each prototype description
2. Compute centroid embedding for each state (mean of 8 prototypes)
3. For each life event:
   - Compute event embedding
   - Calculate cosine similarity to each state centroid
   - Assign event to state with highest similarity
   - Record similarity score as classification confidence

### 3.2 Data-Driven: Clustering Methods

Four clustering algorithms were evaluated at K=4 to match the theoretical state space:

1. **K-Means**: Standard Lloyd's algorithm with k-means++ initialization
   - `n_init=10`, `random_state=42`

2. **Agglomerative Clustering (Ward)**: Hierarchical clustering minimizing within-cluster variance
   - `linkage='ward'`

3. **Agglomerative Clustering (Complete)**: Hierarchical clustering using maximum linkage
   - `linkage='complete'`

4. **Gaussian Mixture Model (GMM)**: Probabilistic clustering assuming Gaussian distributions
   - `n_components=4`, `n_init=3`, `random_state=42`

All data-driven methods operated on the same embedding space as the 4-Animal classifier.

---

## 4. Markov Chain Analysis

### 4.1 Transition Matrix Estimation

Given a set of behavioral sequences, the transition matrix P was estimated using Dirichlet-smoothed maximum likelihood:

$$P_{ij} = \frac{N_{ij} + \alpha}{\sum_k (N_{ik} + \alpha)}$$

where:
- $N_{ij}$ = count of transitions from state i to state j
- $\alpha$ = Dirichlet smoothing parameter (set to 1.0)

### 4.2 Stationary Distribution

The stationary distribution π was computed as the left eigenvector corresponding to eigenvalue 1:

$$\pi^T P = \pi^T$$

### 4.3 Entropy Rate

The entropy rate H(X) measures the average uncertainty in next-state prediction:

$$H(X) = -\sum_i \pi_i \sum_j P_{ij} \log_2 P_{ij}$$

Maximum entropy for K=4 states is 2.0 bits.

### 4.4 Effective Number of States

The effective number of states measures how many states are actually utilized:

$$N_{eff} = 2^{H(\pi)}$$

where $H(\pi) = -\sum_i \pi_i \log_2 \pi_i$ is the entropy of the stationary distribution.

---

## 5. Evaluation Metrics

### 5.1 Cross-Validation Accuracy

**Procedure**: 5-fold cross-validation at the individual level
1. Partition 26 individuals into 5 folds
2. For each fold:
   - Train: Estimate transition matrix from training individuals
   - Test: Predict next state for each transition in test individuals
   - Accuracy = proportion of correct next-state predictions

**Baseline**: Random prediction accuracy = 1/K = 25% for K=4

### 5.2 Silhouette Score

Measures cluster separation in embedding space:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where a(i) = mean intra-cluster distance, b(i) = mean nearest-cluster distance.

Range: [-1, 1], higher is better.

### 5.3 State Efficiency

$$\text{Efficiency} = \frac{N_{eff}}{K}$$

Measures what proportion of the state space is effectively utilized.

---

## 6. Statistical Analysis

### 6.1 Overall Comparison

**Friedman Test**: Non-parametric repeated measures ANOVA for comparing multiple methods across individuals.

$$\chi^2_F = \frac{12n}{k(k+1)} \left[ \sum_j R_j^2 - \frac{k(k+1)^2}{4} \right]$$

### 6.2 Pairwise Comparisons

**Permutation Test**: Non-parametric test for paired differences
- 10,000 permutations
- Two-tailed p-values

**Multiple Comparison Correction**:
- Bonferroni: α_corrected = 0.05 / n_tests
- Benjamini-Hochberg: FDR control at q = 0.05

### 6.3 Effect Sizes

**Cohen's d**: Standardized mean difference

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}$$

Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large

### 6.4 Bootstrap Confidence Intervals

- 10,000 bootstrap samples
- Percentile method for 95% CIs
- Seed: 42 for reproducibility

---

## 7. Software and Reproducibility

### 7.1 Dependencies
```
python>=3.9
numpy>=1.21
scipy>=1.7
scikit-learn>=1.0
sentence-transformers>=2.2
matplotlib>=3.5
seaborn>=0.11
```

### 7.2 Code Availability

All analysis code is available at: [repository URL]

Key scripts:
- `run_embedding_animal_comparison.py`: Main comparison study
- `statistical_analysis.py`: Bootstrap CIs and significance tests
- `publication_figures.py`: Generate publication figures
- `research_pipeline.py`: Reproducible pipeline for ongoing research

### 7.3 Random Seeds

All stochastic procedures used `random_state=42` for reproducibility:
- K-fold split generation
- K-Means initialization
- GMM initialization
- Bootstrap sampling
- Permutation testing

---

## 8. Supplementary Materials

### Table S1: 4-Animal Prototype Descriptions

**Seeking (Self + Explore)**
1. "questioned his own identity and sense of self"
2. "experienced internal conflict and psychological turmoil"
3. "explored dark thoughts and fantasies privately"
4. "struggled with feelings of inadequacy and self-doubt"
5. "became introspective about his violent urges"
6. "searched for meaning and purpose in his actions"
7. "reflected on childhood trauma and its effects"
8. "developed obsessive thought patterns"

**Directing (Other + Exploit)**
1. "manipulated and controlled his victims"
2. "exerted dominance over family members"
3. "committed acts of violence against others"
4. "exploited vulnerable individuals for his purposes"
5. "murdered his victim in a calculated manner"
6. "abused his position of power over others"
7. "tortured and killed multiple victims"
8. "stalked and attacked his targets"

**Conferring (Other + Explore)**
1. "observed potential victims from a distance"
2. "studied criminal cases and violent behavior"
3. "watched documentaries about serial killers"
4. "learned manipulation techniques from others"
5. "observed family dynamics and relationships"
6. "studied victim behavior patterns"
7. "gathered information about targets"
8. "monitored neighborhood activities"

**Revising (Self + Exploit)**
1. "developed ritualistic behaviors around killing"
2. "established strict routines and patterns"
3. "reinforced compulsive habits and urges"
4. "maintained specific methods of operation"
5. "repeated familiar patterns of behavior"
6. "followed established rituals precisely"
7. "practiced signature behaviors consistently"
8. "adhered to personal rules and systems"

---

## References

Senthil, A. (2025). Computational Psychodynamics: A Mathematical Framework for Modeling Human Behavior. [Manuscript in preparation].

---

*Document generated: January 2026*
