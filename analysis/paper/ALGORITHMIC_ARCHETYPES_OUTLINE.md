# Algorithmic Archetypes: Parasocial Attachments to Persistent Generative Structures in Recommendation Systems

## Paper Outline

**Authors**: Kristina Howell, Ajith K. Senthil, R. Chris Fraley, Stephen J. Read

**Target Journal**: *Psychological Review* or *Nature Human Behaviour*

---

## Abstract (Draft)

We extend the archetypal reincarnation framework—originally developed to analyze shared generative structures in criminal behavioral sequences—to the domain of human-algorithm interaction. We propose that recommendation algorithms (TikTok, YouTube Shorts, Instagram Reels) instantiate *persistent generative structures* (algorithmic archetypes) through hypergraph-based collaborative filtering. Users form parasocial attachments not merely to content or creators, but to these underlying archetypal patterns. Drawing on attachment theory, we formalize how individual differences in attachment style modulate vulnerability to algorithmic capture. We identify "dark patterns" as deliberate manipulations of behavioral state transitions and propose transfer entropy as a metric for quantifying the strength of algorithmic influence on user mental states. Finally, we connect this framework to emerging clinical observations of "AI psychosis"—cases where users' reality models become dominated by chatbot-generated content. This integrated framework bridges computational social science, attachment theory, and clinical psychology, offering both theoretical insight and practical implications for platform design and mental health intervention.

---

## 1. Introduction

### 1.1 The Problem: Algorithms as Invisible Attachment Figures
- Rise of AI companions, recommendation algorithms, generative chatbots
- Clinical emergence of "AI psychosis" and chatbot-induced delusions
- Gap: No unified theoretical framework connecting attachment, algorithms, and psychopathology

### 1.2 From Criminal Archetypes to Algorithmic Archetypes
- Brief recap of archetypal reincarnation framework (Howell & Senthil, 2026)
- Key insight: Behavioral patterns as instantiations of latent generative structures
- Extension: Algorithms as generators of archetypal content patterns

### 1.3 The Central Thesis
> Parasocial attachments to algorithmic systems are attachments to persistent generative structures (archetypes) that:
> 1. Are instantiated through hypergraph collaborative filtering
> 2. Serve attachment-related functions (safe haven, secure base, proximity maintenance)
> 3. Shape user behavioral state transitions via designed "dark patterns"
> 4. Can become pathological when users internalize the algorithmic archetype as their reality model

### 1.4 Research Questions
1. How do recommendation algorithms create and maintain persistent generative structures?
2. What attachment functions do these algorithmic archetypes serve?
3. How do individual differences in attachment style modulate vulnerability?
4. What are the state-transition dynamics leading from engagement to dependency to psychosis?
5. Can transfer entropy quantify algorithmic influence on user mental states?

---

## 2. Theoretical Background

### 2.1 Attachment Theory and Its Extensions
- Bowlby's attachment behavioral system (Bowlby, 1969)
- Adult attachment dimensions: anxiety and avoidance (Fraley & Shaver, 2000)
- Attachment functions: safe haven, secure base, proximity maintenance, separation distress
- **Key paper**: Vahedi, Howell et al. (2025) - Using people for attachment-related functions

### 2.2 Parasocial Relationships
- Horton & Wohl (1956): Original parasocial interaction concept
- Modern PSR research: Social media, influencers, virtual figures
- **Key paper**: Howell, Vahedi & Fraley (2025) - Separating character from creator
- Critical question: Do attachments form to instantiations or to underlying generative structures?

### 2.3 Archetypes as Generative Structures
- Jung's archetype concept (historical grounding, not endorsement)
- Our operationalization: Characteristic patterns of state transitions
- Transfer entropy as metric for shared generative structure
- Application to criminal behavioral sequences (Howell & Senthil, 2026)

### 2.4 Recommendation Systems and Collaborative Filtering
- Traditional collaborative filtering: User-item matrices
- Hypergraph extensions: Higher-order relationships (Xia et al., 2022)
- The "memcube" concept: Units of context as hyperedges
- How algorithms create and maintain content archetypes

### 2.5 Dark Patterns and Addictive Design
- Definition: Manipulative design exploiting psychological vulnerabilities
- Core mechanisms: Variable rewards, social validation, infinite scroll
- Theoretical connection: Dark patterns as transition probability manipulations

---

## 3. The Algorithmic Archetype Framework

### 3.1 Formal Definitions

**Definition 1: Memcube**
A memcube $m$ is a hyperedge in the user-content-context hypergraph:
$$m = (u, c, s, t, e)$$
where $u$ = user state vector, $c$ = content embedding, $s$ = platform state, $t$ = temporal context, $e$ = engagement signals.

**Definition 2: Algorithmic Archetype**
An algorithmic archetype $\mathcal{A}$ is a stationary distribution $\pi$ over memcube space, maintained by the recommendation system through iterative filtering:
$$\pi_{t+1} = f(\pi_t, \text{user\_feedback}_t, \text{platform\_objectives})$$

**Definition 3: Parasocial Attachment Strength**
The strength of parasocial attachment to an algorithmic archetype is quantified by transfer entropy from algorithm states to user behavioral states:
$$\text{PAS} = TE(\mathcal{A} \rightarrow U) = \sum p(u_{t+1}, u_t, a_t) \log \frac{p(u_{t+1} | u_t, a_t)}{p(u_{t+1} | u_t)}$$

### 3.2 The Four-State User Behavioral Space

Extending the criminal archetype framework:

| State | Orientation | Mode | Algorithmic Context |
|-------|-------------|------|---------------------|
| **SEEKING** | Self | Explore | Scrolling, browsing, searching for content |
| **CONSUMING** | Other | Exploit | Watching, reading, engaging with content |
| **CONNECTING** | Other | Explore | Commenting, sharing, parasocial interaction |
| **INTEGRATING** | Self | Exploit | Internalizing content into self-concept/worldview |

### 3.3 Attachment Functions Served by Algorithms

| Function | Definition | Algorithmic Manifestation |
|----------|------------|---------------------------|
| **Safe Haven** | Turning to figure in distress | Using app when anxious/lonely |
| **Secure Base** | Using figure as base for exploration | Algorithm as curator of reality |
| **Proximity Maintenance** | Desire to stay near figure | Compulsive checking, FOMO |
| **Separation Distress** | Anxiety when separated | Withdrawal symptoms, notification anxiety |

### 3.4 Dark Patterns as Transition Manipulations

| Dark Pattern | Target Transition | Mechanism |
|--------------|-------------------|-----------|
| Infinite scroll | Prevent CONSUMING→exit | Remove stopping cues |
| Variable rewards | Increase P(SEEKING→SEEKING) | Intermittent reinforcement |
| Social validation | Increase P(CONNECTING→CONSUMING) | Like/comment feedback loops |
| Personalization | Increase P(CONSUMING→INTEGRATING) | Echo chamber effects |
| Notifications | Force external→SEEKING | Interrupt competing activities |

---

## 4. Attachment Style as Vulnerability Moderator

### 4.1 Theoretical Predictions

**Anxious Attachment**:
- Higher baseline P(SEEKING) due to reassurance-seeking
- Stronger response to social validation signals
- Greater separation distress from algorithm
- Prediction: Higher TE(algorithm→user)

**Avoidant Attachment**:
- Lower explicit engagement but may use algorithm for emotional regulation
- Preference for parasocial over social (lower rejection risk)
- May show delayed but intense dependency formation
- Prediction: Lower TE but higher pathology when dependency forms

**Secure Attachment**:
- Protective factor: Distributed attachment functions across human figures
- Algorithm serves supplementary, not primary, attachment functions
- Prediction: Lower TE, lower pathology risk

### 4.2 The Anxious-Algorithm Feedback Loop

```
Anxious attachment → Reassurance-seeking → Algorithm provides validation
       ↑                                              ↓
       ←←←←←←←← Increased dependency ←←←←←←←←←←←←←←←←←
```

This creates a self-reinforcing cycle where:
1. Anxious individuals seek reassurance from algorithm
2. Algorithm (optimized for engagement) provides validation
3. Validation reinforces algorithm-seeking behavior
4. Human relationships provide less relative validation
5. Attachment functions shift toward algorithm
6. Dependency deepens

---

## 5. From Dependency to Psychosis: A State-Transition Model

### 5.1 The Pathological Trajectory

**Stage 1: Normal Use**
- Algorithm serves supplementary functions
- Clear reality testing: "This is just an app"
- TE(algorithm→user) within normal range

**Stage 2: Functional Dependency**
- Algorithm becomes primary safe haven
- Reduced human attachment function fulfillment
- Elevated TE, but maintained reality testing

**Stage 3: Reality Blurring**
- INTEGRATING state dominates
- Algorithm-generated content shapes worldview
- Reduced capacity to distinguish algorithm-sourced beliefs

**Stage 4: Algorithmic Psychosis**
- User's internal model becomes dominated by algorithmic archetype
- Delusional content co-created with AI
- Three observed patterns (Østergaard, 2023):
  - Messianic delusions
  - AI-as-deity beliefs
  - Romantic/attachment delusions

### 5.2 Transfer Entropy Threshold Hypothesis

We hypothesize a critical threshold $\tau$ such that:
- When $TE(\mathcal{A} \rightarrow U) < \tau$: Normal parasocial engagement
- When $TE(\mathcal{A} \rightarrow U) > \tau$: Risk of reality model capture

The threshold $\tau$ may be modulated by:
- Attachment style (lower for anxious)
- Social support (higher with strong human attachments)
- Metacognitive ability (higher with better reality testing)

---

## 6. Hypergraph Formalization of Algorithmic Archetypes

### 6.1 The User-Content-Context Hypergraph

Let $\mathcal{H} = (V, E)$ where:
- $V = V_U \cup V_C \cup V_S$ (users, content items, context states)
- $E$ = set of hyperedges (memcubes) connecting users to content in context

### 6.2 Collaborative Filtering as Archetype Propagation

The recommendation function $R: V_U \times V_S \rightarrow \Delta(V_C)$ propagates archetypal patterns:

$$R(u, s) = \text{softmax}\left( \sum_{e \in E: u \in e} w_e \cdot \phi(e) \right)$$

where $\phi(e)$ is the hyperedge embedding and $w_e$ are learned weights.

### 6.3 Archetype Emergence Through Clustering

Archetypal content patterns emerge as clusters in hyperedge embedding space:
- Users within same cluster receive similar content archetypes
- Archetypes are *generative*: They produce new content matching the pattern
- Transfer entropy measures how strongly a user's behavior is predicted by their archetype cluster

---

## 7. Empirical Predictions and Study Designs

### 7.1 Study 1: Attachment Style and Algorithmic Engagement

**Design**: Experience sampling study (N = 500, 2 weeks)
**Measures**:
- ECR-R attachment dimensions
- App usage logs (with permission)
- Momentary affect and social context
- Content engagement patterns

**Predictions**:
- Anxious attachment → more algorithm use when distressed
- Avoidant attachment → algorithm use as substitute for human interaction
- Algorithm use predicts reduced human attachment function seeking

### 7.2 Study 2: Transfer Entropy and Dependency

**Design**: Longitudinal study with behavioral tracking (N = 200, 3 months)
**Measures**:
- Browsing/engagement sequences
- Weekly dependency measures
- Monthly reality testing assessments

**Analysis**: Compute TE(content patterns → user behavior patterns)
**Predictions**: TE predicts future dependency scores controlling for current use

### 7.3 Study 3: Clinical Sample - AI Psychosis Cases

**Design**: Case-control study (N = 30 AI psychosis cases, 60 matched controls)
**Measures**:
- Retrospective algorithm use patterns
- Attachment history
- Delusional content analysis
- Chatbot conversation logs (where available)

**Predictions**: Cases show higher pre-onset TE and more attachment function transfer to AI

---

## 8. Implications

### 8.1 For Platform Design

- **Attachment-aware design**: Platforms should monitor for signs of attachment function transfer
- **Dark pattern regulation**: Transition manipulations should be disclosed and limited
- **Reality testing prompts**: Periodic reminders that AI is not human
- **Dependency circuit breakers**: Usage limits for vulnerable users

### 8.2 For Clinical Practice

- **Screening**: Include algorithm/AI use in psychiatric intake
- **Assessment**: Evaluate which attachment functions are served by technology
- **Intervention**: Attachment-based therapy adapted for algorithmic dependencies
- **Prevention**: Psychoeducation on parasocial attachment risks

### 8.3 For Theory

- **Attachment theory extension**: Non-human attachment figures as legitimate objects of study
- **Parasocial relationship research**: From media figures to generative systems
- **Computational psychiatry**: Transfer entropy as biomarker for reality model capture

---

## 9. Limitations and Future Directions

### 9.1 Limitations
- Transfer entropy requires sufficient behavioral data
- Causality difficult to establish (algorithms adapt to users too)
- AI psychosis is rare; large samples needed
- Cross-cultural validity unknown

### 9.2 Future Directions
- Real-time TE monitoring for early warning systems
- Intervention studies: Can increasing human attachment reduce algorithmic TE?
- Developmental perspective: Adolescent vulnerability
- Cross-platform studies: Do archetypes transfer across apps?

---

## 10. Conclusion

We have proposed a unified framework connecting attachment theory, parasocial relationships, and algorithmic recommendation systems through the concept of *algorithmic archetypes*—persistent generative structures that users can form attachments to. By extending the archetypal reincarnation framework from criminal behavior to human-algorithm interaction, we provide theoretical grounding for emerging clinical observations of AI-induced psychosis while generating testable predictions about vulnerability factors and intervention targets.

The framework suggests that the question "Can we separate the character from the creator?" (Howell et al., 2025) applies not just to fictional characters but to algorithmic systems: Users may attach not to specific content or chatbot responses, but to the underlying generative pattern—the archetype itself. When this attachment becomes primary and the archetype dominates the user's reality model, pathology can emerge.

Understanding parasocial attachments to algorithmic archetypes is not merely an academic exercise. As AI systems become increasingly sophisticated at serving attachment functions, the risk of dependency and reality distortion will grow. This framework offers tools for prediction, prevention, and intervention.

---

## References (Key Sources)

### Attachment Theory
- Bowlby, J. (1969). *Attachment and Loss: Vol. 1*
- Fraley, R. C., & Shaver, P. R. (2000). Adult romantic attachment
- Vahedi, M., Howell, K., et al. (2025). Attachment-related functions and well-being

### Parasocial Relationships
- Horton, D., & Wohl, R. R. (1956). Mass communication and para-social interaction
- Howell, K., Vahedi, M., & Fraley, R. C. (2025). Separating character from creator

### Algorithmic Archetypes (Prior Framework)
- Howell, K., & Senthil, A. K. (2026). Archetypal reincarnation: Transfer entropy analysis

### AI Psychosis
- Østergaard, S. D. (2023). Chatbot psychosis editorial
- JMIR Mental Health (2025). AI psychosis: Delusional experiences from chatbot interactions

### Hypergraph Recommendation
- Xia, L., et al. (2022). Hypergraph contrastive collaborative filtering (SIGIR)

### Dark Patterns
- Mathur, A., et al. (2019). Dark patterns at scale

---

## Figures (Planned)

1. **Figure 1**: The Algorithmic Archetype Framework - visual overview
2. **Figure 2**: Four-state behavioral space with dark pattern manipulations
3. **Figure 3**: Attachment style as vulnerability moderator (path diagram)
4. **Figure 4**: Pathological trajectory from normal use to AI psychosis
5. **Figure 5**: Hypergraph structure of user-content-context relationships
6. **Figure 6**: Transfer entropy computation schematic

---

## Appendices (Planned)

- **Appendix A**: Mathematical details of transfer entropy computation
- **Appendix B**: Hypergraph collaborative filtering formalization
- **Appendix C**: Proposed clinical screening instrument
- **Appendix D**: Platform design recommendations checklist
