"""
Intervention Protocol Library

Evidence-Based Intervention Protocols for Criminal Behavioral Trajectories

This module provides clinically-grounded intervention protocols mapped to the
4-Animal Behavioral State Space model:

    SEEKING (Fox/Spider) - Internal cognitive processing: fantasy development,
        planning, rumination. Characterized by cognitive rehearsal of offense
        scenarios, victim selection criteria formation, and arousal conditioning.

    DIRECTING (Wolf/Shark) - Active exploitation/harm: execution of planned
        behavior, predatory action, violence. The harmful outcome state.

    CONFERRING (Owl) - External information gathering: surveillance, stalking,
        target reconnaissance, environment scanning. Precursor to action.

    REVISING (Octopus) - Post-event processing: reliving, trophy behavior,
        evidence management, psychological integration of offense experience.

Theoretical Foundations:
------------------------
1. Finkelhor's Precondition Model (1984) - Four preconditions for sexual offending
2. Ward & Siegert's Pathways Model (2002) - Multiple etiological pathways
3. Marshall & Barbaree's Integrated Theory (1990) - Developmental vulnerabilities
4. Yates' Self-Regulation Model (2003) - Approach vs avoidance pathways
5. Andrews & Bonta's RNR Model (2010) - Risk-Need-Responsivity framework
6. Maruna's Desistance Theory (2001) - Identity transformation

Each protocol specifies:
- Theoretical mechanism of action linked to behavioral states
- Target transitions and why intervention disrupts them
- Evidence base with effect sizes from meta-analyses
- Clinical implementation parameters
- Population-specific adaptations

References:
-----------
Hanson, R.K., et al. (2002). First report of the collaborative outcome data
    project on the effectiveness of psychological treatment for sex offenders.
    Sexual Abuse, 14(2), 169-194. [Meta-analysis: d=0.44 for CBT]

Lösel, F., & Schmucker, M. (2005). The effectiveness of treatment for sexual
    offenders: A comprehensive meta-analysis. Journal of Experimental
    Criminology, 1(1), 117-146. [OR=1.70 for organic treatments]

Landenberger, N.A., & Lipsey, M.W. (2005). The positive effects of cognitive–
    behavioral programs for offenders: A meta-analysis. Journal of
    Experimental Criminology, 1(4), 451-476. [d=0.52 for high-risk offenders]

Aos, S., et al. (2006). Evidence-based treatment of alcohol, drug, and mental
    health disorders. Washington State Institute for Public Policy.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ProtocolCategory(Enum):
    """Categories of intervention protocols."""
    THERAPEUTIC = "therapeutic"
    ENVIRONMENTAL = "environmental"
    SUPERVISION = "supervision"
    PHARMACOLOGICAL = "pharmacological"
    COMBINED = "combined"


class EvidenceLevel(Enum):
    """
    Evidence level for protocol effectiveness (GRADE-adapted).

    A: Multiple RCTs or meta-analysis (high confidence)
    B: Single RCT or multiple controlled studies (moderate confidence)
    C: Observational studies or case series (low confidence)
    D: Expert consensus or clinical experience (very low confidence)
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class IntensityLevel(Enum):
    """Intervention intensity levels per RNR principles."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    INTENSIVE = "intensive"


@dataclass
class InterventionProtocol:
    """
    Evidence-based intervention protocol specification.

    Each protocol is grounded in:
    1. Theoretical mechanism explaining WHY it works
    2. Specific behavioral state targets and transition effects
    3. Empirical evidence base with effect sizes
    4. Clinical implementation parameters

    The effect_on_transitions dictionary specifies how the intervention
    modifies state transition probabilities:
    - Negative values REDUCE harmful transition probability
    - Positive values INCREASE protective transition probability
    - Values represent absolute probability changes (e.g., -0.25 = 25% reduction)
    """
    # Identification
    name: str
    display_name: str
    category: ProtocolCategory
    description: str

    # Clinical Theory
    theoretical_basis: str = ""
    mechanism_of_action: str = ""
    state_specific_rationale: Dict[str, str] = field(default_factory=dict)

    # Targeting
    target_states: List[str] = field(default_factory=list)
    target_transitions: List[Tuple[str, str]] = field(default_factory=list)
    contraindicated_states: List[str] = field(default_factory=list)

    # Mechanism of action (transition effects)
    effect_on_transitions: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Clinical parameters
    intensity_levels: List[IntensityLevel] = field(
        default_factory=lambda: [IntensityLevel.LOW, IntensityLevel.MEDIUM, IntensityLevel.HIGH]
    )
    default_intensity: IntensityLevel = IntensityLevel.MEDIUM
    duration_weeks: int = 12
    session_frequency: str = "weekly"
    contraindications: List[str] = field(default_factory=list)

    # Evidence base
    evidence_level: EvidenceLevel = EvidenceLevel.C
    effectiveness_estimate: float = 0.3
    number_needed_to_treat: float = 5.0
    effect_size: float = 0.5
    key_studies: List[str] = field(default_factory=list)

    # Resources
    cost_per_week: float = 200.0
    required_resources: List[str] = field(default_factory=list)
    required_training: str = ""
    monitoring_schedule: str = "bi-weekly"

    # Implementation details
    key_components: List[str] = field(default_factory=list)
    adaptation_notes: str = ""
    population_notes: str = ""

    def get_scaled_effects(self, intensity: IntensityLevel) -> Dict[Tuple[str, str], float]:
        """Get transition effects scaled by intensity (RNR dosage principle)."""
        intensity_multipliers = {
            IntensityLevel.LOW: 0.5,
            IntensityLevel.MEDIUM: 1.0,
            IntensityLevel.HIGH: 1.5,
            IntensityLevel.INTENSIVE: 2.0
        }
        multiplier = intensity_multipliers.get(intensity, 1.0)
        return {
            trans: min(0.95, max(-0.95, effect * multiplier))
            for trans, effect in self.effect_on_transitions.items()
        }

    def is_applicable(self, current_state: str) -> bool:
        """Check if protocol is applicable for current state."""
        if current_state in self.contraindicated_states:
            return False
        return current_state in self.target_states or not self.target_states

    def get_rationale_for_state(self, state: str) -> str:
        """Get clinical rationale for intervention in specific state."""
        return self.state_specific_rationale.get(state, self.mechanism_of_action)

    def to_dict(self) -> Dict:
        """Serialize protocol to dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'category': self.category.value,
            'description': self.description,
            'theoretical_basis': self.theoretical_basis,
            'mechanism_of_action': self.mechanism_of_action,
            'target_states': self.target_states,
            'target_transitions': self.target_transitions,
            'effect_on_transitions': {
                f"{k[0]}->{k[1]}": v for k, v in self.effect_on_transitions.items()
            },
            'evidence_level': self.evidence_level.value,
            'effectiveness_estimate': self.effectiveness_estimate,
            'effect_size': self.effect_size,
            'key_studies': self.key_studies,
            'duration_weeks': self.duration_weeks,
            'cost_per_week': self.cost_per_week,
            'key_components': self.key_components
        }


# =============================================================================
# THERAPEUTIC INTERVENTIONS
# =============================================================================

CBT_FANTASY_MANAGEMENT = InterventionProtocol(
    name="cbt_fantasy_management",
    display_name="CBT for Fantasy Management",
    category=ProtocolCategory.THERAPEUTIC,
    description="""
    Cognitive-Behavioral Therapy targeting deviant fantasy content and the
    fantasy-to-action progression pathway. Based on the cognitive-behavioral
    model of offense chains where deviant fantasies serve as cognitive rehearsals
    that lower inhibitions and increase offense probability.

    This protocol interrupts the SEEKING state's role as an incubator for
    harmful behavior by targeting:
    (1) Fantasy content modification
    (2) Cognitive distortions supporting offense
    (3) The conditioned arousal-fantasy-action sequence
    """,
    theoretical_basis="""
    Grounded in Social Learning Theory (Bandura, 1977) and McGuire's
    Conditioning Theory of Deviant Sexual Interest (1965). Deviant fantasies
    function as cognitive rehearsals that:

    1. Strengthen stimulus-response associations through mental practice
    2. Reduce perceived barriers through cognitive distortion
    3. Create approach motivation via anticipated reward
    4. Desensitize to victim harm through repeated mental simulation

    The SEEKING state represents this internal cognitive workspace. CBT
    disrupts the fantasy->action pathway by modifying cognitive content
    and breaking the conditioned arousal sequence.

    Supports Ward's Pathways Model (2002): addresses intimacy deficit and
    emotional dysregulation pathways through cognitive restructuring.
    """,
    mechanism_of_action="""
    Works through multiple cognitive-behavioral mechanisms:

    1. COGNITIVE RESTRUCTURING: Challenges implicit theories (e.g., "children
       are sexual," "entitlement") that support offense. Targets the
       cognitive distortions that emerge during SEEKING and lower inhibitions.

    2. THOUGHT STOPPING: Interrupts deviant fantasy sequences before
       escalation. Creates aversive association with fantasy initiation.

    3. AROUSAL RECONDITIONING: Pairs deviant imagery with aversive outcomes
       (covert sensitization) or redirects arousal to appropriate stimuli
       (masturbatory reconditioning). Weakens the arousal->fantasy->action chain.

    4. RELAPSE PREVENTION: Identifies high-risk situations, seemingly
       unimportant decisions (SUDs), and lapse->relapse progressions.
       Maps to the SEEKING->DIRECTING transition pathway.

    Net effect: Reduces probability of SEEKING->DIRECTING transition by
    disrupting cognitive processes that enable action initiation.
    """,
    state_specific_rationale={
        'Seeking': """In SEEKING state, individual is cognitively engaged in
            fantasy, planning, or rumination. CBT targets the content of this
            cognitive activity by:
            - Identifying and challenging cognitive distortions
            - Providing alternative thought patterns
            - Breaking conditioned fantasy sequences
            - Building awareness of offense-supportive thinking""",
        'Directing': """Limited direct effect on active DIRECTING state as
            cognitive interventions require reflective capacity. However,
            skills learned in therapy can be accessed during pauses in
            behavioral sequence to enable disengagement.""",
        'Revising': """CBT techniques help process the REVISING state
            productively by reframing post-event cognitions away from
            offense-supportive interpretations toward accountability
            and harm recognition."""
    },
    target_states=['Seeking'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Seeking', 'Conferring')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.25,    # Primary target: fantasy->action
        ('Seeking', 'Conferring'): -0.10,   # Secondary: fantasy->surveillance
        ('Seeking', 'Revising'): +0.20,     # Redirect to reflective processing
        ('Seeking', 'Seeking'): +0.15       # Maintain in safer cognitive state
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.35,
    number_needed_to_treat=4.0,
    effect_size=0.44,
    key_studies=[
        "Hanson et al. (2002): SOTEP meta-analysis, d=0.44 for CBT programs",
        "Lösel & Schmucker (2005): Meta-analysis, OR=1.45 for cognitive-behavioral",
        "Marshall & Marshall (2007): Review of CBT components effectiveness",
        "Beech & Fisher (2002): Component analysis of treatment programs"
    ],
    duration_weeks=16,
    session_frequency="twice weekly",
    cost_per_week=300.0,
    required_resources=[
        "Licensed CBT therapist with forensic specialization",
        "Validated assessment instruments (STABLE-2007, VRS-SO)",
        "Manualized treatment protocol",
        "Arousal assessment capability (optional but recommended)"
    ],
    required_training="CBT certification with forensic/sexual offending specialization",
    key_components=[
        "Fantasy monitoring diary with trigger identification",
        "Cognitive restructuring of offense-supportive beliefs",
        "Covert sensitization for deviant arousal",
        "Seemingly unimportant decisions (SUD) chain analysis",
        "Relapse prevention planning with high-risk situation mapping"
    ],
    contraindications=[
        "Active psychosis (cannot engage in cognitive work)",
        "Severe intellectual disability (IQ<70, requires adaptation)",
        "Complete denial (pre-contemplation stage - motivational work first)",
        "Acute intoxication"
    ],
    population_notes="""
    Most effective for individuals with prominent fantasy/planning component
    (high SEEKING state proportion). Less effective for purely impulsive
    offenders (rapid SEEKING->DIRECTING without elaboration). Consider
    DBT for impulse-dominant presentations.
    """
)

DBT_IMPULSE_CONTROL = InterventionProtocol(
    name="dbt_impulse_control",
    display_name="DBT for Impulse Control",
    category=ProtocolCategory.THERAPEUTIC,
    description="""
    Dialectical Behavior Therapy adapted for impulse control in forensic
    populations. Targets the emotional dysregulation and distress intolerance
    that drive rapid state transitions to DIRECTING without adequate
    cognitive mediation.

    Particularly effective for individuals showing rapid escalation patterns
    (short SEEKING->DIRECTING latency) driven by affective dysregulation
    rather than elaborated fantasy.
    """,
    theoretical_basis="""
    Based on Linehan's Biosocial Theory (1993) of emotional dysregulation,
    adapted for forensic populations per Trupin et al. (2002). The model
    proposes that impulsive harmful behavior results from:

    1. Biological vulnerability to emotional intensity
    2. Invalidating environments that fail to teach emotion regulation
    3. Resulting emotion regulation deficits and distress intolerance
    4. Impulsive action as maladaptive emotion regulation

    In the 4-Animal framework, this manifests as:
    - Rapid SEEKING->DIRECTING transitions triggered by emotional states
    - Sustained DIRECTING as escape from intolerable affect
    - Minimal REVISING engagement (avoidance of difficult emotions)

    DBT addresses the affect-to-action pathway that bypasses cognitive
    mediation, complementing CBT which targets cognitive content.

    Supports Andrews & Bonta's RNR: addresses temperamental risk factors
    (emotional instability, impulsivity) identified in LSI-R/LS-CMI.
    """,
    mechanism_of_action="""
    Four-module intervention targeting emotional-behavioral regulation:

    1. MINDFULNESS: Develops observing stance toward emotional experiences.
       In DIRECTING state, creates microseconds of awareness between impulse
       and action ("urge surfing"). Enables choice point recognition.

    2. DISTRESS TOLERANCE: Provides crisis survival skills for intense
       emotional states without resorting to harmful behavior. Key skills:
       - TIPP (Temperature, Intense exercise, Paced breathing, Paired relaxation)
       - Distraction and self-soothing techniques
       - Radical acceptance of distressing situations

    3. EMOTION REGULATION: Reduces emotional vulnerability and increases
       positive experiences. Long-term reduction in emotional intensity that
       drives impulsive acting.

    4. INTERPERSONAL EFFECTIVENESS: Addresses relationship conflicts that
       trigger emotional dysregulation. Many SEEKING fantasies originate in
       perceived interpersonal injuries.

    Net effect: Extends latency between emotional trigger and behavioral
    response, reducing probability of affect-driven DIRECTING episodes.
    """,
    state_specific_rationale={
        'Directing': """Primary target state. DBT skills interrupt the
            affect->action sequence that maintains DIRECTING. Mindfulness
            creates observing capacity; distress tolerance provides
            alternatives to behavioral acting out.""",
        'Seeking': """For affect-driven SEEKING (rumination, grievance
            cultivation), mindfulness reduces cognitive fusion with
            distressing thoughts. Emotion regulation skills address
            underlying affect driving fantasy content.""",
        'Revising': """REVISING can trigger shame spirals leading back to
            DIRECTING. DBT's radical acceptance and emotion regulation
            prevent REVISING from becoming emotionally intolerable."""
    },
    target_states=['Directing', 'Seeking'],
    target_transitions=[
        ('Directing', 'Directing'),
        ('Seeking', 'Directing'),
        ('Revising', 'Directing')
    ],
    effect_on_transitions={
        ('Directing', 'Directing'): -0.30,  # Break sustained exploitation
        ('Seeking', 'Directing'): -0.20,    # Reduce impulsive acting out
        ('Revising', 'Directing'): -0.15,   # Prevent shame->action spiral
        ('Directing', 'Revising'): +0.20,   # Enable processing after action
        ('Directing', 'Seeking'): +0.10     # Return to cognitive state
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.40,
    number_needed_to_treat=3.5,
    effect_size=0.65,
    key_studies=[
        "Linehan et al. (2006): DBT for substance use, d=0.58",
        "Trupin et al. (2002): DBT in juvenile corrections, significant reduction",
        "Berzins & Trestman (2004): DBT in corrections feasibility",
        "Shelton et al. (2009): DBT for aggression in forensic settings"
    ],
    duration_weeks=24,
    session_frequency="weekly group + individual",
    cost_per_week=400.0,
    required_resources=[
        "DBT-trained therapist (Linehan certification preferred)",
        "Group therapy setting (8-12 participants)",
        "Skills training materials and diary cards",
        "24/7 crisis consultation access",
        "DBT consultation team"
    ],
    required_training="Full DBT Intensive Training (Behavioral Tech)",
    key_components=[
        "Mindfulness practice (wise mind, observe, describe)",
        "Distress tolerance crisis survival skills",
        "Emotion regulation ABC PLEASE skills",
        "Interpersonal effectiveness DEAR MAN, GIVE, FAST",
        "Behavioral chain analysis of harmful episodes"
    ],
    contraindications=[
        "Active substance intoxication during sessions",
        "Severe cognitive impairment preventing skill acquisition",
        "Complete unwillingness to participate in group format"
    ],
    population_notes="""
    Most effective for emotionally dysregulated offenders showing impulsive
    patterns (rapid SEEKING->DIRECTING, minimal planning). Less suited for
    highly planned, predatory offenses (high CONFERRING). Consider for
    'opportunist' or 'impulsive' subtypes.
    """
)

MENTALIZATION_THERAPY = InterventionProtocol(
    name="mentalization_therapy",
    display_name="Mentalization-Based Treatment",
    category=ProtocolCategory.THERAPEUTIC,
    description="""
    Mentalization-Based Treatment (MBT) targeting deficits in understanding
    mental states of self and others. Addresses the empathy deficits, theory
    of mind impairments, and perspective-taking failures common in antisocial
    populations.

    Particularly relevant for CONFERRING state where target is objectified,
    and DIRECTING state where victim perspective is absent.
    """,
    theoretical_basis="""
    Based on Fonagy's mentalization theory (2002) and attachment research.
    Mentalization is the capacity to understand behavior in terms of
    underlying mental states (beliefs, feelings, intentions).

    Offending behavior often occurs when mentalization fails:

    1. TARGET OBJECTIFICATION: In CONFERRING state, targets are viewed as
       objects rather than subjects with mental states. Surveillance focus
       on external characteristics, not internal experience.

    2. EMPATHY FAILURE: In DIRECTING state, victim's fear, pain, distress
       is not represented in perpetrator's mental model. Allows harm without
       experiencing emotional brake of empathy.

    3. SELF-MENTALIZATION DEFICIT: Poor understanding of own mental states
       leads to action rather than reflection. SEEKING becomes behavioral
       rather than thoughtfully processed.

    MBT develops robust mentalizing capacity that makes objectification
    and empathy failure less likely, inserting cognitive-empathic brake
    into offense sequence.

    Supports Ward's Pathways Model: addresses intimacy deficits pathway
    through improved capacity for relationship understanding.
    """,
    mechanism_of_action="""
    Works by strengthening mentalizing capacity across four dimensions:

    1. SELF-MENTALIZATION: Understanding own mental states, emotions,
       motivations. Transforms SEEKING from unreflected fantasy to
       understood experience that can be evaluated and modified.

    2. OTHER-MENTALIZATION: Representing others' mental states, including
       victim perspective. Makes target in CONFERRING a subject with
       feelings rather than an object with characteristics.

    3. IMPLICIT TO EXPLICIT: Moving automatic reactions to conscious
       reflection. Creates space between stimulus and response where
       mentalization can occur.

    4. AFFECT-COGNITION INTEGRATION: Linking emotional states to thoughts
       and behaviors. Enables understanding why one is drawn to harmful
       actions rather than simply acting.

    Net effect: Inserts empathic/reflective capacity into CONFERRING->DIRECTING
    transition where target becomes fully human in perpetrator's mind.
    """,
    state_specific_rationale={
        'Conferring': """Target of MBT intervention. In CONFERRING, individual
            surveys potential victims but fails to mentalize their experience.
            MBT builds capacity to represent target's mental state, making
            transition to DIRECTING less likely as victim becomes subject.""",
        'Directing': """During active DIRECTING, mentalization typically
            collapses. Pre-established mentalizing capacity may create
            moments of victim perspective that enable disengagement. Post-hoc
            processing of DIRECTING events uses mentalization skills.""",
        'Seeking': """Improves self-mentalization during SEEKING, enabling
            reflection on fantasy content and its origins rather than
            unreflective immersion in offensive cognitions."""
    },
    target_states=['Conferring', 'Directing'],
    target_transitions=[
        ('Conferring', 'Directing'),
        ('Directing', 'Directing')
    ],
    effect_on_transitions={
        ('Conferring', 'Directing'): -0.20,  # Can't harm mentalized other
        ('Directing', 'Directing'): -0.15,   # Empathy interrupts sustained harm
        ('Conferring', 'Seeking'): +0.15,    # Redirect to self-reflection
        ('Directing', 'Revising'): +0.10     # Enable processing
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.30,
    number_needed_to_treat=5.0,
    effect_size=0.50,
    key_studies=[
        "Bateman & Fonagy (2008): RCT showing sustained effects",
        "McGauley et al. (2011): MBT for antisocial personality",
        "Taubner et al. (2013): Mentalizing deficits in violent offenders",
        "Newbury-Helps et al. (2017): MBT in forensic settings review"
    ],
    duration_weeks=18,
    session_frequency="twice weekly",
    cost_per_week=350.0,
    required_resources=[
        "MBT-trained therapist (Anna Freud Centre certification)",
        "Group therapy setting (preferred for mentalizing practice)",
        "Supervision structure with focus on therapeutic stance"
    ],
    required_training="MBT certification with forensic adaptation training",
    key_components=[
        "Mentalizing stance (curiosity, not-knowing)",
        "Affect focus and elaboration",
        "Working with transference for mentalizing practice",
        "Group mentalizing exercises",
        "Victim perspective work (advanced stage)"
    ],
    contraindications=[
        "Active psychosis disrupting reality testing",
        "Severe antisocial traits with no capacity for alliance",
        "Complete denial with no acknowledgment of harm"
    ],
    population_notes="""
    Most effective for offenders with identifiable victims where empathy
    deficit is prominent. Less suited for impulsive offenders or those
    with organic empathy deficits. Best for 'predator' subtypes with
    high CONFERRING proportions who objectify targets.
    """
)

TRAUMA_THERAPY = InterventionProtocol(
    name="trauma_therapy",
    display_name="Trauma-Focused Therapy",
    category=ProtocolCategory.THERAPEUTIC,
    description="""
    Trauma processing therapy using EMDR or Prolonged Exposure for individuals
    with significant trauma histories. Targets unresolved trauma that fuels:

    - Revenge fantasies in SEEKING state
    - Identification with aggressor patterns
    - Compulsive reenactment cycles in REVISING
    - Hypervigilance driving CONFERRING behavior
    """,
    theoretical_basis="""
    Based on Herman's Trauma Theory (1992), van der Kolk's Body-Trauma model
    (2014), and research on trauma-crime links (Widom & Maxfield, 2001).

    Many offenders have significant trauma histories that contribute to
    offending through several pathways:

    1. IDENTIFICATION WITH AGGRESSOR: Traumatized individuals may adopt
       perpetrator role as psychological defense. SEEKING content reflects
       internalized perpetrator identity.

    2. REVENGE/RETRIBUTION FANTASY: Unprocessed trauma generates vengeful
       fantasies that fuel SEEKING state content and create approach
       motivation toward DIRECTING.

    3. REENACTMENT COMPULSION: Repetition compulsion drives return to
       trauma-related scenarios. REVISING state becomes stuck in traumatic
       material, cycling back to DIRECTING.

    4. HYPERAROUSAL: Trauma-induced autonomic dysregulation lowers threshold
       for fight response, facilitating transition to DIRECTING.

    Trauma processing resolves underlying material that generates offense-
    related cognitive-emotional content.

    Supports Pathways Model: addresses antisocial cognitions arising from
    traumatic attachment disruptions.
    """,
    mechanism_of_action="""
    Two validated approaches with similar mechanisms:

    EMDR (Shapiro, 2001):
    - Bilateral stimulation during trauma recall
    - Facilitates memory reconsolidation
    - Reduces emotional charge of traumatic memories
    - Allows cognitive reprocessing and integration

    PROLONGED EXPOSURE (Foa & Rothbaum, 1998):
    - Systematic confrontation with trauma memories
    - Habituation reduces emotional response
    - Cognitive processing of meaning
    - Integration into autobiographical narrative

    Both approaches:
    1. Process unresolved trauma fueling offense-related content
    2. Reduce emotional intensity driving dysregulated behavior
    3. Enable cognitive reappraisal of trauma narrative
    4. Break compulsive reenactment cycles

    Net effect: Reduces trauma-driven SEEKING content and REVISING->DIRECTING
    reenactment patterns by processing underlying traumatic material.
    """,
    state_specific_rationale={
        'Seeking': """Trauma-driven SEEKING manifests as revenge fantasies,
            identification with aggressor scenarios, or displaced aggression
            targets. Processing original trauma reduces need for fantasy
            revenge and corrects identification distortions.""",
        'Revising': """REVISING state can become dominated by traumatic
            intrusions and reenactment urges. Trauma processing enables
            healthy integration rather than compulsive return to harmful
            behavioral patterns.""",
        'Conferring': """Hypervigilance from trauma can drive excessive
            CONFERRING (scanning for threats/targets). Processing trauma
            reduces autonomic hyperarousal that maintains surveillance mode."""
    },
    target_states=['Seeking', 'Revising'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Revising', 'Directing'),
        ('Revising', 'Revising')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.15,    # Trauma-driven action reduced
        ('Revising', 'Directing'): -0.20,   # Compulsive reenactment blocked
        ('Revising', 'Revising'): -0.10,    # Break rumination cycles
        ('Seeking', 'Revising'): +0.10,     # Healthy processing pathway
        ('Revising', 'Seeking'): +0.15      # Return to safer exploration
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.35,
    number_needed_to_treat=4.0,
    effect_size=0.55,
    key_studies=[
        "Chen et al. (2014): EMDR for PTSD meta-analysis, d=1.01",
        "Powers et al. (2010): Prolonged exposure meta-analysis, d=1.08",
        "Cusack et al. (2016): Trauma treatments VA/DoD CPG review",
        "Reavis et al. (2013): Adverse childhood experiences in prisoners"
    ],
    duration_weeks=20,
    session_frequency="weekly (with stabilization phase)",
    cost_per_week=250.0,
    required_resources=[
        "Trauma-certified therapist (EMDR International or PE trained)",
        "Safe, private therapeutic environment",
        "Crisis support availability",
        "Comprehensive trauma assessment (CTQ, PCL-5)"
    ],
    required_training="EMDRIA certification or Prolonged Exposure certification",
    key_components=[
        "Comprehensive trauma history assessment",
        "Stabilization phase with coping skills",
        "Trauma processing (EMDR protocol or PE imaginal exposure)",
        "Cognitive restructuring of trauma-related beliefs",
        "Integration and consolidation",
        "Relapse prevention connecting trauma to offense patterns"
    ],
    contraindications=[
        "Active suicidality (stabilize first)",
        "Severe dissociation (requires phased approach)",
        "Unstable living situation (need safety base)",
        "Active substance dependence (treat first or concurrently)",
        "No trauma history (not indicated)"
    ],
    population_notes="""
    Indicated when significant trauma history identified (ACE score ≥4,
    documented abuse/neglect). Most effective when clear trauma->offense
    pathway identified. May not be appropriate as sole intervention -
    combine with offense-specific work.
    """
)

SCHEMA_THERAPY = InterventionProtocol(
    name="schema_therapy",
    display_name="Schema Therapy for Forensic Populations",
    category=ProtocolCategory.THERAPEUTIC,
    description="""
    Schema Therapy addressing early maladaptive schemas that underlie
    offense patterns. Targets the cognitive-affective structures formed
    in childhood that organize perception, emotion, and behavior in ways
    that facilitate offending.

    Particularly effective for characterological issues driving offense
    patterns across all four behavioral states.
    """,
    theoretical_basis="""
    Based on Young's Schema Theory (1990, 2003) integrating cognitive,
    attachment, and experiential approaches. Early Maladaptive Schemas
    (EMS) are self-defeating patterns that:

    1. Develop in childhood through unmet emotional needs
    2. Create cognitive filters distorting perception
    3. Generate coping styles (surrender, avoidance, overcompensation)
    4. Drive behavior in schema-consistent directions

    Schemas relevant to offending include:
    - MISTRUST/ABUSE: Expects harm, justifies preemptive aggression
    - ENTITLEMENT: Believes rules don't apply, victims exist to serve
    - DEFECTIVENESS/SHAME: Compensated through domination/power
    - EMOTIONAL DEPRIVATION: Seeks connection through coercive means
    - SUBJUGATION: Overcompensated through control and dominance

    These schemas shape SEEKING content (entitlement-based fantasy),
    CONFERRING patterns (mistrust-based surveillance), and DIRECTING
    behavior (overcompensation through action).
    """,
    mechanism_of_action="""
    Three-phase intervention:

    1. ASSESSMENT & EDUCATION:
       - Identify core schemas using YSQ-S3 and clinical interview
       - Map schemas to offense patterns and behavioral states
       - Develop schema awareness in daily functioning

    2. COGNITIVE-EXPERIENTIAL WORK:
       - Challenge schema-driven cognitive distortions
       - Experiential techniques (imagery rescripting, chair work)
       - Limited reparenting through therapeutic relationship
       - Mode work identifying Healthy Adult, Vulnerable Child,
         and maladaptive coping modes

    3. BEHAVIORAL PATTERN BREAKING:
       - Identify schema-triggered behaviors across states
       - Develop Healthy Adult responses to schema activation
       - Practice alternative responses to triggering situations

    Net effect: Modifies underlying cognitive-affective structures that
    generate offense-supportive content across all behavioral states.
    """,
    state_specific_rationale={
        'Seeking': """Schema-driven SEEKING reflects themes like entitlement
            fantasies (Entitlement schema), revenge scenarios (Mistrust
            schema), or power/domination themes (Defectiveness compensated).
            Schema work addresses these root patterns.""",
        'Conferring': """Mistrust/Abuse schema drives hypervigilant scanning
            in CONFERRING. Emotional Deprivation schema may drive surveillance
            for connection opportunities. Addressing schemas reduces maladaptive
            surveillance patterns.""",
        'Directing': """Active DIRECTING often represents schema coping mode
            (overcompensation through dominance, bully-attack mode). Schema
            therapy develops Healthy Adult mode that can regulate these impulses.""",
        'Revising': """Defectiveness/Shame schema may generate shame spirals
            in REVISING that paradoxically drive return to DIRECTING. Schema
            work develops self-compassion and Healthy Adult processing."""
    },
    target_states=['Seeking', 'Conferring', 'Directing', 'Revising'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Conferring', 'Directing'),
        ('Revising', 'Directing')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.20,    # Schema content modified
        ('Conferring', 'Directing'): -0.15, # Mistrust patterns addressed
        ('Revising', 'Directing'): -0.15,   # Shame spiral blocked
        ('Seeking', 'Revising'): +0.15,     # Healthy Adult processing
        ('Directing', 'Revising'): +0.10    # Enable reflection
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.35,
    number_needed_to_treat=4.0,
    effect_size=0.52,
    key_studies=[
        "Bernstein et al. (2012): Schema therapy for forensic patients RCT",
        "Keulen-de Vos et al. (2016): Schema modes in forensic patients",
        "Lobbestael et al. (2005): Schema modes and cluster B personality",
        "van Vreeswijk et al. (2012): Group schema therapy efficacy"
    ],
    duration_weeks=52,
    session_frequency="weekly individual + optional group",
    cost_per_week=300.0,
    required_resources=[
        "Schema therapy certified therapist",
        "YSQ-S3 and SMI assessment instruments",
        "Long-term treatment setting (1-3 years optimal)",
        "Supervision with schema therapy focus"
    ],
    required_training="ISST Schema Therapy certification",
    key_components=[
        "Schema assessment and case conceptualization",
        "Cognitive restructuring of schema beliefs",
        "Experiential techniques (imagery, chair work)",
        "Limited reparenting in therapeutic relationship",
        "Mode mapping and Healthy Adult development",
        "Behavioral pattern breaking exercises"
    ],
    contraindications=[
        "Active psychosis",
        "Severe antisocial personality without any vulnerable side access",
        "Unable to form therapeutic alliance"
    ],
    population_notes="""
    Best for offenders with clear personality pathology and early
    developmental adversity. Requires capacity for therapeutic relationship.
    Long-term intervention - not suited for brief treatment settings.
    Consider for 'complex' archetype with characterological issues.
    """
)


# =============================================================================
# SUPERVISION INTERVENTIONS
# =============================================================================

INTENSIVE_SUPERVISION = InterventionProtocol(
    name="intensive_supervision",
    display_name="Intensive Community Supervision",
    category=ProtocolCategory.SUPERVISION,
    description="""
    High-intensity supervision with GPS monitoring, frequent contacts, and
    structured accountability. Creates external structure that disrupts
    opportunity for surveillance (CONFERRING) and action (DIRECTING).

    Based on Rational Choice Theory - increases perceived costs and reduces
    perceived opportunities, shifting cost-benefit analysis away from offending.
    """,
    theoretical_basis="""
    Grounded in multiple theoretical frameworks:

    1. RATIONAL CHOICE THEORY (Cornish & Clarke, 1986): Offending involves
       cost-benefit calculation. Intensive supervision increases perceived
       costs (detection probability) and reduces opportunities.

    2. ROUTINE ACTIVITIES THEORY (Cohen & Felson, 1979): Crime requires
       convergence of motivated offender, suitable target, absence of
       capable guardian. Supervision disrupts this convergence.

    3. SITUATIONAL CRIME PREVENTION: Increases effort and risk, reduces
       rewards, removes excuses. GPS exclusion zones prevent target access.

    4. EXTERNAL MOTIVATION (RNR): For individuals with limited internal
       controls, external structure maintains behavior boundaries while
       internal motivation develops through treatment.

    In behavioral state terms:
    - Prevents CONFERRING->DIRECTING by removing target access
    - Disrupts sustained DIRECTING through monitoring/interruption
    - Creates accountability that influences SEEKING content
    """,
    mechanism_of_action="""
    Multi-component supervision creating environmental constraints:

    1. GPS MONITORING:
       - Continuous location tracking
       - Exclusion zones around high-risk locations (schools, victim
         neighborhoods, former offense locations)
       - Real-time alerts for zone violations
       - Disrupts CONFERRING by preventing target environment access

    2. HIGH-FREQUENCY CONTACTS:
       - Daily check-ins (phone or in-person)
       - Random home/work visits
       - Unpredictable schedule creates constant perceived surveillance
       - Maintains awareness that DIRECTING will be detected

    3. CURFEW & MOVEMENT RESTRICTIONS:
       - Limits unstructured time when offending most likely
       - Reduces opportunity windows
       - Creates accountability for all time periods

    4. COLLATERAL CONTACTS:
       - Verification with employers, family, treatment providers
       - Creates social surveillance network
       - Multiple points of detection for concerning behavior

    Net effect: Creates environmental structure that makes CONFERRING and
    DIRECTING states high-risk and low-opportunity, while allowing positive
    community engagement within boundaries.
    """,
    state_specific_rationale={
        'Conferring': """Direct intervention target. GPS monitoring and
            exclusion zones prevent physical access to potential target
            environments. Cannot engage in surveillance when location is
            tracked and restricted.""",
        'Directing': """Monitoring creates high detection probability during
            any DIRECTING attempt. Frequent contacts reduce time windows
            for sustained DIRECTING. Immediate response to violations.""",
        'Seeking': """Indirect effect through awareness of consequences.
            Knowing that CONFERRING and DIRECTING are highly constrained
            may reduce investment in SEEKING content that cannot be acted upon."""
    },
    target_states=['Conferring', 'Directing'],
    target_transitions=[
        ('Conferring', 'Directing'),
        ('Directing', 'Directing'),
        ('Seeking', 'Conferring')
    ],
    effect_on_transitions={
        ('Conferring', 'Directing'): -0.35,  # Can't access targets
        ('Directing', 'Directing'): -0.25,   # Rapid detection/response
        ('Seeking', 'Conferring'): -0.15,    # Can't scout locations
        ('Conferring', 'Revising'): +0.20,   # Blocked from action->process
        ('Directing', 'Revising'): +0.25     # Intervention forces stop
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.45,
    number_needed_to_treat=3.0,
    effect_size=0.70,
    key_studies=[
        "Aos et al. (2006): WSIPP intensive supervision meta-analysis",
        "Duwe (2013): Minnesota intensive sex offender supervision",
        "Petersilia & Turner (1993): Intensive supervision program evaluation",
        "Lowenkamp et al. (2010): Supervision intensity and recidivism"
    ],
    duration_weeks=52,
    session_frequency="daily contact minimum",
    cost_per_week=500.0,
    required_resources=[
        "GPS monitoring equipment and service",
        "Dedicated supervision officer (low caseload: 20-25)",
        "24/7 monitoring center with alert response",
        "Rapid response capability for violations",
        "Collaboration with law enforcement"
    ],
    required_training="Specialized sex offender supervision certification",
    key_components=[
        "GPS ankle monitoring with exclusion zones",
        "Minimum daily phone/in-person contacts",
        "Curfew enforcement (typically 9pm-6am)",
        "Random home and workplace visits",
        "Employment verification and monitoring",
        "Collateral contacts (family, employer, therapist)",
        "Computer monitoring (if history of internet offenses)",
        "Polygraph examinations (where legally permitted)"
    ],
    contraindicated_states=['Revising'],
    contraindications=[
        "Over-supervision of low-risk individuals (iatrogenic)",
        "Individuals actively engaged in treatment and stable (step down)"
    ],
    population_notes="""
    RNR principle: intensive supervision for high-risk individuals.
    Over-supervision of low-risk cases is counterproductive. Best for
    individuals with prominent CONFERRING patterns (stalker-striker subtype)
    and those with prior supervision failures. Combine with treatment.
    """
)

GEOGRAPHIC_RESTRICTION = InterventionProtocol(
    name="geographic_restriction",
    display_name="Geographic Movement Restriction",
    category=ProtocolCategory.SUPERVISION,
    description="""
    GPS-enforced exclusion from high-risk locations including schools,
    parks, playgrounds, victim neighborhoods, and prior offense locations.
    Creates physical barriers to target access during CONFERRING state.

    Specifically designed to disrupt the target selection and surveillance
    behaviors characteristic of the CONFERRING state.
    """,
    theoretical_basis="""
    Based on Environmental Criminology and Routine Activities Theory:

    1. CRIME PATTERN THEORY (Brantingham & Brantingham, 1991): Offenders
       commit crimes within awareness space near routine activity nodes.
       Restricting access to high-risk nodes disrupts offense opportunity.

    2. JOURNEY TO CRIME: Geographic profiling research shows offenders
       operate within characteristic ranges from home/work. Exclusion zones
       disrupt established patterns.

    3. TARGET HARDENING: Making targets physically inaccessible is most
       direct crime prevention. Cannot surveil/access targets in excluded areas.

    4. SITUATIONAL PREVENTION: Increases effort required for offending by
       eliminating convenient targets and requiring novel, higher-risk
       locations.

    Directly targets CONFERRING state by preventing physical presence in
    environments where target selection occurs.
    """,
    mechanism_of_action="""
    Geographic containment strategy:

    1. EXCLUSION ZONE MAPPING:
       - Schools, daycares, playgrounds (if child victims)
       - Parks, recreational areas with potential victims
       - Previous victim neighborhoods
       - Prior offense locations
       - Any specific high-risk locations identified in assessment

    2. GPS ENFORCEMENT:
       - Real-time tracking against zone boundaries
       - Immediate alerts when zone approached/entered
       - Graduated response protocol (warning -> report -> revocation)

    3. BEHAVIORAL IMPACT:
       - Cannot physically access target environments
       - CONFERRING activities (surveillance, target selection) blocked
       - Forces novel location selection with higher detection risk
       - May reduce SEEKING content related to inaccessible targets

    4. PSYCHOLOGICAL IMPACT:
       - Creates cognitive boundary around restricted areas
       - Awareness of restriction may generalize to reduced approach motivation
       - Defines clear behavioral expectations

    Net effect: Creates physical impossibility of CONFERRING in primary
    target environments, breaking CONFERRING->DIRECTING chain.
    """,
    state_specific_rationale={
        'Conferring': """Primary and direct intervention target. CONFERRING
            state involves reconnaissance of potential targets and their
            environments. Geographic restriction makes this impossible in
            primary risk areas by preventing physical presence.""",
        'Seeking': """Indirect effect: if target access is known to be
            impossible, SEEKING content involving those targets may decrease
            as fantasies become more obviously unattainable."""
    },
    target_states=['Conferring'],
    target_transitions=[
        ('Conferring', 'Directing'),
        ('Seeking', 'Conferring')
    ],
    effect_on_transitions={
        ('Conferring', 'Directing'): -0.30,  # No targets in restricted areas
        ('Seeking', 'Conferring'): -0.25,    # Can't scout restricted locations
        ('Conferring', 'Conferring'): -0.20, # Surveillance disrupted
        ('Conferring', 'Seeking'): +0.15     # Blocked from action->internal
    },
    evidence_level=EvidenceLevel.C,
    effectiveness_estimate=0.35,
    number_needed_to_treat=4.0,
    effect_size=0.55,
    key_studies=[
        "Levenson & Cotter (2005): Sex offender residence restrictions study",
        "Zandbergen & Hart (2006): Restricted zones and housing availability",
        "Nobles et al. (2012): Sex offender proximity research",
        "Socia (2012): Residence restriction effectiveness review"
    ],
    duration_weeks=104,
    session_frequency="continuous monitoring",
    cost_per_week=150.0,
    required_resources=[
        "GPS monitoring system",
        "GIS mapping of exclusion zones",
        "Alert monitoring system",
        "Law enforcement coordination for violations",
        "Legal framework supporting restrictions"
    ],
    key_components=[
        "Comprehensive zone mapping based on victim type",
        "Real-time GPS tracking with geofencing",
        "Alert system with immediate notification",
        "Graduated violation response protocol",
        "Periodic zone review and adjustment",
        "Alternative route planning support"
    ],
    population_notes="""
    Most effective for offenders with specific victim type (children requiring
    school/park restrictions) and geographic offense patterns. Less useful
    for opportunistic offenders without location-specific targeting. Consider
    in combination with intensive supervision for high-risk cases.
    """
)

STRUCTURED_DAILY_SCHEDULE = InterventionProtocol(
    name="structured_schedule",
    display_name="Structured Daily Schedule",
    category=ProtocolCategory.SUPERVISION,
    description="""
    Comprehensive daily schedule with prosocial activities occupying all
    unstructured time. Based on the principle that opportunity requires
    unmonitored time, and that prosocial engagement competes with
    antisocial behavior.

    Targets the 'unstructured time' vulnerability that enables SEEKING
    rumination and CONFERRING surveillance.
    """,
    theoretical_basis="""
    Grounded in multiple behavioral theories:

    1. SOCIAL CONTROL THEORY (Hirschi, 1969): Involvement in conventional
       activities (work, family, recreation) reduces time and motivation
       for deviant behavior. "Idle hands" principle.

    2. SELF-CONTROL THEORY: External structure compensates for deficits in
       internal self-regulation. Schedule provides environmental scaffolding
       for behavior control.

    3. OPERANT CONDITIONING: Prosocial schedule provides reinforcement for
       conventional behavior, competing with and potentially replacing
       offense-related reinforcement.

    4. ROUTINE ACTIVITIES: Removes temporal opportunities by filling time
       periods when offending would occur. Most offenses happen during
       unstructured evening/weekend hours.

    5. BEHAVIORAL ACTIVATION: Engagement in scheduled activities provides
       alternative sources of reward and meaning, reducing reliance on
       offense-related gratification.
    """,
    mechanism_of_action="""
    Time-structured intervention:

    1. OCCUPATION OF RISK PERIODS:
       - Structured activities during high-risk times (evenings, weekends)
       - Verified accountability for all time periods
       - Reduces unmonitored time when SEEKING/CONFERRING occur

    2. PROSOCIAL ENGAGEMENT:
       - Employment or vocational training (8+ hours daily)
       - Treatment appointments (scheduled, verified)
       - Prosocial leisure (sports leagues, volunteer work)
       - Family/relationship time (positive contacts)

    3. ACCOUNTABILITY SYSTEM:
       - Daily schedule submitted in advance
       - Activities verified by employers, family, providers
       - Unexplained time gaps investigated
       - Creates comprehensive monitoring without GPS

    4. COMPETING REINFORCEMENT:
       - Prosocial activities provide legitimate rewards
       - Success in work/relationships builds stake in conformity
       - Alternative identity development (worker, family member vs. offender)

    Net effect: Reduces time available for offense-related activities while
    building prosocial connections that motivate desistance.
    """,
    state_specific_rationale={
        'Seeking': """SEEKING state requires unstructured time for fantasy
            development and planning. Fully occupied schedule eliminates
            the empty time periods when rumination and fantasy escalation
            occur. Engaged mind has less space for deviant content.""",
        'Conferring': """Surveillance and target selection require discretionary
            time and mobility. With all time accounted for, CONFERRING
            activities become impossible without detection.""",
        'Directing': """Scheduled activities and verification points make it
            impossible to disappear for time required for DIRECTING without
            investigation."""
    },
    target_states=['Seeking', 'Conferring'],
    target_transitions=[
        ('Seeking', 'Conferring'),
        ('Seeking', 'Directing'),
        ('Conferring', 'Directing')
    ],
    effect_on_transitions={
        ('Seeking', 'Conferring'): -0.20,   # No time for scouting
        ('Seeking', 'Directing'): -0.15,    # No time for action
        ('Conferring', 'Directing'): -0.20, # No time for action
        ('Seeking', 'Seeking'): -0.15,      # Less rumination time
        ('Seeking', 'Revising'): +0.20      # Structured processing time
    },
    evidence_level=EvidenceLevel.C,
    effectiveness_estimate=0.30,
    number_needed_to_treat=5.0,
    effect_size=0.45,
    key_studies=[
        "Gendreau et al. (2006): Structured programming in corrections",
        "Wilson et al. (2000): Education and employment programs review",
        "MacKenzie (2006): Correctional rehabilitation meta-analysis",
        "Bushway & Reuter (2002): Employment and crime prevention"
    ],
    duration_weeks=26,
    session_frequency="daily schedule review",
    cost_per_week=100.0,
    required_resources=[
        "Case manager for schedule development/monitoring",
        "Access to employment or vocational programs",
        "Prosocial activity resources",
        "Schedule tracking and verification system"
    ],
    key_components=[
        "Comprehensive weekly schedule development",
        "Employment or vocational placement",
        "Scheduled treatment appointments",
        "Prosocial leisure activities",
        "Family/relationship time",
        "Daily activity verification",
        "Unexplained time gap investigation"
    ],
    population_notes="""
    Effective across risk levels when combined with other interventions.
    Particularly important for individuals with limited internal structure
    and those whose offending occurred during discretionary time. Less
    relevant for offenders who offended in work/role contexts.
    """
)


# =============================================================================
# ENVIRONMENTAL INTERVENTIONS
# =============================================================================

STABLE_HOUSING = InterventionProtocol(
    name="stable_housing",
    display_name="Stable Housing Program",
    category=ProtocolCategory.ENVIRONMENTAL,
    description="""
    Provision of stable, supervised housing addressing the environmental
    instability that contributes to behavioral escalation. Housing instability
    is a key criminogenic need associated with recidivism.

    Removes environmental stressors that trigger regression from stable
    states (REVISING, SEEKING) into active DIRECTING.
    """,
    theoretical_basis="""
    Based on General Strain Theory and Social Support Theory:

    1. GENERAL STRAIN THEORY (Agnew, 1992): Crime results from strain
       including negative life events and failure to achieve goals. Housing
       instability is major stressor that increases strain->crime pathway.

    2. SOCIAL SUPPORT THEORY: Social bonds and support reduce crime through
       informal social control and practical resource provision. Stable
       housing provides base for support network development.

    3. DESISTANCE THEORY (Maruna, 2001): Desistance requires "hooks for
       change" - stable housing provides foundation for identity transformation
       and conventional life narrative.

    4. STRESS-VULNERABILITY MODEL: Environmental stressors interact with
       individual vulnerabilities to trigger episodes. Stable housing
       reduces environmental stress load.

    Housing instability specifically relates to behavioral states:
    - Stress drives regression from REVISING to SEEKING (rumination)
    - Extreme stress can trigger direct REVISING->DIRECTING (reactive)
    - Instability prevents consolidation of treatment gains
    """,
    mechanism_of_action="""
    Multi-level housing intervention:

    1. PHYSICAL STABILITY:
       - Consistent residence eliminates housing search stress
       - Safe environment reduces hypervigilance
       - Basic needs met (reduces desperation)

    2. STRUCTURED ENVIRONMENT:
       - House rules provide external behavioral framework
       - Staff presence provides informal monitoring
       - Peer accountability among residents

    3. SERVICE PLATFORM:
       - Stable address enables treatment participation
       - Location for probation contacts
       - Employment more accessible with stable address
       - Foundation for building routine

    4. SOCIAL CONNECTION:
       - Peer support from other residents
       - Staff relationships provide prosocial contacts
       - Base for family reconnection
       - Reduces isolation that fuels SEEKING content

    Net effect: Removes destabilizing environmental factors and provides
    stable foundation for treatment and reintegration.
    """,
    state_specific_rationale={
        'Revising': """Housing stability supports productive REVISING by
            reducing environmental chaos that disrupts reflection. Safe
            environment allows processing without survival-mode thinking.""",
        'Seeking': """Stable housing reduces stress-driven SEEKING (rumination,
            grievance, escape fantasy). When basic needs are met, SEEKING
            can engage with treatment content rather than survival concerns.""",
        'Directing': """Environmental stability removes desperation-driven
            triggers for reactive DIRECTING. Less "nothing to lose" mentality
            when housing is stable."""
    },
    target_states=['Revising', 'Seeking'],
    target_transitions=[
        ('Revising', 'Directing'),
        ('Seeking', 'Directing')
    ],
    effect_on_transitions={
        ('Revising', 'Directing'): -0.20,   # Stress-triggered action reduced
        ('Seeking', 'Directing'): -0.15,    # Desperation action reduced
        ('Revising', 'Revising'): +0.15,    # Stable processing maintained
        ('Seeking', 'Seeking'): +0.10       # Safer exploration
    },
    evidence_level=EvidenceLevel.C,
    effectiveness_estimate=0.25,
    number_needed_to_treat=6.0,
    effect_size=0.40,
    key_studies=[
        "Lutze et al. (2014): Housing and reentry program outcomes",
        "Steiner et al. (2015): Housing instability and recidivism",
        "Roman & Travis (2006): Housing and prisoner reentry",
        "Clark (2016): Residence restrictions and homelessness"
    ],
    duration_weeks=52,
    session_frequency="ongoing residence",
    cost_per_week=250.0,
    required_resources=[
        "Supervised housing facility or slots",
        "House manager/support staff",
        "Basic needs provision (food, utilities)",
        "Connection to community services"
    ],
    key_components=[
        "Stable housing placement",
        "Clear house rules and expectations",
        "Staff support and informal monitoring",
        "Connection to treatment services",
        "Peer support programming",
        "Gradual independence planning"
    ],
    population_notes="""
    Critical for individuals with housing instability history. Address housing
    early in reentry - foundation for other interventions. Less relevant for
    individuals with stable family housing. Consider specialized settings for
    sex offenders given residence restriction laws.
    """
)

EMPLOYMENT_SUPPORT = InterventionProtocol(
    name="employment_support",
    display_name="Supported Employment Program",
    category=ProtocolCategory.ENVIRONMENTAL,
    description="""
    Vocational support and job placement with ongoing coaching. Employment
    provides prosocial identity, structured time, financial stability, and
    stake in conformity - all protective factors against reoffending.

    Addresses both practical (income, time structure) and psychological
    (identity, meaning) factors that influence behavioral states.
    """,
    theoretical_basis="""
    Based on Desistance Theory and Social Bonding Theory:

    1. DESISTANCE THEORY (Maruna, 2001; Sampson & Laub, 2003): Employment
       is key "turning point" in desistance trajectories. Provides:
       - "Hook for change" - new identity opportunity
       - Stake in conformity - something to lose
       - Routine and structure
       - Prosocial relationships

    2. SOCIAL BONDING (Hirschi, 1969): Employment creates bonds through:
       - Attachment to coworkers/supervisors
       - Commitment to career advancement
       - Involvement (time occupied)
       - Belief in conventional system

    3. STRAIN THEORY (Agnew): Employment reduces financial strain that can
       trigger criminal coping.

    4. IDENTITY TRANSFORMATION: Employment provides alternative identity
       ("worker," "provider") competing with offender identity.

    In behavioral state terms:
    - Occupies time that would allow SEEKING rumination
    - Provides alternative reinforcement reducing SEEKING content appeal
    - Creates consequences for DIRECTING that affect employment
    - Supports productive REVISING through sense of accomplishment
    """,
    mechanism_of_action="""
    Comprehensive employment intervention:

    1. VOCATIONAL ASSESSMENT:
       - Skills inventory and interest assessment
       - Criminal record analysis for employment barriers
       - Realistic job target identification
       - Accommodation needs identification

    2. JOB DEVELOPMENT:
       - Employer relationship building
       - Job carving and customization
       - Criminal record disclosure coaching
       - Interview preparation

    3. PLACEMENT SUPPORT:
       - Direct job placement
       - First-day support
       - Employer liaison
       - Problem-solving as issues arise

    4. RETENTION COACHING:
       - Ongoing support post-placement
       - Workplace conflict management
       - Career development planning
       - Crisis intervention if job threatened

    5. IDENTITY DEVELOPMENT:
       - Building worker identity
       - Connecting work success to self-concept
       - Creating "stake in conformity"

    Net effect: Provides alternative reinforcement, time structure, social
    bonds, and identity transformation that compete with offense-related
    states and behaviors.
    """,
    state_specific_rationale={
        'Seeking': """Employment reduces time for and appeal of SEEKING state
            content. Alternative sources of meaning and stimulation compete
            with offense-related fantasy. Worker identity competes with
            offender identity.""",
        'Revising': """Employment success contributes to positive REVISING
            content - processing accomplishments rather than offenses.
            Builds self-efficacy and alternative narrative."""
    },
    target_states=['Seeking', 'Revising'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Seeking', 'Conferring'),
        ('Revising', 'Directing')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.15,    # Alternative identity/reinforcement
        ('Seeking', 'Conferring'): -0.10,   # Time structure reduces opportunity
        ('Revising', 'Directing'): -0.10,   # Stake in conformity
        ('Seeking', 'Revising'): +0.15,     # Productive processing
        ('Revising', 'Revising'): +0.10     # Maintain stability
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.25,
    number_needed_to_treat=6.0,
    effect_size=0.40,
    key_studies=[
        "Uggen (2000): Work as turning point in desistance",
        "Visher et al. (2005): Employment and prisoner reentry",
        "Wilson et al. (2000): Employment programs meta-analysis",
        "Sampson & Laub (1993): Work and crime desistance trajectories"
    ],
    duration_weeks=26,
    session_frequency="weekly coaching",
    cost_per_week=150.0,
    required_resources=[
        "Employment specialist (IPS model preferred)",
        "Employer network willing to hire",
        "Job development capacity",
        "Ongoing retention support"
    ],
    required_training="Individual Placement and Support (IPS) model training",
    key_components=[
        "Comprehensive vocational assessment",
        "Rapid job search (place-train model)",
        "Criminal record disclosure support",
        "Employer liaison and education",
        "Post-placement retention coaching",
        "Career development planning"
    ],
    population_notes="""
    Highly effective when employment is achievable - consider employment
    barriers (sex offender registration, record). Most effective when
    combined with cognitive interventions that support identity change.
    Use IPS model for best outcomes.
    """
)

SOCIAL_SUPPORT_NETWORK = InterventionProtocol(
    name="social_support",
    display_name="Prosocial Network Building",
    category=ProtocolCategory.ENVIRONMENTAL,
    description="""
    Development of prosocial support network through family reconnection,
    mentorship, and community integration. Social bonds serve as protective
    factors through informal social control, emotional support, and
    alternative social reinforcement.

    Directly addresses the social isolation that fuels SEEKING state
    fantasy development and removes social accountability that would
    inhibit CONFERRING and DIRECTING.
    """,
    theoretical_basis="""
    Based on Social Bond Theory and Desistance Research:

    1. SOCIAL BOND THEORY (Hirschi, 1969): Crime is prevented by bonds to
       conventional society through:
       - Attachment: emotional connection to others
       - Commitment: investment in conventional activities
       - Involvement: time spent in prosocial activities
       - Belief: acceptance of conventional values

    2. SOCIAL CAPITAL (Laub & Sampson, 2003): Prosocial relationships provide
       resources for desistance including:
       - Informal social control (others monitoring behavior)
       - Social support (emotional/practical)
       - Access to opportunities

    3. INTIMATE PARTNER DESISTANCE: Quality relationships are key predictor
       of desistance (Laub & Sampson, 2003). Partner provides motivation,
       monitoring, and meaning.

    4. CIRCLES OF SUPPORT AND ACCOUNTABILITY (COSA): Community volunteers
       provide support and monitoring that reduces risk. Combines social
       support with accountability.

    Isolation relates to offense through:
    - Removed social accountability that inhibits CONFERRING/DIRECTING
    - SEEKING content develops without reality check from others
    - Grievance accumulates without social feedback
    - No prosocial sources of connection/intimacy needs
    """,
    mechanism_of_action="""
    Multi-component social intervention:

    1. FAMILY RECONNECTION:
       - Family therapy for willing families
       - Communication skill building
       - Boundary setting with accountability
       - Repairing relationships damaged by offense

    2. MENTORSHIP:
       - Matching with prosocial mentor
       - Regular contact and activities
       - Role modeling conventional lifestyle
       - Crisis support availability

    3. PEER SUPPORT:
       - Connection to support groups (therapy, 12-step, etc.)
       - Prosocial peer activities
       - Mutual aid and accountability

    4. COMMUNITY INTEGRATION:
       - Faith community connection (if desired)
       - Volunteer opportunities
       - Recreational group membership
       - Civic participation where permitted

    5. CIRCLES OF SUPPORT (COSA MODEL):
       - 4-6 trained volunteers
       - Regular meetings with accountability focus
       - Support and monitoring combined
       - Community reintegration assistance

    Net effect: Creates social context that provides alternative attachment,
    informal social control, and reality testing that counters offense-related
    cognitive-emotional processes.
    """,
    state_specific_rationale={
        'Seeking': """Social isolation allows SEEKING content to develop
            without external reality check. Prosocial connections provide:
            - Alternative sources of intimacy/connection needs
            - Social feedback that challenges distortions
            - Reduced rumination time (social engagement)
            - Different content for cognitive processing""",
        'Conferring': """Social accountability makes CONFERRING activities
            more detectable - others notice absences, schedule irregularities.
            Social engagement occupies time otherwise used for surveillance.""",
        'Revising': """Prosocial connections support healthy REVISING by
            providing relationships to maintain and positive events to process,
            competing with offense-related content."""
    },
    target_states=['Seeking', 'Revising'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Seeking', 'Conferring'),
        ('Revising', 'Directing')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.15,    # Social accountability
        ('Seeking', 'Conferring'): -0.15,   # Time occupied, visible
        ('Revising', 'Directing'): -0.10,   # Support buffers
        ('Seeking', 'Seeking'): -0.10,      # Less isolated rumination
        ('Revising', 'Seeking'): +0.10      # Healthy exploration
    },
    evidence_level=EvidenceLevel.C,
    effectiveness_estimate=0.20,
    number_needed_to_treat=7.0,
    effect_size=0.35,
    key_studies=[
        "Wilson et al. (2009): COSA effectiveness evaluation",
        "Duwe (2013): Minnesota COSA program outcomes",
        "Laub & Sampson (2003): Shared Beginnings, Divergent Lives",
        "Visher & Travis (2003): Social support and reentry"
    ],
    duration_weeks=52,
    session_frequency="weekly group + ongoing contacts",
    cost_per_week=75.0,
    required_resources=[
        "Case manager for coordination",
        "Family therapy services",
        "Mentor recruitment and training",
        "Support group access",
        "COSA program (if available)"
    ],
    key_components=[
        "Family relationship assessment and therapy",
        "Mentor matching and supervision",
        "Support group connection",
        "Community activity identification",
        "COSA circle (high-risk cases)",
        "Social skill building (if needed)"
    ],
    population_notes="""
    Critical for socially isolated offenders whose isolation enables offending.
    COSA specifically validated for sex offenders. Consider family dynamics
    carefully - not all families are protective. Screen mentors thoroughly.
    """
)


# =============================================================================
# PHARMACOLOGICAL INTERVENTIONS (ADJUNCT)
# =============================================================================

SSRI_ADJUNCT = InterventionProtocol(
    name="ssri_adjunct",
    display_name="SSRI Medication (Adjunct)",
    category=ProtocolCategory.PHARMACOLOGICAL,
    description="""
    Selective Serotonin Reuptake Inhibitors as adjunct to psychological
    treatment. Can reduce impulsivity, obsessive thoughts, and emotional
    dysregulation that contribute to offense-related behavioral states.

    Never standalone treatment - always combined with psychotherapy and
    supervision as part of comprehensive program.
    """,
    theoretical_basis="""
    Based on Serotonergic Dysfunction Theory and Neurobiological Research:

    1. SEROTONIN AND IMPULSE CONTROL: Low serotonergic function associated
       with impulsivity, aggression, and behavioral disinhibition (Coccaro
       et al., 1989). SSRIs increase serotonin availability.

    2. OBSESSIVE-COMPULSIVE FEATURES: Some offending has obsessive-compulsive
       qualities - intrusive thoughts, urges, ritualized behavior. SSRIs are
       first-line for OCD (Fineberg et al., 2012).

    3. EMOTIONAL DYSREGULATION: SSRIs modulate amygdala reactivity and
       improve emotional regulation (Harmer et al., 2017). Reduces emotional
       triggers for impulsive action.

    4. PARAPHILIC INTERESTS: While not eliminating deviant interests, SSRIs
       may reduce intensity and intrusiveness of deviant sexual thoughts
       (Kafka, 1997).

    In behavioral state terms:
    - Reduces intensity of SEEKING state rumination/obsession
    - Increases latency from urge to action (SEEKING->DIRECTING)
    - Improves emotional regulation reducing affect-driven DIRECTING
    - May reduce compulsive quality of REVISING state
    """,
    mechanism_of_action="""
    Neurobiological intervention affecting multiple systems:

    1. SEROTONIN ENHANCEMENT:
       - Blocks serotonin reuptake
       - Increases synaptic serotonin availability
       - Modulates circuits involved in impulse control
       - Takes 4-6 weeks for full effect

    2. ANXIETY/RUMINATION REDUCTION:
       - Reduces obsessive thought patterns
       - Decreases anxiety that may drive acting out
       - Improves tolerance of distressing thoughts without action

    3. EMOTIONAL STABILIZATION:
       - Reduces emotional reactivity
       - Improves frustration tolerance
       - May reduce anger/irritability

    4. SEXUAL SIDE EFFECTS (THERAPEUTIC):
       - Reduced libido is common side effect
       - May reduce intensity of deviant sexual urges
       - Not primary mechanism but potentially helpful

    IMPORTANT LIMITATIONS:
    - Does NOT change deviant interests or cognitive distortions
    - Does NOT provide coping skills
    - Does NOT address environmental factors
    - Must be combined with psychotherapy
    """,
    state_specific_rationale={
        'Seeking': """SSRIs may reduce intensity and intrusiveness of SEEKING
            state content, particularly obsessive-compulsive features.
            Rumination and fantasy may be less compelling/intrusive.""",
        'Directing': """Improved impulse control increases latency between
            urge and action. Better emotional regulation reduces affect-driven
            acting out. Not effective during active DIRECTING - preventive only."""
    },
    target_states=['Seeking', 'Directing'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Directing', 'Directing'),
        ('Revising', 'Directing')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.10,    # Impulse control improvement
        ('Directing', 'Directing'): -0.15,  # Reduced compulsivity
        ('Revising', 'Directing'): -0.10,   # Emotional regulation
        ('Seeking', 'Seeking'): -0.05       # Reduced rumination intensity
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.20,
    number_needed_to_treat=7.0,
    effect_size=0.35,
    key_studies=[
        "Kafka (1997): SSRI treatment of paraphilias",
        "Adi et al. (2002): SSRIs for sex offenders meta-analysis",
        "Lösel & Schmucker (2005): Pharmacological treatment component",
        "Garcia & Thibaut (2011): Pharmacological treatments review"
    ],
    duration_weeks=52,
    session_frequency="monthly med check",
    cost_per_week=30.0,
    required_resources=[
        "Prescribing physician (psychiatrist preferred)",
        "Medication monitoring protocol",
        "Lab work capability (as indicated)",
        "Coordination with therapy providers"
    ],
    required_training="Medical degree with psychiatric prescribing training",
    key_components=[
        "Comprehensive psychiatric evaluation",
        "Medication selection (consider side effect profile)",
        "Gradual titration to therapeutic dose",
        "Side effect monitoring and management",
        "Coordination with psychotherapy",
        "Regular effectiveness assessment"
    ],
    contraindications=[
        "Bipolar disorder (may trigger mania)",
        "Significant drug interactions",
        "History of serotonin syndrome",
        "Severe liver impairment",
        "Pregnancy (some SSRIs)",
        "As sole treatment (insufficient)"
    ],
    population_notes="""
    Consider for individuals with obsessive-compulsive features to offending,
    significant anxiety/depression comorbidity, or emotional dysregulation.
    Not effective for calculated, predatory offending without impulsive or
    obsessive features. Always combine with psychotherapy.
    """
)

ANTI_ANDROGEN = InterventionProtocol(
    name="anti_androgen",
    display_name="Anti-Androgen Medication",
    category=ProtocolCategory.PHARMACOLOGICAL,
    description="""
    Hormonal treatment to reduce sexual drive and deviant sexual arousal.
    Reserved for high-risk cases with prominent sexual motivation and
    when other interventions insufficient. Requires informed consent,
    careful monitoring, and ongoing psychological treatment.

    Most aggressive pharmacological intervention - significant effects
    but also significant side effects and ethical considerations.
    """,
    theoretical_basis="""
    Based on Androgen-Behavior Research:

    1. TESTOSTERONE AND SEXUAL BEHAVIOR: Testosterone mediates sexual desire
       and arousal. Reducing testosterone reduces both deviant and nondeviant
       sexual motivation (Bradford, 2000).

    2. CONDITIONED AROUSAL: While anti-androgens reduce drive, they do not
       eliminate conditioned associations. Arousal to deviant stimuli may
       return if medication stopped.

    3. WINDOW OF OPPORTUNITY: Reduced drive provides "window" for
       psychological intervention when urges less overwhelming.
       Facilitates therapy engagement.

    4. TWO CATEGORIES:
       - Anti-androgens (CPA, MPA): Block testosterone at receptor level
       - GnRH agonists: Reduce testosterone production centrally

    In behavioral state terms:
    - Reduces intensity of sexually-motivated SEEKING content
    - Decreases drive behind SEEKING->DIRECTING progression
    - May reduce compulsive quality of offense-related REVISING
    - Creates space for cognitive interventions to work
    """,
    mechanism_of_action="""
    Hormonal intervention reducing androgenic drive:

    CYPROTERONE ACETATE (CPA):
    - Anti-androgen blocking testosterone receptor
    - Also reduces testosterone production
    - Available in Europe, not US

    MEDROXYPROGESTERONE ACETATE (MPA/Depo-Provera):
    - Progestational agent reducing testosterone
    - Available in US
    - Intramuscular injection (weekly initially, then longer intervals)

    GnRH AGONISTS (Leuprolide/Lupron):
    - Suppresses pituitary->gonadal axis
    - Produces "chemical castration"
    - Most effective but most significant side effects

    EFFECTS:
    1. Reduced sexual desire and fantasy intensity
    2. Reduced frequency of sexual thoughts
    3. Reduced physiological arousal capacity
    4. General reduction in aggressive impulses (some evidence)

    LIMITATIONS:
    - Does NOT change deviant preferences (direction of attraction)
    - Does NOT address cognitive distortions
    - Effects reverse when medication stopped
    - Significant side effects require monitoring
    - Ethical concerns about consent and coercion
    """,
    state_specific_rationale={
        'Seeking': """Primary target. Reduces intensity of sexually-motivated
            SEEKING content by reducing underlying drive. Fantasy may become
            less compelling and intrusive when testosterone reduced.""",
        'Directing': """For sexually-motivated DIRECTING, reduced drive means
            reduced motivation for acting. However, does not address non-sexual
            motivations for offending.""",
        'Conferring': """If CONFERRING is sexually motivated (stalking for
            sexual targets), drive reduction may reduce surveillance motivation.
            Less effect on non-sexual surveillance."""
    },
    target_states=['Seeking', 'Directing'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Seeking', 'Conferring'),
        ('Directing', 'Directing')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.30,    # Major drive reduction
        ('Seeking', 'Conferring'): -0.25,   # Reduced target-seeking
        ('Directing', 'Directing'): -0.20,  # Reduced persistence
        ('Seeking', 'Seeking'): -0.20       # Reduced fantasy intensity
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.40,
    number_needed_to_treat=3.5,
    effect_size=0.65,
    key_studies=[
        "Lösel & Schmucker (2005): Meta-analysis, OR=1.70 for hormonal",
        "Bradford & Pawlak (1993): Cyproterone acetate study",
        "Meyer et al. (1992): MPA treatment outcomes",
        "Briken et al. (2003): GnRH agonist treatment review"
    ],
    duration_weeks=104,
    session_frequency="monthly monitoring",
    cost_per_week=100.0,
    required_resources=[
        "Endocrinologist or psychiatrist with experience",
        "Regular hormone level monitoring",
        "Bone density monitoring (DEXA)",
        "Cardiovascular monitoring",
        "Documented informed consent process",
        "Mandatory concurrent psychotherapy"
    ],
    required_training="Medical degree with endocrine/psychiatric expertise",
    key_components=[
        "Comprehensive medical evaluation",
        "Clear informed consent documentation",
        "Medication initiation and titration",
        "Regular hormone and side effect monitoring",
        "Bone density monitoring annually",
        "Cardiovascular risk monitoring",
        "Coordination with psychotherapy",
        "Regular assessment of continued need"
    ],
    contraindications=[
        "Liver disease (CPA)",
        "Thromboembolic disorders",
        "Severe cardiovascular disease",
        "Refusal of informed consent",
        "Non-sexual offense pattern",
        "Adolescents (development concerns)",
        "As sole treatment (must include therapy)"
    ],
    population_notes="""
    Reserved for highest-risk cases with clear sexual motivation, when other
    interventions insufficient, and when individual consents. NOT appropriate
    for non-sexual offending or when motivation is primarily non-sexual
    (power, anger). Ethical concerns about coercion in correctional settings.
    Consider for 'Pure Predator' subtype with sexual motivation.
    """
)


# =============================================================================
# COMBINED PROTOCOLS
# =============================================================================

COMPREHENSIVE_TREATMENT_PROGRAM = InterventionProtocol(
    name="comprehensive_program",
    display_name="Comprehensive Treatment Program",
    category=ProtocolCategory.COMBINED,
    description="""
    Multi-modal treatment combining cognitive-behavioral therapy, supervision,
    environmental support, and pharmacotherapy (when indicated). Represents
    highest-intensity, highest-cost intervention for highest-risk individuals.

    Addresses multiple criminogenic needs simultaneously per RNR principles.
    Integrates all intervention modalities into coordinated treatment plan.
    """,
    theoretical_basis="""
    Based on the Risk-Need-Responsivity (RNR) Model (Andrews & Bonta, 2010):

    1. RISK PRINCIPLE: Match intervention intensity to risk level.
       Comprehensive programs for high-risk individuals.

    2. NEED PRINCIPLE: Target criminogenic needs:
       - Antisocial cognition (CBT)
       - Antisocial associates (social support)
       - Family/marital (family therapy)
       - Employment/education (vocational)
       - Substance abuse (treatment)
       - Leisure/recreation (structured activities)
       - Antisocial personality (long-term therapy)

    3. RESPONSIVITY PRINCIPLE: Match treatment style to individual:
       - General: CBT is most effective
       - Specific: Adapt to individual factors

    4. MULTI-MODAL SYNERGY: Addressing multiple needs simultaneously
       produces greater effects than single interventions.

    In behavioral state terms:
    - Multiple intervention points across all state transitions
    - CBT targets SEEKING content
    - Supervision blocks CONFERRING and DIRECTING opportunity
    - Environmental support stabilizes foundation
    - Medication reduces drive/impulsivity (when indicated)
    """,
    mechanism_of_action="""
    Integrated multi-component intervention:

    1. COGNITIVE-BEHAVIORAL THERAPY:
       - CBT for fantasy management (SEEKING)
       - DBT for impulse control (DIRECTING)
       - Trauma processing (if indicated)
       - Offense-specific treatment modules

    2. INTENSIVE SUPERVISION:
       - GPS monitoring and exclusion zones
       - High-frequency contacts
       - Structured schedule
       - Comprehensive accountability

    3. ENVIRONMENTAL SUPPORT:
       - Stable housing provision
       - Employment support
       - Prosocial network building
       - Family involvement (when appropriate)

    4. PHARMACOTHERAPY (when indicated):
       - SSRI for obsessive/impulsive features
       - Anti-androgen for high-risk sexual offenders

    5. CASE MANAGEMENT:
       - Single point of coordination
       - Treatment team meetings
       - Progress monitoring
       - Modification as needed

    Net effect: Creates comprehensive intervention addressing cognitive,
    behavioral, environmental, and biological factors across all
    behavioral states simultaneously.
    """,
    state_specific_rationale={
        'Seeking': """Multiple intervention components target SEEKING:
            - CBT modifies fantasy content and cognitions
            - Structured schedule reduces rumination time
            - Social support provides alternative content
            - Medication may reduce intensity""",
        'Directing': """Multiple blocks on DIRECTING:
            - Supervision prevents opportunity
            - DBT skills for impulse control
            - Medication reduces drive
            - Social accountability""",
        'Conferring': """CONFERRING disrupted by:
            - Geographic restrictions
            - Intensive supervision
            - Time structure
            - Social accountability""",
        'Revising': """Healthy REVISING supported by:
            - Stable housing environment
            - Therapy processing
            - Social support
            - Employment meaning"""
    },
    target_states=['Seeking', 'Directing', 'Conferring', 'Revising'],
    target_transitions=[
        ('Seeking', 'Directing'),
        ('Conferring', 'Directing'),
        ('Directing', 'Directing'),
        ('Revising', 'Directing')
    ],
    effect_on_transitions={
        ('Seeking', 'Directing'): -0.40,    # Combined cognitive+medication+supervision
        ('Conferring', 'Directing'): -0.45, # Supervision+geographic blocks
        ('Directing', 'Directing'): -0.35,  # Multiple interruption points
        ('Revising', 'Directing'): -0.30,   # Environmental stability
        ('Directing', 'Revising'): +0.35,   # Multiple pathways to processing
        ('Conferring', 'Seeking'): +0.20,   # Redirection supported
        ('Seeking', 'Revising'): +0.25      # Healthy processing pathway
    },
    evidence_level=EvidenceLevel.B,
    effectiveness_estimate=0.55,
    number_needed_to_treat=2.5,
    effect_size=0.80,
    key_studies=[
        "Hanson et al. (2009): SOTEP multi-site treatment outcomes",
        "Lösel & Schmucker (2005): Combined treatment meta-analysis",
        "Andrews & Bonta (2010): RNR meta-analysis",
        "Aos et al. (2006): WSIPP comprehensive programs cost-benefit"
    ],
    duration_weeks=52,
    session_frequency="multiple weekly",
    cost_per_week=1000.0,
    required_resources=[
        "Multi-disciplinary treatment team",
        "Primary case manager",
        "CBT/DBT-trained therapist",
        "Psychiatrist (for medication)",
        "Probation officer (specialized)",
        "Housing program",
        "Employment specialist",
        "All modality resources"
    ],
    key_components=[
        "Comprehensive assessment (STABLE-2007, LS-CMI)",
        "Individualized treatment plan",
        "Weekly individual therapy (CBT/DBT)",
        "Weekly group therapy",
        "Intensive supervision with GPS",
        "Stable housing",
        "Employment support",
        "Medication management (if indicated)",
        "Family involvement",
        "Case management coordination",
        "Regular treatment team meetings",
        "Outcome monitoring"
    ],
    population_notes="""
    Reserved for highest-risk cases (Static-99R ≥6, STABLE high) where
    single interventions insufficient. Cost-effective only for high-risk -
    over-treatment of low-risk is counterproductive and costly. Requires
    significant resource investment and coordination capacity.
    """
)


# =============================================================================
# PROTOCOL REGISTRY
# =============================================================================

INTERVENTION_PROTOCOLS: Dict[str, InterventionProtocol] = {
    # Therapeutic
    'cbt_fantasy_management': CBT_FANTASY_MANAGEMENT,
    'dbt_impulse_control': DBT_IMPULSE_CONTROL,
    'mentalization_therapy': MENTALIZATION_THERAPY,
    'trauma_therapy': TRAUMA_THERAPY,
    'schema_therapy': SCHEMA_THERAPY,

    # Supervision
    'intensive_supervision': INTENSIVE_SUPERVISION,
    'geographic_restriction': GEOGRAPHIC_RESTRICTION,
    'structured_schedule': STRUCTURED_DAILY_SCHEDULE,

    # Environmental
    'stable_housing': STABLE_HOUSING,
    'employment_support': EMPLOYMENT_SUPPORT,
    'social_support': SOCIAL_SUPPORT_NETWORK,

    # Pharmacological
    'ssri_adjunct': SSRI_ADJUNCT,
    'anti_androgen': ANTI_ANDROGEN,

    # Combined
    'comprehensive_program': COMPREHENSIVE_TREATMENT_PROGRAM
}


def get_protocol(name: str) -> Optional[InterventionProtocol]:
    """Get protocol by name."""
    return INTERVENTION_PROTOCOLS.get(name)


def get_protocols_for_state(state: str) -> List[InterventionProtocol]:
    """Get all protocols applicable for a given state."""
    return [
        p for p in INTERVENTION_PROTOCOLS.values()
        if p.is_applicable(state)
    ]


def get_protocols_for_transition(
    from_state: str,
    to_state: str
) -> List[InterventionProtocol]:
    """Get protocols that target a specific transition."""
    return [
        p for p in INTERVENTION_PROTOCOLS.values()
        if (from_state, to_state) in p.target_transitions
    ]


def get_protocols_by_category(category: ProtocolCategory) -> List[InterventionProtocol]:
    """Get all protocols in a category."""
    return [
        p for p in INTERVENTION_PROTOCOLS.values()
        if p.category == category
    ]


def get_protocol_summary() -> Dict:
    """Get summary of all protocols for display."""
    return {
        name: {
            'display_name': p.display_name,
            'category': p.category.value,
            'target_states': p.target_states,
            'effectiveness': p.effectiveness_estimate,
            'evidence': p.evidence_level.value,
            'effect_size': p.effect_size,
            'cost_per_week': p.cost_per_week,
            'theoretical_basis': p.theoretical_basis[:200] + '...' if len(p.theoretical_basis) > 200 else p.theoretical_basis
        }
        for name, p in INTERVENTION_PROTOCOLS.items()
    }
