"""
Clinical Intervention Analysis System

A comprehensive framework for identifying optimal intervention points
in criminal behavioral trajectories using causal modeling.

Core Components:
- causal_model: Structural Causal Model (SCM) for behavioral trajectories
- protocols: Evidence-based intervention protocol library
- trajectory_analysis: Critical point and intervention window detection
- intervention_effects: Do-calculus for causal effect estimation (coming)
- counterfactual: Counterfactual simulation engine
- optimization: Optimal intervention timing and protocol selection
- clinical_report: Clinical report generation for decision support

Usage:
    from intervention import (
        BehavioralSCM,
        INTERVENTION_PROTOCOLS,
        TrajectoryAnalyzer
    )

    # Create SCM from transition matrix
    scm = BehavioralSCM(transition_matrix, state_names)

    # Add intervention and compute causal effect
    scm.add_intervention('cbt', target_transitions, effects)
    effect = scm.compute_causal_effect('cbt', 'reached_directing', 'Seeking')

    # Analyze trajectory for intervention opportunities
    analyzer = TrajectoryAnalyzer(transition_matrix=transition_matrix)
    results = analyzer.comprehensive_analysis(trajectory)
"""

from .causal_model import (
    BehavioralSCM,
    CausalNode,
    InterventionNode,
    OutcomeNode,
    NodeType,
    create_scm_from_individual,
    create_population_scm
)

from .protocols import (
    InterventionProtocol,
    ProtocolCategory,
    EvidenceLevel,
    IntensityLevel,
    INTERVENTION_PROTOCOLS,
    get_protocol,
    get_protocols_for_state,
    get_protocols_for_transition,
    get_protocols_by_category,
    get_protocol_summary,
    # Specific protocols
    CBT_FANTASY_MANAGEMENT,
    DBT_IMPULSE_CONTROL,
    INTENSIVE_SUPERVISION,
    COMPREHENSIVE_TREATMENT_PROGRAM
)

from .trajectory_analysis import (
    TrajectoryAnalyzer,
    CriticalTransition,
    TippingPoint,
    InterventionWindow,
    EarlyWarning,
    TransitionRisk,
    CRITICAL_TRANSITIONS,
    analyze_trajectory
)

from .counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    BranchingResult,
    quick_counterfactual
)

from .optimization import (
    InterventionOptimizer,
    InterventionRecommendation,
    OptimizationResult,
    UrgencyScore,
    OptimizationObjective,
    expected_harm_reduction,
    intervention_urgency_score,
    quick_optimize
)

from .clinical_report import (
    ClinicalReportGenerator,
    ClinicalReport,
    ComparisonReport,
    RiskAssessment,
    RiskLevel,
    MonitoringPlan,
    MissedOpportunity,
    quick_report
)


__version__ = "0.1.0"

__all__ = [
    # Causal Model
    'BehavioralSCM',
    'CausalNode',
    'InterventionNode',
    'OutcomeNode',
    'NodeType',
    'create_scm_from_individual',
    'create_population_scm',

    # Protocols
    'InterventionProtocol',
    'ProtocolCategory',
    'EvidenceLevel',
    'IntensityLevel',
    'INTERVENTION_PROTOCOLS',
    'get_protocol',
    'get_protocols_for_state',
    'get_protocols_for_transition',
    'get_protocols_by_category',
    'get_protocol_summary',
    'CBT_FANTASY_MANAGEMENT',
    'DBT_IMPULSE_CONTROL',
    'INTENSIVE_SUPERVISION',
    'COMPREHENSIVE_TREATMENT_PROGRAM',

    # Trajectory Analysis
    'TrajectoryAnalyzer',
    'CriticalTransition',
    'TippingPoint',
    'InterventionWindow',
    'EarlyWarning',
    'TransitionRisk',
    'CRITICAL_TRANSITIONS',
    'analyze_trajectory',

    # Counterfactual
    'CounterfactualEngine',
    'CounterfactualResult',
    'BranchingResult',
    'quick_counterfactual',

    # Optimization
    'InterventionOptimizer',
    'InterventionRecommendation',
    'OptimizationResult',
    'UrgencyScore',
    'OptimizationObjective',
    'expected_harm_reduction',
    'intervention_urgency_score',
    'quick_optimize',

    # Clinical Reports
    'ClinicalReportGenerator',
    'ClinicalReport',
    'ComparisonReport',
    'RiskAssessment',
    'RiskLevel',
    'MonitoringPlan',
    'MissedOpportunity',
    'quick_report'
]
