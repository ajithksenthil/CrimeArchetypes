"""
Clinical Report Generator

Generates actionable clinical reports for intervention planning.

Report Types:
1. Individual Report - Comprehensive assessment for one case
2. Comparison Report - Side-by-side analysis of multiple cases
3. Population Summary - Aggregate intervention opportunities
4. Monitoring Plan - Ongoing surveillance recommendations

Reports integrate:
- Trajectory analysis (critical points, intervention windows)
- Counterfactual analysis (missed opportunities)
- Optimization results (recommended interventions)
- Risk assessment (urgency, escalation patterns)

Output formats: Dict, JSON, Markdown
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import numpy as np
from enum import Enum

from .trajectory_analysis import (
    TrajectoryAnalyzer,
    CriticalTransition,
    TippingPoint,
    InterventionWindow,
    EarlyWarning
)
from .counterfactual import CounterfactualEngine, CounterfactualResult
from .optimization import (
    InterventionOptimizer,
    OptimizationResult,
    InterventionRecommendation,
    UrgencyScore
)
from .protocols import InterventionProtocol, INTERVENTION_PROTOCOLS


class RiskLevel(Enum):
    """Clinical risk levels."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class ReportFormat(Enum):
    """Output format for reports."""
    DICT = "dict"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class RiskAssessment:
    """
    Comprehensive risk assessment.

    Attributes:
        level: Overall risk level
        score: Numeric score (0-1)
        factors: Contributing risk factors
        trajectory_indicators: Indicators from trajectory analysis
        time_to_harm_estimate: Estimated events until harmful outcome
    """
    level: RiskLevel
    score: float
    factors: Dict[str, float]
    trajectory_indicators: List[str]
    time_to_harm_estimate: Optional[float] = None

    @classmethod
    def from_analysis(
        cls,
        current_state: str,
        mfpt_to_directing: float,
        critical_transitions: List[CriticalTransition],
        tipping_points: List[TippingPoint],
        urgency: UrgencyScore = None
    ) -> 'RiskAssessment':
        """Create risk assessment from analysis results."""
        factors = {}

        # State-based risk
        state_risks = {
            'Directing': 1.0,
            'Conferring': 0.6,
            'Seeking': 0.4,
            'Revising': 0.2
        }
        factors['current_state'] = state_risks.get(current_state, 0.3)

        # MFPT-based risk (closer = higher)
        factors['proximity_to_harm'] = max(0, 1 - mfpt_to_directing / 10)

        # Critical transition history
        factors['critical_transitions'] = min(1.0, len(critical_transitions) * 0.2)

        # Tipping point proximity
        if tipping_points:
            max_prob = max(tp.p_directing_after for tp in tipping_points)
            factors['tipping_point_risk'] = max_prob
        else:
            factors['tipping_point_risk'] = 0.3

        # Urgency factors
        if urgency:
            factors['urgency_score'] = urgency.overall
            factors['trajectory_momentum'] = urgency.trajectory_momentum

        # Calculate overall score
        score = np.mean(list(factors.values()))

        # Determine level
        if score >= 0.75:
            level = RiskLevel.CRITICAL
        elif score >= 0.5:
            level = RiskLevel.HIGH
        elif score >= 0.25:
            level = RiskLevel.MODERATE
        else:
            level = RiskLevel.LOW

        # Trajectory indicators
        indicators = []
        if factors.get('current_state', 0) > 0.5:
            indicators.append(f"Currently in high-risk {current_state} state")
        if factors.get('proximity_to_harm', 0) > 0.5:
            indicators.append(f"Low MFPT to Directing ({mfpt_to_directing:.1f} events)")
        if critical_transitions:
            indicators.append(f"{len(critical_transitions)} critical transitions observed")
        if tipping_points:
            indicators.append(f"{len(tipping_points)} tipping points identified")

        return cls(
            level=level,
            score=score,
            factors=factors,
            trajectory_indicators=indicators,
            time_to_harm_estimate=mfpt_to_directing
        )


@dataclass
class MissedOpportunity:
    """A missed intervention opportunity identified retrospectively."""
    event_index: int
    state_at_time: str
    recommended_protocol: str
    estimated_impact: float
    rationale: str


@dataclass
class MonitoringPlan:
    """Recommended monitoring plan."""
    frequency: str  # 'daily', 'weekly', 'biweekly', 'monthly'
    key_indicators: List[str]
    escalation_triggers: List[str]
    reassessment_timeline: str


@dataclass
class ClinicalReport:
    """
    Comprehensive clinical report for intervention planning.

    Designed to be actionable for clinicians while providing
    sufficient detail for decision-making.
    """
    # Header
    individual_id: str
    report_date: str
    report_type: str = "Individual Assessment"

    # Current Assessment
    current_state: str = ""
    recent_trajectory: List[str] = field(default_factory=list)
    archetype: str = ""

    # Risk Analysis
    risk_assessment: RiskAssessment = None
    escalation_score: float = 0.0
    mfpt_to_directing: float = 0.0
    p_directing_next_n: Dict[int, float] = field(default_factory=dict)

    # Critical Events
    critical_transitions: List[CriticalTransition] = field(default_factory=list)
    tipping_points: List[TippingPoint] = field(default_factory=list)
    early_warnings: List[EarlyWarning] = field(default_factory=list)

    # Intervention Recommendations
    intervention_windows: List[InterventionWindow] = field(default_factory=list)
    recommended_interventions: List[InterventionRecommendation] = field(default_factory=list)
    optimal_timing: Optional[int] = None
    optimization_summary: Dict = field(default_factory=dict)

    # Retrospective Analysis
    missed_opportunities: List[MissedOpportunity] = field(default_factory=list)
    counterfactual_analysis: Dict = field(default_factory=dict)

    # Monitoring Plan
    monitoring_plan: MonitoringPlan = None

    # Confidence & Limitations
    confidence_level: float = 0.0
    limitations: List[str] = field(default_factory=list)
    data_quality_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        def serialize(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()
                       if not k.startswith('_')}
            elif isinstance(obj, (list, tuple)):
                return [serialize(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj

        return serialize(self.__dict__)

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        lines = []

        # Header
        lines.append(f"# Clinical Intervention Report")
        lines.append(f"\n**Individual ID:** {self.individual_id}")
        lines.append(f"**Report Date:** {self.report_date}")
        lines.append(f"**Report Type:** {self.report_type}")

        # Risk Summary
        lines.append(f"\n## Risk Assessment")
        if self.risk_assessment:
            level_emoji = {
                RiskLevel.LOW: "🟢",
                RiskLevel.MODERATE: "🟡",
                RiskLevel.HIGH: "🟠",
                RiskLevel.CRITICAL: "🔴"
            }
            emoji = level_emoji.get(self.risk_assessment.level, "⚪")
            lines.append(f"\n**Risk Level:** {emoji} {self.risk_assessment.level.value}")
            lines.append(f"**Risk Score:** {self.risk_assessment.score:.2f}")

            if self.risk_assessment.trajectory_indicators:
                lines.append("\n### Risk Indicators")
                for indicator in self.risk_assessment.trajectory_indicators:
                    lines.append(f"- {indicator}")

        # Current State
        lines.append(f"\n## Current Assessment")
        lines.append(f"\n**Current State:** {self.current_state}")
        if self.archetype:
            lines.append(f"**Archetype:** {self.archetype}")
        lines.append(f"**MFPT to Directing:** {self.mfpt_to_directing:.2f} events")

        if self.recent_trajectory:
            lines.append(f"\n**Recent Trajectory:** {' → '.join(self.recent_trajectory[-5:])}")

        # Critical Events
        if self.critical_transitions:
            lines.append(f"\n## Critical Transitions")
            for ct in self.critical_transitions[:5]:
                lines.append(f"- Event {ct.index}: {ct.from_state} → {ct.to_state} ({ct.name})")

        if self.tipping_points:
            lines.append(f"\n## Tipping Points")
            for tp in self.tipping_points[:3]:
                lines.append(f"- Event {tp.index}: P(Directing) increased to {tp.p_directing_after:.1%}")

        # Intervention Recommendations
        lines.append(f"\n## Intervention Recommendations")

        if self.intervention_windows:
            lines.append(f"\n### Intervention Windows")
            for iw in self.intervention_windows[:3]:
                lines.append(f"\n**Window: Events {iw.start_index}-{iw.end_index}**")
                lines.append(f"- State: {iw.state_at_window}")
                lines.append(f"- Urgency: {iw.urgency:.1%}")
                lines.append(f"- Expected Effectiveness: {iw.expected_effectiveness:.1%}")

        if self.recommended_interventions:
            lines.append(f"\n### Recommended Protocols")
            for i, rec in enumerate(self.recommended_interventions[:5], 1):
                lines.append(f"\n**{i}. {rec.protocol.display_name}**")
                lines.append(f"- Timing: Event {rec.time_index}")
                lines.append(f"- Expected Benefit: {rec.expected_benefit:.1%} harm reduction")
                lines.append(f"- Cost: ${rec.cost:,.0f}")
                lines.append(f"- Urgency: {rec.urgency:.1%}")
                lines.append(f"- Rationale: {rec.rationale}")

        # Retrospective Analysis
        if self.missed_opportunities:
            lines.append(f"\n## Retrospective Analysis: Missed Opportunities")
            for mo in self.missed_opportunities[:5]:
                lines.append(f"\n**Event {mo.event_index} ({mo.state_at_time} state)**")
                lines.append(f"- Recommended: {mo.recommended_protocol}")
                lines.append(f"- Estimated Impact: {mo.estimated_impact:.1%} harm reduction")
                lines.append(f"- {mo.rationale}")

        # Monitoring Plan
        if self.monitoring_plan:
            lines.append(f"\n## Monitoring Plan")
            lines.append(f"\n**Frequency:** {self.monitoring_plan.frequency}")
            lines.append(f"**Reassessment:** {self.monitoring_plan.reassessment_timeline}")

            if self.monitoring_plan.key_indicators:
                lines.append("\n### Key Indicators to Monitor")
                for indicator in self.monitoring_plan.key_indicators:
                    lines.append(f"- {indicator}")

            if self.monitoring_plan.escalation_triggers:
                lines.append("\n### Escalation Triggers")
                for trigger in self.monitoring_plan.escalation_triggers:
                    lines.append(f"- ⚠️ {trigger}")

        # Limitations
        lines.append(f"\n## Confidence & Limitations")
        lines.append(f"\n**Confidence Level:** {self.confidence_level:.1%}")

        if self.limitations:
            lines.append("\n### Limitations")
            for lim in self.limitations:
                lines.append(f"- {lim}")

        # Footer
        lines.append(f"\n---")
        lines.append(f"*This report is for clinical decision support only. "
                    f"All recommendations require professional judgment.*")

        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """
    Comparison report for multiple individuals.

    Enables side-by-side analysis of intervention opportunities
    across similar cases.
    """
    report_date: str
    individuals: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    common_patterns: List[str]
    divergent_patterns: List[str]
    aggregate_recommendations: List[Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'report_date': self.report_date,
            'individuals': self.individuals,
            'comparison_metrics': self.comparison_metrics,
            'common_patterns': self.common_patterns,
            'divergent_patterns': self.divergent_patterns,
            'aggregate_recommendations': self.aggregate_recommendations
        }

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = []
        lines.append(f"# Comparison Report")
        lines.append(f"\n**Date:** {self.report_date}")
        lines.append(f"**Individuals:** {', '.join(self.individuals)}")

        lines.append(f"\n## Comparison Metrics")
        lines.append("\n| Metric | " + " | ".join(self.individuals) + " |")
        lines.append("|" + "---|" * (len(self.individuals) + 1))

        for metric, values in self.comparison_metrics.items():
            row = f"| {metric} |"
            for ind in self.individuals:
                val = values.get(ind, 'N/A')
                if isinstance(val, float):
                    row += f" {val:.2f} |"
                else:
                    row += f" {val} |"
            lines.append(row)

        if self.common_patterns:
            lines.append(f"\n## Common Patterns")
            for pattern in self.common_patterns:
                lines.append(f"- {pattern}")

        if self.divergent_patterns:
            lines.append(f"\n## Divergent Patterns")
            for pattern in self.divergent_patterns:
                lines.append(f"- {pattern}")

        return "\n".join(lines)


class ClinicalReportGenerator:
    """
    Generator for clinical intervention reports.

    Integrates trajectory analysis, counterfactual reasoning,
    and optimization to produce actionable clinical reports.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray = None,
        state_names: List[str] = None
    ):
        """
        Initialize report generator.

        Args:
            transition_matrix: State transition matrix
            state_names: Names of states
        """
        self.transition_matrix = transition_matrix
        self.state_names = state_names or ['Seeking', 'Directing', 'Conferring', 'Revising']

        # Initialize components
        self.trajectory_analyzer = None
        self.counterfactual_engine = None
        self.optimizer = None

        if transition_matrix is not None:
            self._initialize_components()

    def _initialize_components(self):
        """Initialize analysis components."""
        from .causal_model import BehavioralSCM

        # Trajectory analyzer
        self.trajectory_analyzer = TrajectoryAnalyzer(
            transition_matrix=self.transition_matrix,
            state_names=self.state_names
        )

        # SCM and counterfactual engine
        scm = BehavioralSCM(self.transition_matrix, self.state_names)
        self.counterfactual_engine = CounterfactualEngine(scm)
        self.optimizer = InterventionOptimizer(scm)

    def generate_individual_report(
        self,
        individual_id: str,
        trajectory: List[str],
        events: List[str] = None,
        archetype: str = None,
        include_retrospective: bool = True,
        include_optimization: bool = True
    ) -> ClinicalReport:
        """
        Generate comprehensive report for one individual.

        Args:
            individual_id: Identifier for the individual
            trajectory: Sequence of behavioral states
            events: Optional list of event descriptions
            archetype: Behavioral archetype (if known)
            include_retrospective: Include missed opportunity analysis
            include_optimization: Include intervention optimization

        Returns:
            Complete ClinicalReport
        """
        report = ClinicalReport(
            individual_id=individual_id,
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            current_state=trajectory[-1] if trajectory else "",
            recent_trajectory=trajectory[-10:] if trajectory else [],
            archetype=archetype or "Unknown"
        )

        # Trajectory analysis
        if self.trajectory_analyzer:
            analysis = self.trajectory_analyzer.comprehensive_analysis(trajectory)

            report.critical_transitions = analysis.get('critical_transitions', [])
            report.tipping_points = analysis.get('tipping_points', [])
            report.intervention_windows = analysis.get('intervention_windows', [])
            report.early_warnings = analysis.get('early_warnings', [])
            report.mfpt_to_directing = analysis.get('mfpt_to_directing', 5.0)

            # Risk assessment
            urgency = self.optimizer.compute_urgency_score(
                current_state=trajectory[-1],
                recent_trajectory=trajectory[-5:],
                mfpt_to_directing=report.mfpt_to_directing
            ) if self.optimizer and trajectory else None

            report.risk_assessment = RiskAssessment.from_analysis(
                current_state=trajectory[-1] if trajectory else "Unknown",
                mfpt_to_directing=report.mfpt_to_directing,
                critical_transitions=report.critical_transitions,
                tipping_points=report.tipping_points,
                urgency=urgency
            )

        # Optimization
        if include_optimization and self.optimizer and trajectory:
            opt_result = self.optimizer.find_optimal_timing(
                trajectory=trajectory,
                max_interventions=5
            )
            report.recommended_interventions = opt_result.recommendations
            report.optimization_summary = {
                'total_expected_benefit': opt_result.total_expected_benefit,
                'total_cost': opt_result.total_cost,
                'pareto_frontier': opt_result.pareto_frontier
            }

            if opt_result.recommendations:
                report.optimal_timing = opt_result.recommendations[0].time_index

        # Retrospective analysis
        if include_retrospective and self.counterfactual_engine and trajectory:
            report.missed_opportunities = self._identify_missed_opportunities(
                trajectory=trajectory
            )

            if len(trajectory) > 5:
                try:
                    retro = self.counterfactual_engine.retrospective_analysis(
                        observed_trajectory=trajectory,
                        outcome_state='Directing',
                        max_interventions=3
                    )
                    report.counterfactual_analysis = {
                        'best_intervention_point': retro.get('best_intervention_point'),
                        'maximum_harm_reduction': retro.get('maximum_harm_reduction', 0)
                    }
                except Exception:
                    pass

        # Monitoring plan
        report.monitoring_plan = self._generate_monitoring_plan(
            risk_level=report.risk_assessment.level if report.risk_assessment else RiskLevel.MODERATE,
            current_state=report.current_state,
            trajectory=trajectory
        )

        # Confidence and limitations
        report.confidence_level = self._compute_confidence(trajectory)
        report.limitations = self._identify_limitations(trajectory)

        return report

    def generate_comparison_report(
        self,
        individuals: Dict[str, List[str]],
        archetypes: Dict[str, str] = None
    ) -> ComparisonReport:
        """
        Generate comparison report for multiple individuals.

        Args:
            individuals: Dict mapping ID to trajectory
            archetypes: Optional dict mapping ID to archetype

        Returns:
            ComparisonReport
        """
        archetypes = archetypes or {}
        individual_ids = list(individuals.keys())

        # Compute metrics for each individual
        comparison_metrics = {
            'risk_score': {},
            'mfpt_to_directing': {},
            'critical_transitions': {},
            'intervention_windows': {},
            'optimal_benefit': {}
        }

        individual_reports = {}
        for ind_id, trajectory in individuals.items():
            report = self.generate_individual_report(
                individual_id=ind_id,
                trajectory=trajectory,
                archetype=archetypes.get(ind_id),
                include_retrospective=False
            )
            individual_reports[ind_id] = report

            comparison_metrics['risk_score'][ind_id] = (
                report.risk_assessment.score if report.risk_assessment else 0
            )
            comparison_metrics['mfpt_to_directing'][ind_id] = report.mfpt_to_directing
            comparison_metrics['critical_transitions'][ind_id] = len(report.critical_transitions)
            comparison_metrics['intervention_windows'][ind_id] = len(report.intervention_windows)
            comparison_metrics['optimal_benefit'][ind_id] = (
                report.optimization_summary.get('total_expected_benefit', 0)
            )

        # Identify common patterns
        common_patterns = []
        all_have_high_risk = all(
            comparison_metrics['risk_score'].get(i, 0) > 0.5
            for i in individual_ids
        )
        if all_have_high_risk:
            common_patterns.append("All cases show elevated risk scores (>0.5)")

        # Identify divergent patterns
        divergent_patterns = []
        risk_scores = list(comparison_metrics['risk_score'].values())
        if max(risk_scores) - min(risk_scores) > 0.3:
            divergent_patterns.append("Risk scores vary significantly across cases")

        # Aggregate recommendations
        aggregate_recs = []
        all_protocols = {}
        for ind_id, report in individual_reports.items():
            for rec in report.recommended_interventions[:3]:
                protocol_name = rec.protocol.name
                if protocol_name not in all_protocols:
                    all_protocols[protocol_name] = {
                        'count': 0,
                        'total_benefit': 0,
                        'individuals': []
                    }
                all_protocols[protocol_name]['count'] += 1
                all_protocols[protocol_name]['total_benefit'] += rec.expected_benefit
                all_protocols[protocol_name]['individuals'].append(ind_id)

        for protocol_name, data in sorted(
            all_protocols.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ):
            aggregate_recs.append({
                'protocol': protocol_name,
                'recommended_for': data['individuals'],
                'count': data['count'],
                'average_benefit': data['total_benefit'] / data['count']
            })

        return ComparisonReport(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            individuals=individual_ids,
            comparison_metrics=comparison_metrics,
            common_patterns=common_patterns,
            divergent_patterns=divergent_patterns,
            aggregate_recommendations=aggregate_recs[:5]
        )

    def _identify_missed_opportunities(
        self,
        trajectory: List[str]
    ) -> List[MissedOpportunity]:
        """Identify missed intervention opportunities in trajectory."""
        missed = []

        for i, state in enumerate(trajectory[:-1]):
            next_state = trajectory[i + 1]

            # Check for critical escalations
            if state in ['Seeking', 'Conferring'] and next_state == 'Directing':
                # This was a missed opportunity
                protocols = list(INTERVENTION_PROTOCOLS.values())
                applicable = [
                    p for p in protocols
                    if state in p.target_states
                ]

                if applicable:
                    best = max(applicable, key=lambda p: p.effectiveness_estimate)
                    missed.append(MissedOpportunity(
                        event_index=i,
                        state_at_time=state,
                        recommended_protocol=best.display_name,
                        estimated_impact=best.effectiveness_estimate,
                        rationale=f"Intervention before {state}→Directing transition "
                                 f"could have disrupted escalation"
                    ))

        return missed[:10]  # Limit to 10

    def _generate_monitoring_plan(
        self,
        risk_level: RiskLevel,
        current_state: str,
        trajectory: List[str]
    ) -> MonitoringPlan:
        """Generate monitoring plan based on risk level."""
        # Frequency based on risk
        frequency_map = {
            RiskLevel.CRITICAL: 'daily',
            RiskLevel.HIGH: 'twice weekly',
            RiskLevel.MODERATE: 'weekly',
            RiskLevel.LOW: 'biweekly'
        }

        reassessment_map = {
            RiskLevel.CRITICAL: '72 hours',
            RiskLevel.HIGH: '1 week',
            RiskLevel.MODERATE: '2 weeks',
            RiskLevel.LOW: '1 month'
        }

        # Key indicators to monitor
        indicators = [
            f"Transitions from {current_state} state",
            "Changes in behavioral patterns",
            "Response to current interventions"
        ]

        if current_state in ['Seeking', 'Conferring']:
            indicators.append("Signs of escalation toward Directing")

        # Escalation triggers
        triggers = [
            f"Entry into Directing state",
            "Rapid state cycling",
            "Non-compliance with interventions"
        ]

        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            triggers.append("Any new critical transition")

        return MonitoringPlan(
            frequency=frequency_map.get(risk_level, 'weekly'),
            key_indicators=indicators,
            escalation_triggers=triggers,
            reassessment_timeline=reassessment_map.get(risk_level, '2 weeks')
        )

    def _compute_confidence(self, trajectory: List[str]) -> float:
        """Compute confidence level based on data quality."""
        # Base confidence
        confidence = 0.5

        # More data = more confidence
        if len(trajectory) > 20:
            confidence += 0.2
        elif len(trajectory) > 10:
            confidence += 0.1

        # Transition matrix quality
        if self.transition_matrix is not None:
            # Check for sparse rows
            for row in self.transition_matrix:
                if np.sum(row > 0) >= 2:
                    confidence += 0.05

        return min(0.95, confidence)

    def _identify_limitations(self, trajectory: List[str]) -> List[str]:
        """Identify limitations of the analysis."""
        limitations = [
            "Model based on behavioral state transitions only",
            "Intervention effectiveness estimates are illustrative",
            "Does not account for all individual factors"
        ]

        if len(trajectory) < 10:
            limitations.append("Limited trajectory data may reduce accuracy")

        if self.transition_matrix is None:
            limitations.append("Using default transition probabilities")

        return limitations


def quick_report(
    trajectory: List[str],
    individual_id: str = "Unknown",
    output_format: str = "markdown"
) -> str:
    """
    Generate a quick clinical report.

    Args:
        trajectory: State trajectory
        individual_id: Identifier
        output_format: 'markdown', 'json', or 'dict'

    Returns:
        Formatted report
    """
    generator = ClinicalReportGenerator()
    report = generator.generate_individual_report(
        individual_id=individual_id,
        trajectory=trajectory
    )

    if output_format == "markdown":
        return report.to_markdown()
    elif output_format == "json":
        return report.to_json()
    else:
        return report.to_dict()
