"""
Intervention Analysis Page

Clinical decision support for intervention planning using causal modeling.

Features:
- Individual trajectory analysis with intervention windows
- Protocol recommendation and comparison
- Counterfactual "what-if" analysis
- Risk assessment and monitoring plans
- Clinical report generation
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.loader import DashboardDataLoader, STATE_NAMES
from config import DATA_DIR, ANIMAL_COLORS, RISK_COLORS
from components.report_export import generate_html_report

# Import intervention module
try:
    from intervention import (
        BehavioralSCM,
        TrajectoryAnalyzer,
        CounterfactualEngine,
        InterventionOptimizer,
        ClinicalReportGenerator,
        INTERVENTION_PROTOCOLS,
        get_protocol,
        RiskLevel
    )
    INTERVENTION_AVAILABLE = True
except ImportError as e:
    INTERVENTION_AVAILABLE = False
    IMPORT_ERROR = str(e)

st.set_page_config(page_title="Intervention Analysis", layout="wide")

# Colors for intervention components
INTERVENTION_COLORS = {
    'window': '#27ae60',
    'critical': '#e74c3c',
    'tipping_point': '#f39c12',
    'recommended': '#3498db'
}

RISK_LEVEL_COLORS = {
    RiskLevel.LOW: '#27ae60' if INTERVENTION_AVAILABLE else '#27ae60',
    RiskLevel.MODERATE: '#f39c12' if INTERVENTION_AVAILABLE else '#f39c12',
    RiskLevel.HIGH: '#e67e22' if INTERVENTION_AVAILABLE else '#e67e22',
    RiskLevel.CRITICAL: '#c0392b' if INTERVENTION_AVAILABLE else '#c0392b'
}


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


def get_default_transition_matrix():
    """Default 4-state transition matrix for 4-Animal model."""
    return np.array([
        [0.40, 0.25, 0.20, 0.15],  # From Seeking
        [0.10, 0.60, 0.15, 0.15],  # From Directing
        [0.15, 0.30, 0.40, 0.15],  # From Conferring
        [0.30, 0.10, 0.20, 0.40]   # From Revising
    ])


def extract_trajectory_from_signature(signature: dict) -> list:
    """Extract or reconstruct trajectory from behavioral signature."""
    if 'trajectory' in signature:
        return signature['trajectory']

    state_dist = signature.get('state_distribution', {})
    if not state_dist:
        return ['Seeking', 'Seeking', 'Directing']

    states = list(state_dist.keys())
    probs = list(state_dist.values())
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]

    n_events = signature.get('total_events', 20)
    trajectory = list(np.random.choice(states, size=n_events, p=probs))

    return trajectory


def get_attr(obj, key, default=None):
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def create_trajectory_timeline(
    trajectory: list,
    critical_transitions: list = None,
    tipping_points: list = None,
    intervention_windows: list = None
):
    """Create interactive trajectory timeline with annotations."""
    fig = go.Figure()

    state_y = {'Seeking': 1, 'Conferring': 2, 'Directing': 3, 'Revising': 0}
    x_vals = list(range(len(trajectory)))
    y_vals = [state_y.get(s, 1) for s in trajectory]

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        name='Trajectory',
        line=dict(color='#34495e', width=2),
        marker=dict(
            size=10,
            color=[ANIMAL_COLORS.get(s, '#95a5a6') for s in trajectory],
            line=dict(width=1, color='white')
        ),
        hovertemplate='Event %{x}<br>State: %{text}<extra></extra>',
        text=trajectory
    ))

    if intervention_windows:
        for iw in intervention_windows:
            start_idx = get_attr(iw, 'start_index', 0)
            end_idx = get_attr(iw, 'end_index', 1)
            fig.add_vrect(
                x0=start_idx - 0.3,
                x1=end_idx + 0.3,
                fillcolor=INTERVENTION_COLORS['window'],
                opacity=0.2,
                layer='below',
                line_width=0,
                annotation_text="Intervention Window",
                annotation_position="top left"
            )

    if critical_transitions:
        for ct in critical_transitions:
            idx = get_attr(ct, 'index', 0)
            to_state = get_attr(ct, 'to_state', 'Directing')
            fig.add_annotation(
                x=idx,
                y=state_y.get(to_state, 2) + 0.3,
                text="!",
                showarrow=False,
                font=dict(size=16, color=INTERVENTION_COLORS['critical']),
                bgcolor=INTERVENTION_COLORS['critical'],
                borderpad=3
            )

    if tipping_points:
        for tp in tipping_points:
            idx = get_attr(tp, 'index', 0)
            p_after = get_attr(tp, 'p_directing_after', 0.5)
            fig.add_vline(
                x=idx,
                line_dash="dash",
                line_color=INTERVENTION_COLORS['tipping_point'],
                annotation_text=f"Tipping ({p_after:.0%})"
            )

    fig.update_layout(
        title="Behavioral Trajectory with Intervention Opportunities",
        xaxis_title="Event Index",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['Revising', 'Seeking', 'Conferring', 'Directing'],
            title='State'
        ),
        height=400,
        showlegend=False,
        hovermode='x unified'
    )

    return fig


def create_protocol_comparison_chart(recommendations: list):
    """Create comparison chart for recommended protocols."""
    if not recommendations:
        return None

    data = []
    for rec in recommendations[:6]:
        data.append({
            'Protocol': rec.protocol.display_name[:25],
            'Expected Benefit': rec.expected_benefit,
            'Cost ($)': rec.cost,
            'Urgency': rec.urgency,
            'Cost-Effectiveness': rec.cost_effectiveness if rec.cost_effectiveness != float('inf') else rec.expected_benefit
        })

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x='Protocol',
        y='Expected Benefit',
        color='Urgency',
        color_continuous_scale='RdYlGn_r',
        title='Protocol Comparison by Expected Benefit',
        hover_data=['Cost ($)', 'Cost-Effectiveness']
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=400
    )

    return fig


def create_pareto_chart(pareto_frontier: list):
    """Create Pareto frontier visualization."""
    if not pareto_frontier or len(pareto_frontier) < 2:
        return None

    costs = [p[0] for p in pareto_frontier]
    benefits = [p[1] for p in pareto_frontier]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=costs,
        y=benefits,
        mode='lines+markers',
        name='Pareto Frontier',
        line=dict(color='#3498db', width=2),
        marker=dict(size=10, color='#3498db'),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)'
    ))

    fig.update_layout(
        title='Cost vs Harm Reduction Trade-off (Pareto Frontier)',
        xaxis_title='Total Cost ($)',
        yaxis_title='Expected Harm Reduction',
        height=350
    )

    return fig


def create_risk_gauge(risk_score: float, risk_level):
    """Create risk level gauge visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': RISK_LEVEL_COLORS.get(risk_level, '#f39c12')},
            'steps': [
                {'range': [0, 25], 'color': '#d5f4e6'},
                {'range': [25, 50], 'color': '#fdeaa8'},
                {'range': [50, 75], 'color': '#f8c471'},
                {'range': [75, 100], 'color': '#f5b7b1'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))

    fig.update_layout(height=250)
    return fig


data = load_data()
audience = st.session_state.get('audience_mode', 'researcher')

st.title("Clinical Intervention Analysis")

if not INTERVENTION_AVAILABLE:
    st.error(f"Intervention module not available: {IMPORT_ERROR}")
    st.info("Please ensure the intervention module is properly installed.")
    st.stop()

st.markdown("Causal modeling for optimal intervention planning")

st.divider()

# Mode selection
analysis_mode = st.radio(
    "Analysis Mode",
    ["Individual Analysis", "Protocol Explorer", "Population Overview"],
    horizontal=True
)

st.divider()

if analysis_mode == "Individual Analysis":
    # Get individuals with trajectory data available
    individuals_with_data = data.get_individuals_with_trajectory_data()

    if not individuals_with_data:
        # Fall back to all individuals
        individuals_with_data = data.get_all_individuals()
        st.warning("Using default data - no trajectory data found")

    if not individuals_with_data:
        st.warning("No individual data available.")
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col1:
        selected = st.selectbox(
            "Select Individual",
            options=individuals_with_data,
            format_func=lambda x: f"{x.split('_')[0]} {x.split('_')[1]}" if '_' in x else x
        )

    with col2:
        include_retrospective = st.checkbox("Include Retrospective Analysis", value=True)

    if selected:
        # Load real individual data
        ind_data = data.get_individual_data(selected)
        signature = ind_data.get('signature', {})
        classification = ind_data.get('classification', {})

        # Try to load real trajectory and transition matrix
        try:
            trajectory = data.load_individual_trajectory(selected)
            transition_matrix = data.load_individual_transition_matrix(selected)
            data_source = "actual"
        except (ValueError, KeyError) as e:
            # Fall back to signature-based data
            trajectory = extract_trajectory_from_signature(signature)
            transition_matrix = get_default_transition_matrix()
            data_source = "estimated"
            st.info(f"Using estimated data (real data unavailable: {e})")

        state_names = STATE_NAMES

        try:
            scm = BehavioralSCM(transition_matrix, state_names)
            analyzer = TrajectoryAnalyzer(transition_matrix=transition_matrix)
            optimizer = InterventionOptimizer(scm)
            report_gen = ClinicalReportGenerator(transition_matrix, state_names)

            analysis = analyzer.comprehensive_analysis(trajectory)
            critical_transitions = analysis.get('critical_transitions', [])
            tipping_points = analysis.get('tipping_points', [])
            intervention_windows = analysis.get('intervention_windows', [])

            report = report_gen.generate_individual_report(
                individual_id=selected,
                trajectory=trajectory,
                archetype=classification.get('subtype', 'Unknown'),
                include_retrospective=include_retrospective
            )

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if report.risk_assessment:
                    risk_level = report.risk_assessment.level
                    risk_color = RISK_LEVEL_COLORS.get(risk_level, '#f39c12')
                    st.markdown(f"""
                    <div style="background-color: {risk_color}; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: white; margin: 0;">Risk Level: {risk_level.value}</h3>
                        <p style="color: white; margin: 5px 0;">Score: {report.risk_assessment.score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("Risk Level", "Unknown")

            with col2:
                st.metric("Current State", trajectory[-1] if trajectory else "Unknown")
                st.metric("MFPT to Directing", f"{report.mfpt_to_directing:.1f} events")

            with col3:
                st.metric("Critical Transitions", len(critical_transitions))
                st.metric("Intervention Windows", len(intervention_windows))

            st.divider()

            tab1, tab2, tab3, tab4 = st.tabs([
                "Trajectory View",
                "Intervention Recommendations",
                "Retrospective Analysis",
                "Clinical Report"
            ])

            with tab1:
                # Show data source indicator
                if data_source == "actual":
                    st.success(f"Using actual trajectory data ({len(trajectory)} events)")
                else:
                    st.warning("Using estimated trajectory data")

                fig_timeline = create_trajectory_timeline(
                    trajectory=trajectory,
                    critical_transitions=critical_transitions,
                    tipping_points=tipping_points,
                    intervention_windows=intervention_windows
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

                st.subheader("Analysis Details")
                col1, col2 = st.columns(2)

                with col1:
                    if critical_transitions:
                        st.markdown("**Critical Transitions**")
                        for ct in critical_transitions[:5]:
                            idx = get_attr(ct, 'index', 0)
                            from_s = get_attr(ct, 'from_state', '?')
                            to_s = get_attr(ct, 'to_state', '?')
                            name = get_attr(ct, 'name', 'Transition')
                            st.markdown(f"- Event {idx}: {from_s} -> {to_s} ({name})")
                    else:
                        st.info("No critical transitions detected")

                with col2:
                    if intervention_windows:
                        st.markdown("**Intervention Windows**")
                        for iw in intervention_windows[:3]:
                            start_idx = get_attr(iw, 'start_index', 0)
                            end_idx = get_attr(iw, 'end_index', 1)
                            state = get_attr(iw, 'state_at_window', '?')
                            urgency = get_attr(iw, 'urgency', 0.5)
                            st.markdown(
                                f"- Events {start_idx}-{end_idx}: "
                                f"{state} state (Urgency: {urgency:.0%})"
                            )
                    else:
                        st.info("No intervention windows identified")

                # Show event details if real data is available
                if data_source == "actual":
                    st.divider()
                    st.subheader("Event Details")
                    try:
                        events = data.get_individual_events(selected)
                        # Event timeline with expandable details
                        event_idx = st.slider(
                            "Select event to view details",
                            0, len(events) - 1, 0
                        )
                        event = events[event_idx]
                        st.markdown(f"**Event {event_idx}:** {event.get('state', 'Unknown')}")
                        st.markdown(f"**Confidence:** {event.get('confidence', 'N/A')}")
                        with st.expander("Event Description"):
                            st.write(event.get('event', 'No description'))
                            if event.get('reasoning'):
                                st.caption(f"Reasoning: {event.get('reasoning', '')[:300]}...")
                    except Exception as e:
                        st.caption(f"Event details not available: {e}")

            with tab2:
                st.subheader("Recommended Interventions")

                max_budget = st.slider(
                    "Budget Constraint ($)",
                    min_value=1000,
                    max_value=50000,
                    value=20000,
                    step=1000
                )

                opt_result = optimizer.find_optimal_timing(
                    trajectory=trajectory,
                    budget_constraint=max_budget,
                    max_interventions=5
                )

                if opt_result.recommendations:
                    fig_protocols = create_protocol_comparison_chart(opt_result.recommendations)
                    if fig_protocols:
                        st.plotly_chart(fig_protocols, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Expected Benefit", f"{opt_result.total_expected_benefit:.1%}")
                    with col2:
                        st.metric("Total Cost", f"${opt_result.total_cost:,.0f}")

                    st.markdown("### Detailed Recommendations")
                    for i, rec in enumerate(opt_result.recommendations[:5], 1):
                        with st.expander(f"{i}. {rec.protocol.display_name}", expanded=(i == 1)):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Expected Benefit", f"{rec.expected_benefit:.1%}")
                            with col2:
                                st.metric("Cost", f"${rec.cost:,.0f}")
                            with col3:
                                st.metric("Urgency", f"{rec.urgency:.1%}")

                            st.markdown(f"**Timing:** Event {rec.time_index}")
                            st.markdown(f"**Rationale:** {rec.rationale}")

                            if hasattr(rec.protocol, 'description'):
                                st.caption(rec.protocol.description[:200] + "...")

                    if opt_result.pareto_frontier:
                        fig_pareto = create_pareto_chart(opt_result.pareto_frontier)
                        if fig_pareto:
                            st.plotly_chart(fig_pareto, use_container_width=True)
                else:
                    st.info("No intervention recommendations within budget constraint")

            with tab3:
                if include_retrospective and report.missed_opportunities:
                    st.subheader("Missed Intervention Opportunities")

                    for mo in report.missed_opportunities:
                        with st.expander(f"Event {mo.event_index}: {mo.state_at_time} state"):
                            st.markdown(f"**Recommended Protocol:** {mo.recommended_protocol}")
                            st.markdown(f"**Estimated Impact:** {mo.estimated_impact:.1%} harm reduction")
                            st.markdown(f"**Analysis:** {mo.rationale}")
                else:
                    st.info("Enable retrospective analysis to see missed opportunities")

                st.subheader("Counterfactual Simulation")
                st.markdown("*What if we had intervened at a different point in the trajectory?*")

                if len(trajectory) > 5:
                    col1, col2 = st.columns(2)

                    with col1:
                        intervention_point = st.slider(
                            "Select intervention point in trajectory",
                            min_value=0,
                            max_value=len(trajectory) - 1,
                            value=len(trajectory) // 3,
                            help="Choose when in the trajectory to apply the intervention"
                        )
                        # Show context around intervention point
                        context_start = max(0, intervention_point - 2)
                        context_end = min(len(trajectory), intervention_point + 3)
                        context = trajectory[context_start:context_end]
                        st.caption(f"Context: ...{' → '.join(context)}...")

                    with col2:
                        available_protocols = list(INTERVENTION_PROTOCOLS.keys())
                        selected_protocol = st.selectbox(
                            "Select intervention protocol",
                            options=available_protocols,
                            format_func=lambda x: INTERVENTION_PROTOCOLS[x].display_name
                        )
                        # Show protocol details
                        protocol = INTERVENTION_PROTOCOLS[selected_protocol]
                        st.caption(f"Effectiveness: {protocol.effectiveness_estimate:.0%} | Cost: ${protocol.cost_per_week * protocol.duration_weeks:,}")

                    # Show event at intervention point
                    st.markdown("---")
                    st.markdown(f"**Intervention at Event {intervention_point}:** State = `{trajectory[intervention_point]}`")

                    # Try to show actual event description if available
                    if data_source == "actual":
                        try:
                            events = data.get_individual_events(selected)
                            if intervention_point < len(events):
                                event = events[intervention_point]
                                with st.expander("View actual event at this point"):
                                    st.write(event.get('event', 'No description')[:300])
                        except Exception:
                            pass

                    if st.button("Run Counterfactual Simulation", type="primary"):
                        with st.spinner("Simulating counterfactual..."):
                            try:
                                cf_engine = CounterfactualEngine(scm)
                                result = cf_engine.counterfactual_query(
                                    observed_trajectory=trajectory,
                                    intervention_name=selected_protocol,
                                    intervention_time=intervention_point
                                )

                                # Display results in a more visual way
                                st.markdown("### Simulation Results")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Actual P(Directing)",
                                        f"{result.observed_p_harm:.1%}",
                                        help="Probability of reaching harmful state in actual trajectory"
                                    )
                                with col2:
                                    st.metric(
                                        "Counterfactual P(Directing)",
                                        f"{result.counterfactual_p_harm:.1%}",
                                        delta=f"-{(result.observed_p_harm - result.counterfactual_p_harm):.1%}",
                                        delta_color="inverse",
                                        help="Probability with intervention applied"
                                    )
                                with col3:
                                    st.metric(
                                        "Harm Reduction",
                                        f"{result.harm_reduction_estimate:.1%}",
                                        help="Expected reduction in harmful outcomes"
                                    )

                                # Visual comparison
                                comparison_data = pd.DataFrame({
                                    'Scenario': ['Actual', 'With Intervention'],
                                    'P(Directing)': [result.observed_p_harm, result.counterfactual_p_harm]
                                })
                                fig_cf = px.bar(
                                    comparison_data,
                                    x='Scenario',
                                    y='P(Directing)',
                                    color='Scenario',
                                    color_discrete_map={'Actual': '#e74c3c', 'With Intervention': '#27ae60'},
                                    title='Actual vs Counterfactual Outcome'
                                )
                                fig_cf.update_layout(height=300, showlegend=False)
                                st.plotly_chart(fig_cf, use_container_width=True)

                                # Interpretation
                                if result.harm_reduction_estimate > 0.2:
                                    st.success(f"**High Impact:** Intervention at this point would substantially reduce harm ({result.harm_reduction_estimate:.1%})")
                                elif result.harm_reduction_estimate > 0.1:
                                    st.info(f"**Moderate Impact:** Intervention would moderately reduce harm ({result.harm_reduction_estimate:.1%})")
                                else:
                                    st.warning(f"**Low Impact:** Intervention at this point may have limited effect ({result.harm_reduction_estimate:.1%})")

                            except Exception as e:
                                st.error(f"Simulation error: {str(e)}")
                else:
                    st.info("Need at least 5 events in trajectory for counterfactual simulation")

            with tab4:
                st.subheader("Clinical Report")

                report_format = st.radio(
                    "Format",
                    ["Preview", "HTML", "Markdown", "JSON"],
                    horizontal=True
                )

                if report_format == "Preview":
                    if report.risk_assessment:
                        st.markdown(f"### Risk Assessment: {report.risk_assessment.level.value}")
                        for indicator in report.risk_assessment.trajectory_indicators:
                            st.markdown(f"- {indicator}")

                    if report.monitoring_plan:
                        st.markdown("### Monitoring Plan")
                        st.markdown(f"**Frequency:** {report.monitoring_plan.frequency}")
                        st.markdown(f"**Reassessment:** {report.monitoring_plan.reassessment_timeline}")

                        st.markdown("**Key Indicators:**")
                        for ind in report.monitoring_plan.key_indicators:
                            st.markdown(f"- {ind}")

                        st.markdown("**Escalation Triggers:**")
                        for trigger in report.monitoring_plan.escalation_triggers:
                            st.warning(trigger)

                    # Add recommended interventions preview
                    if report.recommended_interventions:
                        st.markdown("### Top Recommended Interventions")
                        for i, rec in enumerate(report.recommended_interventions[:3], 1):
                            st.markdown(f"{i}. **{rec.protocol.display_name}** - "
                                       f"Expected benefit: {rec.expected_benefit:.1%}")

                elif report_format == "HTML":
                    # Generate HTML report with styling
                    html_report = generate_html_report(
                        individual_name=selected,
                        report=report,
                        trajectory=trajectory,
                        classification=classification,
                        data_source=data_source
                    )
                    st.download_button(
                        "Download HTML Report",
                        html_report,
                        file_name=f"clinical_report_{selected}.html",
                        mime="text/html",
                        type="primary"
                    )
                    st.info("Click above to download the formatted HTML report.")
                    with st.expander("Preview HTML"):
                        st.components.v1.html(html_report, height=600, scrolling=True)

                elif report_format == "Markdown":
                    markdown_report = report.to_markdown()
                    st.code(markdown_report, language="markdown")
                    st.download_button(
                        "Download Report",
                        markdown_report,
                        file_name=f"clinical_report_{selected}.md",
                        mime="text/markdown"
                    )

                else:
                    json_report = report.to_json()
                    st.code(json_report, language="json")
                    st.download_button(
                        "Download Report",
                        json_report,
                        file_name=f"clinical_report_{selected}.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            st.exception(e)

elif analysis_mode == "Protocol Explorer":
    st.subheader("Intervention Protocol Library")

    col1, col2 = st.columns([1, 2])

    with col1:
        protocol_names = list(INTERVENTION_PROTOCOLS.keys())
        selected_protocol = st.selectbox(
            "Select Protocol",
            options=protocol_names,
            format_func=lambda x: INTERVENTION_PROTOCOLS[x].display_name
        )

    protocol = INTERVENTION_PROTOCOLS[selected_protocol]

    with col2:
        category_color = {
            'therapeutic': '#3498db',
            'supervision': '#e74c3c',
            'environmental': '#27ae60',
            'pharmacological': '#9b59b6',
            'combined': '#f39c12'
        }
        cat_val = protocol.category.value if hasattr(protocol.category, 'value') else str(protocol.category)
        color = category_color.get(cat_val, '#95a5a6')

        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-bottom: 10px;">
            <span style="background-color: {color}; color: white; padding: 5px 15px; border-radius: 20px;">
                {cat_val.title()}
            </span>
            <span style="background-color: #95a5a6; color: white; padding: 5px 15px; border-radius: 20px;">
                Evidence: {protocol.evidence_level.value if hasattr(protocol.evidence_level, 'value') else protocol.evidence_level}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Effectiveness", f"{protocol.effectiveness_estimate:.0%}")
    with col2:
        st.metric("Duration", f"{protocol.duration_weeks} weeks")
    with col3:
        st.metric("Cost/Week", f"${protocol.cost_per_week:,.0f}")
    with col4:
        st.metric("Total Cost", f"${protocol.cost_per_week * protocol.duration_weeks:,.0f}")

    st.markdown(f"**Description:** {protocol.description}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Target States")
        for state in protocol.target_states:
            st.markdown(f"- {state}")

        st.markdown("### Target Transitions")
        for from_s, to_s in protocol.target_transitions:
            effect = protocol.effect_on_transitions.get((from_s, to_s), 0)
            direction = "reduces" if effect < 0 else "increases"
            st.markdown(f"- {from_s} -> {to_s}: {direction} by {abs(effect):.0%}")

    with col2:
        st.markdown("### Key Components")
        for component in protocol.key_components:
            st.markdown(f"- {component}")

        if protocol.contraindications:
            st.markdown("### Contraindications")
            for contra in protocol.contraindications:
                st.warning(contra)

    st.divider()

    st.subheader("Compare All Protocols")

    comparison_data = []
    for name, p in INTERVENTION_PROTOCOLS.items():
        comparison_data.append({
            'Protocol': p.display_name,
            'Category': p.category.value if hasattr(p.category, 'value') else str(p.category),
            'Effectiveness': p.effectiveness_estimate,
            'Duration (weeks)': p.duration_weeks,
            'Cost': p.cost_per_week * p.duration_weeks,
            'Evidence': p.evidence_level.value if hasattr(p.evidence_level, 'value') else str(p.evidence_level)
        })

    df = pd.DataFrame(comparison_data)

    fig = px.scatter(
        df,
        x='Cost',
        y='Effectiveness',
        color='Category',
        size='Duration (weeks)',
        hover_name='Protocol',
        title='Protocol Comparison: Effectiveness vs Cost'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.sort_values('Effectiveness', ascending=False), use_container_width=True)

else:
    st.subheader("Population Intervention Overview")

    # Prefer individuals with trajectory data
    individuals_with_data = data.get_individuals_with_trajectory_data()
    if individuals_with_data:
        all_individuals = individuals_with_data
        using_real_data = True
    else:
        all_individuals = data.get_all_individuals()
        using_real_data = False

    if not all_individuals:
        st.warning("No data available")
        st.stop()

    if using_real_data:
        st.success(f"Analyzing {len(all_individuals)} individuals with real trajectory data")
    else:
        st.info(f"Analyzing {len(all_individuals)} individuals with estimated data")

    sample_size = min(len(all_individuals), len(all_individuals))  # Use all available
    sample = all_individuals[:sample_size]  # Don't randomize to be deterministic

    population_stats = {
        'risk_levels': {'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0},
        'intervention_opportunities': 0,
        'avg_mfpt': [],
        'recommendations_by_protocol': {},
        'state_distributions': []
    }

    with st.spinner("Analyzing population..."):
        for ind_id in sample:
            try:
                # Try to use real data first
                try:
                    trajectory = data.load_individual_trajectory(ind_id)
                    transition_matrix = data.load_individual_transition_matrix(ind_id)
                except (ValueError, KeyError):
                    ind_data = data.get_individual_data(ind_id)
                    signature = ind_data.get('signature', {})
                    trajectory = extract_trajectory_from_signature(signature)
                    transition_matrix = get_default_transition_matrix()

                report_gen = ClinicalReportGenerator(transition_matrix, STATE_NAMES)

                report = report_gen.generate_individual_report(
                    individual_id=ind_id,
                    trajectory=trajectory,
                    include_retrospective=False
                )

                if report.risk_assessment:
                    level = report.risk_assessment.level.value
                    population_stats['risk_levels'][level] = population_stats['risk_levels'].get(level, 0) + 1

                population_stats['intervention_opportunities'] += len(report.intervention_windows)
                population_stats['avg_mfpt'].append(report.mfpt_to_directing)

                for rec in report.recommended_interventions[:2]:
                    p_name = rec.protocol.display_name
                    if p_name not in population_stats['recommendations_by_protocol']:
                        population_stats['recommendations_by_protocol'][p_name] = 0
                    population_stats['recommendations_by_protocol'][p_name] += 1

            except Exception:
                continue

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg MFPT to Directing", f"{np.mean(population_stats['avg_mfpt']):.1f} events")
    with col2:
        st.metric("Total Intervention Windows", population_stats['intervention_opportunities'])
    with col3:
        critical_count = population_stats['risk_levels'].get('Critical', 0)
        st.metric("Critical Risk Cases", f"{critical_count}/{sample_size}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        risk_data = pd.DataFrame([
            {'Level': k, 'Count': v}
            for k, v in population_stats['risk_levels'].items()
        ])

        fig_risk = px.pie(
            risk_data,
            values='Count',
            names='Level',
            title='Risk Level Distribution',
            color='Level',
            color_discrete_map={
                'Low': '#27ae60',
                'Moderate': '#f39c12',
                'High': '#e67e22',
                'Critical': '#c0392b'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        protocol_data = pd.DataFrame([
            {'Protocol': k, 'Recommendations': v}
            for k, v in sorted(
                population_stats['recommendations_by_protocol'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:8]
        ])

        if not protocol_data.empty:
            fig_protocols = px.bar(
                protocol_data,
                x='Protocol',
                y='Recommendations',
                title='Most Recommended Protocols'
            )
            fig_protocols.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_protocols, use_container_width=True)

st.divider()

st.caption(
    "This analysis is for clinical decision support only. "
    "All recommendations require professional clinical judgment."
)
