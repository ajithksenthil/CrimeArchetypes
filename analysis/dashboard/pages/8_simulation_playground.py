"""
Simulation Playground

Interactive tools for behavioral trajectory simulation and comparison.

Features:
- Individual comparison (side-by-side analysis)
- What-if scenario simulation
- Custom intervention protocol design
- Trajectory prediction
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.loader import DashboardDataLoader, STATE_NAMES
from config import DATA_DIR, ANIMAL_COLORS, RISK_COLORS

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
    from intervention.counterfactual import InterventionScenario
    INTERVENTION_AVAILABLE = True
except ImportError as e:
    INTERVENTION_AVAILABLE = False
    IMPORT_ERROR = str(e)

st.set_page_config(page_title="Simulation Playground", layout="wide")

STATE_COLORS = ANIMAL_COLORS


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


def create_comparison_chart(traj1: list, traj2: list, name1: str, name2: str):
    """Create side-by-side trajectory comparison visualization."""
    state_y = {'Seeking': 1, 'Conferring': 2, 'Directing': 3, 'Revising': 0}

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"{name1} Trajectory", f"{name2} Trajectory"],
        vertical_spacing=0.12
    )

    # First trajectory
    x1 = list(range(len(traj1)))
    y1 = [state_y.get(s, 1) for s in traj1]
    colors1 = [STATE_COLORS.get(s, '#95a5a6') for s in traj1]

    fig.add_trace(
        go.Scatter(
            x=x1, y=y1, mode='lines+markers',
            name=name1, line=dict(color='#34495e', width=2),
            marker=dict(size=8, color=colors1, line=dict(width=1, color='white')),
            hovertemplate='Event %{x}<br>State: %{text}<extra></extra>',
            text=traj1
        ),
        row=1, col=1
    )

    # Second trajectory
    x2 = list(range(len(traj2)))
    y2 = [state_y.get(s, 1) for s in traj2]
    colors2 = [STATE_COLORS.get(s, '#95a5a6') for s in traj2]

    fig.add_trace(
        go.Scatter(
            x=x2, y=y2, mode='lines+markers',
            name=name2, line=dict(color='#34495e', width=2),
            marker=dict(size=8, color=colors2, line=dict(width=1, color='white')),
            hovertemplate='Event %{x}<br>State: %{text}<extra></extra>',
            text=traj2
        ),
        row=2, col=1
    )

    # Update axes
    for i in [1, 2]:
        fig.update_yaxes(
            tickmode='array', tickvals=[0, 1, 2, 3],
            ticktext=['Revising', 'Seeking', 'Conferring', 'Directing'],
            row=i, col=1
        )
        fig.update_xaxes(title_text='Event Index', row=i, col=1)

    fig.update_layout(height=600, showlegend=False)
    return fig


def create_state_distribution_comparison(dist1: dict, dist2: dict, name1: str, name2: str):
    """Create bar chart comparing state distributions."""
    states = list(STATE_NAMES)

    fig = go.Figure()

    # First individual
    vals1 = [dist1.get(s, 0) * 100 for s in states]
    fig.add_trace(go.Bar(
        name=name1[:20],
        x=states,
        y=vals1,
        marker_color='#3498db',
        text=[f'{v:.1f}%' for v in vals1],
        textposition='auto'
    ))

    # Second individual
    vals2 = [dist2.get(s, 0) * 100 for s in states]
    fig.add_trace(go.Bar(
        name=name2[:20],
        x=states,
        y=vals2,
        marker_color='#e74c3c',
        text=[f'{v:.1f}%' for v in vals2],
        textposition='auto'
    ))

    fig.update_layout(
        title='State Distribution Comparison',
        xaxis_title='Behavioral State',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=400
    )

    return fig


def simulate_trajectory(transition_matrix: np.ndarray, initial_state: str,
                        n_steps: int, intervention_time: int = None,
                        intervention_effect: dict = None) -> list:
    """Simulate a trajectory from a transition matrix."""
    state_to_idx = {s: i for i, s in enumerate(STATE_NAMES)}
    idx_to_state = {i: s for i, s in enumerate(STATE_NAMES)}

    current_idx = state_to_idx.get(initial_state, 0)
    trajectory = [initial_state]

    matrix = transition_matrix.copy()

    for step in range(1, n_steps):
        # Apply intervention at specified time
        if intervention_time and step == intervention_time and intervention_effect:
            matrix = apply_intervention_to_matrix(matrix, intervention_effect)

        # Sample next state
        probs = matrix[current_idx]
        probs = probs / probs.sum()  # Normalize
        next_idx = np.random.choice(4, p=probs)
        trajectory.append(idx_to_state[next_idx])
        current_idx = next_idx

    return trajectory


def apply_intervention_to_matrix(matrix: np.ndarray, effect: dict) -> np.ndarray:
    """Apply intervention effect to transition matrix."""
    modified = matrix.copy()
    state_to_idx = {s: i for i, s in enumerate(STATE_NAMES)}

    # Reduce directing transitions
    directing_idx = state_to_idx.get('Directing', 1)
    reduce_factor = effect.get('reduce_directing', 0.3)

    for i in range(4):
        if i != directing_idx:
            # Reduce probability of transitioning to Directing
            reduction = modified[i, directing_idx] * reduce_factor
            modified[i, directing_idx] -= reduction
            # Distribute to other states proportionally
            for j in range(4):
                if j != directing_idx:
                    modified[i, j] += reduction / 3

    # Ensure valid probabilities
    modified = np.clip(modified, 0, 1)
    for i in range(4):
        modified[i] = modified[i] / modified[i].sum()

    return modified


def create_simulation_chart(original: list, simulated: list,
                            intervention_time: int = None):
    """Create chart comparing original and simulated trajectories."""
    state_y = {'Seeking': 1, 'Conferring': 2, 'Directing': 3, 'Revising': 0}

    fig = go.Figure()

    # Original trajectory
    x1 = list(range(len(original)))
    y1 = [state_y.get(s, 1) for s in original]

    fig.add_trace(go.Scatter(
        x=x1, y=y1, mode='lines+markers',
        name='Original', line=dict(color='#95a5a6', width=2, dash='dot'),
        marker=dict(size=6, opacity=0.6)
    ))

    # Simulated trajectory
    x2 = list(range(len(simulated)))
    y2 = [state_y.get(s, 1) for s in simulated]
    colors = [STATE_COLORS.get(s, '#95a5a6') for s in simulated]

    fig.add_trace(go.Scatter(
        x=x2, y=y2, mode='lines+markers',
        name='Simulated', line=dict(color='#3498db', width=2),
        marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
        hovertemplate='Event %{x}<br>State: %{text}<extra></extra>',
        text=simulated
    ))

    # Mark intervention point
    if intervention_time:
        fig.add_vline(
            x=intervention_time, line_dash="dash", line_color="#27ae60",
            annotation_text="Intervention", annotation_position="top right"
        )

    fig.update_layout(
        title='Trajectory Simulation',
        xaxis_title='Event Index',
        yaxis=dict(
            tickmode='array', tickvals=[0, 1, 2, 3],
            ticktext=['Revising', 'Seeking', 'Conferring', 'Directing']
        ),
        height=400,
        hovermode='x unified'
    )

    return fig


def compute_state_distribution(trajectory: list) -> dict:
    """Compute state distribution from trajectory."""
    from collections import Counter
    counts = Counter(trajectory)
    total = len(trajectory)
    return {state: counts.get(state, 0) / total for state in STATE_NAMES}


# Main page content
st.title("Simulation Playground")
st.markdown("""
Interactive tools for exploring behavioral trajectories, comparing individuals,
and testing intervention scenarios.
""")

if not INTERVENTION_AVAILABLE:
    st.error(f"Intervention module not available: {IMPORT_ERROR}")
    st.stop()

data = load_data()
individuals_with_data = data.get_individuals_with_trajectory_data()

if not individuals_with_data:
    st.warning("No individuals with trajectory data available.")
    st.info("Trajectory data is required for simulation features.")
    st.stop()

# Create main tabs
tab1, tab2, tab3 = st.tabs([
    "Individual Comparison",
    "What-If Simulation",
    "Population Explorer"
])

# ============= TAB 1: Individual Comparison =============
with tab1:
    st.header("Compare Two Individuals")
    st.markdown("Select two individuals to compare their behavioral trajectories side by side.")

    col1, col2 = st.columns(2)

    with col1:
        individual1 = st.selectbox(
            "Select First Individual",
            individuals_with_data,
            key='compare_ind1'
        )

    with col2:
        # Default to second person
        default_idx = min(1, len(individuals_with_data) - 1)
        individual2 = st.selectbox(
            "Select Second Individual",
            individuals_with_data,
            index=default_idx,
            key='compare_ind2'
        )

    if individual1 and individual2:
        # Load trajectories
        traj1 = data.load_individual_trajectory(individual1)
        traj2 = data.load_individual_trajectory(individual2)

        ind_data1 = data.get_individual_data(individual1)
        ind_data2 = data.get_individual_data(individual2)

        class1 = ind_data1.get('classification', {})
        class2 = ind_data2.get('classification', {})

        # Display classification comparison
        st.subheader("Classification Comparison")
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown(f"**{individual1.replace('_', ' ')}**")
            st.info(f"Type: {class1.get('primary_type', 'Unknown')}")
            st.info(f"Subtype: {class1.get('subtype', 'Unknown')}")
            st.metric("Events", len(traj1))

        with comp_col2:
            st.markdown(f"**{individual2.replace('_', ' ')}**")
            st.info(f"Type: {class2.get('primary_type', 'Unknown')}")
            st.info(f"Subtype: {class2.get('subtype', 'Unknown')}")
            st.metric("Events", len(traj2))

        st.divider()

        # Trajectory comparison
        st.subheader("Trajectory Comparison")
        fig_comparison = create_comparison_chart(traj1, traj2, individual1, individual2)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # State distribution comparison
        st.subheader("State Distribution Comparison")
        dist1 = compute_state_distribution(traj1)
        dist2 = compute_state_distribution(traj2)

        fig_dist = create_state_distribution_comparison(dist1, dist2, individual1, individual2)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Metrics comparison
        st.subheader("Key Metrics Comparison")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            dir1 = dist1.get('Directing', 0) * 100
            dir2 = dist2.get('Directing', 0) * 100
            st.metric(
                "Directing %",
                f"{dir1:.1f}% vs {dir2:.1f}%",
                delta=f"{dir1 - dir2:+.1f}%" if dir1 != dir2 else None
            )

        with metric_col2:
            seek1 = dist1.get('Seeking', 0) * 100
            seek2 = dist2.get('Seeking', 0) * 100
            st.metric(
                "Seeking %",
                f"{seek1:.1f}% vs {seek2:.1f}%",
                delta=f"{seek1 - seek2:+.1f}%" if seek1 != seek2 else None
            )

        with metric_col3:
            conf1 = dist1.get('Conferring', 0) * 100
            conf2 = dist2.get('Conferring', 0) * 100
            st.metric(
                "Conferring %",
                f"{conf1:.1f}% vs {conf2:.1f}%",
                delta=f"{conf1 - conf2:+.1f}%" if conf1 != conf2 else None,
                delta_color="normal"
            )

        with metric_col4:
            rev1 = dist1.get('Revising', 0) * 100
            rev2 = dist2.get('Revising', 0) * 100
            st.metric(
                "Revising %",
                f"{rev1:.1f}% vs {rev2:.1f}%",
                delta=f"{rev1 - rev2:+.1f}%" if rev1 != rev2 else None
            )


# ============= TAB 2: What-If Simulation =============
with tab2:
    st.header("What-If Scenario Simulation")
    st.markdown("""
    Explore how different interventions might have affected an individual's trajectory.
    This simulation uses the transition matrix to model potential outcomes.
    """)

    # Select individual
    sim_individual = st.selectbox(
        "Select Individual for Simulation",
        individuals_with_data,
        key='sim_individual'
    )

    if sim_individual:
        # Load data
        actual_trajectory = data.load_individual_trajectory(sim_individual)
        transition_matrix = data.load_individual_transition_matrix(sim_individual)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Simulation Parameters")

            # Intervention timing
            intervention_time = st.slider(
                "Intervention at Event #",
                min_value=1,
                max_value=min(len(actual_trajectory) - 1, 50),
                value=min(10, len(actual_trajectory) // 3),
                help="When to apply the intervention in the trajectory"
            )

            # Intervention strength
            intervention_strength = st.slider(
                "Intervention Strength",
                min_value=0.1,
                max_value=0.8,
                value=0.3,
                step=0.1,
                help="How much to reduce transitions to Directing state"
            )

            # Number of simulations
            n_simulations = st.slider(
                "Number of Monte Carlo Simulations",
                min_value=10,
                max_value=100,
                value=30,
                step=10
            )

            run_simulation = st.button("Run Simulation", type="primary")

        with col2:
            if run_simulation:
                with st.spinner("Running simulations..."):
                    intervention_effect = {'reduce_directing': intervention_strength}

                    # Run multiple simulations
                    simulated_outcomes = []
                    initial_state = actual_trajectory[0]

                    for _ in range(n_simulations):
                        sim_traj = simulate_trajectory(
                            transition_matrix,
                            initial_state,
                            len(actual_trajectory),
                            intervention_time,
                            intervention_effect
                        )
                        simulated_outcomes.append(sim_traj)

                    # Compute statistics
                    original_directing = sum(1 for s in actual_trajectory if s == 'Directing') / len(actual_trajectory)

                    simulated_directing_rates = [
                        sum(1 for s in traj if s == 'Directing') / len(traj)
                        for traj in simulated_outcomes
                    ]

                    avg_simulated_directing = np.mean(simulated_directing_rates)
                    std_simulated_directing = np.std(simulated_directing_rates)

                    # Show one sample simulation
                    sample_sim = simulated_outcomes[0]

                    st.subheader("Simulation Results")

                    # Show comparison chart
                    fig_sim = create_simulation_chart(actual_trajectory, sample_sim, intervention_time)
                    st.plotly_chart(fig_sim, use_container_width=True)

                    # Show metrics
                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.metric(
                            "Original Directing %",
                            f"{original_directing * 100:.1f}%"
                        )

                    with result_col2:
                        st.metric(
                            "Avg Simulated Directing %",
                            f"{avg_simulated_directing * 100:.1f}%",
                            delta=f"{(avg_simulated_directing - original_directing) * 100:+.1f}%",
                            delta_color="inverse"
                        )

                    with result_col3:
                        reduction = (original_directing - avg_simulated_directing) / original_directing * 100 if original_directing > 0 else 0
                        st.metric(
                            "Potential Harm Reduction",
                            f"{reduction:.1f}%"
                        )

                    # Show distribution of outcomes
                    st.subheader("Distribution of Simulation Outcomes")

                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=[r * 100 for r in simulated_directing_rates],
                        nbinsx=20,
                        marker_color='#3498db'
                    ))
                    fig_hist.add_vline(
                        x=original_directing * 100,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Original"
                    )
                    fig_hist.update_layout(
                        title=f'Directing State % Across {n_simulations} Simulations',
                        xaxis_title='Directing State %',
                        yaxis_title='Frequency',
                        height=300
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.info(f"""
                    **Interpretation:** With intervention at event {intervention_time} with
                    {intervention_strength:.0%} strength, the directing state percentage
                    could potentially be reduced from {original_directing * 100:.1f}% to
                    {avg_simulated_directing * 100:.1f}% (±{std_simulated_directing * 100:.1f}%).
                    """)
            else:
                st.info("Adjust parameters and click 'Run Simulation' to see results.")


# ============= TAB 3: Population Explorer =============
with tab3:
    st.header("Population Explorer")
    st.markdown("Explore aggregate patterns across all individuals with trajectory data.")

    # Compute population statistics
    all_distributions = []
    all_classifications = []

    for ind_name in individuals_with_data:
        try:
            traj = data.load_individual_trajectory(ind_name)
            ind_data = data.get_individual_data(ind_name)
            classification = ind_data.get('classification', {})

            dist = compute_state_distribution(traj)
            all_distributions.append({
                'name': ind_name,
                'classification': classification.get('primary_type', 'Unknown'),
                'subtype': classification.get('subtype', 'Unknown'),
                **dist
            })
            all_classifications.append(classification)
        except Exception:
            continue

    if all_distributions:
        df = pd.DataFrame(all_distributions)

        # Filter by classification
        st.subheader("Filter by Classification")
        available_types = df['classification'].unique().tolist()
        selected_types = st.multiselect(
            "Select Classification Types",
            available_types,
            default=available_types
        )

        filtered_df = df[df['classification'].isin(selected_types)]

        col1, col2 = st.columns(2)

        with col1:
            # State distribution scatter plot
            st.subheader("State Distribution Scatter")
            fig_scatter = px.scatter(
                filtered_df,
                x='Directing',
                y='Conferring',
                color='classification',
                hover_name='name',
                size='Seeking',
                title='Directing vs Conferring by Classification',
                labels={
                    'Directing': 'Directing %',
                    'Conferring': 'Conferring %'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            # Box plot by classification
            st.subheader("Directing State by Classification")
            fig_box = px.box(
                filtered_df,
                x='classification',
                y='Directing',
                color='classification',
                title='Directing State Distribution by Type'
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # Detailed table
        st.subheader("Individual Details")
        display_df = filtered_df[['name', 'classification', 'subtype', 'Seeking', 'Directing', 'Conferring', 'Revising']].copy()
        for col in ['Seeking', 'Directing', 'Conferring', 'Revising']:
            display_df[col] = (display_df[col] * 100).round(1)

        st.dataframe(
            display_df,
            column_config={
                'name': 'Individual',
                'classification': 'Type',
                'subtype': 'Subtype',
                'Seeking': st.column_config.NumberColumn('Seeking %', format='%.1f'),
                'Directing': st.column_config.NumberColumn('Directing %', format='%.1f'),
                'Conferring': st.column_config.NumberColumn('Conferring %', format='%.1f'),
                'Revising': st.column_config.NumberColumn('Revising %', format='%.1f')
            },
            use_container_width=True,
            hide_index=True
        )

        # Export functionality
        st.subheader("Export Data")
        col1, col2 = st.columns(2)

        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "Download Population Data (CSV)",
                csv_data,
                file_name="population_state_distributions.csv",
                mime="text/csv"
            )

        with col2:
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                "Download Population Data (JSON)",
                json_data,
                file_name="population_state_distributions.json",
                mime="application/json"
            )

st.divider()
st.caption(
    "This simulation playground is for research and exploration purposes. "
    "Results should be interpreted as exploratory, not definitive predictions."
)
