"""
Population Overview Page

Shows high-level distribution of archetypes across the sample.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DashboardDataLoader
from config import (
    DATA_DIR, PRIMARY_COLORS, SUBTYPE_COLORS, RISK_COLORS,
    RISK_DESCRIPTIONS, SUBTYPE_DESCRIPTIONS, PRIMARY_DESCRIPTIONS
)

st.set_page_config(page_title="Population Overview", layout="wide")


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


data = load_data()
stats = data.get_population_stats()
audience = st.session_state.get('audience_mode', 'researcher')

st.title("Population Overview")
st.markdown("Distribution of archetypes across the sample")

st.divider()

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Individuals", stats['total'])

with col2:
    complex_n = stats['primary_distribution'].get('COMPLEX', 0)
    complex_pct = stats['primary_percentages'].get('COMPLEX', 0)
    st.metric("COMPLEX", f"{complex_n} ({complex_pct:.0f}%)")

with col3:
    focused_n = stats['primary_distribution'].get('FOCUSED', 0)
    focused_pct = stats['primary_percentages'].get('FOCUSED', 0)
    st.metric("FOCUSED", f"{focused_n} ({focused_pct:.0f}%)")

with col4:
    hubs_n = stats['roles'].get('hubs', 0)
    st.metric("Network Hubs", hubs_n)

st.divider()

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Primary Type Distribution")

    # Donut chart
    primary_data = stats['primary_distribution']
    fig_primary = go.Figure(data=[go.Pie(
        labels=list(primary_data.keys()),
        values=list(primary_data.values()),
        hole=0.4,
        marker_colors=[PRIMARY_COLORS.get(k, '#95a5a6') for k in primary_data.keys()],
        textinfo='label+percent',
        textposition='outside'
    )])
    fig_primary.update_layout(
        showlegend=False,
        height=350,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_primary, use_container_width=True)

    # Description based on audience
    if audience == 'practitioner':
        st.info("""
        **Key Insight:** 88% of individuals are FOCUSED type - they show clear
        escalation patterns that can be monitored. The 12% COMPLEX types are
        more unpredictable and require broader surveillance strategies.
        """)
    else:
        st.caption(f"COMPLEX: {PRIMARY_DESCRIPTIONS['COMPLEX']}")
        st.caption(f"FOCUSED: {PRIMARY_DESCRIPTIONS['FOCUSED']}")

with col2:
    st.subheader("Subtype Distribution")

    # Prepare subtype data
    subtype_data = []
    for primary, subtypes in stats.get('subtype_distribution', {}).items():
        for subtype, count in subtypes.items():
            subtype_data.append({
                'Primary': primary,
                'Subtype': subtype,
                'Count': count
            })

    if subtype_data:
        df_subtypes = pd.DataFrame(subtype_data)

        fig_subtypes = px.bar(
            df_subtypes,
            x='Subtype',
            y='Count',
            color='Subtype',
            color_discrete_map=SUBTYPE_COLORS,
            text='Count'
        )
        fig_subtypes.update_layout(
            showlegend=False,
            height=350,
            xaxis_tickangle=-45,
            margin=dict(t=20, b=80, l=20, r=20)
        )
        fig_subtypes.update_traces(textposition='outside')
        st.plotly_chart(fig_subtypes, use_container_width=True)

st.divider()

# Risk summary (practitioner view emphasized)
if audience == 'practitioner':
    st.subheader("Risk Assessment Summary")

    risk_data = stats.get('risk_distribution', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        critical_n = risk_data.get('CRITICAL', 0)
        st.error(f"**CRITICAL RISK: {critical_n}**")
        st.markdown("Pure Predator, Fantasy-Actor subtypes")
        st.caption("Highest danger - immediate intervention priority")

    with col2:
        high_n = risk_data.get('HIGH', 0)
        st.warning(f"**HIGH RISK: {high_n}**")
        st.markdown("Strong Escalator, Stalker-Striker subtypes")
        st.caption("Escalation patterns require monitoring")

    with col3:
        unpred_n = risk_data.get('UNPREDICTABLE', 0)
        st.info(f"**UNPREDICTABLE: {unpred_n}**")
        st.markdown("Chameleon, Multi-Modal subtypes")
        st.caption("No consistent pattern - broader surveillance needed")

    st.divider()

# Network roles summary
st.subheader("Influence Network Roles")

col1, col2, col3 = st.columns(3)

roles = data.reincarnation.get('roles', {})

with col1:
    st.markdown("**Pattern Originators (Sources)**")
    sources = roles.get('sources', [])
    if sources:
        for s in sources[:5]:
            name = s.get('name', 'Unknown').split('_')[0]
            if audience == 'researcher':
                st.markdown(f"- {name} (out: {s.get('outgoing', 0):.2f})")
            else:
                st.markdown(f"- {name}")
    else:
        st.caption("No sources identified")

with col2:
    st.markdown("**Central Connectors (Hubs)**")
    hubs = roles.get('hubs', [])
    if hubs:
        for h in hubs[:5]:
            name = h.get('name', 'Unknown').split('_')[0]
            if audience == 'researcher':
                st.markdown(f"- {name} (hub: {h.get('hub_score', 0):.2f})")
            else:
                st.markdown(f"- {name}")
    else:
        st.caption("No hubs identified")

with col3:
    st.markdown("**Pattern Inheritors (Sinks)**")
    sinks = roles.get('sinks', [])
    if sinks:
        for s in sinks[:5]:
            name = s.get('name', 'Unknown').split('_')[0]
            if audience == 'researcher':
                st.markdown(f"- {name} (in: {s.get('incoming', 0):.2f})")
            else:
                st.markdown(f"- {name}")
    else:
        st.caption("No sinks identified")

st.divider()

# ============================================================================
# LLM CLASSIFICATION RESULTS
# ============================================================================
llm_stats = data.get_llm_stats()
if llm_stats:
    st.subheader("LLM-Based 4-Animal State Classification")

    if audience == 'researcher':
        st.caption(f"Model: {llm_stats['model']} | Events: {llm_stats['n_events']:,} | Individuals: {llm_stats['n_individuals']}")

    # State distribution metrics
    col1, col2, col3, col4 = st.columns(4)
    state_dist = llm_stats.get('state_distribution', {})
    total_events = sum(state_dist.values()) if state_dist else 1

    with col1:
        seeking_n = state_dist.get('Seeking', 0)
        seeking_pct = seeking_n / total_events * 100
        st.metric("Seeking", f"{seeking_n} ({seeking_pct:.1f}%)",
                  help="Self + Explore: Introspection, fantasy")

    with col2:
        directing_n = state_dist.get('Directing', 0)
        directing_pct = directing_n / total_events * 100
        st.metric("Directing", f"{directing_n} ({directing_pct:.1f}%)",
                  help="Other + Exploit: Control, manipulation")

    with col3:
        conferring_n = state_dist.get('Conferring', 0)
        conferring_pct = conferring_n / total_events * 100
        st.metric("Conferring", f"{conferring_n} ({conferring_pct:.1f}%)",
                  help="Other + Explore: Observation, stalking")

    with col4:
        revising_n = state_dist.get('Revising', 0)
        revising_pct = revising_n / total_events * 100
        st.metric("Revising", f"{revising_n} ({revising_pct:.1f}%)",
                  help="Self + Exploit: Rituals, habits")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # State distribution pie chart
        state_colors = {'Seeking': '#3498db', 'Directing': '#e74c3c',
                       'Conferring': '#2ecc71', 'Revising': '#9b59b6'}
        fig_states = go.Figure(data=[go.Pie(
            labels=list(state_dist.keys()),
            values=list(state_dist.values()),
            hole=0.4,
            marker_colors=[state_colors.get(k, '#95a5a6') for k in state_dist.keys()],
            textinfo='percent',
            textposition='outside'
        )])
        fig_states.update_layout(
            title="State Distribution",
            showlegend=True,
            height=300,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_states, use_container_width=True)

    with col2:
        # Confidence distribution
        conf_dist = llm_stats.get('confidence_distribution', {})
        if conf_dist:
            conf_colors = {'HIGH': '#27ae60', 'MEDIUM': '#f39c12', 'LOW': '#e74c3c'}
            fig_conf = go.Figure(data=[go.Bar(
                x=list(conf_dist.keys()),
                y=list(conf_dist.values()),
                marker_color=[conf_colors.get(k, '#95a5a6') for k in conf_dist.keys()],
                text=[f"{v:,}" for v in conf_dist.values()],
                textposition='outside'
            )])
            fig_conf.update_layout(
                title="Classification Confidence",
                showlegend=False,
                height=300,
                margin=dict(t=40, b=20, l=20, r=20),
                yaxis_title="Event Count"
            )
            st.plotly_chart(fig_conf, use_container_width=True)

    # Additional researcher metrics
    if audience == 'researcher':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entropy Rate", f"{llm_stats.get('entropy_rate', 0):.3f} bits",
                     help="Predictability of state transitions")
        with col2:
            st.metric("Effective States", f"{llm_stats.get('effective_states', 0):.2f}/4",
                     help="How many states are actively used")
        with col3:
            cv_acc = llm_stats.get('cv_accuracy', 0) * 100
            cv_std = llm_stats.get('cv_std', 0) * 100
            st.metric("CV Accuracy", f"{cv_acc:.1f}% ± {cv_std:.1f}%",
                     help="Cross-validation accuracy for predicting states")

        # Transition matrix heatmap
        trans_matrix = llm_stats.get('transition_matrix', [])
        if trans_matrix:
            import numpy as np
            with st.expander("Transition Matrix", expanded=False):
                states = ['Seeking', 'Directing', 'Conferring', 'Revising']
                fig_trans = go.Figure(data=go.Heatmap(
                    z=trans_matrix,
                    x=states,
                    y=states,
                    colorscale='Blues',
                    text=[[f"{v:.2f}" for v in row] for row in trans_matrix],
                    texttemplate="%{text}",
                    hovertemplate="From %{y} to %{x}: %{z:.3f}<extra></extra>"
                ))
                fig_trans.update_layout(
                    title="State Transition Probabilities",
                    xaxis_title="To State",
                    yaxis_title="From State",
                    height=350
                )
                st.plotly_chart(fig_trans, use_container_width=True)

                st.caption("**Key Insight:** Directing has the highest self-loop (65%), meaning once in 'control mode', individuals tend to stay there.")

    st.divider()

# Individual list
st.subheader("All Individuals")

# Prepare data for table
table_data = []
for c in data.classifications.get('classifications', []):
    name = c.get('name', 'Unknown')
    short_name = name.split('_')[0] + '_' + name.split('_')[1] if '_' in name else name

    # Get network role
    individual_data = data.get_individual_data(name)

    row = {
        'Name': short_name[:25],
        'Primary': c.get('primary_type', '-'),
        'Subtype': c.get('subtype', '-'),
        'Primary Conf.': f"{c.get('primary_confidence', 0):.0%}",
        'Network Role': individual_data.get('network_role', 'GENERAL')
    }

    # Add risk for practitioner
    if audience == 'practitioner':
        subtype = c.get('subtype', '')
        if subtype in ['Pure Predator', 'Fantasy-Actor']:
            row['Risk'] = 'CRITICAL'
        elif subtype in ['Chameleon', 'Multi-Modal']:
            row['Risk'] = 'UNPREDICTABLE'
        else:
            row['Risk'] = 'HIGH'

    table_data.append(row)

df_table = pd.DataFrame(table_data)

# Add filtering
col1, col2 = st.columns(2)
with col1:
    filter_primary = st.selectbox(
        "Filter by Primary Type",
        options=['All'] + list(stats['primary_distribution'].keys())
    )
with col2:
    all_subtypes = []
    for subtypes in stats.get('subtype_distribution', {}).values():
        all_subtypes.extend(subtypes.keys())
    filter_subtype = st.selectbox(
        "Filter by Subtype",
        options=['All'] + list(set(all_subtypes))
    )

# Apply filters
if filter_primary != 'All':
    df_table = df_table[df_table['Primary'] == filter_primary]
if filter_subtype != 'All':
    df_table = df_table[df_table['Subtype'] == filter_subtype]

st.dataframe(df_table, use_container_width=True, hide_index=True)

# Download option for researchers
if audience == 'researcher':
    csv = df_table.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        "archetype_classifications.csv",
        "text/csv"
    )
