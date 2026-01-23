"""
Archetype Comparison Page

Side-by-side comparison of archetypes with radar charts and key metrics.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DashboardDataLoader
from config import (
    DATA_DIR, PRIMARY_COLORS, SUBTYPE_COLORS, ANIMAL_COLORS,
    SUBTYPE_DESCRIPTIONS, PRIMARY_DESCRIPTIONS, RISK_DESCRIPTIONS
)

st.set_page_config(page_title="Archetype Comparison", layout="wide")


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


data = load_data()
audience = st.session_state.get('audience_mode', 'researcher')

st.title("Archetype Comparison")
st.markdown("Compare archetypes side-by-side to understand their differences")

st.divider()

# Get available types
all_types = ['COMPLEX', 'FOCUSED']
all_subtypes = []
for subtypes in data.classifications.get('summary', {}).get('subtype_distribution', {}).values():
    all_subtypes.extend(subtypes.keys())
all_subtypes = list(set(all_subtypes))

# Selection
col1, col2 = st.columns(2)

with col1:
    compare_level = st.radio(
        "Compare",
        options=['Primary Types', 'Subtypes'],
        horizontal=True
    )

options = all_types if compare_level == 'Primary Types' else all_subtypes

with col1:
    type_a = st.selectbox("First Archetype", options=options, index=0)

with col2:
    remaining = [o for o in options if o != type_a]
    type_b = st.selectbox("Second Archetype", options=remaining, index=0 if remaining else None)

if type_a and type_b:
    st.divider()

    # Get comparison data
    comparison = data.get_archetype_comparison(type_a, type_b)

    # Side-by-side cards
    col1, col2 = st.columns(2)

    def render_archetype_card(col, type_name, comp_data, color):
        with col:
            # Header with color
            st.markdown(f"### {type_name}")

            members = comp_data.get('members', [])
            aggregate = comp_data.get('aggregate', {})
            state_dist = aggregate.get('state_distribution', {})
            escalation = aggregate.get('escalation_score', 0)

            # Description
            if type_name in PRIMARY_DESCRIPTIONS:
                st.caption(PRIMARY_DESCRIPTIONS[type_name])
            elif type_name in SUBTYPE_DESCRIPTIONS:
                st.caption(SUBTYPE_DESCRIPTIONS[type_name])

            # Risk (practitioner)
            if audience == 'practitioner' and type_name in RISK_DESCRIPTIONS:
                risk_text = RISK_DESCRIPTIONS[type_name]
                if 'CRITICAL' in risk_text:
                    st.error(risk_text)
                elif 'UNPREDICTABLE' in risk_text:
                    st.info(risk_text)
                else:
                    st.warning(risk_text)

            # Metrics
            st.metric("Members", len(members))

            # State distribution pie
            if state_dist:
                fig = go.Figure(data=[go.Pie(
                    labels=list(state_dist.keys()),
                    values=[v * 100 for v in state_dist.values()],
                    hole=0.3,
                    marker_colors=[ANIMAL_COLORS.get(k, '#95a5a6') for k in state_dist.keys()],
                    textinfo='label+percent',
                    textposition='inside'
                )])
                fig.update_layout(
                    showlegend=False,
                    height=250,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Key metrics
            if state_dist:
                directing_pct = state_dist.get('Directing', 0) * 100
                st.metric("Directing %", f"{directing_pct:.1f}%")

            st.metric("Escalation Score", f"{escalation:+.2f}")

            # Member list
            with st.expander(f"View Members ({len(members)})"):
                for m in members[:10]:
                    short_name = m.split('_')[0]
                    st.markdown(f"- {short_name}")
                if len(members) > 10:
                    st.caption(f"... and {len(members) - 10} more")

    color_a = PRIMARY_COLORS.get(type_a, SUBTYPE_COLORS.get(type_a, '#95a5a6'))
    color_b = PRIMARY_COLORS.get(type_b, SUBTYPE_COLORS.get(type_b, '#95a5a6'))

    render_archetype_card(col1, type_a, comparison['type_a'], color_a)
    render_archetype_card(col2, type_b, comparison['type_b'], color_b)

    st.divider()

    # Comparison radar chart
    st.subheader("Comparison Radar Chart")

    agg_a = comparison['type_a'].get('aggregate', {})
    agg_b = comparison['type_b'].get('aggregate', {})

    dist_a = agg_a.get('state_distribution', {})
    dist_b = agg_b.get('state_distribution', {})

    # Build radar data
    categories = ['Seeking', 'Directing', 'Conferring', 'Revising', 'Escalation']

    values_a = [
        dist_a.get('Seeking', 0) * 100,
        dist_a.get('Directing', 0) * 100,
        dist_a.get('Conferring', 0) * 100,
        dist_a.get('Revising', 0) * 100,
        (agg_a.get('escalation_score', 0) + 1) * 50  # Normalize to 0-100 scale
    ]

    values_b = [
        dist_b.get('Seeking', 0) * 100,
        dist_b.get('Directing', 0) * 100,
        dist_b.get('Conferring', 0) * 100,
        dist_b.get('Revising', 0) * 100,
        (agg_b.get('escalation_score', 0) + 1) * 50
    ]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=values_a + [values_a[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name=type_a,
        line_color=color_a,
        fillcolor=color_a,
        opacity=0.5
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=values_b + [values_b[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=type_b,
        line_color=color_b,
        fillcolor=color_b,
        opacity=0.5
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=450
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # Key differences summary
    st.subheader("Key Differences")

    differences = []

    # Directing difference
    dir_a = dist_a.get('Directing', 0) * 100
    dir_b = dist_b.get('Directing', 0) * 100
    if abs(dir_a - dir_b) > 5:
        higher = type_a if dir_a > dir_b else type_b
        diff = abs(dir_a - dir_b)
        differences.append(f"**Directing behavior**: {higher} is {diff:.1f}% higher")

    # Seeking difference
    seek_a = dist_a.get('Seeking', 0) * 100
    seek_b = dist_b.get('Seeking', 0) * 100
    if abs(seek_a - seek_b) > 5:
        higher = type_a if seek_a > seek_b else type_b
        diff = abs(seek_a - seek_b)
        differences.append(f"**Seeking (introspection)**: {higher} is {diff:.1f}% higher")

    # Escalation difference
    esc_a = agg_a.get('escalation_score', 0)
    esc_b = agg_b.get('escalation_score', 0)
    if abs(esc_a - esc_b) > 0.1:
        higher = type_a if esc_a > esc_b else type_b
        diff = abs(esc_a - esc_b)
        differences.append(f"**Escalation pattern**: {higher} shows stronger escalation (+{diff:.2f})")

    if differences:
        for d in differences:
            st.markdown(f"- {d}")
    else:
        st.info("These archetypes have similar profiles.")

    # Practitioner interpretation
    if audience == 'practitioner':
        st.divider()
        st.subheader("Practical Implications")

        if dir_a > dir_b:
            st.markdown(f"""
            **{type_a}** shows higher exploitation behavior:
            - More active harm phase
            - Higher risk during engagement
            - May require more intensive monitoring
            """)
        else:
            st.markdown(f"""
            **{type_b}** shows higher exploitation behavior:
            - More active harm phase
            - Higher risk during engagement
            - May require more intensive monitoring
            """)

        if esc_a > esc_b:
            st.markdown(f"""
            **{type_a}** shows stronger escalation:
            - Behavior intensifies over time
            - Early intervention is critical
            - Monitor for escalation warning signs
            """)
