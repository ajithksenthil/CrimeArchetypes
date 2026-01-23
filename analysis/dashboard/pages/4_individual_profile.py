"""
Individual Profile Page

Deep-dive into a specific criminal's archetype profile.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DashboardDataLoader
from config import (
    DATA_DIR, PRIMARY_COLORS, SUBTYPE_COLORS, ANIMAL_COLORS, ROLE_COLORS,
    RISK_DESCRIPTIONS, SUBTYPE_DESCRIPTIONS, TERMINOLOGY
)

st.set_page_config(page_title="Individual Profile", layout="wide")


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


data = load_data()
audience = st.session_state.get('audience_mode', 'researcher')

st.title("Individual Profile")
st.markdown("Deep-dive into a specific criminal's behavioral pattern")

st.divider()

# Individual selector
all_individuals = data.get_all_individuals()

# Format names for display
def format_name(name):
    parts = name.split('_')
    return f"{parts[0]} {parts[1]}" if len(parts) > 1 else name

# Check if coming from another page
selected_name = st.session_state.get('selected_individual', all_individuals[0] if all_individuals else None)

col1, col2 = st.columns([3, 1])
with col1:
    selected = st.selectbox(
        "Select Individual",
        options=all_individuals,
        format_func=format_name,
        index=all_individuals.index(selected_name) if selected_name in all_individuals else 0
    )
with col2:
    st.write("")  # Spacer
    idx = all_individuals.index(selected) if selected in all_individuals else 0
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("< Prev") and idx > 0:
            selected = all_individuals[idx - 1]
            st.rerun()
    with col_next:
        if st.button("Next >") and idx < len(all_individuals) - 1:
            selected = all_individuals[idx + 1]
            st.rerun()

if selected:
    # Get all data for this individual
    ind_data = data.get_individual_data(selected)
    classification = ind_data.get('classification', {})
    signature = ind_data.get('signature', {})
    network_role = ind_data.get('network_role', 'GENERAL')
    influence = ind_data.get('influence_metrics', {})
    lineages = ind_data.get('lineages', [])

    st.divider()

    # ============================================================================
    # CLASSIFICATION CARD
    # ============================================================================
    col1, col2 = st.columns([2, 1])

    with col1:
        display_name = format_name(selected)
        st.header(display_name)

        primary = classification.get('primary_type', '-')
        subtype = classification.get('subtype', '-')
        primary_conf = classification.get('primary_confidence', 0)
        subtype_conf = classification.get('subtype_confidence', 0)

        primary_color = PRIMARY_COLORS.get(primary, '#95a5a6')
        subtype_color = SUBTYPE_COLORS.get(subtype, '#95a5a6')
        role_color = ROLE_COLORS.get(network_role, '#95a5a6')

        # Classification badges
        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-bottom: 20px;">
            <span style="background-color: {primary_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                {primary}
            </span>
            <span style="background-color: {subtype_color}; color: white; padding: 5px 15px; border-radius: 20px;">
                {subtype}
            </span>
            <span style="background-color: {role_color}; color: white; padding: 5px 15px; border-radius: 20px;">
                {network_role}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Description
        if subtype in SUBTYPE_DESCRIPTIONS:
            st.caption(SUBTYPE_DESCRIPTIONS[subtype])

        # Risk for practitioner
        if audience == 'practitioner' and subtype in RISK_DESCRIPTIONS:
            risk_text = RISK_DESCRIPTIONS[subtype]
            if 'CRITICAL' in risk_text:
                st.error(f"**Risk Level:** {risk_text}")
            elif 'UNPREDICTABLE' in risk_text:
                st.info(f"**Risk Level:** {risk_text}")
            else:
                st.warning(f"**Risk Level:** {risk_text}")

    with col2:
        # Confidence metrics (researcher)
        if audience == 'researcher':
            st.metric("Primary Confidence", f"{primary_conf:.0%}")
            st.metric("Subtype Confidence", f"{subtype_conf:.0%}")

    st.divider()

    # ============================================================================
    # BEHAVIORAL SIGNATURE
    # ============================================================================
    st.subheader("Behavioral Signature")

    col1, col2 = st.columns(2)

    with col1:
        # State distribution
        st.markdown("**State Distribution**")
        state_dist = signature.get('state_distribution', {})

        if state_dist:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(state_dist.keys()),
                values=[v * 100 for v in state_dist.values()],
                hole=0.3,
                marker_colors=[ANIMAL_COLORS.get(k, '#95a5a6') for k in state_dist.keys()],
                textinfo='label+percent',
                textposition='inside'
            )])
            fig_pie.update_layout(
                showlegend=False,
                height=300,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Key metric
            dominant = signature.get('dominant_state', '-')
            dominant_pct = signature.get('dominant_state_pct', 0)
            st.metric(f"Dominant State: {dominant}", f"{dominant_pct:.1%}")

    with col2:
        # Phase evolution
        st.markdown("**Phase Evolution**")
        phases = signature.get('phase_distributions', {})

        if phases:
            phase_data = []
            for phase, dist in phases.items():
                for state, pct in dist.items():
                    phase_data.append({
                        'Phase': phase.title(),
                        'State': state,
                        'Percentage': pct * 100
                    })

            if phase_data:
                df_phases = pd.DataFrame(phase_data)

                fig_phases = px.bar(
                    df_phases,
                    x='Phase',
                    y='Percentage',
                    color='State',
                    color_discrete_map=ANIMAL_COLORS,
                    barmode='stack'
                )
                fig_phases.update_layout(
                    height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig_phases, use_container_width=True)

        # Escalation metric
        escalation = signature.get('escalation_score', 0)
        shows_esc = signature.get('shows_escalation', False)

        if shows_esc:
            st.metric("Escalation Score", f"{escalation:+.2f}", delta="Escalating")
        else:
            st.metric("Escalation Score", f"{escalation:+.2f}")

    st.divider()

    # ============================================================================
    # TRANSITION PATTERNS
    # ============================================================================
    st.subheader("Transition Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Top bigrams
        st.markdown("**Most Common Transitions (Bigrams)**")
        bigrams = signature.get('top_bigrams', [])

        if bigrams:
            bigram_data = []
            for b in bigrams[:5]:
                if isinstance(b[0], list):
                    transition = f"{b[0][0]} → {b[0][1]}"
                else:
                    transition = str(b[0])
                count = b[1]
                bigram_data.append({'Transition': transition, 'Count': count})

            df_bigrams = pd.DataFrame(bigram_data)

            fig_bigrams = px.bar(
                df_bigrams,
                x='Count',
                y='Transition',
                orientation='h'
            )
            fig_bigrams.update_layout(
                height=250,
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_bigrams, use_container_width=True)

    with col2:
        # Dominant transition
        st.markdown("**Dominant Transition**")
        dom_trans = signature.get('dominant_transition', [])
        dom_prob = signature.get('dominant_transition_prob', 0)

        if dom_trans:
            trans_str = f"{dom_trans[0]} → {dom_trans[1]}" if isinstance(dom_trans, list) else str(dom_trans)
            st.metric(trans_str, f"{dom_prob:.1%}")

        # Top trigrams (researcher)
        if audience == 'researcher':
            st.markdown("**Top 3-grams**")
            trigrams = signature.get('top_trigrams', [])
            if trigrams:
                for t in trigrams[:3]:
                    if isinstance(t[0], (list, tuple)):
                        seq = ' → '.join(t[0])
                    else:
                        seq = str(t[0])
                    st.caption(f"{seq}: {t[1]}x")

    st.divider()

    # ============================================================================
    # NETWORK POSITION
    # ============================================================================
    st.subheader("Position in Influence Network")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Network Role:** {network_role}")

        role_explanations = {
            'SOURCE': "Pattern originator - their patterns appear in many others",
            'SINK': "Pattern receiver - embodies patterns from multiple sources",
            'HUB': "Central connector - both receives and transmits patterns",
            'GENERAL': "Standard network position"
        }
        st.caption(role_explanations.get(network_role, ""))

        if influence:
            if audience == 'researcher':
                st.metric("Outgoing Influence (TE sum)", f"{influence.get('outgoing', 0):.2f}")
                st.metric("Incoming Influence (TE sum)", f"{influence.get('incoming', 0):.2f}")
            else:
                out_rank = "High" if influence.get('outgoing', 0) > 5 else "Moderate" if influence.get('outgoing', 0) > 2 else "Low"
                in_rank = "High" if influence.get('incoming', 0) > 5 else "Moderate" if influence.get('incoming', 0) > 2 else "Low"
                st.metric("Patterns Shared", out_rank)
                st.metric("Patterns Received", in_rank)

    with col2:
        # Who influenced whom
        top_influenced = influence.get('top_influenced', [])
        top_influenced_by = influence.get('top_influenced_by', [])

        if top_influenced:
            st.markdown("**Most influenced by this individual:**")
            for item in top_influenced[:3]:
                name = item['name'].split('_')[0]
                if audience == 'researcher':
                    st.markdown(f"- {name} (TE: {item['te']:.3f})")
                else:
                    st.markdown(f"- {name}")

        if top_influenced_by:
            st.markdown("**Influenced by:**")
            for item in top_influenced_by[:3]:
                name = item['name'].split('_')[0]
                if audience == 'researcher':
                    st.markdown(f"- {name} (TE: {item['te']:.3f})")
                else:
                    st.markdown(f"- {name}")

    # Lineages
    if lineages:
        st.divider()
        st.subheader(f"{TERMINOLOGY['lineage']}s Including This Individual")

        for i, lineage in enumerate(lineages[:5]):
            # Format lineage with this individual highlighted
            formatted = []
            for name in lineage:
                short = name.split('_')[0]
                if name == selected:
                    formatted.append(f"**[{short}]**")
                else:
                    formatted.append(short)

            st.markdown(f"{i+1}. {' → '.join(formatted)}")

        if len(lineages) > 5:
            st.caption(f"... and {len(lineages) - 5} more lineages")

    # ============================================================================
    # RESEARCHER: RAW DATA
    # ============================================================================
    if audience == 'researcher':
        st.divider()
        with st.expander("Raw Signature Data"):
            st.json(signature)

else:
    st.warning("No individuals available. Run the analysis pipeline first.")
