"""
State Space Comparison Page

Compares two approaches for categorizing criminal life events:
1. Theory-Driven: 4-Animal State Space (Seeking, Directing, Conferring, Revising)
2. Data-Driven: 10 Archetypal Event Clusters (LLM-labeled from embeddings)

Shows how these frameworks relate and when to use each.
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

from data.loader import DashboardDataLoader
from config import DATA_DIR, ANIMAL_COLORS

st.set_page_config(page_title="State Space Comparison", layout="wide")


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


# Mapping from archetypal clusters to primary 4-Animal states
# Based on semantic analysis of cluster themes
CLUSTER_TO_ANIMAL_MAP = {
    0: {'primary': 'Conferring', 'secondary': 'Directing',
        'theme': 'Predatory/Geographical Stalking',
        'rationale': 'Surveillance and target selection (Conferring) followed by action (Directing)'},
    1: {'primary': 'Directing', 'secondary': 'Seeking',
        'theme': 'Sexually Motivated Serial Killer',
        'rationale': 'Sexual exploitation (Directing) driven by internal fantasy (Seeking)'},
    2: {'primary': 'Directing', 'secondary': 'Revising',
        'theme': 'Escalating Criminal Behavior',
        'rationale': 'Repeated exploitation (Directing) with pattern refinement (Revising)'},
    3: {'primary': 'Directing', 'secondary': 'Conferring',
        'theme': 'Sadistic Duo/Killer Couple',
        'rationale': 'Coordinated control (Directing) with partner observation (Conferring)'},
    4: {'primary': 'Directing', 'secondary': None,
        'theme': 'Power and Control',
        'rationale': 'Pure exploitation and dominance (Directing)'},
    5: {'primary': 'Directing', 'secondary': 'Seeking',
        'theme': 'Authority Obsession',
        'rationale': 'Control-seeking (Directing) with identity exploration (Seeking)'},
    6: {'primary': 'Revising', 'secondary': None,
        'theme': 'Legal Judgment/Sentencing',
        'rationale': 'Processing consequences (Revising) - system response events'},
    7: {'primary': 'Seeking', 'secondary': 'Revising',
        'theme': 'Search for Identity',
        'rationale': 'Self-exploration (Seeking) and internal processing (Revising)'},
    8: {'primary': 'Directing', 'secondary': 'Revising',
        'theme': 'Domestic Violence/Discord',
        'rationale': 'Interpersonal control (Directing) in cyclical patterns (Revising)'},
    9: {'primary': 'Directing', 'secondary': 'Conferring',
        'theme': 'Angel of Death/Mercy Killer',
        'rationale': 'Control over life/death (Directing) with caretaker observation (Conferring)'}
}

data = load_data()
audience = st.session_state.get('audience_mode', 'researcher')

st.title("State Space Comparison")
st.markdown("**Comparing Theory-Driven vs Data-Driven Event Classification**")

st.divider()

# ============================================================================
# OVERVIEW: TWO APPROACHES
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 4-Animal State Space")
    st.markdown("*Theory-Driven Classification*")

    st.markdown("""
    Based on two psychological dimensions:
    - **Self / Other**: Internal vs external focus
    - **Explore / Exploit**: Discovery vs utilization

    | State | Dimensions | Behavior |
    |-------|------------|----------|
    | **Seeking** | Self + Explore | Fantasy, introspection |
    | **Directing** | Other + Exploit | Control, manipulation |
    | **Conferring** | Other + Explore | Observation, stalking |
    | **Revising** | Self + Exploit | Rituals, habits |
    """)

    # Show LLM state distribution
    llm_stats = data.get_llm_stats()
    if llm_stats:
        state_dist = llm_stats.get('state_distribution', {})
        total = sum(state_dist.values())

        fig_animal = go.Figure(data=[go.Pie(
            labels=list(state_dist.keys()),
            values=list(state_dist.values()),
            hole=0.4,
            marker_colors=[ANIMAL_COLORS.get(k, '#95a5a6') for k in state_dist.keys()],
            textinfo='percent+label',
            textposition='outside'
        )])
        fig_animal.update_layout(
            showlegend=False,
            height=300,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_animal, use_container_width=True)
        st.caption(f"Based on {total:,} events classified by GPT-4o-mini")

with col2:
    st.markdown("### 10 Archetypal Clusters")
    st.markdown("*Data-Driven Classification*")

    st.markdown("""
    Emergent from semantic similarity:
    - Events embedded using SentenceTransformer
    - K-Means clustering groups similar events
    - LLM labels each cluster with archetypal theme

    Captures **contextual meaning** rather than
    abstract psychological dimensions.
    """)

    # Show cluster distribution
    cluster_stats = data.get_cluster_stats()
    if cluster_stats['total_clusters'] > 0:
        df_clusters = pd.DataFrame(cluster_stats['clusters'])

        # Shorten theme names for display
        df_clusters['short_theme'] = df_clusters['theme'].apply(
            lambda x: x[:25] + '...' if len(x) > 25 else x
        )

        fig_clusters = go.Figure(data=[go.Pie(
            labels=df_clusters['short_theme'],
            values=df_clusters['size'],
            hole=0.4,
            textinfo='percent',
            textposition='outside'
        )])
        fig_clusters.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.4, font=dict(size=8)),
            height=300,
            margin=dict(t=20, b=80, l=20, r=20)
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
        st.caption(f"Based on {cluster_stats['total_events']:,} events in 10 clusters")

st.divider()

# ============================================================================
# MAPPING: CLUSTERS TO ANIMAL STATES
# ============================================================================
st.subheader("Mapping: How Clusters Relate to Animal States")

st.markdown("""
Each archetypal cluster can be mapped to a **primary** 4-Animal state based on
the dominant psychological mode it represents. Some clusters also have a
**secondary** state that captures their complexity.
""")

# Create mapping visualization
mapping_data = []
for cid, mapping in CLUSTER_TO_ANIMAL_MAP.items():
    mapping_data.append({
        'Cluster': cid,
        'Theme': mapping['theme'],
        'Primary State': mapping['primary'],
        'Secondary State': mapping['secondary'] or '-',
        'Rationale': mapping['rationale']
    })

df_mapping = pd.DataFrame(mapping_data)

# Add cluster sizes
if cluster_stats['total_clusters'] > 0:
    size_map = {c['id']: c['size'] for c in cluster_stats['clusters']}
    df_mapping['Events'] = df_mapping['Cluster'].map(size_map)

st.dataframe(
    df_mapping,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Cluster': st.column_config.NumberColumn(width="small"),
        'Theme': st.column_config.TextColumn(width="medium"),
        'Primary State': st.column_config.TextColumn(width="small"),
        'Secondary State': st.column_config.TextColumn(width="small"),
        'Events': st.column_config.NumberColumn(width="small"),
        'Rationale': st.column_config.TextColumn(width="large")
    }
)

st.divider()

# ============================================================================
# SANKEY DIAGRAM: CLUSTER TO STATE FLOW
# ============================================================================
st.subheader("Event Flow: Clusters to Animal States")

if cluster_stats['total_clusters'] > 0:
    # Build Sankey data
    cluster_names = [f"C{i}: {CLUSTER_TO_ANIMAL_MAP[i]['theme'][:20]}" for i in range(10)]
    state_names = ['Seeking', 'Directing', 'Conferring', 'Revising']

    all_labels = cluster_names + state_names

    # Create links from clusters to primary states
    sources = []
    targets = []
    values = []
    colors = []

    state_colors_sankey = {
        'Seeking': 'rgba(46, 204, 113, 0.6)',
        'Directing': 'rgba(231, 76, 60, 0.6)',
        'Conferring': 'rgba(52, 152, 219, 0.6)',
        'Revising': 'rgba(155, 89, 182, 0.6)'
    }

    for cid, mapping in CLUSTER_TO_ANIMAL_MAP.items():
        cluster_size = size_map.get(cid, 0)
        primary_state = mapping['primary']
        secondary_state = mapping['secondary']

        # Primary state gets 70% of events (or 100% if no secondary)
        primary_ratio = 0.7 if secondary_state else 1.0

        sources.append(cid)
        targets.append(10 + state_names.index(primary_state))
        values.append(int(cluster_size * primary_ratio))
        colors.append(state_colors_sankey[primary_state])

        # Secondary state gets 30% of events
        if secondary_state:
            sources.append(cid)
            targets.append(10 + state_names.index(secondary_state))
            values.append(int(cluster_size * 0.3))
            colors.append(state_colors_sankey[secondary_state])

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=['#95a5a6']*10 + [ANIMAL_COLORS[s] for s in state_names]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])

    fig_sankey.update_layout(
        title="How Archetypal Clusters Map to 4-Animal States",
        height=500,
        font=dict(size=10)
    )

    st.plotly_chart(fig_sankey, use_container_width=True)

    st.caption("""
    **Reading the diagram:** Each cluster (left) flows to its mapped Animal state(s) (right).
    Band width represents the number of events. Clusters with secondary states split their flow.
    """)

st.divider()

# ============================================================================
# AGGREGATED VIEW: CLUSTER EVENTS BY STATE
# ============================================================================
st.subheader("Aggregated: Events by Primary Animal State")

if cluster_stats['total_clusters'] > 0:
    # Aggregate cluster events by primary state
    state_from_clusters = {'Seeking': 0, 'Directing': 0, 'Conferring': 0, 'Revising': 0}

    for cid, mapping in CLUSTER_TO_ANIMAL_MAP.items():
        cluster_size = size_map.get(cid, 0)
        state_from_clusters[mapping['primary']] += cluster_size

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**From LLM Classification (Direct)**")
        if llm_stats:
            for state, count in llm_stats.get('state_distribution', {}).items():
                total = sum(llm_stats.get('state_distribution', {}).values())
                pct = count / total * 100 if total > 0 else 0
                st.metric(state, f"{count:,} ({pct:.1f}%)")

    with col2:
        st.markdown("**From Cluster Mapping (Indirect)**")
        total_from_clusters = sum(state_from_clusters.values())
        for state, count in state_from_clusters.items():
            pct = count / total_from_clusters * 100 if total_from_clusters > 0 else 0
            st.metric(state, f"{count:,} ({pct:.1f}%)")

    # Comparison bar chart
    comparison_data = []
    for state in ['Seeking', 'Directing', 'Conferring', 'Revising']:
        if llm_stats:
            llm_count = llm_stats.get('state_distribution', {}).get(state, 0)
            llm_total = sum(llm_stats.get('state_distribution', {}).values())
            llm_pct = llm_count / llm_total * 100 if llm_total > 0 else 0
        else:
            llm_pct = 0

        cluster_pct = state_from_clusters[state] / total_from_clusters * 100 if total_from_clusters > 0 else 0

        comparison_data.append({'State': state, 'Method': 'LLM Direct', 'Percentage': llm_pct})
        comparison_data.append({'State': state, 'Method': 'Cluster Mapping', 'Percentage': cluster_pct})

    df_compare = pd.DataFrame(comparison_data)

    fig_compare = px.bar(
        df_compare,
        x='State',
        y='Percentage',
        color='Method',
        barmode='group',
        color_discrete_map={'LLM Direct': '#3498db', 'Cluster Mapping': '#e74c3c'}
    )
    fig_compare.update_layout(
        title="State Distribution: Direct LLM vs Cluster Mapping",
        yaxis_title="Percentage of Events",
        height=350
    )
    st.plotly_chart(fig_compare, use_container_width=True)

st.divider()

# ============================================================================
# WHEN TO USE EACH APPROACH
# ============================================================================
st.subheader("When to Use Each Approach")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Use 4-Animal States When...")
    st.success("""
    - **Theory testing**: Validating psychological models
    - **Cross-study comparison**: Need standardized framework
    - **Transition analysis**: Studying behavioral sequences
    - **Individual profiling**: Assigning archetype classifications
    - **Real-time assessment**: Quick categorization needed
    """)

    st.markdown("**Strengths:**")
    st.markdown("""
    - Theoretically grounded
    - Consistent across datasets
    - Enables Markov chain analysis
    - Interpretable dimensions (Self/Other, Explore/Exploit)
    """)

with col2:
    st.markdown("### Use Archetypal Clusters When...")
    st.info("""
    - **Pattern discovery**: Finding emergent themes in data
    - **Contextual analysis**: Need rich, narrative labels
    - **Case comparison**: Grouping similar crime patterns
    - **Training/education**: Teaching about criminal typologies
    - **New dataset exploration**: Unknown event types
    """)

    st.markdown("**Strengths:**")
    st.markdown("""
    - Data-driven, captures actual patterns
    - Semantically rich labels
    - Captures context (e.g., "killer couple", "mercy killer")
    - Can reveal unexpected groupings
    """)

st.divider()

# ============================================================================
# RESEARCHER VIEW: TECHNICAL DETAILS
# ============================================================================
if audience == 'researcher':
    with st.expander("Technical Details: Mapping Methodology"):
        st.markdown("""
        ### How Clusters Were Mapped to Animal States

        The mapping from 10 archetypal clusters to 4-Animal states was done through
        **semantic analysis** of each cluster's:
        1. LLM-generated archetypal theme
        2. Representative sample events
        3. Psychological interpretation

        **Mapping Rules:**

        | If cluster involves... | Primary State |
        |------------------------|---------------|
        | Active control, violence, exploitation | Directing |
        | Surveillance, observation, target selection | Conferring |
        | Internal fantasy, identity exploration | Seeking |
        | Rituals, habits, processing | Revising |

        **Secondary states** were assigned when clusters showed clear dual-mode patterns
        (e.g., stalking clusters involve Conferring → Directing sequence).

        ### Limitations

        - Mapping is interpretive, not empirically validated
        - Some clusters don't map cleanly to a single state
        - Cluster boundaries are fuzzy (K-means limitation)
        - The two approaches measure different things:
          - 4-Animal: Psychological mode at event occurrence
          - Clusters: Semantic similarity of event descriptions
        """)

        st.markdown("### Raw Mapping Data")
        st.json(CLUSTER_TO_ANIMAL_MAP)
