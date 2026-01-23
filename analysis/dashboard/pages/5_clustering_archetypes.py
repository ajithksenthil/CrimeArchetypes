"""
Clustering & Archetypal Themes Page

Visualizes the life event clustering with LLM-generated archetypal labels.
Shows:
- Cluster distribution
- LLM-assigned archetypal themes
- Representative samples per cluster
- Methodology explanation (embedding + clustering + LLM labeling)
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
from config import DATA_DIR, ANIMAL_COLORS

st.set_page_config(page_title="Clustering & Archetypes", layout="wide")


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


# Color palette for clusters
CLUSTER_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
]


data = load_data()
audience = st.session_state.get('audience_mode', 'researcher')

st.title("Life Event Clustering & Archetypal Themes")

if audience == 'practitioner':
    st.markdown("**How criminal life events are grouped into behavioral patterns**")
else:
    st.markdown("**Embedding-based clustering with LLM archetypal interpretation**")

st.divider()

# Methodology explanation
with st.expander("Methodology: How Clusters Are Created", expanded=False):
    st.markdown("""
    ### Three-Stage Pipeline

    **1. Embedding Generation**
    - Each life event description is converted to a 384-dimensional vector
    - Uses SentenceTransformer (`all-MiniLM-L6-v2`) for semantic embedding
    - Similar events cluster together in embedding space

    **2. Lexical Imputation (Smoothing)**
    - To handle variation in how events are described, we generate multiple paraphrases
    - GPT-4o-mini creates 5 alternative phrasings of each event
    - The final embedding is the centroid (average) of all variants
    - This reduces noise from individual word choices

    **3. K-Means Clustering + LLM Labeling**
    - K-Means groups similar event embeddings into clusters
    - For each cluster, we find the 3-5 most representative samples (closest to centroid)
    - GPT-4o analyzes these representative samples to identify the archetypal theme
    - The LLM provides a psychologically-informed label for each cluster

    ### Why This Approach?
    - **Data-driven**: Clusters emerge from actual event similarity, not predefined categories
    - **Interpretable**: LLM labeling provides human-readable archetypal themes
    - **Robust**: Lexical imputation handles description variation across different case files
    """)

# Load cluster data
cluster_stats = data.get_cluster_stats()

if cluster_stats['total_clusters'] == 0:
    st.warning("No cluster data available. Run the clustering pipeline first.")
    st.info("Expected file: `clusters.json` in the analysis directory")
    st.stop()

# ============================================================================
# OVERVIEW SECTION
# ============================================================================
st.subheader("Cluster Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Clusters", cluster_stats['total_clusters'])

with col2:
    st.metric("Total Events", f"{cluster_stats['total_events']:,}")

with col3:
    avg_size = cluster_stats['total_events'] / cluster_stats['total_clusters'] if cluster_stats['total_clusters'] > 0 else 0
    st.metric("Avg Events/Cluster", f"{avg_size:.0f}")

st.divider()

# ============================================================================
# CLUSTER DISTRIBUTION
# ============================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Cluster Size Distribution")

    # Bar chart of cluster sizes
    df_clusters = pd.DataFrame(cluster_stats['clusters'])

    fig_bar = px.bar(
        df_clusters,
        x='theme',
        y='size',
        color='id',
        color_discrete_sequence=CLUSTER_COLORS,
        labels={'size': 'Number of Events', 'theme': 'Archetypal Theme', 'id': 'Cluster'},
        text='size'
    )

    fig_bar.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickangle=-45,
        margin=dict(b=120)
    )
    fig_bar.update_traces(textposition='outside')

    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### Event Distribution by Theme")

    # Pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=df_clusters['theme'],
        values=df_clusters['size'],
        hole=0.4,
        marker_colors=CLUSTER_COLORS[:len(df_clusters)],
        textinfo='percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Events: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig_pie.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
        margin=dict(t=20, b=80)
    )

    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ============================================================================
# DETAILED CLUSTER VIEW
# ============================================================================
st.subheader("Cluster Details")

# Cluster selector
selected_cluster = st.selectbox(
    "Select Cluster to Explore",
    options=range(len(cluster_stats['clusters'])),
    format_func=lambda x: f"Cluster {cluster_stats['clusters'][x]['id']}: {cluster_stats['clusters'][x]['theme']}"
)

if selected_cluster is not None:
    cluster = cluster_stats['clusters'][selected_cluster]

    # Cluster header
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"### Cluster {cluster['id']}: {cluster['theme']}")

    with col2:
        st.metric("Events", cluster['size'], delta=f"{cluster['percentage']:.1f}%")

    # Full archetypal description
    st.markdown("**LLM Archetypal Analysis:**")
    st.info(cluster['full_theme'])

    # Representative samples
    st.markdown("**Representative Life Events (used for LLM labeling):**")

    samples = cluster.get('representative_samples', [])
    if samples:
        for i, sample in enumerate(samples, 1):
            with st.expander(f"Sample {i}", expanded=(i == 1)):
                st.markdown(f"> {sample}")
    else:
        st.caption("No representative samples available")

st.divider()

# ============================================================================
# CLUSTER COMPARISON
# ============================================================================
st.subheader("Compare Clusters")

col1, col2 = st.columns(2)

with col1:
    compare_a = st.selectbox(
        "Cluster A",
        options=range(len(cluster_stats['clusters'])),
        format_func=lambda x: f"Cluster {cluster_stats['clusters'][x]['id']}: {cluster_stats['clusters'][x]['theme']}",
        key="compare_a"
    )

with col2:
    compare_b = st.selectbox(
        "Cluster B",
        options=range(len(cluster_stats['clusters'])),
        format_func=lambda x: f"Cluster {cluster_stats['clusters'][x]['id']}: {cluster_stats['clusters'][x]['theme']}",
        index=min(1, len(cluster_stats['clusters'])-1),
        key="compare_b"
    )

if compare_a is not None and compare_b is not None:
    cluster_a = cluster_stats['clusters'][compare_a]
    cluster_b = cluster_stats['clusters'][compare_b]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {cluster_a['theme']}")
        st.metric("Size", cluster_a['size'])
        st.markdown("**Theme:**")
        st.caption(cluster_a['full_theme'][:300] + "..." if len(cluster_a['full_theme']) > 300 else cluster_a['full_theme'])

        st.markdown("**Sample Events:**")
        for sample in cluster_a.get('representative_samples', [])[:2]:
            st.markdown(f"- _{sample[:100]}..._" if len(sample) > 100 else f"- _{sample}_")

    with col2:
        st.markdown(f"#### {cluster_b['theme']}")
        st.metric("Size", cluster_b['size'])
        st.markdown("**Theme:**")
        st.caption(cluster_b['full_theme'][:300] + "..." if len(cluster_b['full_theme']) > 300 else cluster_b['full_theme'])

        st.markdown("**Sample Events:**")
        for sample in cluster_b.get('representative_samples', [])[:2]:
            st.markdown(f"- _{sample[:100]}..._" if len(sample) > 100 else f"- _{sample}_")

st.divider()

# ============================================================================
# ALL CLUSTERS TABLE
# ============================================================================
st.subheader("All Clusters Summary")

# Create summary table
table_data = []
for cluster in cluster_stats['clusters']:
    samples_preview = "; ".join(cluster.get('representative_samples', [])[:2])
    if len(samples_preview) > 150:
        samples_preview = samples_preview[:147] + "..."

    table_data.append({
        'Cluster': cluster['id'],
        'Theme': cluster['theme'],
        'Events': cluster['size'],
        '%': f"{cluster['percentage']:.1f}%",
        'Sample Events': samples_preview
    })

df_table = pd.DataFrame(table_data)

st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Cluster': st.column_config.NumberColumn(width="small"),
        'Theme': st.column_config.TextColumn(width="medium"),
        'Events': st.column_config.NumberColumn(width="small"),
        '%': st.column_config.TextColumn(width="small"),
        'Sample Events': st.column_config.TextColumn(width="large")
    }
)

# ============================================================================
# RESEARCHER: ADDITIONAL DETAILS
# ============================================================================
if audience == 'researcher':
    st.divider()

    with st.expander("Technical Details"):
        st.markdown("""
        ### Pipeline Configuration

        | Parameter | Value |
        |-----------|-------|
        | Embedding Model | `all-MiniLM-L6-v2` (384 dim) |
        | Clustering Algorithm | K-Means |
        | Number of Clusters | 10 |
        | Representative Samples | 5 per cluster |
        | LLM for Labeling | GPT-4o-mini |
        | Lexical Imputation Variants | 5 |

        ### Lexical Imputation Details

        Before clustering, each event text is expanded into multiple paraphrases to create
        a more robust embedding. The final embedding is the centroid of:
        - Original text embedding
        - 5 LLM-generated paraphrases

        This addresses lexical variation in how events are described across different
        case files and researchers.

        ### LLM Labeling Prompt

        The LLM receives the representative samples and is asked to identify:
        1. The archetypal pattern or theme
        2. Psychological characteristics
        3. Behavioral signatures

        Temperature is set low (0.1) for deterministic, consistent labeling.
        """)

        # Raw cluster data
        st.markdown("### Raw Cluster Data")
        st.json(cluster_stats['clusters'])
