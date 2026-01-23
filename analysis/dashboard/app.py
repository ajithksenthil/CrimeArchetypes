"""
CrimeArchetypes Interactive Dashboard

Main entry point for the Streamlit dashboard.
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Archetypal Profiling Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
from data.loader import DashboardDataLoader
from config import DATA_DIR, PRIMARY_COLORS, TERMINOLOGY


@st.cache_resource
def load_data():
    """Load and cache all dashboard data."""
    return DashboardDataLoader(DATA_DIR)


# Initialize data
data = load_data()

# Sidebar
with st.sidebar:
    st.title("Archetypal Profiling")
    st.caption("Criminal Behavioral Pattern Analysis")

    st.divider()

    # Audience toggle
    audience = st.radio(
        "View Mode",
        options=['researcher', 'practitioner'],
        format_func=lambda x: x.title(),
        key='audience_mode',
        help="Researcher: Full statistics | Practitioner: Risk-focused"
    )

    st.divider()

    # Quick stats
    stats = data.get_population_stats()
    st.metric("Total Individuals", stats['total'])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("COMPLEX", stats['primary_distribution'].get('COMPLEX', 0))
    with col2:
        st.metric("FOCUSED", stats['primary_distribution'].get('FOCUSED', 0))

    st.divider()

    st.caption("Based on Computational Psychodynamics")
    st.caption("4-Animal State Space Framework")

# Main content - Welcome page
st.title("Archetypal Profiling Dashboard")
st.markdown("**Understanding Criminal Behavioral Patterns Through the 4-Animal Framework**")

st.divider()

# Quick overview
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Population Overview")
    st.markdown("""
    Explore the distribution of archetypes across the sample.
    - Primary types: COMPLEX vs FOCUSED
    - Subtypes: 6 behavioral patterns
    - Risk assessment summaries
    """)
    if st.button("Go to Population Overview", key="nav_pop"):
        st.switch_page("pages/1_population_overview.py")

with col2:
    st.markdown("### Archetype Comparison")
    st.markdown("""
    Compare archetypes side-by-side.
    - State distributions
    - Escalation patterns
    - Key distinguishing features
    """)
    if st.button("Go to Comparison", key="nav_comp"):
        st.switch_page("pages/2_archetype_comparison.py")

with col3:
    st.markdown("### Reincarnation Network")
    st.markdown("""
    Visualize how patterns flow between individuals.
    - Influence network graph
    - Pattern inheritance chains
    - Sources, Hubs, and Sinks
    """)
    if st.button("Go to Network", key="nav_net"):
        st.switch_page("pages/3_reincarnation_network.py")

st.divider()

# Framework explanation
with st.expander("About the 4-Animal Framework", expanded=False):
    st.markdown("""
    ### The Four Behavioral States

    Criminal behavioral sequences follow patterns that can be understood through two dimensions:
    - **Self / Other**: Is behavior directed inward or outward?
    - **Explore / Exploit**: Is behavior exploring new territory or exploiting patterns?

    | State | Dimensions | Description |
    |-------|------------|-------------|
    | **Seeking** | Self + Explore | Introspection, fantasy development |
    | **Directing** | Other + Exploit | Control, manipulation, violence |
    | **Conferring** | Other + Explore | Observation, surveillance, stalking |
    | **Revising** | Self + Exploit | Rituals, habits, compulsive patterns |

    ### Hierarchical Classification

    **Level 1 (Data-Driven):**
    - **COMPLEX** (12%): Multi-modal, unpredictable patterns
    - **FOCUSED** (88%): Directing-dominant, escalating patterns

    **Level 2 (Theory-Driven Subtypes):**
    - Pure Predator, Strong Escalator, Stalker-Striker, Fantasy-Actor, Chameleon, Multi-Modal
    """)

with st.expander("About Archetypal Reincarnation", expanded=False):
    st.markdown(f"""
    ### Pattern Inheritance Across Individuals

    "Archetypal Reincarnation" describes how behavioral patterns appear to "transfer" between criminals.
    Using **{TERMINOLOGY['transfer_entropy']}** analysis, we measure how much knowing one person's
    behavioral sequence helps predict another's.

    **Key Roles:**
    - **{TERMINOLOGY['source']}s**: Individuals whose patterns appear in many others
    - **{TERMINOLOGY['sink']}s**: Individuals who embody patterns from multiple sources
    - **{TERMINOLOGY['hub']}s**: Central connectors who both receive and transmit patterns

    **{TERMINOLOGY['lineage']}s**: Chains showing how patterns "inherit" from one individual to another.
    """)

# Footer
st.divider()
st.caption("Data: 26 serial killers, 1,246 life events | Framework: Senthil, 2025")
