"""
Reincarnation Network Page - Enhanced Version

Visualize how archetypes "flow" between individuals via transfer entropy network.
Features:
- Interactive influence network with ego view
- Path finder between individuals
- Animated flow visualization
- Timeline lineages view
- Sankey pattern flow
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DashboardDataLoader
from config import (
    DATA_DIR, ROLE_COLORS, PRIMARY_COLORS, SUBTYPE_COLORS, TERMINOLOGY
)

st.set_page_config(page_title="Reincarnation Network", layout="wide")


@st.cache_resource
def load_data():
    return DashboardDataLoader(DATA_DIR)


@st.cache_data
def build_network_graph(_te_matrix, criminals, threshold):
    """Build NetworkX graph from transfer entropy matrix."""
    G = nx.DiGraph()
    for i, source in enumerate(criminals):
        G.add_node(source)

    for i, source in enumerate(criminals):
        for j, target in enumerate(criminals):
            if i != j and _te_matrix[i, j] >= threshold:
                G.add_edge(source, target, weight=_te_matrix[i, j])
    return G


def get_short_name(name):
    """Extract short display name."""
    return name.split('_')[0]


data = load_data()
audience = st.session_state.get('audience_mode', 'researcher')

st.title("Archetypal Reincarnation Network")

if audience == 'practitioner':
    st.markdown("**How behavioral patterns spread between individuals**")
else:
    st.markdown("**Transfer entropy analysis of cross-individual behavioral influence**")

st.divider()

# Explanation
with st.expander("What is 'Archetypal Reincarnation'?", expanded=False):
    if audience == 'practitioner':
        st.markdown(f"""
        This analysis shows how criminal behavioral patterns appear to "inherit" from one
        individual to another. When we see similar patterns across criminals, we can trace
        the "influence chains" - not literal contact, but archetypal similarity.

        - **{TERMINOLOGY['source']}s** (Gold): Their patterns appear in many others
        - **{TERMINOLOGY['hub']}s** (Red): Central figures who both receive and transmit patterns
        - **{TERMINOLOGY['sink']}s** (Blue): They embody patterns from multiple sources
        """)
    else:
        st.markdown("""
        **Transfer Entropy (TE)** measures directed information flow between time series.
        TE(A→B) quantifies how much knowing A's behavioral sequence reduces uncertainty
        about B's sequence.

        High TE(A→B) = A's archetype has "reincarnated" in B (A's pattern predicts B's pattern)

        - **Sources**: High outgoing TE (pattern originators)
        - **Hubs**: High both incoming and outgoing TE (central nodes)
        - **Sinks**: High incoming TE (pattern receivers)
        """)

# Load network data
te_matrix = data.te_matrix
criminals = data.criminal_order
roles = data.reincarnation.get('roles', {})

if te_matrix is None or len(criminals) == 0:
    st.warning("Transfer entropy matrix not available. Run archetypal_reincarnation.py first.")
    st.stop()

# Categorize nodes by role
source_names = {r.get('name') for r in roles.get('sources', [])}
sink_names = {r.get('name') for r in roles.get('sinks', [])}
hub_names = {r.get('name') for r in roles.get('hubs', [])}

# Get classifications
classifications = {c['name']: c for c in data.classifications.get('classifications', [])}

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Influence Network",
    "Ego Network",
    "Path Finder",
    "Timeline Lineages",
    "Pattern Flow"
])

# ============================================================================
# TAB 1: FULL INFLUENCE NETWORK
# ============================================================================
with tab1:
    st.subheader("Full Influence Network")

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        threshold_pct = st.slider(
            "Edge Threshold (percentile)",
            min_value=50, max_value=99, value=85,
            help="Only show edges above this percentile of transfer entropy",
            key="full_network_threshold"
        )

    with col2:
        color_by = st.selectbox(
            "Color nodes by",
            options=['Network Role', 'Primary Type', 'Subtype'],
            index=0,
            key="full_network_color"
        )

    with col3:
        show_labels = st.checkbox("Show labels", value=True, key="full_network_labels")

    # Calculate threshold
    nonzero = te_matrix[te_matrix > 0]
    threshold = np.percentile(nonzero, threshold_pct) if len(nonzero) > 0 else 0

    # Build NetworkX graph
    G = build_network_graph(te_matrix, criminals, threshold)

    # Get positions using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    def get_node_color(name, color_mode):
        if color_mode == 'Network Role':
            if name in hub_names:
                return ROLE_COLORS['HUB']
            elif name in source_names:
                return ROLE_COLORS['SOURCE']
            elif name in sink_names:
                return ROLE_COLORS['SINK']
            else:
                return ROLE_COLORS['GENERAL']
        elif color_mode == 'Primary Type':
            c = classifications.get(name, {})
            primary = c.get('primary_type', 'FOCUSED')
            return PRIMARY_COLORS.get(primary, '#95a5a6')
        else:  # Subtype
            c = classifications.get(name, {})
            subtype = c.get('subtype', 'Standard')
            return SUBTYPE_COLORS.get(subtype, '#95a5a6')

    def get_node_size(name):
        if name in hub_names:
            return 25
        elif name in source_names or name in sink_names:
            return 18
        else:
            return 12

    # Create edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.3,
        showlegend=False
    )

    # Group nodes by color for legend
    node_groups = {}
    for node in G.nodes():
        color = get_node_color(node, color_by)
        if color not in node_groups:
            node_groups[color] = {'x': [], 'y': [], 'names': [], 'sizes': [], 'texts': [], 'hovers': []}

        x, y = pos[node]
        node_groups[color]['x'].append(x)
        node_groups[color]['y'].append(y)
        node_groups[color]['names'].append(node)
        node_groups[color]['sizes'].append(get_node_size(node))

        short_name = get_short_name(node)
        node_groups[color]['texts'].append(short_name if show_labels else '')

        # Hover info
        c = classifications.get(node, {})
        idx = criminals.index(node) if node in criminals else -1
        out_te = np.sum(te_matrix[idx, :]) if idx >= 0 else 0
        in_te = np.sum(te_matrix[:, idx]) if idx >= 0 else 0
        hover = f"<b>{short_name}</b><br>Type: {c.get('primary_type', '-')}<br>Subtype: {c.get('subtype', '-')}<br>Out TE: {out_te:.2f}<br>In TE: {in_te:.2f}"
        node_groups[color]['hovers'].append(hover)

    # Color labels
    color_labels = {
        ROLE_COLORS['HUB']: 'Hub',
        ROLE_COLORS['SOURCE']: 'Source',
        ROLE_COLORS['SINK']: 'Sink',
        ROLE_COLORS['GENERAL']: 'General',
        PRIMARY_COLORS.get('COMPLEX', ''): 'COMPLEX',
        PRIMARY_COLORS.get('FOCUSED', ''): 'FOCUSED'
    }

    node_traces = []
    for color, nodes in node_groups.items():
        trace = go.Scatter(
            x=nodes['x'],
            y=nodes['y'],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=nodes['sizes'],
                color=color,
                line=dict(width=1, color='white')
            ),
            text=nodes['texts'],
            textposition='top center',
            textfont=dict(size=8),
            customdata=nodes['names'],
            hovertemplate='%{customdata}<extra></extra>',
            hovertext=nodes['hovers'],
            name=color_labels.get(color, 'Other')
        )
        node_traces.append(trace)

    fig = go.Figure(data=[edge_trace] + node_traces)

    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(t=20, b=20, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Network statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", G.number_of_nodes())
    with col2:
        st.metric("Edges (visible)", G.number_of_edges())
    with col3:
        st.metric("Sources", len(source_names))
    with col4:
        st.metric("Hubs", len(hub_names))

# ============================================================================
# TAB 2: EGO NETWORK VIEW
# ============================================================================
with tab2:
    st.subheader("Ego Network View")
    st.markdown("Select an individual to see only their direct connections")

    col1, col2 = st.columns([2, 1])

    with col1:
        ego_person = st.selectbox(
            "Select Individual",
            options=criminals,
            format_func=get_short_name,
            key="ego_select"
        )

    with col2:
        ego_depth = st.radio("Connection Depth", [1, 2], index=0, horizontal=True,
                             help="1 = direct connections, 2 = connections of connections")

    if ego_person:
        # Build ego network
        ego_idx = criminals.index(ego_person)

        # Find connected nodes
        connected = set([ego_person])

        # Outgoing connections (who ego influences)
        for j, target in enumerate(criminals):
            if te_matrix[ego_idx, j] > 0.1:
                connected.add(target)

        # Incoming connections (who influences ego)
        for i, source in enumerate(criminals):
            if te_matrix[i, ego_idx] > 0.1:
                connected.add(source)

        # If depth 2, add second-level connections
        if ego_depth == 2:
            first_level = connected.copy()
            for person in first_level:
                if person == ego_person:
                    continue
                p_idx = criminals.index(person)
                for j, target in enumerate(criminals):
                    if te_matrix[p_idx, j] > 0.15:
                        connected.add(target)
                for i, source in enumerate(criminals):
                    if te_matrix[i, p_idx] > 0.15:
                        connected.add(source)

        # Build ego subgraph
        ego_G = nx.DiGraph()
        for node in connected:
            ego_G.add_node(node)

        for source in connected:
            s_idx = criminals.index(source)
            for target in connected:
                t_idx = criminals.index(target)
                if source != target and te_matrix[s_idx, t_idx] > 0.05:
                    ego_G.add_edge(source, target, weight=te_matrix[s_idx, t_idx])

        # Layout with ego at center
        pos_ego = nx.spring_layout(ego_G, k=1.5, iterations=50, seed=42)
        # Force ego to center
        pos_ego[ego_person] = np.array([0, 0])

        # Create edge traces with varying thickness
        edge_traces = []
        for edge in ego_G.edges(data=True):
            x0, y0 = pos_ego[edge[0]]
            x1, y1 = pos_ego[edge[1]]
            weight = edge[2].get('weight', 0.1)

            # Color based on direction relative to ego
            if edge[0] == ego_person:
                color = 'rgba(231, 76, 60, 0.6)'  # Red - outgoing
            elif edge[1] == ego_person:
                color = 'rgba(52, 152, 219, 0.6)'  # Blue - incoming
            else:
                color = 'rgba(150, 150, 150, 0.3)'  # Gray - other

            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=max(1, weight * 10), color=color),
                hoverinfo='none',
                showlegend=False
            ))

        # Node trace
        node_x, node_y, node_colors, node_sizes, node_texts, hover_texts = [], [], [], [], [], []

        for node in ego_G.nodes():
            x, y = pos_ego[node]
            node_x.append(x)
            node_y.append(y)

            if node == ego_person:
                node_colors.append('#e74c3c')  # Red for ego
                node_sizes.append(40)
            elif node in hub_names:
                node_colors.append(ROLE_COLORS['HUB'])
                node_sizes.append(25)
            elif node in source_names:
                node_colors.append(ROLE_COLORS['SOURCE'])
                node_sizes.append(20)
            elif node in sink_names:
                node_colors.append(ROLE_COLORS['SINK'])
                node_sizes.append(20)
            else:
                node_colors.append(ROLE_COLORS['GENERAL'])
                node_sizes.append(15)

            node_texts.append(get_short_name(node))

            # Detailed hover
            c = classifications.get(node, {})
            n_idx = criminals.index(node)
            te_to_ego = te_matrix[n_idx, ego_idx] if n_idx != ego_idx else 0
            te_from_ego = te_matrix[ego_idx, n_idx] if n_idx != ego_idx else 0
            hover = f"<b>{get_short_name(node)}</b><br>"
            hover += f"Type: {c.get('subtype', '-')}<br>"
            hover += f"TE to {get_short_name(ego_person)}: {te_to_ego:.3f}<br>"
            hover += f"TE from {get_short_name(ego_person)}: {te_from_ego:.3f}"
            hover_texts.append(hover)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
            text=node_texts,
            textposition='top center',
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False
        )

        fig_ego = go.Figure(data=edge_traces + [node_trace])

        fig_ego.update_layout(
            title=f"Ego Network: {get_short_name(ego_person)}",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            margin=dict(t=40, b=20, l=20, r=20),
            annotations=[
                dict(text="<span style='color:#e74c3c'>Red edges</span> = outgoing influence | <span style='color:#3498db'>Blue edges</span> = incoming influence",
                     xref="paper", yref="paper", x=0.5, y=-0.05, showarrow=False, font=dict(size=10))
            ]
        )

        st.plotly_chart(fig_ego, use_container_width=True)

        # Ego stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Direct Connections", len(connected) - 1)
        with col2:
            st.metric("Outgoing Influence", f"{np.sum(te_matrix[ego_idx, :]):.2f}")
        with col3:
            st.metric("Incoming Influence", f"{np.sum(te_matrix[:, ego_idx]):.2f}")

        # Top connections table
        st.markdown("**Top Connections**")
        conn_data = []
        for node in connected:
            if node == ego_person:
                continue
            n_idx = criminals.index(node)
            conn_data.append({
                'Name': get_short_name(node),
                'Type': classifications.get(node, {}).get('subtype', '-'),
                'TE to Ego': te_matrix[n_idx, ego_idx],
                'TE from Ego': te_matrix[ego_idx, n_idx],
                'Net Flow': te_matrix[ego_idx, n_idx] - te_matrix[n_idx, ego_idx]
            })

        if conn_data:
            df_conn = pd.DataFrame(conn_data).sort_values('Net Flow', ascending=False)
            st.dataframe(df_conn, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: PATH FINDER
# ============================================================================
with tab3:
    st.subheader("Influence Path Finder")
    st.markdown("Find the shortest influence path between two individuals")

    col1, col2 = st.columns(2)

    with col1:
        path_start = st.selectbox(
            "From",
            options=criminals,
            format_func=get_short_name,
            key="path_start"
        )

    with col2:
        path_end = st.selectbox(
            "To",
            options=criminals,
            format_func=get_short_name,
            index=min(1, len(criminals)-1),
            key="path_end"
        )

    if path_start and path_end and path_start != path_end:
        # Build weighted graph for path finding
        path_G = nx.DiGraph()
        for i, source in enumerate(criminals):
            for j, target in enumerate(criminals):
                if i != j and te_matrix[i, j] > 0.05:
                    # Use inverse of TE as weight (higher TE = shorter path)
                    path_G.add_edge(source, target, weight=1.0 / (te_matrix[i, j] + 0.01))

        try:
            # Find shortest path
            path = nx.shortest_path(path_G, path_start, path_end, weight='weight')
            path_length = len(path) - 1

            st.success(f"Found path with {path_length} step(s)")

            # Visualize path
            # Create figure with path highlighted
            fig_path = go.Figure()

            # Draw all edges faintly
            for i, source in enumerate(criminals):
                for j, target in enumerate(criminals):
                    if i != j and te_matrix[i, j] > 0.1:
                        x0, y0 = pos[source]
                        x1, y1 = pos[target]
                        fig_path.add_trace(go.Scatter(
                            x=[x0, x1], y=[y0, y1],
                            mode='lines',
                            line=dict(width=0.3, color='rgba(200,200,200,0.2)'),
                            hoverinfo='none',
                            showlegend=False
                        ))

            # Draw path edges boldly with arrows
            path_te_values = []
            for i in range(len(path) - 1):
                source, target = path[i], path[i+1]
                x0, y0 = pos[source]
                x1, y1 = pos[target]

                s_idx = criminals.index(source)
                t_idx = criminals.index(target)
                te_val = te_matrix[s_idx, t_idx]
                path_te_values.append(te_val)

                # Draw edge
                fig_path.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=4, color='#e74c3c'),
                    hoverinfo='none',
                    showlegend=False
                ))

                # Add arrow annotation
                fig_path.add_annotation(
                    x=x1, y=y1,
                    ax=x0, ay=y0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='#e74c3c'
                )

            # Draw all nodes
            for node in criminals:
                x, y = pos[node]
                if node in path:
                    if node == path_start:
                        color = '#27ae60'  # Green for start
                        size = 30
                    elif node == path_end:
                        color = '#8e44ad'  # Purple for end
                        size = 30
                    else:
                        color = '#e74c3c'  # Red for path
                        size = 22
                else:
                    color = 'rgba(150,150,150,0.3)'
                    size = 8

                fig_path.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=size, color=color, line=dict(width=2, color='white')),
                    text=[get_short_name(node)] if node in path else [''],
                    textposition='top center',
                    hovertext=get_short_name(node),
                    hoverinfo='text',
                    showlegend=False
                ))

            fig_path.update_layout(
                title=f"Path: {get_short_name(path_start)} → {get_short_name(path_end)}",
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
                margin=dict(t=40, b=20, l=20, r=20)
            )

            st.plotly_chart(fig_path, use_container_width=True)

            # Path details
            st.markdown("**Path Details**")
            path_details = []
            for i, node in enumerate(path):
                c = classifications.get(node, {})
                step_info = {
                    'Step': i,
                    'Individual': get_short_name(node),
                    'Type': c.get('primary_type', '-'),
                    'Subtype': c.get('subtype', '-'),
                }
                if i < len(path) - 1:
                    step_info['TE to Next'] = f"{path_te_values[i]:.3f}"
                else:
                    step_info['TE to Next'] = '-'
                path_details.append(step_info)

            st.dataframe(pd.DataFrame(path_details), use_container_width=True, hide_index=True)

            # Cumulative TE
            total_te = sum(path_te_values)
            avg_te = total_te / len(path_te_values) if path_te_values else 0
            st.metric("Total Path TE", f"{total_te:.3f}", delta=f"Avg: {avg_te:.3f}")

        except nx.NetworkXNoPath:
            st.warning(f"No path found from {get_short_name(path_start)} to {get_short_name(path_end)}")
            st.info("Try lowering the edge threshold or selecting different individuals")

# ============================================================================
# TAB 4: TIMELINE LINEAGES
# ============================================================================
with tab4:
    st.subheader("Influence Chains (Timeline View)")

    if audience == 'practitioner':
        st.markdown("""
        These chains show how behavioral patterns appear to "inherit" from one criminal to another.
        Read left to right: each person's pattern influenced the next.
        """)
    else:
        st.markdown("""
        Lineages are chains of high transfer entropy, showing directed influence paths.
        Each chain represents a path where each individual's pattern significantly predicts the next.
        """)

    lineages = data.reincarnation.get('lineages', [])

    if lineages:
        n_show = st.slider("Number of lineages to show", 1, min(15, len(lineages)), 5, key="timeline_slider")

        for i, lineage in enumerate(lineages[:n_show]):
            # Create timeline figure
            fig_timeline = go.Figure()

            n_nodes = len(lineage)
            x_positions = list(range(n_nodes))
            y_position = 0

            # Draw connecting lines with TE values
            for j in range(n_nodes - 1):
                source = lineage[j]
                target = lineage[j + 1]
                s_idx = criminals.index(source) if source in criminals else -1
                t_idx = criminals.index(target) if target in criminals else -1

                te_val = te_matrix[s_idx, t_idx] if s_idx >= 0 and t_idx >= 0 else 0

                # Line
                fig_timeline.add_trace(go.Scatter(
                    x=[x_positions[j], x_positions[j+1]],
                    y=[y_position, y_position],
                    mode='lines',
                    line=dict(width=max(2, te_val * 15), color='#3498db'),
                    hoverinfo='none',
                    showlegend=False
                ))

                # Arrow
                fig_timeline.add_annotation(
                    x=x_positions[j+1] - 0.15,
                    y=y_position,
                    ax=x_positions[j] + 0.15,
                    ay=y_position,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowcolor='#3498db'
                )

                # TE label
                if audience == 'researcher':
                    fig_timeline.add_annotation(
                        x=(x_positions[j] + x_positions[j+1]) / 2,
                        y=y_position + 0.15,
                        text=f"TE: {te_val:.2f}",
                        showarrow=False,
                        font=dict(size=9, color='#666')
                    )

            # Draw nodes
            for j, name in enumerate(lineage):
                c = classifications.get(name, {})
                subtype = c.get('subtype', '-')

                # Color by role
                if name in hub_names:
                    color = ROLE_COLORS['HUB']
                elif name in source_names:
                    color = ROLE_COLORS['SOURCE']
                elif name in sink_names:
                    color = ROLE_COLORS['SINK']
                else:
                    color = ROLE_COLORS['GENERAL']

                fig_timeline.add_trace(go.Scatter(
                    x=[x_positions[j]],
                    y=[y_position],
                    mode='markers',
                    marker=dict(size=35, color=color, line=dict(width=2, color='white')),
                    hovertext=f"{get_short_name(name)}<br>{subtype}",
                    hoverinfo='text',
                    showlegend=False
                ))

                # Name label
                fig_timeline.add_annotation(
                    x=x_positions[j],
                    y=y_position - 0.35,
                    text=f"<b>{get_short_name(name)}</b><br><span style='font-size:10px'>{subtype[:15]}</span>",
                    showarrow=False,
                    font=dict(size=11)
                )

            fig_timeline.update_layout(
                title=f"Chain {i+1}",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, n_nodes - 0.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 0.5]),
                height=180,
                margin=dict(t=30, b=10, l=10, r=10)
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

        # Summary
        st.divider()
        st.markdown(f"**Total lineages identified:** {len(lineages)}")

        # Most frequent in lineages
        from collections import Counter
        all_in_lineages = [name for l in lineages for name in l]
        freq = Counter(all_in_lineages)
        most_common = freq.most_common(5)

        st.markdown("**Most frequent in lineages:**")
        for name, count in most_common:
            short_name = get_short_name(name)
            role = "Hub" if name in hub_names else "Source" if name in source_names else "Sink" if name in sink_names else "General"
            st.markdown(f"- **{short_name}** ({role}): appears in {count} chains")

    else:
        st.info("No lineages available. Run archetypal_reincarnation.py to compute them.")

# ============================================================================
# TAB 5: PATTERN FLOW (SANKEY)
# ============================================================================
with tab5:
    st.subheader("Pattern Flow (Sankey Diagram)")

    st.markdown("""
    This diagram shows how behavioral patterns flow from Sources through Hubs to Sinks.
    The width of each band represents the cumulative influence (transfer entropy).
    """)

    sources = roles.get('sources', [])
    hubs = roles.get('hubs', [])
    sinks = roles.get('sinks', [])

    if sources and sinks and te_matrix is not None:
        # Build Sankey data
        labels = []
        source_indices = []
        target_indices = []
        values = []

        # Add source names
        for s in sources:
            labels.append(get_short_name(s.get('name', '')) + ' (S)')
        source_start = 0
        source_end = len(sources)

        # Add hub names
        for h in hubs:
            labels.append(get_short_name(h.get('name', '')) + ' (H)')
        hub_start = source_end
        hub_end = hub_start + len(hubs)

        # Add sink names
        for s in sinks:
            labels.append(get_short_name(s.get('name', '')) + ' (K)')
        sink_start = hub_end
        sink_end = sink_start + len(sinks)

        # Source → Hub flows
        for i, src in enumerate(sources):
            src_idx = criminals.index(src['name']) if src['name'] in criminals else -1
            for j, hub in enumerate(hubs):
                hub_idx = criminals.index(hub['name']) if hub['name'] in criminals else -1
                if src_idx >= 0 and hub_idx >= 0:
                    te = te_matrix[src_idx, hub_idx]
                    if te > 0.1:
                        source_indices.append(source_start + i)
                        target_indices.append(hub_start + j)
                        values.append(float(te))

        # Hub → Sink flows
        for i, hub in enumerate(hubs):
            hub_idx = criminals.index(hub['name']) if hub['name'] in criminals else -1
            for j, snk in enumerate(sinks):
                snk_idx = criminals.index(snk['name']) if snk['name'] in criminals else -1
                if hub_idx >= 0 and snk_idx >= 0:
                    te = te_matrix[hub_idx, snk_idx]
                    if te > 0.1:
                        source_indices.append(hub_start + i)
                        target_indices.append(sink_start + j)
                        values.append(float(te))

        # Also add direct Source → Sink flows
        for i, src in enumerate(sources):
            src_idx = criminals.index(src['name']) if src['name'] in criminals else -1
            for j, snk in enumerate(sinks):
                snk_idx = criminals.index(snk['name']) if snk['name'] in criminals else -1
                if src_idx >= 0 and snk_idx >= 0:
                    te = te_matrix[src_idx, snk_idx]
                    if te > 0.2:
                        source_indices.append(source_start + i)
                        target_indices.append(sink_start + j)
                        values.append(float(te))

        if values:
            node_colors = (
                [ROLE_COLORS['SOURCE']] * len(sources) +
                [ROLE_COLORS['HUB']] * len(hubs) +
                [ROLE_COLORS['SINK']] * len(sinks)
            )

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=labels,
                    color=node_colors
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color='rgba(150, 150, 150, 0.4)'
                )
            )])

            fig_sankey.update_layout(
                title_text="Pattern Flow: Sources → Hubs → Sinks",
                font_size=10,
                height=500
            )

            st.plotly_chart(fig_sankey, use_container_width=True)

            # Legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<span style='color:{ROLE_COLORS['SOURCE']}'>&#9632;</span> **Sources (S)**: Pattern originators", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='color:{ROLE_COLORS['HUB']}'>&#9632;</span> **Hubs (H)**: Central connectors", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<span style='color:{ROLE_COLORS['SINK']}'>&#9632;</span> **Sinks (K)**: Pattern receivers", unsafe_allow_html=True)

        else:
            st.info("Not enough connections to display Sankey diagram.")

    else:
        st.info("Insufficient data for Sankey diagram. Need Sources, Hubs, Sinks, and transfer entropy matrix.")

# Role summary at bottom
st.divider()
st.subheader("Role Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"### {TERMINOLOGY['source']}s")
    for s in roles.get('sources', [])[:5]:
        name = get_short_name(s.get('name', ''))
        if audience == 'researcher':
            st.markdown(f"- **{name}** (out: {s.get('outgoing', 0):.2f})")
        else:
            st.markdown(f"- **{name}**")

with col2:
    st.markdown(f"### {TERMINOLOGY['hub']}s")
    for h in roles.get('hubs', [])[:5]:
        name = get_short_name(h.get('name', ''))
        if audience == 'researcher':
            st.markdown(f"- **{name}** (score: {h.get('hub_score', 0):.2f})")
        else:
            st.markdown(f"- **{name}**")

with col3:
    st.markdown(f"### {TERMINOLOGY['sink']}s")
    for s in roles.get('sinks', [])[:5]:
        name = get_short_name(s.get('name', ''))
        if audience == 'researcher':
            st.markdown(f"- **{name}** (in: {s.get('incoming', 0):.2f})")
        else:
            st.markdown(f"- **{name}**")
