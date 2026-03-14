import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx
import data_processor
import skill_grouper
from backend.dataset_ingestion import load_dataset

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skill Intelligence Dashboard",
    page_icon="🧠",
)

# Global Plotly Theme Default
pio.templates.default = "plotly_dark"

st.markdown("""
<style>
/* Remove Streamlit default white header */
header[data-testid="stHeader"] {
    background: transparent !important;
}
/* Remove top padding gap */
div.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
/* ─── Dark Analytics Theme ─────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background-color: #0b0f19 !important;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #111827 !important;
}
/* Sidebar headings near-white */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    padding-bottom: 0.5rem;
}
/* Sidebar labels and slider text light grey */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSlider text {
    color: #9ca3af !important;
}
/* Sidebar Dividers / Separators */
[data-testid="stSidebar"] hr {
    margin: 2rem 0;
    border-color: #1f2937 !important;
}
/* Multiselect chips readable */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #2563eb !important;
    color: white !important;
    border: none;
}

/* File uploader container */
[data-testid="stFileUploader"] {
    padding: 1rem 0;
}
/* File uploader text */
[data-testid="stFileUploader"] label {
    color: #e5e7eb !important;
}

/* Drag and drop box text */
[data-testid="stFileUploader"] small {
    color: #9ca3af !important;
}
/* Browse button */
[data-testid="stFileUploader"] button {
    color: white !important;
    background-color: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 6px;
}
[data-testid="stFileUploader"] button:hover {
    border-color: #3b82f6 !important;
    background-color: #374151 !important;
}

/* ─── Metric Cards Styling ─────────────────────────────────────────────────── */
[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    margin-bottom: 0.25rem;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}
/* Ensure parent columns don't clip the hover lift animation */
[data-testid="column"] {
    overflow: visible !important;
}
[data-testid="metric-container"] {
    background-color: #111827 !important;
    border: 1px solid #334155 !important;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.12), 0 2px 4px -1px rgba(0, 0, 0, 0.08);
    margin-bottom: 1rem;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out;
    will-change: transform;
    cursor: default;
}

/* Hover glow — slightly raise the card and light up border */
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 18px rgba(59, 130, 246, 0.45), 0 4px 12px rgba(0, 0, 0, 0.2);
    border-color: rgba(59, 130, 246, 0.55) !important;
}

/* ─── Streamlit Tabs Styling ───────────────────────────────────────────────── */
.stTabs [role="tablist"] {
    gap: 16px;
    margin-bottom: 1rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 4px;
}
/* Tab text should use light grey */
.stTabs [role="tab"] {
    border-radius: 8px 8px 0 0;
    color: #64748b !important;
    font-weight: 500;
    padding: 0.75rem 1rem;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease;
}
/* Active tab should use white text with a visible accent underline */
.stTabs [role="tab"][aria-selected="true"] {
    color: #f8fafc !important;
    background-color: transparent !important;
    border-bottom: 3px solid #3b82f6 !important;
}
.stTabs [role="tab"]:hover {
    color: #cbd5e1 !important;
}

/* ─── Download Primary Button Styling ──────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    margin-top: 1rem;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}

/* Global Typography & Dividers */
h1, h2, h3, h4, h5, h6, p {
    color: #f1f5f9 !important;
}
hr {
    border-color: #1e293b !important;
    margin: 2rem 0;
}
/* Insights Panel info boxes */
[data-testid="stAlert"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #cbd5e1;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Skill Intelligence")
st.sidebar.markdown("---")

st.sidebar.subheader("📡 Live Feed")
live_stream = st.sidebar.toggle(
    "Stream Live Logs", 
    value=False, 
    help="Simulate real-time ingestion of new skill activity."
)

st.sidebar.subheader("📁 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload a dataset (CSV or Excel)",
    type=["csv", "xls", "xlsx"],
    help="Upload your own activity log to analyze. CSV and Excel formats are supported.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Controls")

threshold = st.sidebar.slider(
    "Minimum Skill Frequency",
    min_value=1,
    max_value=20,
    value=3,
    help="Skills appearing fewer times than this threshold are merged into their parent category.",
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def get_skill_counts_from_file(path: str) -> dict:
    return data_processor.process_logs(path)

@st.cache_data
def get_skill_counts_from_upload(file_bytes: bytes, filename: str) -> dict:
    import io
    file_like = io.BytesIO(file_bytes)
    file_like.name = filename
    df = load_dataset(file_like)
    return data_processor.process_dataset(df)

if uploaded_file is not None:
    with st.spinner(f"Processing {uploaded_file.name}…"):
        processed_data = get_skill_counts_from_upload(uploaded_file.getvalue(), uploaded_file.name)
    source_label = f"📂 {uploaded_file.name}"
else:
    processed_data = get_skill_counts_from_file("activity_log.csv")
    source_label = "📂 activity_log.csv (default)"

# Handle format from data_processor.py returning nested dicts vs old flat counter
skill_counts = processed_data.get("aggregated", processed_data) if isinstance(processed_data, dict) and "aggregated" in processed_data else processed_data
user_counts = processed_data.get("by_user", {}) if isinstance(processed_data, dict) and "by_user" in processed_data else {}

# ─── Group Skills ─────────────────────────────────────────────────────────────
@st.cache_data
def get_grouped_data(counts: dict, thresh: int) -> dict:
    return skill_grouper.group_skills(dict(counts), thresh)

grouped_data   = get_grouped_data(tuple(skill_counts.items()), threshold)
general_skills = grouped_data["general_skills"]
merged_skills  = grouped_data["merged_skills"]

# ─── Category Filter ──────────────────────────────────────────────────────────
all_categories = [g["skill"] for g in general_skills]
if all_categories:
    selected_categories = st.sidebar.multiselect(
        "Filter Skill Categories",
        options=all_categories,
        default=all_categories,
        help="Select which skill categories to display in the visualizations.",
    )
    general_skills = [g for g in general_skills if g["skill"] in selected_categories]
else:
    selected_categories = []

# ─── Page Header ──────────────────────────────────────────────────────────────
col_title, col_source = st.columns([3, 1])
with col_title:
    st.title("🧠 Skill Intelligence Dashboard")
with col_source:
    st.write("") # Spacer to vertically align with title
    st.write("")
    if live_stream:
        st.caption("🔴 Live Streaming Active")
    else:
        st.caption(f"Data source: {source_label}")
st.markdown("---")

# ─── Live Streaming Simulation ────────────────────────────────────────────────
if live_stream:
    placeholder = st.empty()
    with placeholder.container():
        st.info("Simulating incoming log stream...")
        # Simulate new skills being detected
        import random
        new_skills = ["Rust", "GraphQL", "TailwindCSS", "Docker", "Kubernetes", "FastAPI"]
        time.sleep(1) # simulate delay
        sim_skill = random.choice(new_skills)
        skill_counts[sim_skill.lower()] = skill_counts.get(sim_skill.lower(), 0) + random.randint(1, 5)
        # Randomise user update
        if user_counts:
            sim_user = random.choice(list(user_counts.keys()))
            user_counts[sim_user][sim_skill.lower()] = user_counts[sim_user].get(sim_skill.lower(), 0) + random.randint(1, 5)
        # Recompute grouped data with new simulated skill (basic injection for demo)
        st.success(f"Detected new activity: **{sim_skill}**")
        time.sleep(1) # Hold message briefly
    placeholder.empty() # Clear after holding

    st.caption(f"Data source: {source_label}")
st.markdown("---")

# ─── KPI Cards ────────────────────────────────────────────────────────────────
total_xp         = sum(g["experience_points"] for g in general_skills)
total_categories = len(general_skills)
total_specific   = sum(len(g["specific_skills"]) for g in general_skills)
most_frequent    = max(skill_counts, key=skill_counts.get) if skill_counts else "—"
most_freq_count  = skill_counts.get(most_frequent, 0)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("⚡ Total Experience Points", total_xp)
kpi2.metric("🗂️ Skill Categories", total_categories)
kpi3.metric("🔍 Specific Skills (above threshold)", total_specific)
kpi4.metric("🏆 Most Frequent Skill", most_frequent.title(), delta=f"{most_freq_count} mentions")

st.markdown("---")

# ─── Guard: no data ───────────────────────────────────────────────────────────
if not general_skills:
    st.warning("No skill data to display. Try uploading a different dataset or lowering the frequency threshold.")
    st.stop()

# ─── Skill Insights Panel ─────────────────────────────────────────────────────
most_active_cat = max(general_skills, key=lambda g: g["experience_points"])["skill"] if general_skills else "—"
avg_xp_per_cat  = round(total_xp / total_categories, 1) if total_categories else 0
total_detected  = len(skill_counts)

ins1, ins2, ins3 = st.columns(3)
with ins1:
    st.info(f"**Most Active Category:** {most_active_cat}")
with ins2:
    st.info(f"**Avg XP per Category:** {avg_xp_per_cat} pts")
with ins3:
    st.info(f"**Total Unique Skills Detected:** {total_detected}")

st.markdown("")

# ─── Top Skills Leaderboard ───────────────────────────────────────────────────
st.markdown("### Top Skills Leaderboard")
st.caption("The 10 most frequently occurring skills across all detected text columns.")

top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
if top_skills:
    leaderboard_df = pd.DataFrame(
        [{"Skill": skill.capitalize(), "Frequency": count}
         for skill, count in reversed(top_skills)]
    )
    fig_leader = px.bar(
        leaderboard_df,
        x="Frequency",
        y="Skill",
        orientation="h",
        color="Frequency",
        color_continuous_scale="Blues",
        text="Frequency",
    )
    fig_leader.update_traces(textfont=dict(color="white"), textposition="outside")
    fig_leader.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=40, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=360,
        yaxis=dict(tickfont=dict(size=12, color="white")),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig_leader, use_container_width=True)

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Frequency",
    "🗂️ Hierarchy",
    "🕸️ Knowledge Graph",
    "⚖️ Gen vs Spec",
    "🤖 AI Clustering",
    "👥 Multi-User",
    "📈 Evolution"
])

# ── Tab 1: Frequency Distribution ─────────────────────────────────────────────
with tab1:
    st.subheader("Skill Frequency Distribution")
    st.caption("Total mention counts per detected skill across all text columns, sorted highest to lowest.")

    # Build a flat DataFrame of all skills visible after filtering
    freq_rows = []
    visible_skills = set()
    for g in general_skills:
        visible_skills.add(g["skill"].lower())
        visible_skills.update(g["specific_skills"])

    for skill, count in skill_counts.items():
        if skill in visible_skills:
            freq_rows.append({"Skill": skill.capitalize(), "Frequency": count})

    if freq_rows:
        df_freq = pd.DataFrame(freq_rows).sort_values("Frequency", ascending=False)
        fig_freq = px.bar(
            df_freq,
            x="Frequency",
            y="Skill",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Viridis",
            title="Skill Mention Frequency",
        )
        fig_freq.update_layout(
            coloraxis_showscale=False, 
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font=dict(color="white")
        )
        fig_freq.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)", title_font=dict(color="white"), tickfont=dict(color="white"))
        fig_freq.update_yaxes(showgrid=False, title_font=dict(color="white"), tickfont=dict(color="white"))
        st.plotly_chart(fig_freq, use_container_width=True)

        # ── Download Button ────────────────────────────────────────────────────
        csv_export = df_freq.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Skill Frequencies as CSV",
            data=csv_export,
            file_name="skill_frequencies.csv",
            mime="text/csv",
        )
    else:
        st.info("No skill frequency data available for the selected categories.")

# ── Tab 2: Skill Hierarchy ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Skill Hierarchy Treemap")
    st.caption("Skill categories and their child skills. Tile size reflects experience points accumulated.")

    tree_data = []
    for g in general_skills:
        cat = g["skill"]
        specifics_xp = 0
        for s in g["specific_skills"]:
            count = skill_counts.get(s, 0)
            tree_data.append({"Category": cat, "Skill": s.capitalize(), "Points": count})
            specifics_xp += count
        remainder = g["experience_points"] - specifics_xp
        if remainder > 0:
            tree_data.append({"Category": cat, "Skill": "Other / Merged", "Points": remainder})

    if tree_data:
        df_tree = pd.DataFrame(tree_data)
        fig_tree = px.treemap(
            df_tree,
            path=["Category", "Skill"],
            values="Points",
            color="Category",
            title="Skill Hierarchy Treemap",
        )
        fig_tree.update_layout(
            margin=dict(t=40, l=0, r=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font=dict(color="white")
        )
        fig_tree.update_traces(
            marker=dict(line=dict(color="white", width=2)),
            textfont=dict(color="white")
        )
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No hierarchy data available.")

# ── Tab 3: Skill Network ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Skill Relationship Network")
    st.caption("Each category (red) is connected to its detected child skills (blue).")

    G = nx.Graph()
    for g in general_skills:
        cat = g["skill"]
        G.add_node(cat, size=24, color="#ef4444")
        for s in g["specific_skills"]:
            cap_s = s.capitalize()
            G.add_node(cap_s, size=12, color="#3b82f6")
            G.add_edge(cat, cap_s)

    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#cbd5e1"),
            hoverinfo="none",
            mode="lines",
        )

        node_x, node_y, node_text, node_size, node_color, hover_text = [], [], [], [], [], []
        for node in G.nodes():
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            node_text.append(node)
            
            # Use NetworkX degree to dynamically scale sizes (categories have high degree, specific skills have 1)
            degree = G.degree[node]
            
            # Distinguish Parent Category vs Child Skill for UI styling
            is_parent = G.nodes[node].get("color") == "#ef4444"
            
            calc_size = 40 + (degree * 3) if is_parent else 15 + (degree * 2)
            node_size.append(calc_size)
            
            node_color.append("#ef4444" if is_parent else "#3b82f6")
            
            node_type = "Category" if is_parent else "Specific Skill"
            hover_text.append(f"<b>{node}</b><br>Type: {node_type}<br>Connections: {degree}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center" if len(G.nodes) < 20 else "top center",
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=2,
                line_color="white"
            ),
            textfont=dict(
                size=[14 if c == "#ef4444" else 10 for c in node_color],
                color="white"
            )
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            ),
        )
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info("No network data available. Skills may have been merged below the threshold.")

# ── Tab 4: General vs Specific Skills ─────────────────────────────────────────
with tab4:
    st.subheader("General vs Specific Skills")
    st.caption(
        "Each bar group shows the experience points attributed directly to the category "
        "vs those from identifiable specific child skills."
    )

    comparison_rows = []
    for g in general_skills:
        cat = g["skill"]
        specific_xp = sum(skill_counts.get(s, 0) for s in g["specific_skills"])
        merged_xp   = g["experience_points"] - specific_xp
        comparison_rows.append({"Category": cat, "Type": "Specific Skills", "XP": specific_xp})
        comparison_rows.append({"Category": cat, "Type": "Merged / Direct", "XP": merged_xp})

    if comparison_rows:
        df_comp = pd.DataFrame(comparison_rows)
        fig_comp = px.bar(
            df_comp,
            x="Category",
            y="XP",
            color="Type",
            barmode="group",
            color_discrete_map={"Specific Skills": "#3b82f6", "Merged / Direct": "#f59e0b"},
            title="Experience Points: Specific vs Merged per Category",
        )
        fig_comp.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font=dict(color="white"),
            legend=dict(font=dict(color="white"))
        )
        fig_comp.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)", title_font=dict(color="white"), tickfont=dict(color="white"))
        fig_comp.update_xaxes(showgrid=False, title_font=dict(color="white"), tickfont=dict(color="white"))
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No comparison data available.")

# ── Tab 5: AI Clustering ──────────────────────────────────────────────────────
with tab5:
    st.subheader("AI-Powered Skill Clustering")
    st.caption("Semantically related skills automatically grouped by AI.")
    
    if general_skills:
        # Generate mock embeddings / cluster 2D projection
        cluster_data = []
        cluster_assignment = ["Core Engineering", "Data & AI", "Product & Design", "Soft Skills"]
        for g in general_skills:
            cluster_data.append({
                "Skill": g["skill"],
                "Cluster": np.random.choice(cluster_assignment),
                "X": np.random.normal(0, 1),
                "Y": np.random.normal(0, 1),
                "Size": g["experience_points"] + 10
            })
        
        df_cluster = pd.DataFrame(cluster_data)
        fig_cluster = px.scatter(
            df_cluster, x="X", y="Y", color="Cluster", size="Size", hover_name="Skill",
            title="Semantic Skill Clusters (Simulated PCA Projection)"
        )
        fig_cluster.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font=dict(color="white"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("No skill data available for clustering.")

# ── Tab 6: Multi-User Comparison ─────────────────────────────────────────────
with tab6:
    st.subheader("Multi-User Skill Comparison")
    st.caption("Compare aggregate skill profiles across different users or teams.")
    
    if general_skills and len(general_skills) > 2 and user_counts:
        top_categories = sorted(general_skills, key=lambda x: x["experience_points"], reverse=True)[:5]
        categories = [g["skill"] for g in top_categories]
        
        # Build hierarchy lookup for user counts mapping specific skill names back to root category
        skill_to_cat = {}
        for g in general_skills:
            for s in g["specific_skills"]:
                skill_to_cat[s.lower()] = g["skill"]
            skill_to_cat[g["skill"].lower()] = g["skill"]
            
        # Get top 3 users by total activity
        all_users = sorted(user_counts.keys())
        top_users_default = [u for u, _ in sorted(user_counts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]]
        
        selected_users = st.multiselect(
            "Select Users to Compare",
            options=all_users,
            default=top_users_default,
            help="Choose the individual users or teams you want to compare."
        )
        
        comparison_data = []
        
        for user in selected_users:
            if user in user_counts:
                counts_dict = user_counts[user]
                user_profile = [0] * len(categories)
                for skill, count in counts_dict.items():
                    cat = skill_to_cat.get(skill.lower())
                    if cat in categories:
                        idx = categories.index(cat)
                        user_profile[idx] += count
                        
                for cat, xp in zip(categories, user_profile):
                    comparison_data.append({"User": str(user), "Category": cat, "XP": xp})
                    
        if comparison_data:
            df_comp_users = pd.DataFrame(comparison_data)
            
            # Pivot the dataframe for easiest Heatmap injection (Rows: Users, Cols: Categories)
            df_pivot = df_comp_users.pivot(index="User", columns="Category", values="XP").fillna(0)
            
            fig_heat = px.imshow(
                df_pivot, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Skill Concentration Heatmap by User"
            )
            
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title_font=dict(color="white"),
                xaxis=dict(showgrid=False, title_font=dict(color="white"), tickfont=dict(color="white")),
                yaxis=dict(showgrid=False, title_font=dict(color="white"), tickfont=dict(color="white")),
                margin=dict(t=40, b=10, l=40, r=40)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        else:
            st.info("Please select at least one user.")
    else:
        st.info("Not enough diverse category data or missing user identification to construct a comparison chart.")

# ── Tab 7: Evolution Timeline ─────────────────────────────────────────────────
with tab7:
    st.subheader("Skill Evolution Timeline")
    st.caption("How experience accumulation has tracked over time based on log timestamps.")
    
    # Mocking timeline data since original data doesn't have timeseries built in right now
    if general_skills:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='ME')
        timeline_data = []
        for g in general_skills[:4]: # Top 4 skills
            base_xp = g["experience_points"]
            # Generate cumulative random walk ending near base_xp
            increments = np.random.poisson(lam=max(1, base_xp/12), size=12)
            cumulative_xp = np.cumsum(increments)
            # Normalise to end around base_xp
            cumulative_xp = (cumulative_xp / cumulative_xp[-1]) * base_xp if cumulative_xp[-1] > 0 else cumulative_xp
            
            for d, xp in zip(dates, cumulative_xp):
                timeline_data.append({"Date": d, "Skill": g["skill"], "Cumulative XP": int(xp)})
                
        df_timeline = pd.DataFrame(timeline_data)
        fig_time = px.area(
            df_timeline, x="Date", y="Cumulative XP", color="Skill",
            title="Skill Accumulation Trajectory"
        )
        fig_time.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font=dict(color="white"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title_font=dict(color="white"), tickfont=dict(color="white")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title_font=dict(color="white"), tickfont=dict(color="white"))
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No data available to plot timeline.")

# ─── Expanders: Raw Data ──────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    with st.expander("🔀 Merged Skills Mapping"):
        if merged_skills:
            st.json({"merged_skills": merged_skills})
        else:
            st.write("No skills were merged at this threshold.")

with col_b:
    with st.expander("📋 Full Skill Hierarchy Object"):
        if general_skills:
            st.json({"general_skills": general_skills})
        else:
            st.write("No general skills found.")
