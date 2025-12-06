# streamlit_app/app.py

import streamlit as st
from app_backend import data_loader, DataCleaning, EDAEngine, VizEngine, safe_execute

# Load custom CSS
st.markdown(
    '<link rel="stylesheet" href="streamlit_app/assets/styles.css">', 
    unsafe_allow_html=True
)

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(
    page_title="DA Project Dashboard",
    page_icon="🖥️",
    layout="wide",
)

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Visualizations", "Correlation Analysis"]
)

# ------------------------------
# File upload (persistent)
# ------------------------------
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    if "df" not in st.session_state or st.session_state.get("uploaded_file") != uploaded_file:
        loader = data_loader(uploaded_file)
        df = loader.df_copy()
        st.session_state.df = df
        st.session_state.uploaded_file = uploaded_file
    else:
        df = st.session_state.df
else:
    df = None

# ------------------------------
# Helper function to wrap sections in cards
# ------------------------------
def card_section(title, icon, content_func, **kwargs):
    st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"{icon} {title}")
    content_func(**kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Page Content
# ------------------------------
def show_overview():
    if df is None:
        st.info("Please upload a CSV file to begin.")
        return

    def dataset_snapshot():
        st.write(df.head(10))

    def basic_overview():
        eda = EDAEngine(df)
        overview = eda.get_overview()
        st.write(overview)

    def summary_stats():
        eda = EDAEngine(df)
        stats = eda.summary_stats()
        st.dataframe(stats)

    def preview_rows():
        idx = st.number_input("Enter row index to preview", min_value=0, max_value=len(df)-1, value=0)
        st.write(df.iloc[idx])

    def simple_viz():
        viz = VizEngine(df)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select a numeric column for histogram", numeric_cols)
            hist_fig = viz.histogram(col, bins=20)
            st.pyplot(hist_fig)
        else:
            st.info("No numeric columns found for histogram.")

    card_section("Dataset Snapshot", "✅", dataset_snapshot)
    card_section("Basic Overview", "📌", basic_overview)
    card_section("Summary Statistics", "🔢", summary_stats)
    card_section("Sample Rows by Index", "🕵️", preview_rows)
    card_section("Simple Visualizations", "📉", simple_viz)

def show_visualizations():
    if df is None:
        st.info("Please upload a CSV file to begin visualizations.")
        return

    viz = VizEngine(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def numeric_viz():
        if not numeric_cols:
            st.info("No numeric columns found.")
            return
        col = st.selectbox("Select numeric column for histogram", numeric_cols)
        hist_fig = viz.histogram(col, bins=20, kde=True)
        if isinstance(hist_fig, dict):
            st.write("Histogram data & KDE preview:")
            st.json(hist_fig)
        else:
            st.pyplot(hist_fig)

        box_fig, box_info = viz.boxplot(col)
        st.pyplot(box_fig)
        st.write("Boxplot Info:", box_info)

        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, index=0)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)
            scatter_fig, scatter_data = viz.scatter(x_col, y_col)
            st.pyplot(scatter_fig)
            st.write(f"Number of points: {scatter_data['n_points']}")

        pairplot_fig = viz.pairplot(cols=numeric_cols[:6])
        st.pyplot(pairplot_fig)

        corr_matrix = df[numeric_cols].corr()
        heatmap_fig = viz.heatmap(corr_matrix, annot=True)
        st.pyplot(heatmap_fig)

    def categorical_viz():
        if not categorical_cols:
            st.info("No categorical columns found.")
            return
        cat_col = st.selectbox("Select categorical column for bar chart", categorical_cols)
        bar_fig, bar_data = viz.bar_chart(cat_col)
        st.pyplot(bar_fig)
        st.write("Top values and counts:", dict(zip(bar_data["labels"], bar_data["values"])))

    card_section("Numeric Visualizations", "🔢", numeric_viz)
    card_section("Categorical Visualizations", "🟠", categorical_viz)

def show_correlation():
    if df is None:
        st.info("Please upload a CSV file to analyze correlations.")
        return

    eda = EDAEngine(df)
    viz = VizEngine(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for correlation analysis.")
        return

    def corr_matrix_section():
        corr_method = st.selectbox("Select correlation method", ["pearson", "spearman", "kendall"])
        corr_matrix = eda.correlation_matrix(method=corr_method)
        st.dataframe(corr_matrix)
        return corr_matrix

    def top_corr_section(corr_matrix):
        target_col = st.selectbox("Select target column (optional)", [None] + numeric_cols)
        top_n = st.slider("Number of top correlations", 3, 20, 10)
        top_corr = eda.top_correlations(target_col=target_col, n=top_n)
        st.dataframe(top_corr)

    def heatmap_section(corr_matrix):
        annot = st.checkbox("Annotate values on heatmap?", value=True)
        heatmap_fig = viz.heatmap(corr_matrix, annot=annot)
        st.pyplot(heatmap_fig)

    corr_matrix = corr_matrix_section()
    card_section("Top Correlations", "🌟", top_corr_section, corr_matrix=corr_matrix)
    card_section("Correlation Heatmap", "📊", heatmap_section, corr_matrix=corr_matrix)

# ------------------------------
# Page Router
# ------------------------------
if page == "Overview":
    show_overview()
elif page == "Visualizations":
    show_visualizations()
elif page == "Correlation Analysis":
    show_correlation()
