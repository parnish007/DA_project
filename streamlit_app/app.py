import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="DA Project Dashboard",
    page_icon="🖥️",
    layout="wide",
)

# ------------------------------
# Load custom CSS
# ------------------------------
st.markdown(
    '<link rel="stylesheet" href="streamlit_app/assets/styles.css">',
    unsafe_allow_html=True
)

# ------------------------------
# Imports
# ------------------------------
from app_backend.data_loader import DataLoader
from app_backend.data_cleaning import DataCleaning
from app_backend.eda_analysis import EDAEngine
from app_backend.viz_charts import VizEngine

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Visualizations", "Correlation Analysis"]
)

# ------------------------------
# File uploader (persistent)
# ------------------------------
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    if "df" not in st.session_state or st.session_state.get("uploaded_file") != uploaded_file:
        loader = DataLoader(uploaded_file)
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
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"{icon} {title}")
    content_func(**kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Page Functions
# ------------------------------
def show_overview():
    if df is None:
        st.info("Please upload a CSV file to begin.")
        return

    def dataset_snapshot():
        st.write(df.head(10))

    def basic_overview():
        eda = EDAEngine(df)
        st.write(eda.get_overview())

    def summary_stats():
        eda = EDAEngine(df)
        st.dataframe(eda.summary_stats())

    def preview_rows():
        idx = st.number_input("Enter row index to preview", min_value=0, max_value=len(df)-1, value=0)
        st.write(df.iloc[idx])

    def simple_viz():
        viz = VizEngine(df)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column for histogram", numeric_cols)
            st.pyplot(viz.histogram(col, bins=20))
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
        st.pyplot(viz.histogram(col, bins=20, kde=True))
        box_fig, box_info = viz.boxplot(col)
        st.pyplot(box_fig)
        st.write("Boxplot Info:", box_info)

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
        method = st.selectbox("Select correlation method", ["pearson", "spearman", "kendall"])
        corr_matrix = eda.correlation_matrix(method=method)
        st.dataframe(corr_matrix)
        return corr_matrix

    def top_corr_section(corr_matrix):
        target_col = st.selectbox("Select target column (optional)", [None] + numeric_cols)
        top_n = st.slider("Number of top correlations", 3, 20, 10)
        st.dataframe(eda.top_correlations(target_col=target_col, n=top_n))

    def heatmap_section(corr_matrix):
        annot = st.checkbox("Annotate values on heatmap?", value=True)
        st.pyplot(viz.heatmap(corr_matrix, annot=annot))

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
