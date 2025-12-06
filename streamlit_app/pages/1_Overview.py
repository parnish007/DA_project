import streamlit as st
from app_backend.data_loader import DataLoader
from app_backend.eda_analysis import EDAEngine
from app_backend.viz_charts import VizEngine

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Dataset Overview",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Helper function for card section
# ------------------------------
def card_section(title, icon, content_func, **kwargs):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"{icon} {title}")
    content_func(**kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Section functions
# ------------------------------
def dataset_snapshot(df):
    st.dataframe(df.head(10))

def basic_overview(df):
    eda = EDAEngine(df)
    st.json(eda.get_overview())

def summary_statistics(df):
    eda = EDAEngine(df)
    st.dataframe(eda.summary_stats())

def sample_rows(df):
    idx = st.number_input("Enter row index to preview", min_value=0, max_value=len(df)-1, value=0)
    st.write(df.iloc[idx])

def simple_visualizations(df):
    viz = VizEngine(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col = st.selectbox("Select a numeric column for histogram", numeric_cols)
        hist_fig = viz.histogram(col, bins=20, kde=True)
        st.pyplot(hist_fig)

        st.subheader("Boxplot")
        box_fig, box_info = viz.boxplot(col)
        st.pyplot(box_fig)
        st.write("Boxplot info:", box_info)
    else:
        st.info("No numeric columns found for visualizations.")

# ------------------------------
# Main function
# ------------------------------
def main():
    st.title("📊 Dataset Overview")
    st.markdown("""
        Quick overview of your dataset:
        - Shape, columns, missing values
        - Summary statistics
        - Sample rows
        - Basic visualizations
    """)

    # ------------------------------
    # File uploader
    # ------------------------------
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="overview_uploader")

    if uploaded_file:
        # Load dataset only if not already loaded or file changed
        if "df" not in st.session_state or st.session_state.get("uploaded_file") != uploaded_file:
            loader = DataLoader(uploaded_file)  # Works with Streamlit uploaded files
            df = loader.df_copy()
            st.session_state.df = df
            st.session_state.uploaded_file = uploaded_file
        else:
            df = st.session_state.df

        # ------------------------------
        # Render sections in cards
        # ------------------------------
        card_section("Dataset Snapshot", "✅", dataset_snapshot, df=df)
        card_section("Basic Overview", "📌", basic_overview, df=df)
        card_section("Summary Statistics", "🔢", summary_statistics, df=df)
        card_section("Sample Rows by Index", "🕵️", sample_rows, df=df)
        card_section("Simple Visualizations", "📉", simple_visualizations, df=df)
    else:
        st.info("Please upload a CSV file to begin.")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    main()
