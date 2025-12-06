# streamlit_app/pages/3_Correlation.py

import streamlit as st
from app_backend import data_loader, EDAEngine, VizEngine

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Correlation Analysis",
    page_icon="📉",
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
def correlation_matrix_section(df):
    eda = EDAEngine(df)
    viz = VizEngine(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found for correlation analysis.")
        return

    corr_method = st.selectbox("Select correlation method", ["pearson", "spearman", "kendall"])
    corr_matrix = eda.correlation_matrix(method=corr_method)
    st.write("Correlation Matrix:")
    st.dataframe(corr_matrix)

    # Top correlations
    st.subheader("🌟 Top Correlations")
    target_col = st.selectbox("Select target column (optional)", [None] + numeric_cols)
    top_n = st.slider("Number of top correlations", 3, 20, 10)

    if target_col is None:
        top_corr = eda.top_correlations(target_col=None, n=top_n)
    else:
        top_corr = eda.top_correlations(target_col=target_col, n=top_n)

    st.dataframe(top_corr)

    # Heatmap
    st.subheader("📊 Correlation Heatmap")
    annot = st.checkbox("Annotate values on heatmap?", value=True)
    heatmap_fig = viz.heatmap(corr_matrix, annot=annot)
    st.pyplot(heatmap_fig)

# ------------------------------
# Main
# ------------------------------
def main():
    st.title("📉 Correlation Analysis")
    st.markdown(
        """
        Explore correlations between numeric features:
        - Correlation matrix
        - Top correlated features
        - Heatmap visualization
        """
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="corr_uploader")

    if uploaded_file:
        # Persist dataset in session state
        if "df" not in st.session_state or st.session_state.get("uploaded_file") != uploaded_file:
            loader = data_loader(uploaded_file)
            df = loader.df_copy()
            st.session_state.df = df
            st.session_state.uploaded_file = uploaded_file
        else:
            df = st.session_state.df

        # Render correlation section in a card
        card_section("Correlation Analysis", "📉", correlation_matrix_section, df=df)
    else:
        st.info("Please upload a CSV file to analyze correlations.")

# Call main
if __name__ == "__main__":
    main()
