# streamlit_app/pages/2_Visualizations.py

import streamlit as st
from app_backend import data_loader, VizEngine, EDAEngine

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Data Visualizations",
    page_icon="📈",
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
def numeric_visualizations(df):
    viz = VizEngine(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col = st.selectbox("Select numeric column for histogram", numeric_cols)
        hist_fig = viz.histogram(col, bins=20, kde=True)
        if isinstance(hist_fig, dict):
            st.write("Histogram data & KDE preview:")
            st.json(hist_fig)
        else:
            st.pyplot(hist_fig)

        st.subheader("Boxplot")
        box_fig, box_info = viz.boxplot(col)
        st.pyplot(box_fig)
        st.write("Boxplot Info:", box_info)

        if len(numeric_cols) >= 2:
            st.subheader("Scatter Plot")
            x_col = st.selectbox("X-axis", numeric_cols, index=0)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)
            scatter_fig, scatter_data = viz.scatter(x_col, y_col)
            st.pyplot(scatter_fig)
            st.write(f"Number of points: {scatter_data['n_points']}")

        st.subheader("Pairplot")
        pairplot_fig = viz.pairplot(cols=numeric_cols[:6])
        st.pyplot(pairplot_fig)

        st.subheader("Correlation Heatmap")
        corr_matrix = df[numeric_cols].corr()
        heatmap_fig = viz.heatmap(corr_matrix, annot=True)
        st.pyplot(heatmap_fig)
    else:
        st.info("No numeric columns found.")

def categorical_visualizations(df):
    viz = VizEngine(df)
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if categorical_cols:
        cat_col = st.selectbox("Select categorical column for bar chart", categorical_cols)
        bar_fig, bar_data = viz.bar_chart(cat_col)
        st.pyplot(bar_fig)
        st.write("Top values and counts:", dict(zip(bar_data["labels"], bar_data["values"])))
    else:
        st.info("No categorical columns found.")

# ------------------------------
# Main
# ------------------------------
def main():
    st.title("📈 Data Visualizations")
    st.markdown(
        """
        Explore your dataset visually:
        - Histograms & Boxplots for numeric data
        - Bar charts for categorical data
        - Scatter plots & pairplots for numeric relationships
        - Correlation heatmaps
        """
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="viz_uploader")

    if uploaded_file:
        # Persist dataset in session state
        if "df" not in st.session_state or st.session_state.get("uploaded_file") != uploaded_file:
            loader = data_loader(uploaded_file)
            df = loader.df_copy()
            st.session_state.df = df
            st.session_state.uploaded_file = uploaded_file
        else:
            df = st.session_state.df

        # Render sections as cards
        card_section("Numeric Visualizations", "🔢", numeric_visualizations, df=df)
        card_section("Categorical Visualizations", "🟠", categorical_visualizations, df=df)

    else:
        st.info("Please upload a CSV file to begin visualizations.")

# Call main
if __name__ == "__main__":
    main()
