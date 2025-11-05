import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def eda_page():
    """Exploratory Data Analysis page functionality"""

    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe

        st.subheader("📊 Basic Data Overview")

        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        # Expandable sections
        with st.expander("📊 Sample Data Views"):
            show_sample_data_views(df)

        with st.expander("📋 Column Information"):
            show_column_information(df)

        st.markdown("---")

        # EDA Analysis
        st.subheader("🎨 Choose EDA Analysis Type")

        eda_options = [
            "Histogram",
            "Bar Chart",
            "Scatter Plot",
            "Pair Plot",
            "Heatmap",
            "Line Chart",
            "Pie Chart",
            "Radar Plot"
        ]

        selected_eda = st.selectbox("Select Analysis Type:", eda_options)

        # Generate chart based on selection
        if selected_eda == "Histogram":
            create_histogram(df)
        elif selected_eda == "Bar Chart":
            create_bar_chart(df)
        elif selected_eda == "Scatter Plot":
            create_scatter_plot(df)
        elif selected_eda == "Pair Plot":
            create_pair_plot(df)
        elif selected_eda == "Heatmap":
            create_heatmap(df)
        elif selected_eda == "Line Chart":
            create_line_chart(df)
        elif selected_eda == "Pie Chart":
            create_pie_chart(df)
        elif selected_eda == "Radar Plot":
            create_radar_plot(df)

    else:
        st.warning("⚠️ No data available. Please upload a file first.")


def show_sample_data_views(df):
    """Show sample data views with Complete Data option"""

    view_option = st.radio(
        "Choose view:",
        ["First 10 rows", "Last 10 rows", "Random 10 rows", "Custom range", "Complete Data"]
    )

    if view_option == "First 10 rows":
        st.dataframe(df.head(10), use_container_width=True)
    elif view_option == "Last 10 rows":
        st.dataframe(df.tail(10), use_container_width=True)
    elif view_option == "Random 10 rows":
        st.dataframe(df.sample(min(10, len(df))), use_container_width=True)
    elif view_option == "Custom range":
        start = st.number_input("Start row:", min_value=0, max_value=len(df)-1, value=0)
        end = st.number_input("End row:", min_value=start+1, max_value=len(df), value=min(start+10, len(df)))
        st.dataframe(df.iloc[start:end], use_container_width=True)
    elif view_option == "Complete Data":
        st.dataframe(df, use_container_width=True)


def show_column_information(df):
    """Display column information"""

    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes,
        "Non-Null Count": df.count(),
        "Null Count": df.isnull().sum(),
        "Unique Values": df.nunique()
    })

    st.dataframe(col_info, use_container_width=True)


def create_histogram(df):
    """Create histogram with download option"""
    st.subheader("📊 Histogram")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns available")
        return

    col = st.selectbox("Select column:", numeric_cols, key="hist_col")
    bins = st.slider("Number of bins:", 5, 50, 20)

    fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        png_data = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Download as PNG",
            data=png_data,
            file_name=f"histogram_{col}.png",
            mime="image/png"
        )
    with col2:
        pdf_data = fig.to_image(format="pdf", width=1200, height=600)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_data,
            file_name=f"histogram_{col}.pdf",
            mime="application/pdf"
        )


def create_bar_chart(df):
    """Create bar chart with download option"""
    st.subheader("📊 Bar Chart")

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not cat_cols:
        st.error("No categorical columns available")
        return

    col = st.selectbox("Select column:", cat_cols, key="bar_col")
    sort_order = st.radio("Sort by:", ["Ascending", "Descending", "Alphabetical"], horizontal=True)

    value_counts = df[col].value_counts()
    if sort_order == "Ascending":
        value_counts = value_counts.sort_values()
    elif sort_order == "Descending":
        value_counts = value_counts.sort_values(ascending=False)
    else:
        value_counts = value_counts.sort_index()

    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                 title=f"Bar Chart of {col}", labels={"x": col, "y": "Count"})
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        png_data = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Download as PNG",
            data=png_data,
            file_name=f"barchart_{col}.png",
            mime="image/png"
        )
    with col2:
        pdf_data = fig.to_image(format="pdf", width=1200, height=600)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_data,
            file_name=f"barchart_{col}.pdf",
            mime="application/pdf"
        )


def create_scatter_plot(df):
    """Create scatter plot with download option"""
    st.subheader("📊 Scatter Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return

    col1 = st.selectbox("Select X column:", numeric_cols, key="scatter_x")
    col2 = st.selectbox("Select Y column:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")

    fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        png_data = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Download as PNG",
            data=png_data,
            file_name=f"scatter_{col1}_{col2}.png",
            mime="image/png"
        )
    with col_b:
        pdf_data = fig.to_image(format="pdf", width=1200, height=600)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_data,
            file_name=f"scatter_{col1}_{col2}.pdf",
            mime="application/pdf"
        )


def create_pair_plot(df):
    """Create pair plot with download option"""
    st.subheader("📊 Pair Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return

    selected_cols = st.multiselect("Select columns (2-6):", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns")
        return

    fig, axes = plt.subplots(len(selected_cols), len(selected_cols), figsize=(12, 12))

    for i, col1 in enumerate(selected_cols):
        for j, col2 in enumerate(selected_cols):
            ax = axes[i, j] if len(selected_cols) > 1 else plt.gca()
            if i == j:
                ax.hist(df[col1], bins=20, edgecolor='black')
            else:
                ax.scatter(df[col2], df[col1], alpha=0.6)
            ax.set_xlabel(col2 if i == len(selected_cols) - 1 else "")
            ax.set_ylabel(col1 if j == 0 else "")

    plt.tight_layout()
    st.pyplot(fig)

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
        buf_png.seek(0)
        st.download_button(
            label="📥 Download as PNG",
            data=buf_png,
            file_name="pairplot.png",
            mime="image/png"
        )
    with col_b:
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", dpi=300, bbox_inches='tight')
        buf_pdf.seek(0)
        st.download_button(
            label="📥 Download as PDF",
            data=buf_pdf,
            file_name="pairplot.pdf",
            mime="application/pdf"
        )


def create_heatmap(df):
    """Create correlation heatmap with download option"""
    st.subheader("📊 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        st.error("No numeric columns available")
        return

    corr_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
    corr_matrix = numeric_df.corr(method=corr_method)

    fig = px.imshow(corr_matrix, title=f"{corr_method.capitalize()} Correlation Heatmap",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        png_data = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Download as PNG",
            data=png_data,
            file_name=f"heatmap_{corr_method}.png",
            mime="image/png"
        )
    with col_b:
        pdf_data = fig.to_image(format="pdf", width=1200, height=600)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_data,
            file_name=f"heatmap_{corr_method}.pdf",
            mime="application/pdf"
        )


def create_line_chart(df):
    """Create line chart with download option"""
    st.subheader("📊 Line Chart")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns available")
        return

    col = st.selectbox("Select column:", numeric_cols, key="line_col")

    fig = px.line(df, y=col, title=f"Trend of {col}")
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        png_data = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Download as PNG",
            data=png_data,
            file_name=f"linechart_{col}.png",
            mime="image/png"
        )
    with col_b:
        pdf_data = fig.to_image(format="pdf", width=1200, height=600)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_data,
            file_name=f"linechart_{col}.pdf",
            mime="application/pdf"
        )


def create_pie_chart(df):
    """Create pie chart with download option"""
    st.subheader("📊 Pie Chart")

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not cat_cols:
        st.error("No categorical columns available")
        return

    col = st.selectbox("Select column:", cat_cols, key="pie_col")

    fig = px.pie(df, names=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        png_data = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Download as PNG",
            data=png_data,
            file_name=f"piechart_{col}.png",
            mime="image/png"
        )
    with col_b:
        pdf_data = fig.to_image(format="pdf", width=1200, height=600)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_data,
            file_name=f"piechart_{col}.pdf",
            mime="application/pdf"
        )


def create_radar_plot(df):
    """Create radar plot with download option"""
    st.subheader("📊 Radar Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 3:
        st.error("Need at least 3 numeric columns")
        return

    selected_cols = st.multiselect("Select columns (3+):", numeric_cols, 
                                   default=numeric_cols[:min(5, len(numeric_cols))])

    if len(selected_cols) < 3:
        st.warning("Please select at least 3 columns")
        return

    means = df[selected_cols].mean()

    fig = go.Figure(data=go.Scatterpolar(
        r=means.values,
        theta=means.index,
        fill='toself'
    ))

    fig.update_layout(title="Radar Plot")
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        png_data = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label="📥 Download as PNG",
            data=png_data,
            file_name="radarplot.png",
            mime="image/png"
        )
    with col_b:
        pdf_data = fig.to_image(format="pdf", width=1200, height=600)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_data,
            file_name="radarplot.pdf",
            mime="application/pdf"
        )
