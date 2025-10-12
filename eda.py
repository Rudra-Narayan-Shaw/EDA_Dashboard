import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def eda_page():
    """
    Exploratory Data Analysis page functionality
    Updated design with direct metrics display and expandable sections
    """

    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe

        # Basic Data Overview
        st.subheader("🔧 Basic Data Overview")

        # Display key metrics directly
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Rows",
                value=f"{df.shape[0]:,}"
            )

        with col2:
            st.metric(
                label="Columns", 
                value=f"{df.shape[1]:,}"
            )

        with col3:
            st.metric(
                label="Missing Values",
                value=f"{df.isnull().sum().sum():,}"
            )

        # Expandable sections below the metrics
        with st.expander("👀 Sample Data Views"):
            show_sample_data_views(df)

        with st.expander("🏷️ Column Information"):
            show_column_information(df)

        st.markdown("---")

        # EDA Options - Fixed: Removed "Choose Option"
        st.subheader("🔍 Choose EDA Analysis Type")

        eda_options = [
            "Choose Option",
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

        if selected_eda == "Histogram":
            create_histogram(df)
        elif selected_eda == "Bar Chart":
            create_bar_chart(df)
        elif selected_eda == "Scatter Plot":
            create_scatter_plot(df)
        elif selected_eda == "Pair Plot":
            create_pair_plot(df)
        elif selected_eda == "Heatmap":
            create_correlation_heatmap(df)
        elif selected_eda == "Line Chart":
            create_line_chart(df)
        elif selected_eda == "Pie Chart":
            create_pie_chart(df)
        elif selected_eda == "Radar Plot":
            create_radar_plot(df)
    else:
        st.warning("⚠️ No data available. Please upload a file first.")

def show_sample_data_views(df):
    """Show various sample data views"""
    view_option = st.radio(
        "Choose view:",
        ["First 10 rows", "Last 10 rows", "Random 10 rows", "Custom range"]
    )

    if view_option == "First 10 rows":
        st.dataframe(df.head(10), use_container_width=True)
    elif view_option == "Last 10 rows":
        st.dataframe(df.tail(10), use_container_width=True)
    elif view_option == "Random 10 rows":
        if len(df) > 0:
            st.dataframe(df.sample(min(10, len(df))), use_container_width=True)
        else:
            st.warning("Dataset is empty.")
    elif view_option == "Custom range":
        if len(df) > 0:
            col1, col2 = st.columns(2)
            with col1:
                start_row = st.number_input("Start row", min_value=0, max_value=len(df)-1, value=0)
            with col2:
                end_row = st.number_input("End row", min_value=start_row+1, max_value=len(df), value=min(10, len(df)))
            st.dataframe(df.iloc[start_row:end_row], use_container_width=True)
        else:
            st.warning("Dataset is empty.")

def show_column_information(df):
    """Show detailed column information"""
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique(),
        'Sample Value': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A' for col in df.columns]
    })

    st.dataframe(col_info, use_container_width=True)

    # Data type distribution
    st.subheader("📊 Data Type Distribution")
    dtype_counts = df.dtypes.value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(dtype_counts.reset_index().rename(columns={'index': 'Data Type', 0: 'Count'}), use_container_width=True)
    with col2:
        fig = px.bar(
            x=dtype_counts.index.astype(str),
            y=dtype_counts.values,
            title="Data Types Distribution",
            labels={'x': 'Data Type', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

def check_column_compatibility(df, column, plot_type):
    """Check if a column is compatible with a specific plot type and return appropriate message"""

    if column not in df.columns:
        return False, f"Column '{column}' not found in dataset."

    col_dtype = df[column].dtype
    null_count = df[column].isnull().sum()
    unique_count = df[column].nunique()

    # Check for empty column
    if null_count == len(df):
        return False, f"Column '{column}' contains only null values. Cannot generate {plot_type}."

    # Check based on plot type
    if plot_type == "Histogram":
        if col_dtype in ['object', 'category'] and unique_count > 50:
            return False, f"Column '{column}' has too many categories ({unique_count}) for histogram. Consider using Bar Chart or reduce categories."
        return True, ""

    elif plot_type == "Bar Chart":
        if col_dtype in ['int64', 'float64'] and unique_count > 100:
            return False, f"Numeric column '{column}' has too many unique values ({unique_count}) for bar chart. Consider using Histogram instead."
        return True, ""

    elif plot_type == "Scatter Plot":
        return True, ""  # Scatter plot can handle any data type

    elif plot_type == "Pair Plot":
        if col_dtype not in ['int64', 'float64']:
            return False, f"Column '{column}' is not numeric. Pair plot requires numeric columns only."
        return True, ""

    elif plot_type == "Heatmap":
        if col_dtype not in ['int64', 'float64']:
            return False, f"Column '{column}' is not numeric. Heatmap requires numeric columns for correlation analysis."
        return True, ""

    elif plot_type == "Line Chart":
        return True, ""  # Line chart can handle any data type

    elif plot_type == "Pie Chart":
        if col_dtype in ['int64', 'float64'] and unique_count > 20:
            return False, f"Numeric column '{column}' has too many unique values ({unique_count}) for pie chart. Consider binning the data first."
        elif unique_count < 2:
            return False, f"Column '{column}' has only {unique_count} unique value(s). Pie chart needs at least 2 categories."
        return True, ""

    elif plot_type == "Radar Plot":
        if col_dtype not in ['int64', 'float64']:
            return False, f"Column '{column}' is not numeric. Radar plot requires numeric columns only."
        return True, ""

    return True, ""

def create_histogram(df):
    """Create histogram for selected column with compatibility checking"""
    st.subheader("📊 Histogram")

    # Column selection - prefer numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_cols = df.columns.tolist()

    if len(numeric_cols) > 0:
        default_col = numeric_cols[0]
        selected_col = st.selectbox("Select column for histogram:", all_cols, 
                                  index=all_cols.index(default_col) if default_col in all_cols else 0)
    else:
        selected_col = st.selectbox("Select column for histogram:", all_cols)

    # Check compatibility
    is_compatible, error_msg = check_column_compatibility(df, selected_col, "Histogram")

    if not is_compatible:
        st.error(f"❌ Cannot generate histogram: {error_msg}")
        st.info("💡 Suggestions: Try selecting a numeric column or use Bar Chart for categorical data.")
        return

    if df[selected_col].dtype in ['int64', 'float64']:
        # Numeric histogram
        bins = st.slider("Number of bins:", 5, 50, 30)
        try:
            fig = px.histogram(
                df, 
                x=selected_col, 
                title=f"Histogram of {selected_col}",
                nbins=bins
            )
            st.plotly_chart(fig, use_container_width=True)

            # Distribution metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[selected_col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[selected_col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[selected_col].std():.2f}")
            with col4:
                st.metric("Skewness", f"{df[selected_col].skew():.2f}")
        except Exception as e:
            st.error(f"❌ Error creating histogram: {str(e)}")
    else:
        # For non-numeric data, show frequency histogram
        st.info("Selected column is not numeric. Showing frequency distribution.")
        try:
            value_counts = df[selected_col].value_counts()
            max_categories = st.slider("Maximum categories to display:", 5, 50, 20)

            fig = px.bar(
                x=value_counts.head(max_categories).index,
                y=value_counts.head(max_categories).values,
                title=f"Frequency Distribution of {selected_col}",
                labels={'x': selected_col, 'y': 'Count'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating frequency distribution: {str(e)}")

def create_bar_chart(df):
    """Create bar chart for selected column with compatibility checking"""
    st.subheader("📊 Bar Chart")

    # Column selection - prefer categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    all_cols = df.columns.tolist()

    if len(categorical_cols) > 0:
        default_col = categorical_cols[0]
        selected_col = st.selectbox("Select column for bar chart:", all_cols, 
                                  index=all_cols.index(default_col) if default_col in all_cols else 0)
    else:
        selected_col = st.selectbox("Select column for bar chart:", all_cols)

    # Check compatibility
    is_compatible, error_msg = check_column_compatibility(df, selected_col, "Bar Chart")

    if not is_compatible:
        st.error(f"❌ Cannot generate bar chart: {error_msg}")
        st.info("💡 Suggestion: Try using Histogram for numeric data with many unique values.")
        return

    try:
        if df[selected_col].dtype in ['object', 'category']:
            # Categorical bar chart
            value_counts = df[selected_col].value_counts()
            max_categories = st.slider("Maximum categories to display:", 5, 50, 20)

            # Sort options
            sort_option = st.radio("Sort by:", ["Count (Descending)", "Count (Ascending)", "Alphabetical"])

            if sort_option == "Count (Ascending)":
                value_counts = value_counts.sort_values()
            elif sort_option == "Alphabetical":
                value_counts = value_counts.sort_index()

            fig = px.bar(
                x=value_counts.head(max_categories).index,
                y=value_counts.head(max_categories).values,
                title=f"Bar Chart of {selected_col}",
                labels={'x': selected_col, 'y': 'Count'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Categories", len(value_counts))
            with col2:
                st.metric("Most Common", f"{value_counts.index[0]} ({value_counts.iloc[0]})")
            with col3:
                st.metric("Least Common", f"{value_counts.index[-1]} ({value_counts.iloc[-1]})")

        else:
            # Numeric bar chart (binned)
            st.info("Selected column is numeric. Creating binned bar chart.")
            bins = st.slider("Number of bins:", 5, 30, 10)
            fig = px.histogram(
                df, 
                x=selected_col, 
                title=f"Bar Chart of {selected_col} (Binned)",
                nbins=bins
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error creating bar chart: {str(e)}")
        st.info("💡 Please try selecting a different column or check your data quality.")

def create_scatter_plot(df):
    """Create scatter plot between two variables with compatibility checking"""
    st.subheader("🔗 Scatter Plot")

    # Column selections
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(all_cols) < 2:
        st.error("❌ Cannot generate scatter plot: Need at least 2 columns in the dataset.")
        return

    if len(numeric_cols) >= 2:
        # Default to first two numeric columns
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column:", all_cols, 
                               index=all_cols.index(numeric_cols[0]) if len(numeric_cols) > 0 else 0)
        with col2:
            y_col = st.selectbox("Select Y-axis column:", all_cols,
                               index=all_cols.index(numeric_cols[1]) if len(numeric_cols) > 1 else 1)
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column:", all_cols)
        with col2:
            y_col = st.selectbox("Select Y-axis column:", all_cols)

    # Check if columns are different
    if x_col == y_col:
        st.warning("⚠️ X and Y columns are the same. Please select different columns for meaningful scatter plot.")
        return

    try:
        # Optional enhancements
        col3, col4 = st.columns(2)
        with col3:
            color_col = st.selectbox("Color by (optional):", ["None"] + all_cols)
        with col4:
            size_col = st.selectbox("Size by (optional):", ["None"] + list(numeric_cols))

        # Create scatter plot
        kwargs = {
            'data_frame': df,
            'x': x_col,
            'y': y_col,
            'title': f"Scatter Plot: {x_col} vs {y_col}"
        }

        if color_col != "None":
            kwargs['color'] = color_col
        if size_col != "None":
            kwargs['size'] = size_col

        fig = px.scatter(**kwargs)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation coefficient (if both are numeric)
        if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
            correlation = df[x_col].corr(df[y_col])
            st.metric("Correlation Coefficient", f"{correlation:.3f}")

            # Interpretation
            if abs(correlation) > 0.7:
                strength = "Strong"
            elif abs(correlation) > 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"

            direction = "Positive" if correlation > 0 else "Negative"
            st.info(f"Relationship: {strength} {direction} correlation")

    except Exception as e:
        st.error(f"❌ Error creating scatter plot: {str(e)}")
        st.info("💡 Check if selected columns contain valid data for plotting.")

def create_pair_plot(df):
    """Create pair plot for multiple variables with compatibility checking"""
    st.subheader("🔗 Pair Plot Matrix")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        st.error("❌ Cannot generate pair plot: Need at least 2 numeric columns in the dataset.")
        st.info("💡 Suggestion: Convert some columns to numeric type in the Data Manipulation section.")
        return

    try:
        # Variable selection
        max_vars = min(6, len(numeric_cols))  # Limit for performance
        default_selection = list(numeric_cols[:min(4, len(numeric_cols))])

        selected_vars = st.multiselect(
            f"Select variables for pair plot (2-{max_vars} recommended for performance):",
            numeric_cols,
            default=default_selection
        )

        if len(selected_vars) < 2:
            st.warning("⚠️ Please select at least 2 variables for pair plot.")
            return
        elif len(selected_vars) > max_vars:
            st.warning(f"⚠️ Please select maximum {max_vars} variables for optimal performance.")
            return

        # Check for invalid data
        for var in selected_vars:
            is_compatible, error_msg = check_column_compatibility(df, var, "Pair Plot")
            if not is_compatible:
                st.error(f"❌ {error_msg}")
                return

        # Optional color coding
        color_col = st.selectbox("Color by (optional):", ["None"] + df.columns.tolist())

        # Create pair plot
        if color_col == "None":
            fig = px.scatter_matrix(
                df[selected_vars],
                title="Pair Plot Matrix"
            )
        else:
            fig = px.scatter_matrix(
                df,
                dimensions=selected_vars,
                color=color_col,
                title="Pair Plot Matrix"
            )

        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation summary
        st.subheader("📊 Correlation Summary")
        corr_matrix = df[selected_vars].corr()
        st.dataframe(corr_matrix.round(3), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error creating pair plot: {str(e)}")
        st.info("💡 Try selecting fewer variables or check data quality.")

def create_correlation_heatmap(df):
    """Create correlation heatmap with compatibility checking"""
    st.subheader("🔥 Correlation Heatmap")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        st.error("❌ Cannot generate correlation heatmap: Need at least 2 numeric columns in the dataset.")
        st.info("💡 Suggestion: Convert some columns to numeric type in the Data Manipulation section.")
        return

    try:
        # Column selection
        selected_cols = st.multiselect(
            "Select columns for correlation analysis:",
            numeric_cols,
            default=list(numeric_cols)
        )

        if len(selected_cols) < 2:
            st.warning("⚠️ Please select at least 2 columns for correlation analysis.")
            return

        # Check for columns with no variance
        zero_var_cols = []
        for col in selected_cols:
            if df[col].var() == 0:
                zero_var_cols.append(col)

        if zero_var_cols:
            st.warning(f"⚠️ Columns with zero variance detected: {zero_var_cols}. These will be excluded from correlation analysis.")
            selected_cols = [col for col in selected_cols if col not in zero_var_cols]

        if len(selected_cols) < 2:
            st.error("❌ Not enough columns with variance for correlation analysis.")
            return

        # Correlation method
        method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])

        # Calculate correlation
        correlation_matrix = df[selected_cols].corr(method=method)

        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title=f"Correlation Heatmap ({method.title()})",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

        # Strong correlations analysis
        st.subheader("🔍 Strong Correlations Analysis")

        threshold = st.slider("Correlation threshold:", 0.5, 1.0, 0.7, 0.05)

        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) >= threshold and not pd.isna(corr_val):
                    strong_corr.append({
                        'Variable 1': correlation_matrix.columns[i],
                        'Variable 2': correlation_matrix.columns[j],
                        'Correlation': round(corr_val, 3),
                        'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                    })

        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(strong_corr_df, use_container_width=True)
        else:
            st.info(f"No correlations found above threshold {threshold}")

    except Exception as e:
        st.error(f"❌ Error creating correlation heatmap: {str(e)}")
        st.info("💡 Check if selected columns contain valid numeric data.")

def create_line_chart(df):
    """Create line chart with compatibility checking"""
    st.subheader("📈 Line Chart")

    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.error("❌ Cannot generate line chart: Need at least 2 columns in the dataset.")
        return

    try:
        # Column selections
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column:", all_cols, key="line_x")
        with col2:
            y_col = st.selectbox("Select Y-axis column:", all_cols, key="line_y")

        # Check if columns are different
        if x_col == y_col:
            st.warning("⚠️ X and Y columns are the same. Please select different columns.")
            return

        # Optional enhancements
        col3, col4 = st.columns(2)
        with col3:
            color_col = st.selectbox("Color/Group by (optional):", ["None"] + all_cols, key="line_color")
        with col4:
            sort_data = st.checkbox("Sort by X-axis", value=True)

        # Prepare data
        plot_df = df.copy()
        if sort_data:
            try:
                plot_df = plot_df.sort_values(x_col)
            except:
                st.info("Could not sort data by X-axis column.")

        # Create line chart
        if color_col == "None":
            fig = px.line(plot_df, x=x_col, y=y_col, title=f"Line Chart: {y_col} vs {x_col}")
        else:
            fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, 
                         title=f"Line Chart: {y_col} vs {x_col} (grouped by {color_col})")

        st.plotly_chart(fig, use_container_width=True)

        # Statistics for numeric Y column
        if df[y_col].dtype in ['int64', 'float64']:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Value", f"{df[y_col].min():.2f}")
            with col2:
                st.metric("Max Value", f"{df[y_col].max():.2f}")
            with col3:
                st.metric("Mean Value", f"{df[y_col].mean():.2f}")
            with col4:
                trend_indicator = "↗️" if df[y_col].iloc[-1] > df[y_col].iloc[0] else "↘️"
                st.metric("Trend", trend_indicator)

    except Exception as e:
        st.error(f"❌ Error creating line chart: {str(e)}")
        st.info("💡 Check if selected columns contain valid data for plotting.")

def create_pie_chart(df):
    """Create pie chart for categorical data with compatibility checking"""
    st.subheader("🥧 Pie Chart")

    # Column selection - prefer categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    all_cols = df.columns.tolist()

    if len(categorical_cols) > 0:
        default_col = categorical_cols[0]
        selected_col = st.selectbox("Select categorical column:", all_cols,
                                  index=all_cols.index(default_col) if default_col in all_cols else 0)
    else:
        selected_col = st.selectbox("Select column for pie chart:", all_cols)
        st.info("Selected column may not be ideal for pie chart. Consider categorical columns.")

    # Check compatibility
    is_compatible, error_msg = check_column_compatibility(df, selected_col, "Pie Chart")

    if not is_compatible:
        st.error(f"❌ Cannot generate pie chart: {error_msg}")
        st.info("💡 Suggestions: Try grouping numeric data into categories or select a categorical column.")
        return

    try:
        # Chart options
        col1, col2 = st.columns(2)
        with col1:
            max_categories = st.slider("Maximum categories to display:", 3, 20, 8)
        with col2:
            show_percentages = st.checkbox("Show percentages", value=True)

        # Prepare data
        value_counts = df[selected_col].value_counts()

        # Handle too many categories
        if len(value_counts) > max_categories:
            top_categories = value_counts.head(max_categories - 1)
            others_count = value_counts.iloc[max_categories - 1:].sum()
            plot_data = pd.concat([top_categories, pd.Series([others_count], index=['Others'])])
        else:
            plot_data = value_counts

        # Create pie chart
        fig = px.pie(
            values=plot_data.values,
            names=plot_data.index,
            title=f"Pie Chart: Composition of {selected_col}"
        )

        if show_percentages:
            fig.update_traces(textposition='inside', textinfo='percent+label')
        else:
            fig.update_traces(textposition='inside', textinfo='label')

        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("📊 Category Breakdown")
        breakdown_df = pd.DataFrame({
            'Category': plot_data.index,
            'Count': plot_data.values,
            'Percentage': (plot_data.values / plot_data.sum() * 100).round(2)
        })
        st.dataframe(breakdown_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error creating pie chart: {str(e)}")
        st.info("💡 Try selecting a categorical column with fewer unique values.")

def create_radar_plot(df):
    """Create radar plot for multidimensional analysis with compatibility checking"""
    st.subheader("🌐 Radar Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 3:
        st.error("❌ Cannot generate radar plot: Need at least 3 numeric columns in the dataset.")
        st.info("💡 Suggestion: Convert some columns to numeric type in the Data Manipulation section.")
        return

    try:
        # Variable selection
        selected_cols = st.multiselect(
            "Select columns for radar plot (3-8 recommended):",
            numeric_cols,
            default=list(numeric_cols[:min(5, len(numeric_cols))])
        )

        if len(selected_cols) < 3:
            st.warning("⚠️ Please select at least 3 columns for radar plot.")
            return

        # Check for columns with invalid data
        valid_cols = []
        for col in selected_cols:
            is_compatible, error_msg = check_column_compatibility(df, col, "Radar Plot")
            if is_compatible:
                valid_cols.append(col)
            else:
                st.warning(f"⚠️ Excluding {col}: {error_msg}")

        if len(valid_cols) < 3:
            st.error("❌ Not enough valid columns for radar plot.")
            return

        selected_cols = valid_cols

        # Analysis type
        analysis_type = st.radio(
            "Select radar plot type:", 
            ["Sample Rows", "Statistical Summary", "Custom Values"]
        )

        if analysis_type == "Sample Rows":
            # Row selection
            max_rows = min(10, len(df))
            n_rows = st.slider("Number of rows to display:", 1, max_rows, min(3, max_rows))

            # Sample selection method
            sample_method = st.radio("Sample method:", ["First N", "Last N", "Random"])

            if sample_method == "First N":
                sample_data = df[selected_cols].head(n_rows)
            elif sample_method == "Last N":
                sample_data = df[selected_cols].tail(n_rows)
            else:
                sample_data = df[selected_cols].sample(n_rows)

            # Normalize data (0-1 scale)
            min_vals = sample_data.min()
            max_vals = sample_data.max()
            range_vals = max_vals - min_vals

            # Handle zero range
            range_vals = range_vals.replace(0, 1)
            normalized_data = (sample_data - min_vals) / range_vals
            normalized_data = normalized_data.fillna(0)

            # Create radar plot
            fig = go.Figure()

            for i, (idx, row) in enumerate(normalized_data.iterrows()):
                fig.add_trace(go.Scatterpolar(
                    r=row.values.tolist() + [row.values[0]],
                    theta=selected_cols + [selected_cols[0]],
                    fill='toself',
                    name=f'Row {idx}',
                    opacity=0.6
                ))

        elif analysis_type == "Statistical Summary":
            # Statistical measures
            stats_data = pd.DataFrame({
                'Mean': df[selected_cols].mean(),
                'Median': df[selected_cols].median(),
                'Max': df[selected_cols].max(),
                'Min': df[selected_cols].min()
            })

            # Select which statistics to show
            selected_stats = st.multiselect(
                "Select statistics to display:",
                ['Mean', 'Median', 'Max', 'Min'],
                default=['Mean', 'Max']
            )

            if not selected_stats:
                st.warning("⚠️ Please select at least one statistic.")
                return

            # Normalize for comparison
            normalized_stats = stats_data[selected_stats].div(stats_data.max())
            normalized_stats = normalized_stats.fillna(0)

            fig = go.Figure()

            for stat_name in selected_stats:
                values = normalized_stats[stat_name].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=selected_cols + [selected_cols[0]],
                    fill='toself',
                    name=stat_name,
                    opacity=0.6
                ))

        else:  # Custom Values
            st.subheader("Enter Custom Values")
            custom_values = {}

            col_groups = [selected_cols[i:i+3] for i in range(0, len(selected_cols), 3)]

            for group in col_groups:
                cols = st.columns(len(group))
                for i, col in enumerate(group):
                    with cols[i]:
                        custom_values[col] = st.number_input(f"{col}:", value=0.5, min_value=0.0, max_value=1.0, step=0.1)

            values = list(custom_values.values())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=selected_cols + [selected_cols[0]],
                fill='toself',
                name='Custom Values',
                opacity=0.7
            ))

        # Update layout and display
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Radar Plot - Multidimensional Analysis",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data summary
        if analysis_type in ["Sample Rows", "Statistical Summary"]:
            st.subheader("📊 Data Summary")
            summary_df = df[selected_cols].describe()
            st.dataframe(summary_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error creating radar plot: {str(e)}")
        st.info("💡 Try selecting different columns or check data quality.")