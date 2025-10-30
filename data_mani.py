import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

def data_manipulation_page():
    """
    Data Manipulation page functionality
    This module provides comprehensive data cleaning and transformation capabilities
    """

    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe.copy()

        st.subheader("🛠️ Choose Data Manipulation Operation")

        manipulation_options = [
            "Data Cleaning",
            "Column Operations", 
            "Row Operations",
            "Data Transformation",
            "Feature Engineering",
            "Data Filtering"
        ]

        selected_operation = st.selectbox("Select Operation Type:", manipulation_options)

        if selected_operation == "Data Cleaning":
            data_cleaning(df)
        elif selected_operation == "Column Operations":
            column_operations(df)
        elif selected_operation == "Row Operations":
            row_operations(df)
        elif selected_operation == "Data Transformation":
            data_transformation(df)
        elif selected_operation == "Feature Engineering":
            feature_engineering(df)
        elif selected_operation == "Data Filtering":
            data_filtering(df)
    else:
        st.warning("⚠️ No data available. Please upload a file first.")

def data_cleaning(df):
    """Data cleaning operations"""
    st.subheader("🧹 Data Cleaning Operations")

    cleaning_option = st.radio(
        "Select cleaning operation:",
        [
            "Handle Missing Values",
            "Remove Duplicates", 
            "Fix Data Types",
            "Clean Text Data",
            "Handle Outliers"
        ]
    )

    if cleaning_option == "Handle Missing Values":
        handle_missing_values(df)
    elif cleaning_option == "Remove Duplicates":
        remove_duplicates(df)
    elif cleaning_option == "Fix Data Types":
        fix_data_types(df)
    elif cleaning_option == "Clean Text Data":
        clean_text_data(df)
    elif cleaning_option == "Handle Outliers":
        handle_outliers(df)

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    st.subheader("🕳️ Handle Missing Values")

    # Show missing value summary
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    if len(missing_cols) > 0:
        st.write("**Columns with missing values:**")
        st.dataframe(pd.DataFrame({
            'Column': missing_cols.index,
            'Missing Count': missing_cols.values,
            'Missing %': (missing_cols / len(df) * 100).round(2)
        }))

        # Select columns to handle
        selected_cols = st.multiselect(
            "Select columns to handle missing values:",
            missing_cols.index.tolist()
        )

        if selected_cols:
            # Choose strategy
            strategy = st.selectbox(
                "Select strategy:",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value", "Forward fill", "Backward fill"]
            )

            if st.button("Apply Missing Value Treatment"):
                df_cleaned = df.copy()

                for col in selected_cols:
                    if strategy == "Drop rows":
                        df_cleaned = df_cleaned.dropna(subset=[col])
                    elif strategy == "Fill with mean" and df[col].dtype in ['int64', 'float64']:
                        df_cleaned[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == "Fill with median" and df[col].dtype in ['int64', 'float64']:
                        df_cleaned[col].fillna(df[col].median(), inplace=True)
                    elif strategy == "Fill with mode":
                        df_cleaned[col].fillna(df[col].mode()[0], inplace=True)
                    elif strategy == "Fill with custom value":
                        custom_value = st.text_input(f"Enter custom value for {col}:")
                        if custom_value:
                            df_cleaned[col].fillna(custom_value, inplace=True)
                    elif strategy == "Forward fill":
                        df_cleaned[col].fillna(method='ffill', inplace=True)
                    elif strategy == "Backward fill":
                        df_cleaned[col].fillna(method='bfill', inplace=True)

                st.session_state.current_dataframe = df_cleaned
                st.success(f"✅ Missing values handled! New shape: {df_cleaned.shape}")

                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before:**")
                    st.write(f"Missing values: {df.isnull().sum().sum()}")
                with col2:
                    st.write("**After:**")
                    st.write(f"Missing values: {df_cleaned.isnull().sum().sum()}")
    else:
        st.success("🎉 No missing values found!")

def remove_duplicates(df):
    """Remove duplicate rows"""
    st.subheader("🔄 Remove Duplicates")

    duplicate_count = df.duplicated().sum()
    st.metric("Duplicate Rows Found", duplicate_count)

    if duplicate_count > 0:
        # Option to select columns for duplicate checking
        subset_cols = st.multiselect(
            "Select columns to consider for duplicates (leave empty for all columns):",
            df.columns.tolist()
        )

        keep_option = st.radio(
            "Which duplicates to keep:",
            ["first", "last", "none"]
        )

        if st.button("Remove Duplicates"):
            if subset_cols:
                df_cleaned = df.drop_duplicates(subset=subset_cols, keep=keep_option if keep_option != "none" else False)
            else:
                df_cleaned = df.drop_duplicates(keep=keep_option if keep_option != "none" else False)

            st.session_state.current_dataframe = df_cleaned
            st.success(f"✅ Duplicates removed! Removed {len(df) - len(df_cleaned)} rows")
    else:
        st.success("🎉 No duplicates found!")

def fix_data_types(df):
    """Fix data types of columns"""
    st.subheader("🔧 Fix Data Types")

    # Show current data types
    st.write("**Current Data Types:**")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Current Type': df.dtypes.astype(str)
    })
    st.dataframe(dtype_df)

    # Select column to change type
    selected_col = st.selectbox("Select column to change type:", df.columns)

    new_type = st.selectbox(
        "Select new data type:",
        ["int64", "float64", "object", "datetime64", "category", "bool"]
    )

    if st.button("Change Data Type"):
        try:
            df_updated = df.copy()

            if new_type == "datetime64":
                df_updated[selected_col] = pd.to_datetime(df_updated[selected_col])
            elif new_type == "category":
                df_updated[selected_col] = df_updated[selected_col].astype('category')
            elif new_type == "bool":
                df_updated[selected_col] = df_updated[selected_col].astype('bool')
            else:
                df_updated[selected_col] = df_updated[selected_col].astype(new_type)

            st.session_state.current_dataframe = df_updated
            st.success(f"✅ Data type changed for {selected_col} to {new_type}")

        except Exception as e:
            st.error(f"❌ Error changing data type: {str(e)}")

def clean_text_data(df):
    """Clean text data"""
    st.subheader("📝 Clean Text Data")

    text_cols = df.select_dtypes(include=['object']).columns

    if len(text_cols) > 0:
        selected_col = st.selectbox("Select text column to clean:", text_cols)

        cleaning_options = st.multiselect(
            "Select cleaning operations:",
            [
                "Remove leading/trailing whitespace",
                "Convert to lowercase",
                "Convert to uppercase",
                "Remove special characters",
                "Remove numbers",
                "Remove extra spaces"
            ]
        )

        if st.button("Apply Text Cleaning") and cleaning_options:
            df_cleaned = df.copy()

            for option in cleaning_options:
                if option == "Remove leading/trailing whitespace":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.strip()
                elif option == "Convert to lowercase":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.lower()
                elif option == "Convert to uppercase":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.upper()
                elif option == "Remove special characters":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                elif option == "Remove numbers":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.replace(r'\d+', '', regex=True)
                elif option == "Remove extra spaces":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.replace(r'\s+', ' ', regex=True)

            st.session_state.current_dataframe = df_cleaned
            st.success("✅ Text cleaning applied!")

            # Show before/after sample
            st.write("**Sample Before/After:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before:")
                st.write(df[selected_col].head())
            with col2:
                st.write("After:")
                st.write(df_cleaned[selected_col].head())
    else:
        st.warning("No text columns found!")

def handle_outliers(df):
    """Handle outliers in numeric columns"""
    st.subheader("🎯 Handle Outliers")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select numeric column:", numeric_cols)

        # Calculate outliers using IQR method
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)
        outlier_count = outliers_mask.sum()

        st.metric("Outliers Found", outlier_count)

        if outlier_count > 0:
            handling_method = st.radio(
                "Select outlier handling method:",
                ["Remove outliers", "Cap outliers", "Transform with log", "Replace with median"]
            )

            if st.button("Handle Outliers"):
                df_handled = df.copy()

                if handling_method == "Remove outliers":
                    df_handled = df_handled[~outliers_mask]
                elif handling_method == "Cap outliers":
                    df_handled[selected_col] = np.clip(df_handled[selected_col], lower_bound, upper_bound)
                elif handling_method == "Transform with log":
                    # Add 1 to handle zero values
                    df_handled[selected_col] = np.log1p(np.abs(df_handled[selected_col]))
                elif handling_method == "Replace with median":
                    median_val = df[selected_col].median()
                    df_handled.loc[outliers_mask, selected_col] = median_val

                st.session_state.current_dataframe = df_handled
                st.success("✅ Outliers handled!")
        else:
            st.success("🎉 No outliers found!")
    else:
        st.warning("No numeric columns found!")

def column_operations(df):
    """Column manipulation operations"""
    st.subheader("📊 Column Operations")

    operation = st.radio(
        "Select column operation:",
        [
            "Add New Column",
            "Rename Columns",
            "Drop Columns",
            "Reorder Columns"
        ]
    )

    if operation == "Add New Column":
        add_new_column(df)
    elif operation == "Rename Columns":
        rename_columns(df)
    elif operation == "Drop Columns":
        drop_columns(df)
    elif operation == "Reorder Columns":
        reorder_columns(df)

def add_new_column(df):
    """Add a new column"""
    st.subheader("➕ Add New Column")

    new_col_name = st.text_input("Enter new column name:")

    if new_col_name:
        column_type = st.radio(
            "How to create the column:",
            ["Constant value", "Mathematical operation", "Conditional logic", "From existing columns"]
        )

        if column_type == "Constant value":
            const_value = st.text_input("Enter constant value:")
            if st.button("Add Column") and const_value:
                df_updated = df.copy()
                df_updated[new_col_name] = const_value
                st.session_state.current_dataframe = df_updated
                st.success(f"✅ Column '{new_col_name}' added!")

        elif column_type == "Mathematical operation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Select first column:", numeric_cols)
                operation_type = st.selectbox("Select operation:", ["+", "-", "*", "/", "**"])
                col2 = st.selectbox("Select second column:", numeric_cols)

                if st.button("Add Column"):
                    df_updated = df.copy()
                    if operation_type == "+":
                        df_updated[new_col_name] = df_updated[col1] + df_updated[col2]
                    elif operation_type == "-":
                        df_updated[new_col_name] = df_updated[col1] - df_updated[col2]
                    elif operation_type == "*":
                        df_updated[new_col_name] = df_updated[col1] * df_updated[col2]
                    elif operation_type == "/":
                        df_updated[new_col_name] = df_updated[col1] / df_updated[col2]
                    elif operation_type == "**":
                        df_updated[new_col_name] = df_updated[col1] ** df_updated[col2]

                    st.session_state.current_dataframe = df_updated
                    st.success(f"✅ Column '{new_col_name}' added!")
            else:
                st.warning("Need at least 2 numeric columns for mathematical operations")

def rename_columns(df):
    """Rename columns"""
    st.subheader("✏️ Rename Columns")

    selected_col = st.selectbox("Select column to rename:", df.columns)
    new_name = st.text_input("Enter new name:", value=selected_col)

    if st.button("Rename Column") and new_name != selected_col:
        df_updated = df.copy()
        df_updated.rename(columns={selected_col: new_name}, inplace=True)
        st.session_state.current_dataframe = df_updated
        st.success(f"✅ Column renamed from '{selected_col}' to '{new_name}'")

def drop_columns(df):
    """Drop columns"""
    st.subheader("🗑️ Drop Columns")

    columns_to_drop = st.multiselect("Select columns to drop:", df.columns)

    if columns_to_drop and st.button("Drop Columns"):
        df_updated = df.copy()
        df_updated.drop(columns=columns_to_drop, inplace=True)
        st.session_state.current_dataframe = df_updated
        st.success(f"✅ Dropped {len(columns_to_drop)} columns")

def reorder_columns(df):
    """Reorder columns"""
    st.subheader("🔄 Reorder Columns")

    # Show current order
    st.write("**Current column order:**")
    for i, col in enumerate(df.columns):
        st.write(f"{i+1}. {col}")

    # Allow manual reordering
    new_order = st.multiselect(
        "Select columns in desired order:",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

    if len(new_order) == len(df.columns) and st.button("Apply New Order"):
        df_updated = df[new_order].copy()
        st.session_state.current_dataframe = df_updated
        st.success("✅ Columns reordered!")

def row_operations(df):
    """Row manipulation operations"""
    st.subheader("📋 Row Operations")

    operation = st.radio(
        "Select row operation:",
        ["Filter Rows", "Sort Rows", "Sample Rows", "Add New Row"]
    )

    if operation == "Filter Rows":
        filter_rows(df)
    elif operation == "Sort Rows":
        sort_rows(df)
    elif operation == "Sample Rows":
        sample_rows(df)
    elif operation == "Add New Row":
        add_new_row(df)

def filter_rows(df):
    """Filter rows based on conditions"""
    st.subheader("🔍 Filter Rows")

    filter_col = st.selectbox("Select column to filter by:", df.columns)

    if df[filter_col].dtype in ['int64', 'float64']:
        # Numeric filtering
        min_val = float(df[filter_col].min())
        max_val = float(df[filter_col].max())

        filter_range = st.slider(
            "Select value range:",
            min_val, max_val,
            (min_val, max_val)
        )

        if st.button("Apply Filter"):
            df_filtered = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])].copy()
            st.session_state.current_dataframe = df_filtered
            st.success(f"✅ Filtered to {len(df_filtered)} rows")

    else:
        # Categorical filtering
        unique_values = df[filter_col].unique()
        selected_values = st.multiselect("Select values to keep:", unique_values)

        if selected_values and st.button("Apply Filter"):
            df_filtered = df[df[filter_col].isin(selected_values)].copy()
            st.session_state.current_dataframe = df_filtered
            st.success(f"✅ Filtered to {len(df_filtered)} rows")

def sort_rows(df):
    """Sort rows"""
    st.subheader("📈 Sort Rows")

    sort_cols = st.multiselect("Select columns to sort by:", df.columns)
    ascending = st.checkbox("Ascending order", value=True)

    if sort_cols and st.button("Sort"):
        df_sorted = df.sort_values(by=sort_cols, ascending=ascending).copy()
        st.session_state.current_dataframe = df_sorted
        st.success("✅ Data sorted!")

def sample_rows(df):
    """Sample rows from dataset"""
    st.subheader("🎲 Sample Rows")

    sample_method = st.radio("Select sampling method:", ["Random sample", "First N rows", "Last N rows"])

    if sample_method == "Random sample":
        sample_size = st.number_input("Sample size:", min_value=1, max_value=len(df), value=min(100, len(df)))

        if st.button("Sample"):
            df_sampled = df.sample(n=sample_size).copy()
            st.session_state.current_dataframe = df_sampled
            st.success(f"✅ Sampled {sample_size} rows")

    elif sample_method == "First N rows":
        n_rows = st.number_input("Number of rows:", min_value=1, max_value=len(df), value=min(100, len(df)))

        if st.button("Get First N"):
            df_sampled = df.head(n_rows).copy()
            st.session_state.current_dataframe = df_sampled
            st.success(f"✅ Selected first {n_rows} rows")

    else:  # Last N rows
        n_rows = st.number_input("Number of rows:", min_value=1, max_value=len(df), value=min(100, len(df)))

        if st.button("Get Last N"):
            df_sampled = df.tail(n_rows).copy()
            st.session_state.current_dataframe = df_sampled
            st.success(f"✅ Selected last {n_rows} rows")

def add_new_row(df):
    """Add a new row"""
    st.subheader("➕ Add New Row")

    st.write("Enter values for new row:")
    new_row_data = {}

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            new_row_data[col] = st.number_input(f"{col}:", value=0.0)
        else:
            new_row_data[col] = st.text_input(f"{col}:")

    if st.button("Add Row"):
        df_updated = df.copy()
        new_row_df = pd.DataFrame([new_row_data])
        df_updated = pd.concat([df_updated, new_row_df], ignore_index=True)
        st.session_state.current_dataframe = df_updated
        st.success("✅ New row added!")

def data_transformation(df):
    """Data transformation operations"""
    st.subheader("🔄 Data Transformation")

    transformation = st.radio(
        "Select transformation:",
        ["Normalize/Scale Data", "Encode Categorical Variables", "Binning", "Log Transformation"]
    )

    if transformation == "Normalize/Scale Data":
        normalize_data(df)
    elif transformation == "Encode Categorical Variables":
        encode_categorical(df)
    elif transformation == "Binning":
        binning_operation(df)
    elif transformation == "Log Transformation":
        log_transformation(df)

def normalize_data(df):
    """Normalize numeric data"""
    st.subheader("📊 Normalize/Scale Data")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        selected_cols = st.multiselect("Select columns to normalize:", numeric_cols)

        scaling_method = st.radio(
            "Select scaling method:",
            ["Min-Max Scaling (0-1)", "Z-Score Standardization", "Robust Scaling"]
        )

        if selected_cols and st.button("Apply Scaling"):
            df_scaled = df.copy()

            for col in selected_cols:
                if scaling_method == "Min-Max Scaling (0-1)":
                    df_scaled[col] = (df_scaled[col] - df_scaled[col].min()) / (df_scaled[col].max() - df_scaled[col].min())
                elif scaling_method == "Z-Score Standardization":
                    df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()
                elif scaling_method == "Robust Scaling":
                    median = df_scaled[col].median()
                    mad = np.median(np.abs(df_scaled[col] - median))
                    df_scaled[col] = (df_scaled[col] - median) / mad

            st.session_state.current_dataframe = df_scaled
            st.success("✅ Data scaled successfully!")
    else:
        st.warning("No numeric columns found!")

def encode_categorical(df):
    """Encode categorical variables"""
    st.subheader("🏷️ Encode Categorical Variables")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        selected_col = st.selectbox("Select categorical column:", categorical_cols)

        encoding_method = st.radio(
            "Select encoding method:",
            ["One-Hot Encoding", "Label Encoding", "Binary Encoding"]
        )

        if st.button("Apply Encoding"):
            df_encoded = df.copy()

            if encoding_method == "One-Hot Encoding":
                # Get dummies
                dummies = pd.get_dummies(df_encoded[selected_col], prefix=selected_col)
                df_encoded = pd.concat([df_encoded.drop(selected_col, axis=1), dummies], axis=1)

            elif encoding_method == "Label Encoding":
                # Simple label encoding
                unique_values = df_encoded[selected_col].unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                df_encoded[selected_col + '_encoded'] = df_encoded[selected_col].map(encoding_map)

            elif encoding_method == "Binary Encoding":
                # Simple binary encoding for two categories
                if len(df_encoded[selected_col].unique()) == 2:
                    df_encoded[selected_col + '_binary'] = (df_encoded[selected_col] == df_encoded[selected_col].unique()[0]).astype(int)
                else:
                    st.warning("Binary encoding works best with 2 categories")
                    return

            st.session_state.current_dataframe = df_encoded
            st.success("✅ Encoding applied successfully!")
    else:
        st.warning("No categorical columns found!")

def binning_operation(df):
    """Create bins from numeric data"""
    st.subheader("📦 Binning Operation")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select numeric column for binning:", numeric_cols)

        binning_method = st.radio(
            "Select binning method:",
            ["Equal Width", "Equal Frequency", "Custom Bins"]
        )

        if binning_method in ["Equal Width", "Equal Frequency"]:
            n_bins = st.number_input("Number of bins:", min_value=2, max_value=20, value=5)
        else:  # Custom Bins
            bin_edges = st.text_input("Enter bin edges (comma-separated):", "0,25,50,75,100")

        if st.button("Create Bins"):
            df_binned = df.copy()

            try:
                if binning_method == "Equal Width":
                    df_binned[selected_col + '_binned'] = pd.cut(df_binned[selected_col], bins=n_bins)
                elif binning_method == "Equal Frequency":
                    df_binned[selected_col + '_binned'] = pd.qcut(df_binned[selected_col], q=n_bins)
                else:  # Custom Bins
                    bins = [float(x.strip()) for x in bin_edges.split(',')]
                    df_binned[selected_col + '_binned'] = pd.cut(df_binned[selected_col], bins=bins)

                st.session_state.current_dataframe = df_binned
                st.success("✅ Binning applied successfully!")

            except Exception as e:
                st.error(f"❌ Error in binning: {str(e)}")
    else:
        st.warning("No numeric columns found!")

def log_transformation(df):
    """Apply log transformation"""
    st.subheader("📈 Log Transformation")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        selected_cols = st.multiselect("Select columns for log transformation:", numeric_cols)

        if selected_cols and st.button("Apply Log Transformation"):
            df_transformed = df.copy()

            for col in selected_cols:
                # Handle negative values by adding constant
                min_val = df_transformed[col].min()
                if min_val <= 0:
                    df_transformed[col + '_log'] = np.log1p(df_transformed[col] - min_val + 1)
                else:
                    df_transformed[col + '_log'] = np.log(df_transformed[col])

            st.session_state.current_dataframe = df_transformed
            st.success("✅ Log transformation applied!")
    else:
        st.warning("No numeric columns found!")

def feature_engineering(df):
    """Feature engineering operations"""
    st.subheader("🔧 Feature Engineering")

    st.info("Feature engineering operations help create new meaningful features from existing data.")

    # Date/Time feature extraction
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        st.subheader("📅 DateTime Feature Extraction")
        datetime_col = st.selectbox("Select datetime column:", datetime_cols)

        features_to_extract = st.multiselect(
            "Select features to extract:",
            ["Year", "Month", "Day", "Hour", "Minute", "DayOfWeek", "Quarter", "IsWeekend"]
        )

        if features_to_extract and st.button("Extract DateTime Features"):
            df_featured = df.copy()

            for feature in features_to_extract:
                if feature == "Year":
                    df_featured[f'{datetime_col}_year'] = df_featured[datetime_col].dt.year
                elif feature == "Month":
                    df_featured[f'{datetime_col}_month'] = df_featured[datetime_col].dt.month
                elif feature == "Day":
                    df_featured[f'{datetime_col}_day'] = df_featured[datetime_col].dt.day
                elif feature == "Hour":
                    df_featured[f'{datetime_col}_hour'] = df_featured[datetime_col].dt.hour
                elif feature == "Minute":
                    df_featured[f'{datetime_col}_minute'] = df_featured[datetime_col].dt.minute
                elif feature == "DayOfWeek":
                    df_featured[f'{datetime_col}_dayofweek'] = df_featured[datetime_col].dt.dayofweek
                elif feature == "Quarter":
                    df_featured[f'{datetime_col}_quarter'] = df_featured[datetime_col].dt.quarter
                elif feature == "IsWeekend":
                    df_featured[f'{datetime_col}_is_weekend'] = (df_featured[datetime_col].dt.dayofweek >= 5).astype(int)

            st.session_state.current_dataframe = df_featured
            st.success("✅ DateTime features extracted!")

    # Text feature extraction
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        st.subheader("📝 Text Feature Extraction")
        text_col = st.selectbox("Select text column:", text_cols)

        text_features = st.multiselect(
            "Select text features to extract:",
            ["Character Count", "Word Count", "Sentence Count", "Contains Email", "Contains URL"]
        )

        if text_features and st.button("Extract Text Features"):
            df_featured = df.copy()

            for feature in text_features:
                if feature == "Character Count":
                    df_featured[f'{text_col}_char_count'] = df_featured[text_col].astype(str).str.len()
                elif feature == "Word Count":
                    df_featured[f'{text_col}_word_count'] = df_featured[text_col].astype(str).str.split().str.len()
                elif feature == "Sentence Count":
                    df_featured[f'{text_col}_sentence_count'] = df_featured[text_col].astype(str).str.count(r'[.!?]+')
                elif feature == "Contains Email":
                    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                    df_featured[f'{text_col}_contains_email'] = df_featured[text_col].astype(str).str.contains(email_pattern, regex=True).astype(int)
                elif feature == "Contains URL":
                    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                    df_featured[f'{text_col}_contains_url'] = df_featured[text_col].astype(str).str.contains(url_pattern, regex=True).astype(int)

            st.session_state.current_dataframe = df_featured
            st.success("✅ Text features extracted!")

def data_filtering(df):
    """Advanced data filtering operations"""
    st.subheader("🔍 Advanced Data Filtering")

    filter_type = st.radio(
        "Select filter type:",
        ["Condition-based Filter", "Statistical Filter", "Pattern-based Filter"]
    )

    if filter_type == "Condition-based Filter":
        condition_filter(df)
    elif filter_type == "Statistical Filter":
        statistical_filter(df)
    elif filter_type == "Pattern-based Filter":
        pattern_filter(df)

def condition_filter(df):
    """Apply condition-based filtering"""
    st.subheader("⚡ Condition-based Filter")

    # Multiple condition support
    st.write("Build your filter conditions:")

    conditions = []

    # Allow up to 3 conditions
    for i in range(3):
        with st.expander(f"Condition {i+1}"):
            col = st.selectbox(f"Select column:", ["None"] + df.columns.tolist(), key=f"col_{i}")

            if col != "None":
                if df[col].dtype in ['int64', 'float64']:
                    operator = st.selectbox("Operator:", ["==", "!=", "<", "<=", ">", ">="], key=f"op_{i}")
                    value = st.number_input("Value:", key=f"val_{i}")
                else:
                    operator = st.selectbox("Operator:", ["==", "!=", "contains", "startswith", "endswith"], key=f"op_{i}")
                    value = st.text_input("Value:", key=f"val_{i}")

                if st.checkbox(f"Include condition {i+1}", key=f"include_{i}"):
                    conditions.append({
                        'column': col,
                        'operator': operator,
                        'value': value
                    })

    if conditions:
        logic_operator = st.radio("Combine conditions with:", ["AND", "OR"])

        if st.button("Apply Conditions"):
            df_filtered = df.copy()

            # Build filter mask
            masks = []
            for condition in conditions:
                col, op, val = condition['column'], condition['operator'], condition['value']

                if op == "==":
                    mask = df_filtered[col] == val
                elif op == "!=":
                    mask = df_filtered[col] != val
                elif op == "<":
                    mask = df_filtered[col] < val
                elif op == "<=":
                    mask = df_filtered[col] <= val
                elif op == ">":
                    mask = df_filtered[col] > val
                elif op == ">=":
                    mask = df_filtered[col] >= val
                elif op == "contains":
                    mask = df_filtered[col].astype(str).str.contains(str(val), na=False)
                elif op == "startswith":
                    mask = df_filtered[col].astype(str).str.startswith(str(val), na=False)
                elif op == "endswith":
                    mask = df_filtered[col].astype(str).str.endswith(str(val), na=False)

                masks.append(mask)

            # Combine masks
            if logic_operator == "AND":
                final_mask = masks[0]
                for mask in masks[1:]:
                    final_mask = final_mask & mask
            else:  # OR
                final_mask = masks[0]
                for mask in masks[1:]:
                    final_mask = final_mask | mask

            df_filtered = df_filtered[final_mask]
            st.session_state.current_dataframe = df_filtered
            st.success(f"✅ Filter applied! {len(df_filtered)} rows remaining.")

def statistical_filter(df):
    """Apply statistical filtering"""
    st.subheader("📊 Statistical Filter")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column for statistical filtering:", numeric_cols)

        filter_method = st.radio(
            "Select statistical filter:",
            ["Remove Outliers (IQR)", "Keep Top/Bottom Percentile", "Remove Extreme Values"]
        )

        if filter_method == "Remove Outliers (IQR)":
            if st.button("Remove Outliers"):
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df_filtered = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)].copy()
                st.session_state.current_dataframe = df_filtered
                st.success(f"✅ Outliers removed! {len(df_filtered)} rows remaining.")

        elif filter_method == "Keep Top/Bottom Percentile":
            percentile = st.slider("Select percentile to keep:", 1, 50, 10)
            keep_option = st.radio("Keep:", ["Top percentile", "Bottom percentile", "Both ends"])

            if st.button("Apply Percentile Filter"):
                if keep_option == "Top percentile":
                    threshold = df[selected_col].quantile(1 - percentile/100)
                    df_filtered = df[df[selected_col] >= threshold].copy()
                elif keep_option == "Bottom percentile":
                    threshold = df[selected_col].quantile(percentile/100)
                    df_filtered = df[df[selected_col] <= threshold].copy()
                else:  # Both ends
                    lower_threshold = df[selected_col].quantile(percentile/100)
                    upper_threshold = df[selected_col].quantile(1 - percentile/100)
                    df_filtered = df[(df[selected_col] <= lower_threshold) | (df[selected_col] >= upper_threshold)].copy()

                st.session_state.current_dataframe = df_filtered
                st.success(f"✅ Percentile filter applied! {len(df_filtered)} rows remaining.")

        elif filter_method == "Remove Extreme Values":
            std_threshold = st.slider("Standard deviations from mean:", 1.0, 5.0, 3.0, 0.1)

            if st.button("Remove Extreme Values"):
                mean = df[selected_col].mean()
                std = df[selected_col].std()
                lower_bound = mean - std_threshold * std
                upper_bound = mean + std_threshold * std

                df_filtered = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)].copy()
                st.session_state.current_dataframe = df_filtered
                st.success(f"✅ Extreme values removed! {len(df_filtered)} rows remaining.")
    else:
        st.warning("No numeric columns found!")

def pattern_filter(df):
    """Apply pattern-based filtering"""
    st.subheader("🔍 Pattern-based Filter")

    text_cols = df.select_dtypes(include=['object']).columns

    if len(text_cols) > 0:
        selected_col = st.selectbox("Select text column:", text_cols)

        pattern_type = st.radio(
            "Select pattern type:",
            ["Regex Pattern", "Contains Substring", "Length Filter", "Character Type Filter"]
        )

        if pattern_type == "Regex Pattern":
            regex_pattern = st.text_input("Enter regex pattern:", "^[A-Za-z]+$")

            if st.button("Apply Regex Filter") and regex_pattern:
                try:
                    df_filtered = df[df[selected_col].astype(str).str.match(regex_pattern, na=False)].copy()
                    st.session_state.current_dataframe = df_filtered
                    st.success(f"✅ Regex filter applied! {len(df_filtered)} rows remaining.")
                except Exception as e:
                    st.error(f"❌ Invalid regex pattern: {str(e)}")

        elif pattern_type == "Contains Substring":
            substring = st.text_input("Enter substring to search:")
            case_sensitive = st.checkbox("Case sensitive")

            if st.button("Apply Substring Filter") and substring:
                if case_sensitive:
                    df_filtered = df[df[selected_col].astype(str).str.contains(substring, na=False)].copy()
                else:
                    df_filtered = df[df[selected_col].astype(str).str.lower().str.contains(substring.lower(), na=False)].copy()

                st.session_state.current_dataframe = df_filtered
                st.success(f"✅ Substring filter applied! {len(df_filtered)} rows remaining.")

        elif pattern_type == "Length Filter":
            min_length = st.number_input("Minimum length:", min_value=0, value=0)
            max_length = st.number_input("Maximum length:", min_value=1, value=100)

            if st.button("Apply Length Filter"):
                df_filtered = df[
                    (df[selected_col].astype(str).str.len() >= min_length) &
                    (df[selected_col].astype(str).str.len() <= max_length)
                ].copy()
                st.session_state.current_dataframe = df_filtered
                st.success(f"✅ Length filter applied! {len(df_filtered)} rows remaining.")

        elif pattern_type == "Character Type Filter":
            char_filter = st.radio(
                "Keep rows with:",
                ["Only alphabetic characters", "Only numeric characters", "Only alphanumeric", "Contains digits", "Contains special characters"]
            )

            if st.button("Apply Character Filter"):
                if char_filter == "Only alphabetic characters":
                    df_filtered = df[df[selected_col].astype(str).str.isalpha()].copy()
                elif char_filter == "Only numeric characters":
                    df_filtered = df[df[selected_col].astype(str).str.isnumeric()].copy()
                elif char_filter == "Only alphanumeric":
                    df_filtered = df[df[selected_col].astype(str).str.isalnum()].copy()
                elif char_filter == "Contains digits":
                    df_filtered = df[df[selected_col].astype(str).str.contains(r'\d', regex=True, na=False)].copy()
                elif char_filter == "Contains special characters":
                    df_filtered = df[df[selected_col].astype(str).str.contains(r'[^a-zA-Z0-9\s]', regex=True, na=False)].copy()

                st.session_state.current_dataframe = df_filtered
                st.success(f"✅ Character filter applied! {len(df_filtered)} rows remaining.")
    else:
        st.warning("No text columns found!")
