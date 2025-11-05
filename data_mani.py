import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

def data_manipulation_page():
    """
    Data Manipulation page functionality
    """

    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe.copy()

        st.subheader("🛠️ Data Manipulation Operations")

        # Choose operation category using RADIO BUTTONS - REMOVED Data Transformation
        manipulation_options = [
            "Data Cleaning",
            "Column Operations",
            "Row Operations",
            "Data Filtering"
        ]

        selected_operation = st.radio(
            "Select Operation Type:",
            manipulation_options,
            horizontal=False
        )

        if selected_operation == "Data Cleaning":
            data_cleaning(df)
        elif selected_operation == "Column Operations":
            column_operations(df)
        elif selected_operation == "Row Operations":
            row_operations(df)
        elif selected_operation == "Data Filtering":
            data_filtering(df)

    else:
        st.warning("⚠️ No data available. Please upload a file first.")


def data_cleaning(df):
    """Data Cleaning Operations"""
    st.subheader("🧹 Data Cleaning Operations")

    cleaning_options = [
        "Handle Missing Values",
        "Remove Duplicates",
        "Change Data Type",
        "Clean Text Data"
    ]

    # Using RADIO BUTTONS instead of selectbox
    selected_cleaning = st.radio(
        "Select cleaning operation:",
        cleaning_options,
        horizontal=True
    )

    if selected_cleaning == "Handle Missing Values":
        handle_missing_values(df)

    elif selected_cleaning == "Remove Duplicates":
        st.write("### Remove Duplicate Rows")

        if st.button("Remove Duplicates"):
            df_cleaned = df.drop_duplicates()
            removed_count = len(df) - len(df_cleaned)
            st.success(f"✅ Removed {removed_count} duplicate rows")
            st.session_state.current_dataframe = df_cleaned
            st.dataframe(df_cleaned, use_container_width=True)

    elif selected_cleaning == "Change Data Type":
        st.write("### Change Column Data Type")

        col = st.selectbox("Select column:", df.columns)
        new_type = st.selectbox("Select new data type:", ["int", "float", "str", "datetime"])

        if st.button("Convert Type"):
            try:
                if new_type == "datetime":
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(new_type)

                st.session_state.current_dataframe = df
                st.success(f"✅ Successfully converted '{col}' to {new_type}")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif selected_cleaning == "Clean Text Data":
        st.write("### Clean Text Data")

        col = st.selectbox("Select text column:", df.select_dtypes(include='object').columns)

        text_operations = st.multiselect(
            "Select operations:",
            ["Remove spaces", "Convert to lowercase", "Remove special characters", "Remove leading/trailing spaces"]
        )

        if st.button("Apply Text Cleaning"):
            try:
                for operation in text_operations:
                    if operation == "Remove spaces":
                        df[col] = df[col].str.replace(" ", "")
                    elif operation == "Convert to lowercase":
                        df[col] = df[col].str.lower()
                    elif operation == "Remove special characters":
                        df[col] = df[col].str.replace(r"[^a-zA-Z0-9]", "")
                    elif operation == "Remove leading/trailing spaces":
                        df[col] = df[col].str.strip()

                st.session_state.current_dataframe = df
                st.success("✅ Text cleaning applied successfully")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def handle_missing_values(df):
    """Handle missing values"""
    st.write("### Handle Missing Values")

    # Show missing values summary
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        st.write("**Missing Values Count:**")
        st.dataframe(missing_summary[missing_summary > 0])
    else:
        st.success("✅ No missing values found!")
        return

    col = st.selectbox("Select column:", df.columns[missing_summary > 0])

    missing_options = ["Drop rows", "Fill with mean", "Fill with median"]
    selected_option = st.radio("Select handling method:", missing_options, horizontal=True)

    if st.button("Apply"):
        try:
            if selected_option == "Drop rows":
                df_cleaned = df.dropna(subset=[col])
                removed = len(df) - len(df_cleaned)
                st.success(f"✅ Removed {removed} rows with missing values in '{col}'")
                st.session_state.current_dataframe = df_cleaned
                st.dataframe(df_cleaned, use_container_width=True)

            elif selected_option == "Fill with mean":
                if df[col].dtype in ['int64', 'float64']:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    st.success(f"✅ Filled missing values with mean ({mean_val:.2f})")
                    st.session_state.current_dataframe = df
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error("❌ Column must be numeric for mean fill")

            elif selected_option == "Fill with median":
                if df[col].dtype in ['int64', 'float64']:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    st.success(f"✅ Filled missing values with median ({median_val:.2f})")
                    st.session_state.current_dataframe = df
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error("❌ Column must be numeric for median fill")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def column_operations(df):
    """Column Operations"""
    st.subheader("📊 Column Operations")

    col_options = ["Rename Column", "Drop Column", "Add Column"]
    # Using RADIO BUTTONS
    selected_col_op = st.radio("Select column operation:", col_options, horizontal=True)

    if selected_col_op == "Rename Column":
        st.write("### Rename Column")
        col = st.selectbox("Select column to rename:", df.columns)
        new_name = st.text_input("Enter new column name:")

        if st.button("Rename"):
            try:
                df = df.rename(columns={col: new_name})
                st.session_state.current_dataframe = df
                st.success(f"✅ Column '{col}' renamed to '{new_name}'")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif selected_col_op == "Drop Column":
        st.write("### Drop Column")
        col = st.selectbox("Select column to drop:", df.columns)

        if st.button("Drop Column"):
            try:
                df = df.drop(columns=[col])
                st.session_state.current_dataframe = df
                st.success(f"✅ Column '{col}' dropped successfully")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif selected_col_op == "Add Column":
        st.write("### Add Column")
        new_col_name = st.text_input("Enter new column name:")
        default_value = st.text_input("Enter default value:")

        if st.button("Add Column"):
            try:
                df[new_col_name] = default_value
                st.session_state.current_dataframe = df
                st.success(f"✅ Column '{new_col_name}' added successfully")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def row_operations(df):
    """Row Operations"""
    st.subheader("📋 Row Operations")

    row_options = ["Drop Rows", "Update Rows"]
    # Using RADIO BUTTONS
    selected_row_op = st.radio("Select row operation:", row_options, horizontal=True)

    if selected_row_op == "Drop Rows":
        st.write("### Drop Rows")

        drop_method = st.radio("Drop method:", ["By index", "By condition"], horizontal=True)

        if drop_method == "By index":
            indices = st.text_input("Enter row indices to drop (comma-separated):")
            if st.button("Drop Rows"):
                try:
                    indices_list = [int(i.strip()) for i in indices.split(",")]
                    df = df.drop(indices_list)
                    st.session_state.current_dataframe = df
                    st.success(f"✅ Dropped {len(indices_list)} rows")
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        else:  # By condition
            col = st.selectbox("Select column:", df.columns)
            condition = st.selectbox("Condition:", ["==", "!=", ">", "<", ">=", "<=", "contains"])
            value = st.text_input("Enter value:")

            if st.button("Drop Rows"):
                try:
                    if condition == "==":
                        df = df[df[col] != value]
                    elif condition == "!=":
                        df = df[df[col] == value]
                    elif condition == ">":
                        df = df[df[col] <= float(value)]
                    elif condition == "<":
                        df = df[df[col] >= float(value)]
                    elif condition == ">=":
                        df = df[df[col] < float(value)]
                    elif condition == "<=":
                        df = df[df[col] > float(value)]
                    elif condition == "contains":
                        df = df[~df[col].str.contains(value, na=False)]

                    st.session_state.current_dataframe = df
                    st.success(f"✅ Rows dropped successfully")
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    elif selected_row_op == "Update Rows":
        st.write("### Update Rows")

        col = st.selectbox("Select column to update:", df.columns)
        find_value = st.text_input("Find value:")
        replace_value = st.text_input("Replace with:")

        if st.button("Update Rows"):
            try:
                updated_count = (df[col] == find_value).sum()
                df[col] = df[col].replace(find_value, replace_value)
                st.session_state.current_dataframe = df
                st.success(f"✅ Updated {updated_count} rows")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def data_filtering(df):
    """Data Filtering"""
    st.subheader("🔍 Data Filtering")

    filter_options = ["Filter by Value", "Filter by Range", "Filter by Null"]
    # Using RADIO BUTTONS
    selected_filter = st.radio("Select filter type:", filter_options, horizontal=True)

    if selected_filter == "Filter by Value":
        st.write("### Filter by Value")
        col = st.selectbox("Select column:", df.columns)
        value = st.text_input("Enter value to filter:")

        if st.button("Apply Filter"):
            try:
                filtered_df = df[df[col] == value]
                st.session_state.current_dataframe = filtered_df
                st.success(f"✅ Filtered {len(filtered_df)} rows")
                st.dataframe(filtered_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif selected_filter == "Filter by Range":
        st.write("### Filter by Range")
        col = st.selectbox("Select numeric column:", df.select_dtypes(include=[np.number]).columns)

        min_val = st.number_input("Enter minimum value:", value=float(df[col].min()))
        max_val = st.number_input("Enter maximum value:", value=float(df[col].max()))

        if st.button("Apply Range Filter"):
            try:
                filtered_df = df[(df[col] >= min_val) & (df[col] <= max_val)]
                st.session_state.current_dataframe = filtered_df
                st.success(f"✅ Filtered {len(filtered_df)} rows")
                st.dataframe(filtered_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif selected_filter == "Filter by Null":
        st.write("### Filter by Null Values")

        null_option = st.radio("Select option:", ["Show rows with nulls", "Show rows without nulls"], horizontal=True)
        col = st.selectbox("Select column:", df.columns)

        if st.button("Apply Null Filter"):
            try:
                if null_option == "Show rows with nulls":
                    filtered_df = df[df[col].isnull()]
                else:
                    filtered_df = df[df[col].notnull()]

                st.session_state.current_dataframe = filtered_df
                st.success(f"✅ Filtered {len(filtered_df)} rows")
                st.dataframe(filtered_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
