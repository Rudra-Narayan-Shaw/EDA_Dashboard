import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import duckdb

# Import the separate modules
try:
    #from ingestion import ingestion_page
    from eda import eda_page
    from data_mani import data_manipulation_page
except ImportError:
    st.error("Required modules not found. Please ensure ingestion.py, eda.py, and data_mani.py are in the same directory.")

def initialize_session_state():
    """Initialize session state variables"""
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []

    if 'current_dataframe' not in st.session_state:
        st.session_state.current_dataframe = None

    # Simple DuckDB integration
    if 'duckdb_conn' not in st.session_state:
        st.session_state.duckdb_conn = duckdb.connect(':memory:')

def save_upload_history(filename, file_type, upload_time, rows, columns):
    """Save upload history to session state"""
    history_entry = {
        'file_name': filename,
        'file_type': file_type,
        'upload_date': upload_time.strftime('%Y-%m-%d'),
        'upload_time': upload_time.strftime('%H:%M:%S'),
        'rows': rows,
        'columns': columns
    }
    st.session_state.upload_history.append(history_entry)

def display_upload_history():
    """Display upload history table"""
    if st.session_state.upload_history:
        st.subheader("📈 Upload History")

        # Create DataFrame from upload history
        history_df = pd.DataFrame(st.session_state.upload_history)

        # Display as a nice table
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # Add download button for history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Upload History",
            data=csv,
            file_name="upload_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No upload history available.")

def file_upload_section():
    """Handle file upload functionality with updated file types"""
    st.subheader("📁 File Upload")

    # Fixed: Change "Choose Option" to "Choose the File Type"
    file_types = ["Choose the File Type", ".csv", ".xlsx/.xls", ".txt"]
    selected_file_type = st.selectbox(
        "Select File Type",
        file_types,
        help="Choose the type of file you want to upload"
    )

    if selected_file_type == "Choose the File Type":
        st.info("Please select a file type to proceed with upload.")
        return

    # Determine accepted file extensions
    if selected_file_type == ".csv":
        accepted_types = ["csv"]
    elif selected_file_type == ".xlsx/.xls":
        accepted_types = ["xlsx", "xls"]
    elif selected_file_type == ".txt":
        accepted_types = ["txt"]

    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=accepted_types,
        help=f"Upload a {selected_file_type} file for analysis"
    )

    if uploaded_file is not None:
        try:
            # Read file based on type with proper encoding handling
            if selected_file_type == ".csv":
                try:
                    df = pd.read_csv(uploaded_file)  # Try UTF-8 first
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding="latin1")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding="cp1252")
            elif selected_file_type == ".xlsx/.xls":
                df = pd.read_excel(uploaded_file)
            elif selected_file_type == ".txt":
                df = pd.read_csv(uploaded_file, delimiter='\t')

            # Store in session state
            st.session_state.current_dataframe = df

            # Simple DuckDB storage (just the basics)
            try:
                table_name = "uploaded_data"
                st.session_state.duckdb_conn.register(table_name, df)
            except:
                pass  # Silent fail for database issues

            # Save to upload history
            upload_time = datetime.now()
            save_upload_history(
                uploaded_file.name,
                selected_file_type,
                upload_time,
                len(df),
                len(df.columns)
            )

            # Display success message
            st.success(f"✅ File uploaded successfully!")
            st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            # Show preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="EDA Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Main header
    st.title("📊 EDA Dashboard")
    st.markdown("---")
    st.markdown(" ")

    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")

    # Radio buttons for page selection
    page_selection = st.sidebar.radio(
        "Select a page:",
        ["Ingestion", "EDA", "Data Manipulation"],
        help="Choose which functionality you want to use"
    )

    # Add some information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Instructions")
    st.sidebar.markdown("""
    - **Ingestion**: Upload and preview your data
    - **EDA**: Perform exploratory data analysis  
    - **Data Manipulation**: Clean and transform your data
    """)

    # Main content based on selection
    if page_selection == "Ingestion":
        st.header("📥 Data Ingestion")
        st.markdown(" ")

        # File upload section
        file_upload_section()

        # Upload history
        st.markdown("---")
        display_upload_history()

    elif page_selection == "EDA":
        st.markdown(" ")

        if st.session_state.current_dataframe is not None:
            try:
                eda_page()
            except NameError:
                st.warning("eda.py module not found. Please upload eda.py file.")
                st.info("Basic EDA functionality would be implemented in the eda.py module.")
        else:
            st.warning("⚠️ No data available. Please upload a file in the Ingestion section first.")

    elif page_selection == "Data Manipulation":
        st.markdown(" ")

        if st.session_state.current_dataframe is not None:
            try:
                data_manipulation_page()
            except NameError:
                st.warning("data_mani.py module not found. Please upload data_mani.py file.")
                st.info("Data manipulation functionality would be implemented in the data_mani.py module.")
        else:
            st.warning("⚠️ No data available. Please upload a file in the Ingestion section first.")

if __name__ == "__main__":
    main()