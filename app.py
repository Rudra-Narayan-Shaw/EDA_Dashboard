import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from database import DatabaseManager
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import the separate modules
try:
    from eda import eda_page
    from data_mani import data_manipulation_page
except ImportError as e:
    st.error(f"Required modules not found: {str(e)}")

def initialize_session_state():
    """Initialize session state variables"""
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []

    if 'current_dataframe' not in st.session_state:
        st.session_state.current_dataframe = None

    # Initialize persistent database manager (only once per session)
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager("eda_dashboard.db")

def get_file_size_formatted(file_size_bytes):
    """Convert file size from bytes to human readable format"""
    if file_size_bytes < 1024:
        return f"{file_size_bytes} B"
    elif file_size_bytes < 1024**2:
        return f"{file_size_bytes/1024:.1f} KB"
    elif file_size_bytes < 1024**3:
        return f"{file_size_bytes/(1024**2):.1f} MB"
    else:
        return f"{file_size_bytes/(1024**3):.1f} GB"

def save_upload_history(filename, file_type, upload_time, file_size_bytes):
    """Save upload history to persistent database"""
    file_size_formatted = get_file_size_formatted(file_size_bytes)

    # Verify database is connected
    if not st.session_state.db_manager.is_db_connected():
        return False

    # Save to database using DatabaseManager
    success = st.session_state.db_manager.insert_upload_history(
        filename,
        file_type,
        upload_time.strftime('%Y-%m-%d'),
        upload_time.strftime('%H:%M:%S'),
        file_size_formatted
    )

    return success

def load_upload_history_from_db():
    """Load upload history from persistent database"""
    return st.session_state.db_manager.get_upload_history()

def display_upload_history():
    """Display upload history table from persistent database with delete button"""
    # Load history from database
    history_df = load_upload_history_from_db()

    if not history_df.empty:
        st.subheader("📈 Upload History")

        # Display as a nice table
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # Add buttons for download and delete in two columns
        col1, col2 = st.columns(2)

        with col1:
            # Download button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Upload History",
                data=csv,
                file_name="upload_history.csv",
                mime="text/csv"
            )

        with col2:
            # Delete button
            if st.button("🗑️ Delete Upload History", key="delete_history"):
                # Delete and show confirmation
                if st.session_state.db_manager.clear_upload_history():
                    st.success("✅ Upload history deleted successfully!")
                    st.info("📋 All upload records have been removed from the database.")
                    # Refresh the page to show updated history
                    st.rerun()
                else:
                    st.error("❌ Failed to delete upload history")

        # Show total files uploaded
        total_files = len(history_df)
        st.info(f"📊 Total files uploaded: {total_files}")

    else:
        st.info("No upload history available.")

def file_upload_section():
    """Handle file upload functionality with updated file types"""
    st.subheader("📁 File Upload")

    # File type selection
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
            # Suppress warnings during file processing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Get file size
                file_size_bytes = uploaded_file.size

                # Read file based on type with proper encoding handling
                if selected_file_type == ".csv":
                    try:
                        df = pd.read_csv(uploaded_file)
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
                    df = pd.read_csv(uploaded_file, delimiter='	')

                # Store in session state
                st.session_state.current_dataframe = df

                # Register dataframe in database for analysis
                # This now handles duplicate uploads gracefully and silently
                st.session_state.db_manager.register_dataframe(df, "uploaded_data")

                # Save to upload history in persistent database
                # This allows duplicate file uploads with different timestamps
                upload_time = datetime.now()
                success = save_upload_history(
                    uploaded_file.name,
                    selected_file_type,
                    upload_time,
                    file_size_bytes
                )

                # Display success message (only success, no errors)
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

                st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.info(f"📁 File Size: {get_file_size_formatted(file_size_bytes)}")

                # Show preview
                with st.expander("📄 Data Preview"):
                    st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def display_database_status():
    """Display detailed database status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🦆 Database Status")

    if st.session_state.db_manager and st.session_state.db_manager.is_db_connected():
        # Get status details
        status = st.session_state.db_manager.get_connection_status()

        # Connection status
        st.sidebar.success("✅ Database Connected")

        # Upload history count
        upload_count = st.session_state.db_manager.get_upload_count()
        st.sidebar.metric("Upload History Records", upload_count)

        # Table existence
        if status['upload_history_exists']:
            st.sidebar.success("✅ upload_history table exists")
        else:
            st.sidebar.warning("⚠️ upload_history table not found")

        # Database file info
        if status['db_file_exists']:
            db_size = get_file_size_formatted(status['db_file_size'])
            st.sidebar.text(f"Database Size: {db_size}")
            st.sidebar.text(f"Database Path: {status['db_path']}")

        st.sidebar.success("✅ Stateful Database Active")
        st.sidebar.caption("📝 History persists across sessions")

    else:
        st.sidebar.error("❌ Database Disconnected")
        st.sidebar.caption("Unable to connect to database")

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="EDA Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state (includes database initialization)
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

    # Display database status in sidebar
    display_database_status()

    # Main content based on selection
    if page_selection == "Ingestion":
        st.header("📥 Data Ingestion")
        st.markdown(" ")

        # File upload section
        file_upload_section()

        # Upload history (loaded from persistent database)
        st.markdown("---")
        display_upload_history()

    elif page_selection == "EDA":
        st.markdown(" ")

        if st.session_state.current_dataframe is not None:
            try:
                eda_page()
            except Exception as e:
                st.error(f"Error in EDA module: {str(e)}")
        else:
            st.warning("⚠️ No data available. Please upload a file in the Ingestion section first.")

    elif page_selection == "Data Manipulation":
        st.markdown(" ")

        if st.session_state.current_dataframe is not None:
            try:
                data_manipulation_page()
            except Exception as e:
                st.error(f"Error in Data Manipulation module: {str(e)}")
        else:
            st.warning("⚠️ No data available. Please upload a file in the Ingestion section first.")

if __name__ == "__main__":
    main()
