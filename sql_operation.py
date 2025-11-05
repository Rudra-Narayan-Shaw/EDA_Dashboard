import streamlit as st
import pandas as pd
import duckdb
import os
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')

class SQLOperationManager:
    """Manage SQL operations on uploaded datasets using DuckDB"""

    def __init__(self, db_path="eda_dashboard.db"):
        self.db_path = db_path
        self.conn = None
        self.table_name = "uploaded_data"
        self.initialize_connection()

    def initialize_connection(self):
        """Initialize DuckDB connection"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=False)
            st.session_state.sql_conn = self.conn
        except Exception as e:
            st.error(f"❌ Failed to connect to database: {str(e)}")

    def register_dataframe(self, df, table_name="uploaded_data"):
        """Register pandas DataFrame as DuckDB table"""
        if self.conn is None:
            return False

        try:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass

            self.conn.register(table_name, df)
            self.table_name = table_name
            return True
        except Exception as e:
            st.error(f"❌ Error registering table: {str(e)}")
            return False

    def execute_query(self, query):
        """Execute SQL query and return results"""
        if self.conn is None:
            return None, "❌ No database connection"

        try:
            result = self.conn.execute(query).fetchdf()
            return result, None
        except Exception as e:
            return None, f"❌ Query Error: {str(e)}"

    def get_table_info(self):
        """Get information about registered table"""
        if self.conn is None:
            return None

        try:
            info = self.conn.execute(f"DESCRIBE {self.table_name}").fetchdf()
            return info
        except Exception as e:
            return None

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Query Templates
QUERY_TEMPLATES = {
    "Select All": "SELECT * FROM uploaded_data;",
    "Count Rows": "SELECT COUNT(*) as total_rows FROM uploaded_data;",
    "Column Summary": "SELECT COUNT(*), COUNT(DISTINCT *) FROM uploaded_data;",
    "Top 10 Rows": "SELECT * FROM uploaded_data LIMIT 10;",
    "Group By": "SELECT column_name, COUNT(*) as count FROM uploaded_data GROUP BY column_name;",
    "Average": "SELECT AVG(numeric_column) as average FROM uploaded_data;",
    "Filter Records": "SELECT * FROM uploaded_data WHERE condition;",
    "Sort Data": "SELECT * FROM uploaded_data ORDER BY column_name DESC;",
}


def initialize_sql_session():
    """Initialize SQL operation session state"""
    if 'sql_query_history' not in st.session_state:
        st.session_state.sql_query_history = []

    if 'sql_manager' not in st.session_state:
        st.session_state.sql_manager = SQLOperationManager()

    if 'sql_query_box' not in st.session_state:
        st.session_state.sql_query_box = "SELECT * FROM uploaded_data LIMIT 10;"

    if 'show_copy_message' not in st.session_state:
        st.session_state.show_copy_message = False

    if 'switch_to_sql_tab' not in st.session_state:
        st.session_state.switch_to_sql_tab = False


def sql_operation_page():
    """SQL Operation Page"""

    initialize_sql_session()

    st.subheader("🗄️ SQL Operations")
    st.markdown("Execute custom SQL queries on your uploaded dataset using DuckDB")

    if st.session_state.current_dataframe is None:
        st.warning("⚠️ Please upload a dataset first in the Ingestion page")
        return

    # Register dataframe in DuckDB
    df = st.session_state.current_dataframe
    manager = st.session_state.sql_manager

    if not manager.register_dataframe(df, "uploaded_data"):
        st.error("❌ Failed to register dataframe in database")
        return

    st.success("✅ Dataset registered as table 'uploaded_data'")

    # REARRANGED TABS: Table Info, SQL Query, Templates, History
    tab1, tab2, tab3, tab4 = st.tabs(
        ["ℹ️ Table Info", "📝 SQL Query", "📚 Templates", "📋 History"]
    )

    # TAB 1: Table Information (moved to first)
    with tab1:
        st.markdown("### Dataset Information")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

        st.markdown("---")
        st.markdown("### Column Information")

        table_info = manager.get_table_info()
        if table_info is not None:
            st.dataframe(table_info, use_container_width=True)

        st.markdown("---")
        st.markdown("### Data Types")
        st.dataframe(
            pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum()
            }),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

    # TAB 2: SQL Query Input and Execution (moved to second)
    with tab2:
        st.markdown("### Execute SQL Query")

        # SQL Query Input with key to maintain state
        sql_query = st.text_area(
            "Enter your SQL query:",
            value=st.session_state.sql_query_box,
            height=200,
            placeholder="SELECT * FROM uploaded_data LIMIT 10;",
            key="sql_input_box"
        )

        # Update session state when text changes
        st.session_state.sql_query_box = sql_query

        col1, col2, col3 = st.columns(3)

        with col1:
            run_query = st.button("▶️ Run Query", use_container_width=True)

        with col2:
            clear_query = st.button("🔄 Clear", use_container_width=True)

        with col3:
            download_results = st.button("⬇️ Download Results", use_container_width=True)

        if clear_query:
            st.session_state.sql_query_box = ""
            st.rerun()

        # Execute Query
        if run_query:
            if not sql_query.strip():
                st.warning("⚠️ Please enter a query")
            else:
                with st.spinner("🔄 Executing query..."):
                    result_df, error = manager.execute_query(sql_query)

                    if error:
                        st.error(error)
                    else:
                        st.success("✅ Query executed successfully!")
                        st.dataframe(result_df, use_container_width=True)

                        # Add to history
                        if sql_query not in st.session_state.sql_query_history:
                            st.session_state.sql_query_history.append(sql_query)

                        # Store results for download
                        st.session_state.last_query_results = result_df

                        # Show query statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows Returned", len(result_df))
                        with col2:
                            st.metric("Columns", len(result_df.columns))
                        with col3:
                            st.metric("Memory Usage", f"{result_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

        # Download Results
        if download_results and 'last_query_results' in st.session_state:
            csv = st.session_state.last_query_results.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # TAB 3: Query Templates (moved to third)
    with tab3:
        st.markdown("### Query Templates")
        st.markdown("Choose a template and modify as needed")

        selected_template = st.selectbox(
            "Select a template:",
            list(QUERY_TEMPLATES.keys())
        )

        template_query = QUERY_TEMPLATES[selected_template]

        st.markdown(f"**Template:** {selected_template}")
        st.code(template_query, language="sql")

        # Fixed: Properly paste template to query box with auto-switch
        if st.button("📋 Copy to Query Box"):
            # Set the query in session state
            st.session_state.sql_query_box = template_query

            # Show success messages
            st.success("✅ Template copied! Go to SQL Query tab to see it.")
            st.info("💡 The query has been inserted into the SQL Query box above. You can modify it and run it.")

            # Show messages for 5 seconds
            time.sleep(5)

            # Switch to SQL Query tab by rerunning and setting flag
            st.session_state.switch_to_sql_tab = True
            st.rerun()

        st.markdown("---")
        st.markdown("### Quick Reference")
        st.markdown("""
        **Common SQL Commands:**
        - `SELECT * FROM uploaded_data` - Get all data
        - `SELECT COUNT(*) FROM uploaded_data` - Count rows
        - `SELECT DISTINCT column FROM uploaded_data` - Get unique values
        - `SELECT * FROM uploaded_data WHERE condition` - Filter data
        - `SELECT * FROM uploaded_data ORDER BY column DESC` - Sort data
        - `SELECT column, COUNT(*) FROM uploaded_data GROUP BY column` - Aggregate
        - `SELECT * FROM uploaded_data LIMIT 10` - Get first 10 rows

        **Aggregate Functions:**
        - `COUNT()`, `SUM()`, `AVG()`, `MIN()`, `MAX()`, `STDDEV()`, `VARIANCE()`

        **Operators:**
        - Comparison: `=`, `!=`, `<`, `>`, `<=`, `>=`
        - Logical: `AND`, `OR`, `NOT`
        - String: `LIKE`, `IN`, `BETWEEN`
        """)

    # TAB 4: Query History (moved to fourth)
    with tab4:
        st.markdown("### Query History")

        if st.session_state.sql_query_history:
            for i, query in enumerate(reversed(st.session_state.sql_query_history), 1):
                with st.expander(f"Query #{i}", expanded=False):
                    st.code(query, language="sql")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📋 Use Query #{i}", key=f"use_{i}"):
                            st.session_state.sql_query_box = query
                            st.success(f"✅ Query #{i} copied to SQL Query box!")
                            st.info("Go to 'SQL Query' tab to run it.")

                    with col2:
                        if st.button(f"🗑️ Delete Query #{i}", key=f"delete_{i}"):
                            st.session_state.sql_query_history.remove(query)
                            st.success("✅ Query deleted!")
                            st.rerun()

            if st.button("🗑️ Clear All History"):
                st.session_state.sql_query_history = []
                st.success("✅ History cleared!")
                st.rerun()
        else:
            st.info("ℹ️ No query history yet. Run some queries to see them here!")


def sql_operation_display():
    """Display SQL Operation page"""
    sql_operation_page()
