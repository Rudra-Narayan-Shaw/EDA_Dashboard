import streamlit as st
import duckdb
import pandas as pd
from datetime import datetime
import os

class DatabaseManager:
    """
    Database Manager with proper connection and table management workflow
    1. Check if database is connected
    2. If not connected, connect it
    3. Check if upload_history table exists
    4. If exists, use it; if not, create it
    """

    def __init__(self, db_path="eda_dashboard.db"):
        """Initialize database connection and tables"""
        self.db_path = db_path
        self.conn = None
        self.is_connected = False

        # Step 1: Check and establish connection
        self.check_and_connect()

        # Step 2: Check and manage tables
        if self.is_connected:
            self.check_and_manage_tables()

    def check_and_connect(self):
        """
        Step 1: Check if database is connected.
        If not connected, then connect it.
        """
        try:
            # Try to establish connection
            self.conn = duckdb.connect(self.db_path, read_only=False)

            # Verify connection is working
            self.conn.execute("SELECT 1").fetchone()

            self.is_connected = True
            st.sidebar.success("✅ Database Connected")

        except Exception as e:
            self.is_connected = False
            st.sidebar.error(f"❌ Database Connection Failed: {str(e)}")
            self.conn = None

    def check_and_manage_tables(self):
        """
        Step 2: Check if 'upload_history' table exists.
        If exists, use it.
        If not exists, create new table.
        """
        if not self.is_connected or not self.conn:
            st.sidebar.error("❌ Cannot manage tables - Database not connected")
            return False

        try:
            # Check if table exists
            table_exists = self._table_exists("upload_history")

            if table_exists:
                # Table exists - use it
                st.sidebar.info("📋 Using existing upload_history table")
                return True
            else:
                # Table doesn't exist - create new one
                return self._create_upload_history_table()

        except Exception as e:
            st.sidebar.error(f"❌ Error managing tables: {str(e)}")
            return False

    def _table_exists(self, table_name):
        """Check if a table exists in the database"""
        if not self.is_connected or not self.conn:
            return False

        try:
            query = f"SELECT * FROM information_schema.tables WHERE table_name = '{table_name}' LIMIT 1"
            result = self.conn.execute(query).fetchall()
            return len(result) > 0
        except:
            # If information_schema doesn't work, try alternative method
            try:
                self.conn.execute(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
                return True
            except:
                return False

    def _create_upload_history_table(self):
        """Create the upload_history table if it doesn't exist"""
        if not self.is_connected or not self.conn:
            return False

        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS upload_history (
                file_name VARCHAR NOT NULL,
                file_type VARCHAR NOT NULL,
                upload_date VARCHAR NOT NULL,
                upload_time VARCHAR NOT NULL,
                file_size VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.conn.execute(create_table_query)
            self.conn.commit()
            st.sidebar.success("✅ Created new upload_history table")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Error creating table: {str(e)}")
            return False

    def is_db_connected(self):
        """Get current database connection status"""
        return self.is_connected

    def table_exists(self, table_name):
        """Check if a specific table exists"""
        return self._table_exists(table_name)

    def insert_upload_history(self, file_name, file_type, upload_date, upload_time, file_size):
        """Insert new upload record into database"""
        if not self.is_connected or not self.conn:
            st.error("❌ Database not connected - cannot save upload history")
            return False

        try:
            # Verify table exists
            if not self._table_exists("upload_history"):
                st.error("❌ Upload history table does not exist")
                return False

            insert_query = """
            INSERT INTO upload_history (file_name, file_type, upload_date, upload_time, file_size)
            VALUES (?, ?, ?, ?, ?)
            """
            self.conn.execute(insert_query, [file_name, file_type, upload_date, upload_time, file_size])
            self.conn.commit()
            return True

        except Exception as e:
            st.error(f"❌ Error saving upload history: {str(e)}")
            return False

    def get_upload_history(self):
        """Retrieve all upload history from database"""
        if not self.is_connected or not self.conn:
            return pd.DataFrame(columns=['file_name', 'file_type', 'upload_date', 'upload_time', 'file_size'])

        try:
            # Check if table exists first
            if not self._table_exists("upload_history"):
                return pd.DataFrame(columns=['file_name', 'file_type', 'upload_date', 'upload_time', 'file_size'])

            query = """
            SELECT file_name, file_type, upload_date, upload_time, file_size
            FROM upload_history 
            ORDER BY created_at DESC
            """
            result = self.conn.execute(query).fetchdf()
            return result

        except Exception as e:
            st.error(f"❌ Error loading upload history: {str(e)}")
            return pd.DataFrame(columns=['file_name', 'file_type', 'upload_date', 'upload_time', 'file_size'])

    def get_upload_count(self):
        """Get total number of uploaded files"""
        if not self.is_connected or not self.conn:
            return 0

        try:
            if not self._table_exists("upload_history"):
                return 0

            count_query = "SELECT COUNT(*) as count FROM upload_history"
            result = self.conn.execute(count_query).fetchone()
            return result[0] if result else 0
        except Exception as e:
            return 0

    def clear_upload_history(self):
        """Clear all upload history (for testing/cleanup)"""
        if not self.is_connected or not self.conn:
            return False

        try:
            if not self._table_exists("upload_history"):
                return False

            self.conn.execute("DELETE FROM upload_history")
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"❌ Error clearing upload history: {str(e)}")
            return False

    def register_dataframe(self, df, table_name="uploaded_data"):
        """Register pandas DataFrame as table in DuckDB"""
        if not self.is_connected or not self.conn:
            return False

        try:
            # Drop if exists and recreate (this is for temporary analysis data)
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.register(table_name, df)
            return True
        except Exception as e:
            st.error(f"❌ Error registering dataframe: {str(e)}")
            return False

    def get_connection_status(self):
        """Get detailed connection status"""
        status = {
            'is_connected': self.is_connected,
            'db_path': self.db_path,
            'upload_history_exists': self._table_exists("upload_history") if self.is_connected else False,
            'db_file_exists': os.path.exists(self.db_path),
            'db_file_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        }
        return status

    def close(self):
        """Close database connection"""
        if self.conn:
            try:
                self.conn.commit()
                self.conn.close()
            except:
                pass
            self.conn = None
            self.is_connected = False

def database_page():
    """Database page - blank as requested"""
    st.header("🦆 Database")
    st.info("🚧 This section is under development. Features will be added soon!")