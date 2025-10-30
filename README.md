📊 EDA Dashboard
A comprehensive, interactive data exploration tool built with Streamlit and DuckDB



🎯 Overview
EDA Dashboard is a no-code, web-based application for exploratory data analysis (EDA). Upload your data files (CSV, Excel, TXT) and instantly explore them through interactive visualizations, statistical analysis, and comprehensive data insights.



✨ Key Features
📥 Data Ingestion
✅ Multi-format support (CSV, Excel, TXT)

✅ Automatic encoding detection (UTF-8, Latin1, CP1252)

✅ Drag-and-drop file upload

✅ Persistent upload history with database storage

✅ Support for duplicate file uploads (same file multiple times)

✅ Real-time file metadata display

✅ Data preview with configurable rows


🔍 Exploratory Data Analysis (EDA)
✅ Basic data overview (rows, columns, missing values)

✅ Column-level statistics and data type analysis

✅ Sample data exploration (first, last, random, custom range)

📊 8 Interactive Visualization Types
Histogram - Distribution analysis with statistical metrics

Bar Chart - Categorical data with sorting options

Scatter Plot - Bivariate relationships with correlation

Pair Plot - Multi-variable relationship matrix

Heatmap - Correlation analysis with multiple methods

Line Chart - Time series and trend visualization

Pie Chart - Composition and percentage analysis

Radar Plot - Multi-dimensional comparison

💾 Stateful Database System
✅ Persistent DuckDB storage

✅ Automatic upload history tracking

✅ Complete data retention across sessions

✅ Timestamp tracking for each upload

✅ Download history as CSV

✅ Delete history management

🎨 User-Friendly Interface
✅ Clean, intuitive dashboard design

✅ Interactive Plotly charts

✅ Real-time status indicators

✅ Professional error handling

✅ Responsive layout

✅ Mobile-friendly design

⚡ Performance & Reliability
✅ Fast data processing with Pandas & NumPy

✅ Smooth interactive visualizations

✅ Robust error handling

✅ No external dependencies (local processing)

✅ Private data (no cloud upload)


📖 Usage
Step 1: Upload Data
Navigate to Ingestion page

Select file type (CSV, Excel, or TXT)

Upload your data file

View file preview and metadata

Step 2: Explore Data
Go to EDA page

View basic data overview

Explore sample data

Check column information

Step 3: Visualize
Choose visualization type

Select columns to analyze

Customize chart settings

Interact with charts (zoom, pan, hover)

Step 4: Track History
View all uploaded files

Download history as CSV

Delete old entries

Re-upload files anytime

🏗️ Project Structure
text
eda-dashboard/
├── app.py                    # Main Streamlit application
├── database.py              # DuckDB database manager
├── eda.py                   # EDA visualization functions
├── data_mani.py            # Data manipulation
├── requirements.txt         # Python dependencies
├── eda_dashboard.db        # DuckDB database (auto-created)
├── README.md               # This file

🔧 Technical Stack
Component	Technology	Version
Frontend	Streamlit	1.29.0
Data Processing	Pandas	2.1.4
Numerical Computing	NumPy	1.26.4
Visualization	Plotly	5.17.0
Database	DuckDB	0.9.2
File Handling	openpyxl, xlrd	Latest
Language	Python	3.8+
📊 Database Schema
Upload History Table


Features:

Allows duplicate entries (same file multiple times)

Persistent storage across sessions

Efficient querying

Automatic timestamp tracking







Plotly for beautiful interactive visualizations

Pandas for powerful data manipulation

