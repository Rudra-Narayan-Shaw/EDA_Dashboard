📊 EDA Dashboard
A comprehensive, interactive data exploration tool built with Streamlit and DuckDB

Python
Streamlit
DuckDB
License

🎯 Overview
EDA Dashboard is a no-code, web-based application for exploratory data analysis (EDA). Upload your data files (CSV, Excel, TXT) and instantly explore them through interactive visualizations, statistical analysis, and comprehensive data insights—all without writing a single line of code.

Perfect for data scientists, analysts, students, and anyone who works with data!

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

🚀 Quick Start
Prerequisites
Python 3.8 or higher

pip (Python package manager)

Installation
Clone the repository

bash
git clone https://github.com/yourusername/eda-dashboard.git
cd eda-dashboard
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
streamlit run app.py
Open in browser

text
http://localhost:8501
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
├── data_mani.py            # Data manipulation (future)
├── requirements.txt         # Python dependencies
├── eda_dashboard.db        # DuckDB database (auto-created)
├── README.md               # This file
└── PROJECT_DESCRIPTION.md  # Detailed project documentation
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
sql
CREATE TABLE upload_history (
    file_name VARCHAR NOT NULL,        -- File name
    file_type VARCHAR NOT NULL,        -- File extension
    upload_date VARCHAR NOT NULL,      -- Upload date (YYYY-MM-DD)
    upload_time VARCHAR NOT NULL,      -- Upload time (HH:MM:SS)
    file_size VARCHAR NOT NULL,        -- Human-readable size
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
Features:

Allows duplicate entries (same file multiple times)

Persistent storage across sessions

Efficient querying

Automatic timestamp tracking

💡 Use Cases
👨‍🔬 Data Scientists
Explore new datasets quickly

Understand data distributions

Identify patterns and outliers

Generate preliminary statistics

📈 Business Analysts
Analyze sales and performance data

Track trends over time

Create visual reports

Export analysis history

🎓 Students & Researchers
Learn EDA concepts

Practice data analysis

Generate visualizations for papers

Understand data before modeling

📊 Anyone Working with Data
No coding required

Intuitive interface

Professional visualizations

Persistent history tracking

🎯 Features in Detail
Data Ingestion
Multi-format support - CSV, Excel (.xlsx, .xls), TXT

Smart encoding detection - Automatically detects UTF-8, Latin1, CP1252

File preview - View first rows before full import

Metadata tracking - Rows, columns, file size, timestamps

Upload history - Persistent database of all uploads

Duplicate uploads - Upload same file multiple times with different timestamps

Exploratory Data Analysis
Overview metrics - Total rows, columns, missing values

Column analysis - Data types, non-null counts, unique values

Sample views - First, last, random, custom range

Data type distribution - Visual breakdown of column types

Visualizations
Histogram - Bins: 5-50, metrics: mean, median, std dev, skewness

Bar Chart - Sorting: ascending, descending, alphabetical

Scatter Plot - With correlation coefficient and custom coloring

Pair Plot - 2-6 variable comparison with optional grouping

Heatmap - Pearson, Spearman, Kendall correlations

Line Chart - Data sorting, grouping, trend indicators

Pie Chart - Percentage display, "Others" category handling

Radar Plot - Multi-dimensional analysis with multiple modes

Database Features
Persistent storage - File-based DuckDB database

Automatic connection - Establishes on app startup

Table management - Auto-creates tables if missing

Status monitoring - Real-time database health in sidebar

Error recovery - Graceful handling of connection issues

🔐 Privacy & Security
✅ Local Processing - All data processed on your machine

✅ No Cloud Upload - Files never sent to external servers

✅ No Tracking - No user data collection

✅ Private Database - Local DuckDB storage

✅ File Size Limit - 200MB per file for safety

⚙️ Configuration
System Requirements
RAM - 2GB minimum (4GB recommended)

Disk - 500MB free space for database

Browser - Modern browser (Chrome, Firefox, Safari, Edge)

Optional Settings
Edit app.py to customize:

Database file location

File size limits

Encoding preferences

Visualization settings

🐛 Troubleshooting
Issue: Port 8501 already in use
bash
streamlit run app.py --server.port 8502
Issue: Database connection failed
bash
# Delete and recreate database
rm eda_dashboard.db
streamlit run app.py
Issue: Encoding error on CSV import
The app automatically tries UTF-8, Latin1, and CP1252. If still failing:

Open CSV in text editor

Save with UTF-8 encoding

Re-upload

Issue: Large file takes too long
Files over 100MB may be slow

Try reducing file size first

Process in chunks if possible

🤝 Contributing
Contributions are welcome! Here's how to contribute:

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

Areas for Contribution:
New visualization types

Enhanced statistics

Data cleaning tools

Performance optimizations

Documentation improvements

Bug fixes

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🔮 Roadmap
Upcoming Features
 Data cleaning and transformation tools

 Advanced statistical analysis

 Machine learning model integration

 Custom report generation (PDF)

 Multi-file comparison

 Data quality scoring

 Anomaly detection

 Time series analysis

In Development
Data Manipulation page (framework ready)

Enhanced statistical tests

Export to various formats

📚 Documentation
Detailed Project Description - Comprehensive overview

Quick Start Guide - Installation and usage

Technical Stack - Technologies used

API Reference - Function documentation (coming soon)

🆘 Support
Getting Help
Check Troubleshooting section

Review PROJECT_DESCRIPTION.md

Open an Issue

Check Discussions

Report a Bug
Go to Issues

Click "New Issue"

Describe the problem with screenshots

Include your Python version and OS

Feature Request
Go to Discussions

Start a new discussion

Describe the feature and use case

📊 Project Stats
Total Features - 20+

Visualization Types - 8

File Formats - 3

Lines of Code - 1000+

Dependencies - 12

Python Version - 3.8+

🎓 Learning Resources
Getting Started with EDA
Pandas Documentation

Streamlit Docs

Plotly Guide

DuckDB Tutorial

Data Analysis Concepts
Exploratory Data Analysis Guide

Statistical Analysis Basics

Data Visualization Best Practices

👥 Authors
Project Developer - [Rudra Narayan Shaw]

Contributors - [List of contributors]

🙏 Acknowledgments
Streamlit for the amazing web framework

DuckDB for high-performance analytics

Plotly for beautiful interactive visualizations

Pandas for powerful data manipulation

