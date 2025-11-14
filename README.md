📊 EDA Dashboard - Exploratory Data Analysis Tool
A comprehensive Exploratory Data Analysis (EDA) dashboard built with Streamlit and Python. This tool provides an intuitive interface for data exploration, visualization, SQL queries, and data manipulation.

✨ Features
📥 Ingestion Page
Upload CSV and Excel files

Preview data before processing

View data summary and statistics

Upload history tracking

📊 EDA Page (Exploratory Data Analysis)
8 Chart Types: Histogram, Box Plot, Scatter Plot, Line Chart, Bar Chart, Heatmap, Violin Plot, Area Chart

Interactive visualizations with Plotly

Correlation heatmap analysis

Download charts as PNG or PDF

Statistical summaries

🔍 SQL Operation Page
Execute custom SQL queries on your data

3 Tabs:

ℹ️ Table Info: View table structure and metadata

📝 SQL Query: Write and execute queries

📚 Templates: Pre-built query templates for quick access

Copy templates to query box with one click

Auto-tab switching after template selection

🛠️ Data Manipulation Page
4 Operation Types:

Filter Data: Remove rows based on conditions

Sort Data: Arrange by one or more columns

Aggregate Data: Group and calculate statistics

Select Columns: Choose specific columns

Real-time data preview

Operation history

🚀 Quick Start
Prerequisites
Python 3.8 or higher

Docker (optional, but recommended)

Installation
Option 1: Local Setup (Python)
bash
# Clone the repository
git clone https://github.com/yourusername/eda-dashboard.git
cd eda-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
The app will open in your browser at http://localhost:8501

Option 2: Docker (Recommended)
bash
# Build Docker image
docker build -t eda-dashboard .

# Run container
docker run -p 8501:8501 eda-dashboard

# Open browser
# http://localhost:8501
📁 Project Structure
text
eda-dashboard/
├── app.py                      # Main Streamlit application
├── eda.py                      # EDA page module
├── data_mani.py                # Data manipulation module
├── sql_operation.py            # SQL operations module
├── ingestion.py                # Data ingestion module
├── requirements.txt            # Python dependencies
└── README.md                   # This file
📋 Dependencies
The main dependencies are:

Package	Version	Purpose
streamlit	1.28.1	Web framework
pandas	2.0.3	Data manipulation
numpy	1.24.3	Numerical computing
plotly	5.17.0	Interactive charts
matplotlib	3.7.2	Static visualizations
seaborn	0.12.2	Statistical graphics
duckdb	0.8.1	SQL database
openpyxl	3.1.2	Excel handling
kaleido	0.2.1	PNG export
reportlab	4.0.4	PDF export
See requirements.txt for complete list.

🎯 Usage Guide
1. Ingestion
Click on "Ingestion" in the sidebar

Upload your CSV or Excel file

Preview the data

Proceed to analysis

2. Exploratory Data Analysis (EDA)
Select columns and chart type

Customize chart parameters

View interactive visualizations

Download charts in PNG or PDF format

3. SQL Operations
Write custom SQL queries

Use pre-built templates

Execute queries on uploaded data

View results instantly

4. Data Manipulation
Apply filters, sorting, or aggregations

Preview results in real-time

Download modified data


🔐 Configuration
Streamlit Config (.streamlit/config.toml)
Customize the application behavior:

Theme settings

Server configuration

Client settings

Environment Variables
Set custom ports and configurations:

bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_LOGGER_LEVEL=error
📊 Features in Detail
EDA Analytics
Data Profiling: Automatically calculate statistics

Correlation Analysis: Identify relationships between variables

Distribution Analysis: Visualize data distributions

Outlier Detection: Identify anomalies

Missing Value Analysis: Handle missing data

SQL Capabilities
Full SQL query support (via DuckDB)

Built-in templates for common queries

Query history

Results export

Data Operations
Filtering: Multiple conditions supported

Sorting: Single and multi-column sorting

Aggregation: GROUP BY operations

Column Selection: Choose specific columns

🐛 Troubleshooting
Port Already in Use
bash
# Change port in command
streamlit run app.py --server.port 8502
Docker Build Issues
bash
# Clear cache and rebuild
docker build --no-cache -t eda-dashboard .
Out of Memory
bash
# Limit Docker memory
docker run -m 4g -p 8501:8501 eda-dashboard
📚 Documentation
For detailed guides, see:

Streamlit Documentation

Pandas Documentation

Plotly Documentation

🤝 Contributing
Contributions are welcome!

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open a Pull Request

📝 Version History
v2.1 (Current)
✅ Fixed template copying bug in SQL operations

✅ Auto-switches to SQL Query tab after template selection

✅ Query automatically appears in query box

✅ Success messages display for 5 seconds

✅ Chart download errors fixed

✅ Enhanced error handling

✅ Streamlined data manipulation page

v2.0
Full dashboard implementation

All core features added

Docker support

v1.0
Initial release

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💼 Author
Rudra

GitHub: @yourusername

Email: your.email@example.com

🙋 Support
For issues, questions, or suggestions:

Check Existing Issues

Create a New Issue

Start a Discussion

🎯 Roadmap
 Advanced ML features (clustering, regression)

 Real-time data streaming support

 User authentication

 Multi-file operations

 API endpoint support

 Custom visualization plugins

 Performance optimization for large datasets

📊 Project Stats
Total Lines of Code: ~15,000+

Supported Chart Types: 8

Data Operations: 4

Python Modules: 5

Dependencies: 10

🚀 Performance
Startup Time: <10 seconds

Data Loading: <5 seconds (typical CSV)

Chart Rendering: <2 seconds

Query Execution: Varies by complexity

Max Dataset Size: Limited by system RAM

💡 Tips & Tricks
Fast Data Loading
Use CSV format (faster than Excel)

Pre-filter large datasets

Use chunked loading for big files

Optimization
Close unused tabs

Clear cache periodically

Use filters before aggregation

Best Practices
Always preview data after upload

Validate SQL queries in templates

Use meaningful column names

Document custom queries

🎓 Learning Resources
Streamlit: https://streamlit.io

Pandas: https://pandas.pydata.org

Plotly: https://plotly.com

DuckDB: https://duckdb.org

📞 Contact & Community
Email: rudranarayan.shawji@gmail.com

LinkedIn: (https://www.linkedin.com/in/rudra-07-nararyan/)


⭐ Show Your Support
If this project helped you, please consider:

⭐ Star the repository

🐛 Report bugs

💡 Suggest features

📢 Share with others

🤝 Contribute code

Made with ❤️ by Rudra

Last Updated: November 2025
