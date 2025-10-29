Global VaR Backtesting Platform - Setup Guide

Follow these steps to get your full-stack VaR backtesting platform running locally.

Step 1: Set up PostgreSQL Database

Install PostgreSQL: If you don't have it, download and install PostgreSQL from postgresql.org.

Create a Database:

Open the psql command-line tool or use a GUI like pgAdmin.

Create a new user and database. For example:

CREATE USER var_admin WITH PASSWORD 'your_secure_password';
CREATE DATABASE vardb OWNER var_admin;



Note: Remember the host, port, database, user, and password you set.

Step 2: Set up Python Environment

Create a Virtual Environment:

python -m venv venv



Activate the Environment:

macOS/Linux: source venv/bin/activate

Windows: .\venv\Scripts\activate

Install Dependencies:

Make sure all the files (var_backtesting_app.py, schema.sql, requirements.txt, README.md) are in the same directory.

Run:

pip install -r requirements.txt



Step 3: Create Streamlit Secrets File

Streamlit uses a secrets file to securely store database credentials.

Create a new folder in your project directory named .streamlit.

Inside that folder, create a new file named secrets.toml.

Add your PostgreSQL connection details to this file. It must match this format:

# .streamlit/secrets.toml

[postgres]
host = "localhost"
port = 5432
dbname = "vardb"
user = "var_admin"
password = "your_secure_password"



Replace the values with your own from Step 1.

Step 4: Run the Application

Make sure your virtual environment is active.

In your terminal, run the Streamlit app:

streamlit run var_backtesting_app.py



Your browser should automatically open to the application (usually at http://localhost:8501).

Step 5: First-Time Use (Inside the App)

When you first launch the app, the database will be empty.

On the sidebar, click the "1. Setup Database Schema" button. This will run the schema.sql script to create the var_data table.

After the schema is created, click the "2. Populate with Simulated Data" button. This will generate and insert ~7,500 rows of demo data (15 indices * ~500 days).

The app will auto-refresh, and you can now select an index and start backtesting!

Next Steps: Connecting Your Data

The simulate_var_data() and populate_database() functions are just for demonstration.

To use your real data, you will need to:

Build Your Data Pipeline: Create a separate Python script (e.g., data_pipeline.py) that fetches your actual P&L and VaR data from its source.

Write to the Database: Have that script connect to the same PostgreSQL database and INSERT or UPDATE the var_data table. You can run this script on a schedule (e.g., daily using cron or Airflow).

Remove Simulation: You can then remove the "Populate" buttons and simulation functions from the Streamlit app, as it will just read directly from the auto-refreshing database.