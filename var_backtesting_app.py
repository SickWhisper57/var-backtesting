import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from scipy.stats import chi2, norm
import altair as alt
from datetime import date, timedelta
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Global VaR Backtesting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Database Connection ---

# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_db_connection():
    """Initializes and returns a connection to the PostgreSQL database."""
    try:
        return psycopg2.connect(**st.secrets["postgres"])
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.stop()

# Function to get the connection
def get_db_connection():
    """Gets the database connection from cache."""
    return init_db_connection()

# Function to run the schema
def setup_database(conn):
    """Creates the var_data table if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            with open("schema.sql", "r") as f:
                cur.execute(f.read())
        conn.commit()
    except Exception as e:
        st.error(f"Error setting up database: {e}")
        conn.rollback()

# --- Data Simulation (To replace with your real pipeline) ---
def simulate_var_data(index_name, num_days=504): # approx 2 years of trading days
    """
    Generates realistic-looking P&L and VaR data for demonstration.
    Replace this with your actual data pipeline.
    """
    dates = [date.today() - timedelta(days=i) for i in range(num_days)][::-1]
    
    # Simulate P&L (e.g., mean 0.05%, std 1.5%)
    returns = np.random.normal(0.0005, 0.015, num_days)
    pnl = 1_000_000 * returns # P&L on a $1M portfolio
    
    # Simulate VaR (e.g., as a rolling std deviation)
    pnl_series = pd.Series(pnl)
    rolling_std = pnl_series.rolling(window=60).std().bfill()
    z_score = norm.ppf(0.01) # For 99% VaR
    var_99 = -rolling_std * abs(z_score) * 1_000_000
    
    # Make VaR a bit noisy so it's not perfect
    var_99 = var_99 * np.random.normal(1.0, 0.1, num_days)
    
    return pd.DataFrame({
        'index_name': index_name,
        'date': dates,
        'pnl': pnl,
        'var_99': var_99
    })

def populate_database(conn):
    """Populates the database with simulated data if it's empty."""
    indices = [
        'S&P 500', 'FTSE 100', 'Nikkei 225', 'DAX', 'Hang Seng',
        'NASDAQ 100', 'Euro Stoxx 50', 'CAC 40', 'SMI', 'AEX',
        'IBEX 35', 'ASX 200', 'KOSPI', 'BSE Sensex', 'MSCI World'
    ]
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM var_data")
            count = cur.fetchone()[0]
            
            if count > 0:
                st.toast("Database already contains data.")
                return

            st.toast("Database is empty. Populating with simulated data...")
            all_data = []
            for index in indices:
                df = simulate_var_data(index)
                for record in df.to_dict('records'):
                    all_data.append((
                        record['index_name'],
                        record['date'],
                        record['pnl'],
                        record['var_99']
                    ))
            
            # Use execute_values for efficient batch insertion
            execute_values(
                cur,
                "INSERT INTO var_data (index_name, date, pnl, var_99) VALUES %s",
                all_data
            )
            conn.commit()
            st.success(f"Successfully populated database with {len(all_data)} records for 15 indices.")
            
    except Exception as e:
        st.error(f"Error populating database: {e}")
        conn.rollback()

# --- Data Fetching ---

@st.cache_data(ttl=300) # Cache data for 5 minutes
def fetch_data_from_db(_conn, index_name, confidence_level):
    """Fetches P&L and VaR data from the database for a given index."""
    # Note: We fetch all data and calculate VaR exceptions in Python
    # to handle different confidence levels.
    # In a real system, you might store VaR for *multiple* levels.
    # For this demo, we'll scale the stored 'var_99' to the selected level.
    
    z_99 = norm.ppf(0.01)
    z_selected = norm.ppf(1 - confidence_level)
    scale_factor = z_selected / z_99
    
    query = f"""
    SELECT 
        date, 
        pnl, 
        var_99 * %(scale_factor)s as var_model
    FROM var_data
    WHERE index_name = %(index_name)s
    ORDER BY date;
    """
    try:
        df = pd.read_sql(query, _conn, params={"index_name": index_name, "scale_factor": scale_factor})
        if df.empty:
            return pd.DataFrame()
            
        df['exception'] = df['pnl'] < df['var_model']
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_available_indices(_conn):
    """Fetches the list of distinct index names from the database."""
    try:
        df = pd.read_sql("SELECT DISTINCT index_name FROM var_data ORDER BY index_name", _conn)
        return df['index_name'].tolist()
    except Exception:
        return [] # Return empty list if table doesn't exist yet

# --- Backtesting Logic ---

def kupiec_pof_test(df, confidence_level):
    """
    Performs the Kupiec POF (Proportion of Failures) test.
    This tests if the *number* of exceptions is consistent with the confidence level.
    """
    alpha = 1 - confidence_level
    n = len(df)
    x = df['exception'].sum()
    
    if n == 0:
        return 0, 1, 0, 0 # Handle empty data
        
    observed_freq = x / n
    
    # Handle edge cases where log(0) would occur
    if x == 0 or x == n:
        if x == 0 and alpha == 0: return 0, 1, x, n # Perfect model, no exceptions expected
        if x == n and alpha == 1: return 0, 1, x, n # Perfect model, all exceptions expected
        # If x=0 but alpha > 0, or x=n but alpha < 1, this is max LR
        if x == 0:
            log_likelihood_unrestricted = (n - x) * np.log(1 - observed_freq)
        else:
            log_likelihood_unrestricted = x * np.log(observed_freq)
    else:
         log_likelihood_unrestricted = (n - x) * np.log(1 - observed_freq) + x * np.log(observed_freq)

    log_likelihood_restricted = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    
    lr_pof = -2 * (log_likelihood_restricted - log_likelihood_unrestricted)
    p_value = 1 - chi2.cdf(lr_pof, df=1)
    
    return lr_pof, p_value, x, n

def christoffersen_tuff_test(df, confidence_level):
    """
    Performs the Christoffersen TUFF (Time Until First Failure) test,
    which includes tests for independence and conditional coverage.
    """
    # 1. Kupiec POF test (required for conditional coverage)
    lr_pof, p_value_pof, x, n = kupiec_pof_test(df, confidence_level)

    if n == 0 or x == 0:
        # Not enough data or no exceptions to test for clustering
        return lr_pof, p_value_pof, 0, 1, lr_pof, p_value_pof

    # 2. Independence Test
    exceptions = df['exception'].astype(int).tolist()
    
    # Create transition matrix
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, n):
        if exceptions[i-1] == 0 and exceptions[i] == 0: n00 += 1
        elif exceptions[i-1] == 0 and exceptions[i] == 1: n01 += 1
        elif exceptions[i-1] == 1 and exceptions[i] == 0: n10 += 1
        elif exceptions[i-1] == 1 and exceptions[i] == 1: n11 += 1

    # Handle cases with no transitions (e.g., only one exception)
    if (n01 + n11 == 0) or (n00 + n10 == 0):
        # Not enough data to test independence
        return lr_pof, p_value_pof, 0, 1, lr_pof, p_value_pof

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_overall = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Avoid log(0) issues
    log_l_unrestricted = (
        (n00 * np.log(1 - pi0) if pi0 < 1 else 0) +
        (n01 * np.log(pi0) if pi0 > 0 else 0) +
        (n10 * np.log(1 - pi1) if pi1 < 1 else 0) +
        (n11 * np.log(pi1) if pi1 > 0 else 0)
    )
    
    log_l_restricted = (
        ((n00 + n10) * np.log(1 - pi_overall) if pi_overall < 1 else 0) +
        ((n01 + n11) * np.log(pi_overall) if pi_overall > 0 else 0)
    )

    lr_ind = -2 * (log_l_restricted - log_l_unrestricted)
    p_value_ind = 1 - chi2.cdf(lr_ind, df=1)

    # 3. Conditional Coverage Test
    lr_cc = lr_pof + lr_ind
    p_value_cc = 1 - chi2.cdf(lr_cc, df=2) # 2 degrees of freedom

    return lr_pof, p_value_pof, lr_ind, p_value_ind, lr_cc, p_value_cc

def format_result(p_value, critical_level=0.05):
    """Formats a p-value result with emojis."""
    if p_value < critical_level:
        return f"**REJECTED** (p-value: {p_value:.4f}) ❌"
    else:
        return f"**ACCEPTED** (p-value: {p_value:.4f}) ✅"

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()

# --- Main Application UI ---

def run_app():
    st.title("🌎 Global VaR Backtesting Platform")
    
    conn = get_db_connection()
    
    # --- Sidebar Controls ---
    st.sidebar.title("Controls")
    
    # Check if DB setup is needed
    try:
        available_indices = get_available_indices(conn)
    except:
        available_indices = []

    if not available_indices:
        st.sidebar.warning("Database seems empty or uninitialized.")
        if st.sidebar.button("1. Setup Database Schema"):
            setup_database(conn)
            st.sidebar.success("Database schema created!")
            st.rerun()
            
        if st.sidebar.button("2. Populate with Simulated Data"):
            populate_database(conn)
            st.rerun()
        
        st.info("Please setup and populate the database using the sidebar controls to begin.")
        st.stop()
        
    else:
        if st.sidebar.button("Re-populate Simulated Data"):
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE var_data")
            conn.commit()
            populate_database(conn)
            st.cache_data.clear() # Clear data caches
            st.rerun()

    # --- Main App ---
    
    selected_index = st.sidebar.selectbox(
        "Select Index/Model",
        available_indices
    )
    
    confidence_level = st.sidebar.slider(
        "VaR Confidence Level",
        min_value=0.950,
        max_value=0.995,
        value=0.990,
        step=0.005,
        format="%.3f"
    )
    alpha = 1 - confidence_level
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.info(
        """
        This app demonstrates a full-stack VaR backtesting system
        using simulated P&L and VaR data stored in a PostgreSQL database.
        """
    )

    # --- Fetch Data ---
    data = fetch_data_from_db(conn, selected_index, confidence_level)

    if data.empty:
        st.error(f"No data found for {selected_index}. Try populating the database.")
        st.stop()

    # --- Run Tests ---
    (
        lr_pof, p_pof, lr_ind, p_ind, lr_cc, p_cc
    ) = christoffersen_tuff_test(data, confidence_level)
    
    n = len(data)
    x = data['exception'].sum()
    expected_exceptions = n * alpha

    # --- Dashboard Display ---
    
    st.header(f"Backtesting Results: {selected_index}")
    st.markdown(f"**Model:** 1-Day VaR at **{confidence_level*100:.1f}%** confidence level. "
                f"**History:** {n} trading days.")

    # --- Key Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations (N)", n)
    col2.metric("Expected Exceptions (N * α)", f"{expected_exceptions:.2f}")
    
    # Conditional formatting for Actual Exceptions
    delta_exceptions = x - expected_exceptions
    if p_pof < 0.05: # If Kupiec test fails, show it as bad
        col3.metric("Actual Exceptions (X)", x, f"{delta_exceptions:+.2f} (Poor Fit)", delta_color="inverse")
    else:
        col3.metric("Actual Exceptions (X)", x, f"{delta_exceptions:+.2f} (Good Fit)", delta_color="normal")
        
    # --- P&L vs VaR Chart ---
    st.subheader("P&L vs. Value-at-Risk")
    
    # Create a 'color' column for the chart
    chart_data = data.copy()
    chart_data['color'] = np.where(chart_data['exception'], 'Exception', 'No Exception')
    
    # Base chart
    base = alt.Chart(chart_data).encode(
        x=alt.X('date', title='Date')
    )
    
    # VaR Line
    var_line = base.mark_line(color='red', strokeDash=[5,5]).encode(
        y=alt.Y('var_model', title='P&L / VaR'),
        tooltip=[
            alt.Tooltip('date', format='%Y-%m-%d'),
            alt.Tooltip('var_model', format=',.2f', title='VaR')
        ]
    ).properties(
        title='VaR Line'
    )

    # P&L Bars (colored by exception)
    pnl_bars = base.mark_bar().encode(
        y=alt.Y('pnl', title='P&L / VaR'),
        color=alt.Color('color', 
                        scale={'domain': ['Exception', 'No Exception'],
                               'range': ['#e45756', '#4c78a8']}),
        tooltip=[
            alt.Tooltip('date', format='%Y-%m-%d'),
            alt.Tooltip('pnl', format=',.2f'),
            alt.Tooltip('var_model', format=',.2f', title='VaR'),
            'exception'
        ]
    ).properties(
        title='Daily P&L'
    )
    
    # Combine and make interactive
    chart = (pnl_bars + var_line).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # --- Test Results Table ---
    st.subheader("Regulatory Test Details")
    
    results_df = pd.DataFrame({
        "Test": [
            "Kupiec (POF) Test",
            "Christoffersen (Independence) Test",
            "Christoffersen (Conditional Coverage) Test"
        ],
        "Description": [
            "Tests if the *frequency* of exceptions is correct.",
            "Tests if exceptions are *clustered* together.",
            "Combines both tests (frequency and independence)."
        ],
        "Likelihood Ratio (LR)": [lr_pof, lr_ind, lr_cc],
        "p-value": [p_pof, p_ind, p_cc],
        "Result (at 5% level)": [
            format_result(p_pof),
            format_result(p_ind),
            format_result(p_cc)
        ]
    })
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # --- Data Export ---
    st.subheader("Export Data")
    
    # Create a summary of the results
    summary = {
        "Index": selected_index,
        "Confidence Level": confidence_level,
        "Days": n,
        "Expected Exceptions": expected_exceptions,
        "Actual Exceptions": x,
        "Kupiec LR": lr_pof,
        "Kupiec p-value": p_pof,
        "Independence LR": lr_ind,
        "Independence p-value": p_ind,
        "Conditional Coverage LR": lr_cc,
        "Conditional Coverage p-value": p_cc,
        "Final Model Status": "REJECTED" if p_cc < 0.05 or p_pof < 0.05 else "ACCEPTED"
    }
    
    # Add exception details to the main data
    export_df = data.copy()
    export_df['confidence_level'] = confidence_level
    export_df['alpha'] = alpha
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        # Download button for the summary
        st.download_button(
            label="Export Test Summary (CSV)",
            data=convert_df_to_csv(pd.DataFrame([summary])),
            file_name=f"{selected_index}_summary_{confidence_level*100:.0f}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_export2:
        # Download button for the full data
        st.download_button(
            label="Export Full Raw Data (CSV)",
            data=convert_df_to_csv(export_df),
            file_name=f"{selected_index}_raw_data.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- Run the App ---
if __name__ == "__main__":
    run_app()
