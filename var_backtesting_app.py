import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from scipy.stats import chi2
from datetime import datetime, timedelta

# --- Configuration ---

# Load database credentials securely from Streamlit secrets
@st.cache_resource
def get_db_secrets():
    # Use st.secrets if running on Streamlit Cloud, otherwise expect a local file.
    if 'postgres' not in st.secrets:
        st.error("Database secrets not found. Please ensure your .streamlit/secrets.toml is correctly configured.")
        return None
    return st.secrets['postgres']

# --- Database Connection and Utility Functions ---

def get_db_connection():
    """Establishes and returns a new database connection."""
    secrets = get_db_secrets()
    if not secrets:
        return None
    
    # Connection args needed for cloud hosting like Neon (sslmode='require')
    connect_args = {}
    if 'sslmode' not in secrets and 'host' in secrets and 'neon.tech' in secrets['host']:
        connect_args['sslmode'] = 'require'
    
    try:
        # We establish a new connection every time for quick, atomic operations
        conn = psycopg2.connect(
            host=secrets['host'],
            port=secrets['port'],
            dbname=secrets['dbname'],
            user=secrets['user'],
            password=secrets['password'],
            connect_timeout=10,
            **connect_args
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        st.stop()
        return None

def setup_database():
    """Runs the SQL schema to create the table, using a new connection."""
    conn = get_db_connection()
    if not conn:
        return

    st.info("Creating database schema...")
    try:
        with conn.cursor() as cur:
            # SQL to create the table
            sql_schema = """
            CREATE TABLE IF NOT EXISTS var_data (
                date DATE NOT NULL,
                index_name VARCHAR(50) NOT NULL,
                pnl FLOAT NOT NULL,
                var_99 FLOAT NOT NULL,
                var_95 FLOAT NOT NULL,
                PRIMARY KEY (date, index_name)
            );
            """
            cur.execute(sql_schema)
        conn.commit()
        st.success("Database schema created successfully (Table: var_data).")
        # Trigger a rerun to refresh the app state
        st.rerun() 
    except Exception as e:
        conn.rollback()
        st.error(f"Error setting up database: {e}")
    finally:
        if conn:
            conn.close()

# --- Simulation and Population Functions ---

def simulate_var_data(index_name):
    """Generates a DataFrame of simulated PnL and VaR data."""
    np.random.seed(index_name.__hash__() % 100) # Deterministic simulation per index
    
    # 2 years (approx 504 trading days)
    days = 504 
    dates = [datetime.today().date() - timedelta(days=i) for i in range(days)]
    dates.reverse() # Sort in chronological order

    # Simulate PnL: Normally distributed returns with varying mean/volatility per index
    mean = np.random.uniform(-0.0005, 0.0015)
    volatility = np.random.uniform(0.015, 0.035)
    returns = np.random.normal(mean, volatility, days)
    pnl = 100000 * returns # Assuming a position size of 100,000

    # Calculate VaR (Simplified Historical VaR method for simulation)
    # 99% VaR is the 1st percentile loss (VaR is expressed as a positive loss value)
    var_99 = np.percentile(pnl, 1) * -1
    var_95 = np.percentile(pnl, 5) * -1
    
    # Introduce random breaches (losses greater than VaR)
    # Add random spikes to simulate market stress/breaches
    for i in range(int(days * np.random.uniform(0.01, 0.03))): # 1-3% random breaches
        if np.random.rand() < 0.2: # 20% chance of a 99% VaR breach
            pnl[np.random.randint(0, days)] = np.random.uniform(-1.5 * var_99, -1.1 * var_99)
        else: # 80% chance of a 95% VaR breach
            pnl[np.random.randint(0, days)] = np.random.uniform(-1.5 * var_95, -1.1 * var_95)

    df = pd.DataFrame({
        'date': dates,
        'index_name': index_name,
        'pnl': pnl,
        'var_99': var_99,
        'var_95': var_95
    })
    return df

def populate_database():
    """Generates simulated data for all indices and inserts it, using a new connection."""
    conn = get_db_connection()
    if not conn:
        return
    
    indices = [
        "S&P 500", "NASDAQ 100", "FTSE 100", "DAX", "Nikkei 225", "Hang Seng", 
        "CSI 300", "Nifty 50", "TSX", "Euro Stoxx 50", "CAC 40", "IBOVESPA",
        "ASX 200", "Sensex", "OMX Stockholm 30"
    ]
    
    total_records = 0
    st.info("Generating and inserting simulated data for 15 indices...")

    try:
        with conn.cursor() as cur:
            # Clear existing data before inserting new simulated data
            cur.execute("TRUNCATE TABLE var_data;")
            
            for index_name in indices:
                df = simulate_var_data(index_name)
                total_records += len(df)
                
                # Convert DataFrame to a list of tuples for fast insertion
                tuples = [tuple(x) for x in df.to_numpy()]
                
                # Use psycopg2.extras.execute_values for performance
                query = """
                INSERT INTO var_data (date, index_name, pnl, var_99, var_95) 
                VALUES %s ON CONFLICT (date, index_name) DO NOTHING;
                """
                psycopg2.extras.execute_values(cur, query, tuples)
        
        conn.commit()
        st.success(f"Database populated with {total_records} simulated records successfully.")
        # Trigger a rerun to display the newly populated data
        st.rerun() 
    except Exception as e:
        conn.rollback()
        st.error(f"Error populating database: {e}")
    finally:
        if conn:
            conn.close()

def load_data(index_name):
    """Loads all data for a specific index from the database."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    try:
        query = f"""
        SELECT date, pnl, var_99, var_95 
        FROM var_data 
        WHERE index_name = '{index_name}' 
        ORDER BY date;
        """
        # Read the data directly into a DataFrame
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date']).dt.date # Keep date format clean
        return df
    except Exception as e:
        # If the table doesn't exist yet, an error occurs (expected on first run)
        st.error("Database seems empty or uninitialized. Please run the setup steps in the sidebar.")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_available_indices():
    """Fetches unique index names from the database."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        query = "SELECT DISTINCT index_name FROM var_data ORDER BY index_name;"
        with conn.cursor() as cur:
            cur.execute(query)
            indices = [row[0] for row in cur.fetchall()]
        return indices
    except Exception:
        # Expected if the table doesn't exist yet
        return []
    finally:
        if conn:
            conn.close()

# --- VaR Backtesting Logic ---

def kupiec_pof_test(N, T, p):
    """
    Kupiec's Proportion of Failures (POF) Test (Unconditional Coverage).
    Tests if the observed failure rate (x/N) equals the expected failure rate (p).
    
    N: number of breaches (failures)
    T: total number of observations
    p: expected failure rate (e.g., 0.01 for 99% VaR)
    """
    if N == 0:
        # No breaches, so L_N and L_p are 0. LL ratio is 0. 
        # Chi-square stat is effectively 0, p-value is 1.0.
        return 0.0, 1.0

    # Expected number of breaches
    N_exp = T * p
    
    # Observed failure rate
    p_obs = N / T
    
    # Log-likelihood ratio statistic
    if N == T:
        # Avoid log(0) if N=T (all observations are breaches)
        LR_POF = -2 * ( (T - N_exp) * np.log(1 - p) + N_exp * np.log(p) )
    else:
        LR_POF = -2 * ( 
            (T - N) * np.log(1 - p_obs) + N * np.log(p_obs) -
            (T - N_exp) * np.log(1 - p) - N_exp * np.log(p)
        )

    # Chi-square with 1 degree of freedom
    p_value = 1.0 - chi2.cdf(LR_POF, 1)
    
    return LR_POF, p_value

def christoffersen_tuff_test(N_00, N_01, N_10, N_11, N, T, p):
    """
    Christoffersen's Test for Independence (Time Until First Failure, TUFF).
    Tests if breaches are clustered (dependent) or independent.
    
    N_ij: Number of days where state i (t-1) transitions to state j (t).
          State 0: No breach. State 1: Breach.
    N: total number of breaches (N_01 + N_11)
    T: total number of observations (N_00 + N_01 + N_10 + N_11)
    p: expected failure rate
    """
    if N <= 1:
        # Cannot perform independence test with 0 or 1 breach
        return 0.0, 1.0

    # Unconditional probability of a breach (observed failure rate)
    pi_unc = N / T

    # Transition probabilities: P(0|0) and P(1|0)
    pi_01 = N_01 / (N_00 + N_01) if (N_00 + N_01) > 0 else 0
    pi_11 = N_11 / (N_10 + N_11) if (N_10 + N_11) > 0 else 0
    
    # Log-likelihood terms for the Independent model (L_I)
    # L_I is essentially the Log-Likelihood of the POF test
    L_I = (T - N) * np.log(1 - pi_unc) + N * np.log(pi_unc) if pi_unc > 0 and pi_unc < 1 else 0

    # Log-likelihood terms for the Dependent model (L_D)
    # L_D uses the transition probabilities
    L_D = (
        N_00 * np.log(1 - pi_01) + N_01 * np.log(pi_01) if pi_01 > 0 and pi_01 < 1 else 0 +
        N_10 * np.log(1 - pi_11) + N_11 * np.log(pi_11) if pi_11 > 0 and pi_11 < 1 else 0
    )

    # Log-likelihood Ratio statistic
    LR_TUFF = -2 * (L_I - L_D)
    
    # Chi-square with 1 degree of freedom
    p_value = 1.0 - chi2.cdf(LR_TUFF, 1)
    
    return LR_TUFF, p_value

def run_backtests(df, confidence_level):
    """Performs both Kupiec and Christoffersen backtests."""
    
    if df.empty:
        return None, None, 0, 0, 0

    # 1. Identify Breaches (Failures)
    var_col = f'var_{confidence_level}'
    
    # Breach occurs if PnL (loss) is greater than VaR (positive value)
    # PnL is negative for a loss, VaR is positive for the expected loss magnitude
    # We must convert PnL to a positive loss value for the comparison
    losses = -df['pnl']
    breaches = (losses > df[var_col]).astype(int)
    
    T = len(df)
    N = breaches.sum() # Total breaches (N_1)
    p = 1 - (confidence_level / 100) # Expected failure rate (e.g., 0.01 for 99%)
    
    # --- Kupiec POF Test (Unconditional Coverage) ---
    lr_pof, p_pof = kupiec_pof_test(N, T, p)
    
    # --- Christoffersen TUFF Test (Independence) ---
    
    # State transition counting: 
    # Current day breach (t=1) or no breach (t=0)
    # Previous day breach (t-1=1) or no breach (t-1=0)
    
    breaches_prev = breaches.shift(1).fillna(0).astype(int)
    
    # N_ij = (previous state i) AND (current state j)
    N_00 = ((breaches_prev == 0) & (breaches == 0)).sum() # No breach -> No breach
    N_01 = ((breaches_prev == 0) & (breaches == 1)).sum() # No breach -> Breach
    N_10 = ((breaches_prev == 1) & (breaches == 0)).sum() # Breach -> No breach
    N_11 = ((breaches_prev == 1) & (breaches == 1)).sum() # Breach -> Breach
    
    lr_tuff, p_tuff = christoffersen_tuff_test(N_00, N_01, N_10, N_11, N, T, p)

    # Compile Results
    results = {
        'Total Observations (T)': T,
        'Total Breaches (N)': N,
        'Expected Breaches (T*p)': round(T * p, 2),
        'Observed %': f"{round(N / T * 100, 2)}%",
        'Expected %': f"{round(p * 100, 2)}%"
    }
    
    pof_result = {
        'Test': 'Kupiec POF (Unconditional Coverage)',
        'LR Statistic': round(lr_pof, 4),
        'P-Value': round(p_pof, 4),
        'Result': 'PASS (Accept Null Hypothesis: Expected breaches are met)' if p_pof >= 0.05 else 'FAIL (Reject Null Hypothesis: Too many/few breaches)'
    }

    tuff_result = {
        'Test': 'Christoffersen TUFF (Independence)',
        'LR Statistic': round(lr_tuff, 4),
        'P-Value': round(p_tuff, 4),
        'Result': 'PASS (Accept Null Hypothesis: Breaches are independent)' if p_tuff >= 0.05 else 'FAIL (Reject Null Hypothesis: Breaches are clustered)'
    }
    
    return results, pd.DataFrame([pof_result, tuff_result]), breaches, losses, df[var_col]

# --- Streamlit UI ---

def display_results(results_summary, test_df, breaches, losses, var_series, df_raw):
    """Displays backtesting results and plots."""
    
    st.markdown("### Backtesting Results")
    
    # 1. Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Observations (T)", results_summary['Total Observations (T)'])
    col2.metric("Total Breaches (N)", results_summary['Total Breaches (N)'])
    col3.metric("Expected Breaches (T*p)", results_summary['Expected Breaches (T*p)'])
    col4.metric("Observed Breach Rate", results_summary['Observed %'])
    col5.metric("Expected Breach Rate", results_summary['Expected %'])

    st.divider()

    # 2. Test Results Table
    st.markdown("#### Regulatory Tests ($\alpha=0.05$)")
    st.dataframe(test_df, use_container_width=True, hide_index=True)

    # Conditional Warning/Pass Message
    if 'FAIL' in test_df['Result'].values:
        st.error("⚠️ REGULATORY FAILURE: At least one test failed the 95% confidence level. Review the VaR model.")
    else:
        st.success("✅ REGULATORY PASS: Both Kupiec POF and Christoffersen TUFF tests passed.")

    st.divider()

    # 3. Plotting VaR and Losses
    st.markdown("#### VaR vs. Daily Loss Plot")
    
    plot_df = pd.DataFrame({
        'Date': df_raw['date'],
        'Daily Loss': losses, # Use positive value for plotting loss
        'VaR Limit': var_series 
    }).set_index('Date')
    
    # Add a column to identify breaches for visualization
    plot_df['Breach'] = np.where(breaches == 1, losses, np.nan)
    
    st.line_chart(plot_df[['Daily Loss', 'VaR Limit']], use_container_width=True)
    
    # Scatter plot for breaches
    st.scatter_chart(plot_df[['Breach']], color='#ff4b4b', use_container_width=True)
    st.caption("Red dots indicate days where the daily loss breached the VaR limit.")

    # 4. CSV Export
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_raw)

    st.download_button(
        label="Download Full Backtesting Data (CSV)",
        data=csv,
        file_name='VaR_Backtest_Data.csv',
        mime='text/csv',
    )


def run_app():
    """Main Streamlit application function."""
    st.set_page_config(layout="wide", page_title="Global VaR Backtesting Platform")

    # --- Sidebar for Setup and Selection ---
    
    with st.sidebar:
        st.title("VaR Backtesting Controls")
        
        # Database Setup Section
        st.markdown("---")
        st.header("Database Utilities")
        
        # Button 1: Setup Schema
        if st.button("1. Setup Database Schema"):
            setup_database()

        # Button 2: Populate Data
        if st.button("2. Populate with Simulated Data"):
            populate_database()

        st.markdown("---")
        st.header("Backtesting Parameters")

        available_indices = get_available_indices()
        
        if not available_indices:
            st.warning("No data found. Please complete the Database Utilities steps above.")
            st.stop() # Stop execution if no data is available
            
        # Select Index
        selected_index = st.selectbox(
            "Select Global Index:",
            options=available_indices,
            index=0
        )
        
        # Select Confidence Level
        confidence_level = st.radio(
            "VaR Confidence Level:",
            options=[99, 95],
            index=0,
            horizontal=True
        )

        st.caption(f"Testing {confidence_level}% VaR: Expected Failure Rate (p) is {100-confidence_level}%.")

    # --- Main Dashboard ---
    
    st.title(f"VaR Backtesting Dashboard: {selected_index} ({confidence_level}% VaR)")
    st.markdown("""
        **Objective:** Validate the 2-year Value-at-Risk (VaR) model by comparing the number and clustering of actual losses (breaches) against expected values.
    """)

    # 1. Load Data
    data_load_state = st.text("Loading data...")
    df_raw = load_data(selected_index)
    data_load_state.text(f"Data loaded: {len(df_raw)} records.")
    
    if df_raw.empty:
        # If data_load_state shows 0 records, this message is visible.
        st.error("No data available for selected index. Please populate the database.")
        return

    # 2. Run Backtests
    results_summary, test_results_df, breaches, losses, var_series = run_backtests(df_raw, confidence_level)

    # 3. Display Results
    display_results(results_summary, test_results_df, breaches, losses, var_series, df_raw)

if __name__ == "__main__":
    # Ensure logs are visible for debugging
    # st.set_log_level("DEBUG") 
    run_app()
