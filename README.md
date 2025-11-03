This Streamlit-based application is a comprehensive Value at Risk (VaR) backtesting platform that combines the analytical power of Python with the reliability of a PostgreSQL database. It evaluates how accurately VaR models predict potential portfolio losses — a critical component in financial risk management.

The system performs regulatory backtesting using two industry-standard statistical tests: Kupiec’s Proportion of Failures (POF) test and Christoffersen’s Conditional Coverage test. These tests help determine whether a VaR model correctly estimates risk levels and whether observed exceptions occur randomly or in clusters during volatile market periods.

The application works with both stored financial data and synthetic simulations, allowing users to explore model performance under real or hypothetical market conditions. All results — including daily Profit & Loss (PnL) figures, VaR estimates, and breach events — are stored in a PostgreSQL database for future analysis and reproducibility.

The interactive Streamlit dashboard provides a user-friendly interface to:

* Visualize PnL and VaR values across time.

* Identify and highlight days where actual losses exceeded the predicted VaR (exceptions).

* View detailed results from the Kupiec and Christoffersen backtests, including p-values and pass/fail indicators.

* Export datasets for deeper statistical analysis or reporting in external tools like Excel or Power BI.

Overall, this project integrates quantitative finance, data engineering, and data visualization to deliver a full-fledged VaR validation workflow. It mirrors the kind of backtesting and reporting frameworks used in investment banks, trading firms, and risk management teams, but in a simplified, transparent, and interactive format.
