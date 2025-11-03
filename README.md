This Streamlit-based application is a comprehensive Value at Risk (VaR) backtesting platform that combines the analytical power of Python with the robustness of a PostgreSQL database. It is designed to evaluate the performance and reliability of VaR models — a cornerstone of modern risk management — by comparing the model’s predicted losses with actual portfolio outcomes.

At its core, the application performs regulatory backtesting using two widely recognized statistical tests:

Kupiec’s Proportion of Failures (POF) test, which assesses whether the frequency of VaR breaches aligns with the expected confidence level (e.g., 99% or 95%).

Christoffersen’s Conditional Coverage test, which further examines whether these breaches occur randomly over time or cluster during periods of market stress.

The tool can operate on stored financial data or generate simulated market data, allowing users to test the robustness of VaR models under controlled or hypothetical scenarios. Through its PostgreSQL integration, all simulated or real data points — including daily Profit & Loss (PnL) values and their corresponding VaR estimates — are stored efficiently for future retrieval, analysis, or comparison across different models and timeframes.

A key feature of the application is its interactive Streamlit dashboard, which provides a visual and intuitive interface for exploring backtesting results. Users can dynamically select indices, confidence levels, and test parameters, then view:

Time-series charts comparing daily PnL against VaR thresholds

Highlighted exceptions (days when losses exceeded VaR)

Detailed statistical outputs, including test statistics, p-values, and pass/fail flags

Exportable datasets for extended analysis in Excel, Python, or BI tools

In essence, this project bridges quantitative finance, data engineering, and data visualization — offering a practical, end-to-end solution for VaR model validation, educational demonstrations, or internal risk reporting. It replicates the kind of backtesting workflow used in banks, trading firms, and risk management teams, but in a streamlined and transparent way.
