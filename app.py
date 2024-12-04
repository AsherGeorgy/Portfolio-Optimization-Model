import streamlit as st
import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Initialize session state for inputs and run button
if "inputs_ready" not in st.session_state:
    st.session_state.inputs_ready = False
if "mode" not in st.session_state:
    st.session_state.mode = "custom"  # Default to custom mode

# Header section
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>Portfolio Optimization Model</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size: 18px;'>This is a Python-based application for analyzing and optimizing financial portfolios. "
    "It uses modern portfolio theory to identify optimal allocations and visualize performances.</p>",
    unsafe_allow_html=True,
)

# Separator line
st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)

# Instructional text
#st.markdown("<h2>Choose an option to start:</h2>", unsafe_allow_html=True)

# Custom input fields
st.markdown("### Configure Your Portfolio:")
user_input = st.text_input(
    "Enter stock tickers separated by commas: ",
    placeholder="aapl, msft, tsla",
    key="user_input",
).strip()

min_return = st.slider(
        "Desired return (%):",
        min_value=2.0,
        max_value=20.0,
        value=5.0,
        key="min_return",
        help="Annual return you'd like your portfolio to achieve while minimizing volatility.",
    )/100

# Sliders for optional settings
with st.expander("Additional Settings"):
    benchmark_index = st.selectbox(
    "Select an index to benchmark the portfolio:",
    ["^GSPC", "^DJI", "^IXIC"],
    key="benchmark_index",
    help="Performance of your portfolio will be benchmarked against this index's performance over the past 10 years."
    )

    risk_free_rate = st.slider(
        "Risk-free rate (%):",
        min_value=0.0,
        max_value=10.0,
        value=4.1,
        key="risk_free_rate",
        help="Represents the return on a risk-free investment (e.g. government bonds). It is used in the calculation of the Sharpe ratio to measure the risk-adjusted return."
    )/100

# Button layout for Run and Random Inputs
col1, col2 = st.columns([1, 0.4])

with col1:
    # Run button for custom inputs
    run_button = st.button("Run")
with col2:
    # Random Button for testing with random tickers
    random_button = st.button("Test with random tickers")

assets_list = None

# Logic for Custom Inputs after validation
if run_button:
    assets_list = utils.run_button(user_input)
# Logic for random Inputs 
elif random_button:
    assets_list = utils.random_button()

no_of_years = 10            # Number of years of historical data to be used for analysis
no_of_iterations = 1000     # Number of Monte Carlo simulations

st.write('________________________________________________________________________')

if assets_list is not None:
    # Retrieve Data
    adj_close, benchmark_df, combined_df, benchmark_name = utils.retrieve_data(assets_list, risk_free_rate, benchmark_index, no_of_years)

    # Calculate Returns and Risk Statistics
    if adj_close is None or benchmark_df is None or combined_df is None or benchmark_name is None:
        None
    else:
        returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark, returns_all_corr = utils.return_stats(
            adj_close, benchmark_df, combined_df, assets_list, benchmark_index, no_of_years
        )

        # Perform Efficient Frontier Analysis
        pfolio_volatility, pfolio_return, weights, sharpe_ratios = utils.eff_frontier(
            assets_list, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations
        )

        # Optimize Portfolio Using Mean-Variance Optimization
        optimal_weights, min_return_valid = utils.opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, min_return)

        # Outputs
        # Output Results
        utils.opt_portfolio_results(optimal_weights, returns_assets_ann, returns_assets_cov, risk_free_rate, assets_list, min_return_valid)
        
        # Visualize Results
        plt.style.use('bmh')
        utils.visualize_analyses(
            pfolio_volatility, pfolio_return, weights, sharpe_ratios, 
            returns_assets, optimal_weights, returns_benchmark, benchmark_name, 
            assets_list, benchmark_index, no_of_iterations
        )   

        # Prepare data for Table 1: Return, Volatility, Sharpe Ratio of Each Asset
        table1_data = {
            "Metric": ["Average Return", "Average Volatility", "Sharpe Ratio"]
        }
        for ticker in assets_list:
            asset_return = returns_assets_ann[ticker].mean()
            asset_volatility = returns_assets[ticker].std() * (250 ** 0.5)
            asset_sharpe = (asset_return - risk_free_rate) / asset_volatility

            table1_data[ticker] = [
                f"{asset_return * 100:.2f}%",
                f"{asset_volatility * 100:.2f}%",
                f"{asset_sharpe:.4f}",
            ]

        # Display Table 1
        st.subheader("Asset Metrics")
        table1_df = pd.DataFrame(table1_data)
        st.table(table1_df.set_index("Metric"))

        # Prepare data for Table 2: Portfolio Weights and Metrics
        table2_data = {
        "Portfolio": ["Optimized Portfolio", "Maximum Return Portfolio", "Minimum Volatility Portfolio", "Maximum Sharpe Ratio Portfolio"],
        **{ticker: [] for ticker in assets_list},  # Initialize empty lists for each ticker
        "Portfolio Return": [],
        "Portfolio Volatility": [],
        "Portfolio Sharpe Ratio": [],
        }

        portfolios = [
            ("Optimized Portfolio", optimal_weights),
            ("Maximum Return Portfolio", weights[np.argmax(pfolio_return)]),
            ("Minimum Volatility Portfolio", weights[np.argmin(pfolio_volatility)]),
            ("Maximum Sharpe Ratio Portfolio", weights[np.argmax(sharpe_ratios)]),
        ]

        # Populate the table with data
        for name, portfolio_weights in portfolios:
            portfolio_return, portfolio_volatility, portfolio_sharpe = utils.portfolio_stats(
                portfolio_weights, returns_assets_ann, returns_assets_cov, risk_free_rate
            )
            for i, ticker in enumerate(assets_list):
                table2_data[ticker].append(f"{portfolio_weights[i]:.2f}")  # Add weight for each ticker
            table2_data["Portfolio Return"].append(f"{portfolio_return * 100:.2f}%")
            table2_data["Portfolio Volatility"].append(f"{portfolio_volatility * 100:.2f}%")
            table2_data["Portfolio Sharpe Ratio"].append(f"{portfolio_sharpe:.4f}")

        # Display Table 2
        st.subheader("Portfolio Weights and Metrics")
        table2_df = pd.DataFrame(table2_data)
        st.table(table2_df.set_index("Portfolio"))

        # Correlation Heatmap
        st.subheader("Asset Correlation Heatmap:")
        fig = plt.figure(figsize=(15, 8))
        sns.heatmap(returns_all_corr, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig)  # Display the heatmap in Streamlit

else:
    None 

