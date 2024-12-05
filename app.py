import streamlit as st
import utils
import matplotlib.pyplot as plt

# Header section
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>Portfolio Optimization Model</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size: 18px;'>This is a Python-based application for analyzing and optimizing financial portfolios. "
    "It uses modern portfolio theory and convex optimization techniques to identify optimal allocations and visualize performances.</p>",
    unsafe_allow_html=True,
)

# Separator line
st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)

# Custom input fields
st.markdown("### Configure Your Portfolio:")
user_input = st.text_input(
    "Enter stock tickers separated by commas: ",
    placeholder="aapl, msft, tsla",
    key="user_input",
).strip()

target_cagr = st.slider(
        "Target CAGR (%):",
        min_value=2.0,
        max_value=100.0,
        value=50.0,
        key="target_cagr",
        help="Compound Annual Growth Rate (%) you'd like your portfolio to achieve while minimizing volatility.",
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

st.markdown('________________________________________________________________________')

assets_list = None

# Logic for Custom Inputs after validation
if run_button:
    assets_list = utils.run_button(user_input)
# Logic for random Inputs 
elif random_button:
    assets_list = utils.random_button()

no_of_years = 10            # Number of years of historical data to be used for analysis
no_of_iterations = 1000     # Number of Monte Carlo simulations


if assets_list is not None:
    # Retrieve Data
    adj_close, benchmark_df, combined_df, benchmark_name = utils.retrieve_data(assets_list, risk_free_rate, benchmark_index, no_of_years)

    # Calculate Returns and Risk Statistics
    if adj_close is None or benchmark_df is None or combined_df is None or benchmark_name is None:
        None
    else:
        returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark, returns_all_corr = utils.return_stats(
            adj_close, benchmark_df, combined_df
        )

        # Perform Efficient Frontier Analysis
        pfolio_volatility, pfolio_return, weights, sharpe_ratios = utils.eff_frontier(
            assets_list, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations
        )

        # Optimize Portfolio Using Mean-Variance Optimization
        optimal_weights, target_cagr_valid = utils.opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, target_cagr)

        # Outputs
        # Output Results
        utils.opt_portfolio_results(
            optimal_weights, returns_assets, returns_assets_ann, returns_assets_cov, risk_free_rate, assets_list, returns_benchmark, benchmark_index, benchmark_name, target_cagr_valid
        )
        
        # Visualize Results
        plt.style.use('bmh')
        utils.visualize_analyses(
            pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, benchmark_index, no_of_iterations, assets_list, returns_assets_ann, returns_all_corr, returns_assets_cov, risk_free_rate
        )

else:
    None 

