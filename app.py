import streamlit as st
import utils.data_processing as dp
import utils.optimization as opt
import utils.visualization as vis

# Header section
st.markdown("<h1 style='text-align:center; color:#003366;'>Portfolio Optimization Model</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size: 18px;'>This is a Python-based application for analyzing and optimizing financial portfolios. "
    "It uses modern portfolio theory and convex optimization techniques to identify optimal allocations, backtest performance, and visualize results.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size: 15px;'>"
    "<a href='https://github.com/AsherGeorgy/portfolio-optimization-tool/tree/streamlit_app' target='_blank'>View Source Code on GitHub</a>"
    "</p>",
    unsafe_allow_html=True,
)


# Sidebar disclaimer
with st.sidebar:
    # Developer Information
    st.markdown("<h1 style='color: #003366;'>Developed by:</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; border-top: 2px solid #003366; margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True)  # Underline-style line below the heading
    st.markdown("<h3 style='font-weight: bold; color: #333;'>Asher Georgy</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #555;'>Finance professional turned data enthusiast. <br><a href='https://ashergeorgy.github.io/' target='_blank' style='color: #007BFF; text-decoration: none;'>Visit my Website</a></p>", unsafe_allow_html=True)

    # Disclaimer Section
    st.markdown("<h1 style='color: #003366;'>Disclaimer:</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; border-top: 2px solid #003366; margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True)  # Underline-style line below the heading
    st.markdown("""
    <div style="color: #555; font-size: 14px;">
        <p>This application is for educational and informational purposes only. The analysis, data, and results provided by this tool are based on historical data and theoretical models. 
        <u><strong>They do not predict future performance.</strong></u></p>
        <p>Users should conduct their own due diligence, verify results, and consult a qualified financial advisor before making any investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Link to full disclaimer
    st.markdown("<p style='font-size: 13px; text-align: center; margin-top: 20px;'><a href='https://ashergeo/full-disclaimer' target='_blank' style='color: #007BFF; text-decoration: none;'>View full Disclaimer and Limitations</a></p>", unsafe_allow_html=True)

# Separator line
st.markdown("<hr style='border: 1px solid #003366;'>", unsafe_allow_html=True)

# Custom input fields
st.markdown("<h3 style='color: #003366;'>Configure Your Portfolio:</h3>", unsafe_allow_html=True)
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
    ["^IXIC", "^GSPC", "^DJI"],
    key="benchmark_index",
    help="Performance of your portfolio will be benchmarked against this index's performance over the time period of analysis."
    )

    no_of_years = st.slider(
        "Period of analysis (years):",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        key="no_of_years",
        help="The number of years of historical data that will be retrieved to conduct the analysis."
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
    assets_list = dp.run_button(user_input)
# Logic for random Inputs 
elif random_button:
    assets_list = dp.random_button()

no_of_iterations = 1000     # Number of Monte Carlo simulations


if assets_list is not None:
    # Retrieve Data
    adj_close, benchmark_df, combined_df, benchmark_name = dp.retrieve_data(assets_list, risk_free_rate, benchmark_index, no_of_years)

    # Calculate Returns and Risk Statistics
    if adj_close is None or benchmark_df is None or combined_df is None or benchmark_name is None:
        None
    else:
        returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark, returns_all_corr = opt.return_stats(
            adj_close, benchmark_df, combined_df
        )

        # Perform Efficient Frontier Analysis
        pfolio_volatility, pfolio_return, weights, sharpe_ratios = opt.eff_frontier(
            assets_list, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations
        )

        # Optimize Portfolio Using Mean-Variance Optimization
        optimal_weights, target_cagr_valid = opt.opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, target_cagr)

        # Outputs
        # Output Results
        vis.opt_portfolio_results(
            optimal_weights, returns_assets, returns_assets_ann, returns_assets_cov, risk_free_rate, assets_list, returns_benchmark, benchmark_index, benchmark_name, target_cagr_valid
        )
        
        # Visualize Results
        vis.visualize_analyses(
            pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, benchmark_index, no_of_iterations, assets_list, returns_assets_ann, returns_all_corr, returns_assets_cov, risk_free_rate
        )

else:
    None 





