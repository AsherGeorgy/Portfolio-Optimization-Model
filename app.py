import streamlit as st
import utils.data_processing as dp
import utils.optimization as opt
import utils.visualization as vis

st.markdown(
    "<p style='text-align:center; font-size: 0.9rem; font-style:italic; color: #666; margin-top: 0.625rem;'>"
    "For the best experience, please switch to landscape mode if you're using a mobile device."
    "</p>",
    unsafe_allow_html=True,
)

# Header section
st.markdown("<h1 style='text-align:center; color:#003366;'>Portfolio Optimization Model</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size: 1.125rem;'>This is a Python-based application for analyzing and optimizing financial portfolios. "
    "It uses modern portfolio theory and convex optimization techniques to identify optimal allocations, backtest performance, and visualize results.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size: 0.9375rem;'>"
    "<a href='https://github.com/AsherGeorgy/Portfolio-Optimization-Model/tree/main' target='_blank'>View Source Code on GitHub</a>"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center; font-size: 0.8rem; color: #666; margin-top: 0.625rem;'>"
    "View sidebar for developer info and disclaimer"
    "</p>",
    unsafe_allow_html=True,
)


# Sidebar disclaimer
with st.sidebar:
    # Developer Information
    st.markdown("<h1 style='color: #003366;'>Developed by:</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; border-top: 0.125rem solid #003366; margin-top: 0px; margin-bottom: 0.625rem;'>", unsafe_allow_html=True)  # Underline-style line below the heading
    st.markdown("<h3 style='font-weight: bold; color: #333;'>Asher Georgy</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #555;'>Finance professional turned data enthusiast. <br><a href='https://ashergeorgy.github.io/' target='_blank' style='color: #007BFF; text-decoration: none;'>Visit my Website</a></p>", unsafe_allow_html=True)

    # Disclaimer Section
    st.markdown("<h1 style='color: #003366;'>Disclaimer:</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; border-top: 0.125rem solid #003366; margin-top: 0px; margin-bottom: 0.625rem;'>", unsafe_allow_html=True)  # Underline-style line below the heading
    st.markdown("""
    <div style="color: #555; font-size: 0.875rem;">
        <p>This application is for educational and informational purposes only. The analysis, data, and results provided by this tool are based on historical data and theoretical models. 
        <u><strong>They do not predict future performance.</strong></u></p>
        <p>Users should conduct their own due diligence, verify results, and consult a qualified financial advisor before making any investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Link to full disclaimer
    st.markdown("<p style='font-size: 0.8125rem; text-align: center; margin-top: 1.25rem;'><a href='https://ashergeorgy.github.io/blog/Limitations%20and%20Disclaimer.html'>View full Disclaimer and Limitations</a></p>", unsafe_allow_html=True)

# Separator line
st.markdown("<hr style='border: 0.0625rem solid #003366;'>", unsafe_allow_html=True)


# Instructions
st.markdown(
    "<h5 style='color: #003366; background-color: #e6f7ff; padding: 0.625rem; font-weight: bold;'>Instructions:</h5>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="color: #555; background-color: #e6f7ff; padding: 0.9375rem; border-radius: 0.3125rem; box-shadow: 0 0.125rem 0.3125rem rgba(0, 0, 0, 0.1); margin-bottom: 1.25rem;">
        <p><strong>1. Enter Stock Tickers:</strong> Input the stock tickers separated by commas (e.g., <i>'AAPL, MSFT, TSLA'</i>) in the input field below.</p>
        <p><strong>2. Set Desired Target Return:</strong> Use the slider to set the target Annual Return you desire for your portfolio.</p>
        <p><strong>3. Run the Optimization:</strong> Click the <b>Run</b> button to generate the optimized portfolio.</p>
        <p><em>Alternatively, Click the Test with random tickers button to use randomly selected stock tickers for testing.</em></p>
    </div>
    """, unsafe_allow_html=True
)


# Program code starts here

# Custom input fields
st.markdown("<h3 style='color: #003366;'>Configure Your Portfolio:</h3>", unsafe_allow_html=True)
user_input = st.text_input(
                            "Enter stock tickers separated by commas: ",
                            placeholder="aapl, msft, tsla",
                            key="user_input",
                        ).strip()

target_return = st.slider(
        "Target Expected Annual Return (%):",
        min_value=2.0,
        max_value=100.0,
        value=50.0,
        step=0.5,
        key="target_return",
        help="Target annual return (%) you'd like your portfolio to achieve while minimizing volatility."
    )/100

# Sliders for optional settings
with st.expander("Additional Settings"):
    benchmark_index = st.selectbox(
    "Select an index to benchmark the portfolio:",
    ["^GSPC", "^IXIC", "^DJI"],
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

no_of_iterations = 10000     # Number of Monte Carlo simulations


if assets_list is not None:
    # Retrieve Data
    adj_close, benchmark_df, combined_df, benchmark_name = dp.retrieve_data(
        assets_list, risk_free_rate, benchmark_index, no_of_years
    )

    # Control Flow
    if adj_close is None or benchmark_df is None or combined_df is None or benchmark_name is None:
        None
    else:
        # Calculate Returns and Risk Statistics
        returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark, returns_all_corr = opt.return_stats(
            adj_close, benchmark_df, combined_df
        )

        # Perform Efficient Frontier Analysis
        pfolio_volatility, pfolio_return, weights, sharpe_ratios = opt.eff_frontier(
            assets_list, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations
        )

        # Optimize Portfolio Using Mean-Variance Optimization
        optimal_weights, target_return_valid = opt.opt_portfolio_cvxpy(
            returns_assets_ann, returns_assets_cov, target_return
        )

        # Outputs
        # Output Results
        vis.opt_portfolio_results(
            optimal_weights, returns_assets, returns_assets_ann, returns_assets_cov, risk_free_rate, assets_list, returns_benchmark, benchmark_index, benchmark_name, target_return_valid
        )
        
        # Visualize Results
        vis.visualize_analyses(
            pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, benchmark_index, no_of_iterations, assets_list, returns_assets_ann, returns_all_corr, returns_assets_cov, risk_free_rate
        )

else:
    None 

# Separator line
st.markdown("<hr style='border: 0.0625rem solid #003366;'>", unsafe_allow_html=True)

# Data source and error warnings 
st.markdown(
    """<br><br><br><h6 style="color: #ff6600;">Note:</h6>
    <ul style="font-size: 0.75rem; color: #ff6600; line-height: 1.6;">
        <li>This application utilizes the free Yahoo Finance (yfinance) API, which may have reliability limitations compared to premium data sources. As a result, it may occasionally encounter issues due to web scraping.</li>
        <li>The API generally provides accurate data for stocks listed on NASDAQ and NYSE. However, for stocks on other exchanges, exchange-specific suffixes (e.g., <i>.L</i> for the London Stock Exchange) may be required. Please note that this data may still be inconsistent or unreliable in some cases.</li>
        <li>If you experience errors with Streamlit or yfinance, refreshing the page typically resolves the issue.</li>
    </ul>""",
    unsafe_allow_html=True
)











