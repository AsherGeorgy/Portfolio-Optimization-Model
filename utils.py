import streamlit as st
import yfinance as yf
import numpy as np

# Inputs

# Function to generate random inputs
def generate_random_inputs():
    # Generate random stock tickers
    sample_assets = {
    "AAPL": "Apple Inc.",  # Tech stock
    "MSFT": "Microsoft Corp.",  # Tech stock
    "GOOGL": "Alphabet Inc.",  # Tech stock
    "AMZN": "Amazon.com Inc.",  # Consumer Discretionary
    "TSLA": "Tesla Inc.",  # Consumer Discretionary
    "NVDA": "NVIDIA Corporation",  # Tech stock
    "META": "Meta Platforms Inc.",  # Tech stock
    "NFLX": "Netflix Inc.",  # Communication Services
    "V": "Visa Inc.",  # Financials
    "JPM": "JPMorgan Chase & Co.",  # Financials
    "DIS": "The Walt Disney Company",  # Communication Services
    "KO": "The Coca-Cola Company",  # Consumer Staples
    "PEP": "PepsiCo, Inc.",  # Consumer Staples
    "PFE": "Pfizer Inc.",  # Healthcare
    "JNJ": "Johnson & Johnson",  # Healthcare
    "UNH": "UnitedHealth Group",  # Healthcare
    "WMT": "Walmart Inc.",  # Consumer Staples
    "XOM": "Exxon Mobil Corporation",  # Energy
    "CVX": "Chevron Corporation",  # Energy
    "BA": "Boeing Company",  # Industrials
    "CAT": "Caterpillar Inc.",  # Industrials
    "GM": "General Motors Company",  # Consumer Discretionary
    "IBM": "International Business Machines",  # Tech stock
    "CSCO": "Cisco Systems, Inc.",  # Tech stock
    "SPG": "Simon Property Group",  # Real Estate
    "SLB": "Schlumberger Limited",  # Energy
    "LMT": "Lockheed Martin Corporation",  # Aerospace & Defense
    "GS": "Goldman Sachs Group, Inc.",  # Financials
    "MRK": "Merck & Co., Inc.",  # Healthcare
    "BNS": "Bank of Nova Scotia",  # Financials
    "T": "AT&T Inc.",  # Communication Services
    "VZ": "Verizon Communications Inc.",  # Communication Services
    }

    size = np.random.randint(3,11)
    assets_list = np.random.choice(list(sample_assets.keys()), size=size, replace=False)  # Select random tickers
    
    st.markdown("##### Preparing random tickers...")
    st.markdown(f"**Tickers selected**: {(', ').join(assets_list)}")
    for ticker in assets_list:
        st.success(f"Ticker validated: {sample_assets[ticker]} ({ticker}) ")

    st.markdown('________________________________________________________________________')
    st.markdown("##### Running optimization...")

    return assets_list

def assets(user_input):
    st.markdown("##### Processing input tickers...")
    if not user_input:
        st.error("Input cannot be empty.")
        return None

    # Split and strip tickers, removing empty or whitespace-only entries
    input_split = [ticker.strip().replace("'", "").upper() for ticker in user_input.split(',') if ticker.strip()]
    if not input_split:
        st.error("No valid tickers found in input. Try again.")

        return None

    invalid_tickers = []
    valid_assets = []
    stock_names = {}

    for ticker in input_split:
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period="1mo")
            stock_name = stock.info.get("longName")
            if stock_data.empty:  # Check if data exists for the ticker
                invalid_tickers.append(ticker)
            else:
                valid_assets.append(ticker)
                stock_names[ticker] = stock_name
        except Exception:
            invalid_tickers.append(ticker)

    if invalid_tickers:
        st.error(f"Invalid ticker found: {', '.join(invalid_tickers)}")
        st.error("Try again with valid tickers!")
        return None
    elif valid_assets:
        for ticker, name in stock_names.items():
            st.success(f"Ticker validated: {name} ({ticker})")
        st.markdown('________________________________________________________________________')
        st.markdown("##### Running optimization...")
        return valid_assets

def run_button(user_input):
    # Validate that there are at least two tickers
    tickers = [ticker.strip() for ticker in user_input.split(',') if ticker.strip()]  
    if len(tickers) < 2:
        st.error("Please enter at least two stock tickers.")
        return None
    else:
        assets_list = assets(user_input)
        return assets_list

def random_button():
    assets_list = generate_random_inputs()
    return assets_list


# Processes
import pandas as pd
from datetime import datetime, timedelta

def retrieve_data(assets_list, risk_free_rate, benchmark_index, no_of_years):  
    # Determine start_date and end_date based on no_of_years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(no_of_years*365))
    
    # Assets data
    adj_close = pd.DataFrame()
    for t in assets_list:
        adj_close[t] = yf.download(t, start=start_date, end=end_date)['Adj Close']
        if adj_close[t].empty:
            st.error(f"No data available for asset: {t}. Try again.")
            return None, None, None, None
    
    # Benchmark index data
    benchmark_df = pd.DataFrame()
    benchmark_df[benchmark_index] = yf.download(benchmark_index, start=start_date, end=end_date)['Adj Close']
    if benchmark_df.empty:
        st.error(f"No data available for benchmark index: {benchmark_index}.  Try again.")
        return None, None, None, None
    
    # Retrieve long names of assets 
    asset_names = []
    for t in assets_list:
        t = yf.Ticker(t)
        company_name = t.info['longName']
        asset_names.append(company_name)

    # Retrieve long name of benchmark index
    benchmark_ticker = yf.Ticker(benchmark_index)
    benchmark_name = benchmark_ticker.info['longName']

    # Combine asset and benchmark data on common index.
    combined_df = pd.merge(adj_close, benchmark_df, left_index=True, right_index=True)

    # Outputs
    # Separator line
    st.markdown('________________________________________________________________________')
    st.markdown("<h2 style='text-align:center;'><u>Analysis</u></h2>",
    unsafe_allow_html=True,)
    st.markdown(f'##### The following analysis was conducted on {no_of_years}Y daily adjusted closing price data from Yahoo Finance.')
    st.markdown(f'**Time period of analysis:**    {(start_date).strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    st.markdown(f'**Index used as benchmark:**    {benchmark_name}')
    st.markdown(f'**Risk free rate used:**        {risk_free_rate*100:.2f}%')
    st.markdown(f'**Assets analysed:**           {",  ".join(asset_names)}')
    st.markdown(f'**Techniques used:**           Convex Optimization, Modern Portfolio Theory')
    st.markdown('________________________________________________________________________')
        
    return adj_close, benchmark_df, combined_df, benchmark_name

def return_stats(adj_close, benchmark_df, combined_df, assets, benchmark_index, no_of_years):
    """
    Calculates return statistics (annualized returns, volatility, covariance, and correlation) for assets and a benchmark.
    Returns pd DataFrames.
    """
    # Calculate simple returns and covariance of the assets
    returns_assets = adj_close.pct_change().dropna()
    returns_assets_ann = returns_assets * 250   # annualised
    returns_assets_cov = returns_assets.cov() * 250

    # Calculate mean of Annualized Total Returns and Volatility of the assets
    ann_total_returns_mean = returns_assets_ann.mean()
    ann_volatility_mean = (returns_assets.std() * 250 ** 0.5).mean()
    
    # Calculate simple returns of the benchmark
    returns_benchmark = benchmark_df.pct_change().dropna()

    # Create a df with both assets and benchmark combined 
    returns_all = combined_df.pct_change().dropna()
    returns_all_ann = returns_all * 250
    returns_all_corr = returns_all.corr()
    
    # Header
    # st.markdown("### I. Individual Asset Analysis:")

    # Tickers (including the benchmark index)
    tickers = assets + [benchmark_index]

    # CAGR Table (Daily Compounding)
    cagr_data = [(ticker, f"{mean*100:.2f}%") for ticker, mean in zip(tickers, returns_all.mean())]
    cagr_df = pd.DataFrame(cagr_data, columns=["Asset", f"{no_of_years}-Year CAGR (Daily Compounding)"])
    # st.subheader(f"CAGR Table ({no_of_years}-Year Daily Compounding):")
    # st.table(cagr_df)

    # Display Average (excluding benchmark)
    avg_cagr = ann_total_returns_mean.mean() * 100
    # st.markdown(f"**Average CAGR (excluding {benchmark_index}):** {avg_cagr:.2f}%")

    # Volatility Table
    volatility_data = [(ticker, f"{(std * 100 * 250 ** 0.5):.2f}%") for ticker, std in zip(tickers, returns_all.std())]
    volatility_df = pd.DataFrame(volatility_data, columns=["Asset", "Annualized Volatility"])
    # st.subheader("Volatility Table:")
    # st.table(volatility_df)

    # Display Average Volatility (excluding benchmark)
    avg_volatility = ann_volatility_mean.mean() * 100
    # st.markdown(f"**Average Volatility (excluding {benchmark_index}):** {avg_volatility:.2f}%")
    
    return returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark, returns_all_corr

def portfolio_stats(weights, returns_assets, cov_assets, risk_free_rate):
    """
    Calculates portfolio return, volatility, and Sharpe ratio based on asset weights, returns, covariance, and risk-free rate.
    Returns floats.
    """

    # Portfolio Return
    portfolio_return = np.dot(weights, returns_assets.mean())
    
    # Portfolio Volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_assets, weights)))
    
    # Portfolio Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def eff_frontier(assets, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations=1000):
    """
    Generates the efficient frontier of a portfolio using Monte Carlo simulation.
    Returns numpy arrays.
    """
    # Initialize list to append portfolio returns, volatility and weights from each iteration.    
    pfolio_return = []
    pfolio_volatility = []
    weights_list = []
    
    # Monte Carlo Simulation
    for i in range (no_of_iterations):
        # Generate random weights which add up to 1 and append to weight_list
        no_of_tickers = len(assets)
        random_floats = np.random.random(no_of_tickers)
        weights = random_floats/np.sum(random_floats)
        weights_list.append(weights)

        # Calculate and append portfolio return and volatility at the generated weight
        pfolio_return.append(np.dot(returns_assets_ann.mean(), weights))
        pfolio_volatility.append(np.sqrt(np.dot(weights.T, np.dot(returns_assets_cov, weights))))
        
    # Convert each lists to numpy arrays. Mulitply by 100 to percentage.
    weights = np.array(weights_list)
    pfolio_return = np.array(pfolio_return)*100
    pfolio_volatility = np.array(pfolio_volatility)*100
    sharpe_ratios = (pfolio_return - risk_free_rate*100) / pfolio_volatility
    
    return pfolio_volatility, pfolio_return, weights, sharpe_ratios

import cvxpy as cp

def opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, target_cagr):
    """
    Optimizes portfolio allocation using Mean-Variance Optimization with convex optimization (cvxpy).
    Returns floats (or None if the optimization fails).
    """

     # Display the portfolio optimization results
    st.markdown("<h2><u>Convex Optimization</u></h2>",
    unsafe_allow_html=True,)
    st.markdown(f"#####  **Results**:")

    # Initialize variables
    n = len(returns_assets_ann.columns)  # Number of assets (columns in the dataframe)
    w = cp.Variable(n)  # Portfolio weights as a cvxpy variable

    # Check if the input target_cagr is feasible (it needs to be equal to or lower than the max average return of individual assets)
    max_expected_return = returns_assets_ann.mean().max()
    min_expected_return = returns_assets_ann.mean().min()

    if target_cagr > max_expected_return:
        st.error(f"Warning: The target CAGR of {target_cagr*100:.2f}% exceeds the maximum feasible portfolio CAGR of {max_expected_return*100:.2f}%, which is based on the highest historical return of the portfolio's assets during this time period.")
        st.success(f"CAGR has been adjusted to the maximum feasible return of {(max_expected_return - 0.0001)*100:.2f}%.")
        target_cagr_valid = max_expected_return - 0.0001  # Adjust target_cagr to the highest feasible value
    elif target_cagr < min_expected_return:
        st.warning(f"Warning: The target CAGR of {target_cagr*100:.2f}% is below the minimum feasible return of {min_expected_return*100:.2f}%, which is based on the lowest historical return of the portfolio's assets during this time period.")
        st.success(f"CAGR has been adjusted to the minimum feasible return of {(min_expected_return + 0.0001)*100:.2f}%.")

        target_cagr_valid = min_expected_return + 0.0001  # Adjust target_cagr to the lowest feasible value
    else:
        target_cagr_valid = target_cagr


    
    # Calculate expected Return and Risk
    ret = returns_assets_ann.mean().values @ w  # Expected return: dot product of weights and asset returns ("@" is the matrix multiplication operator)
    risk = cp.quad_form(w, returns_assets_cov.values)  # Portfolio risk: computes a quadratic form giving the variance
    
    # Define the objective: Minimize portfolio risk
    objective = cp.Minimize(risk)
    
    # Define the constraints: 
    constraints = [
        cp.sum(w) == 1,  # Sum of weights equals 1 (fully invested)
        ret == target_cagr_valid,  # Target minimum return constraint
        w >= 0  # Non-negative weights (no short positions)
    ]
    
    # CVXPY solver
    prob = cp.Problem(objective, constraints)  # The optimization problem
    prob.solve()  
    optimal_weights = w.value
    
    # Check if the optimization was successful and the optimal weights are valid
    if prob.status != cp.OPTIMAL or optimal_weights is None or any(np.isnan(optimal_weights)):
        st.write(f"Optimization failed. Solver status: {prob.status} or Invalid portfolio weights.")
        return None, target_cagr_valid  

    return optimal_weights, target_cagr_valid


def opt_portfolio_results(optimal_weights, returns_assets, returns_assets_ann, returns_assets_cov, risk_free_rate, assets_list, returns_benchmark, benchmark_index, benchmark_name, target_cagr_valid):
    """
    Displays the results of the portfolio optimization if the optimal weights are valid.
    """
    # Return if weights are invalid
    if optimal_weights is None or np.any(np.isnan(optimal_weights)):
        st.warning("Optimal weights are invalid. Please check your inputs.")
        return

    # Calculate stats of portfolio with optimal weights
    optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio = portfolio_stats(optimal_weights, returns_assets_ann, returns_assets_cov, risk_free_rate)

    # Create a DataFrame for weights with better formatting
    results_df = pd.DataFrame({
        "Asset": assets_list,
        "Weight": [f"{weight * 100:.2f}%" for weight in optimal_weights]  # Format weights as percentages
    })

    # Set 'Asset' column as the index
    results_df.set_index('Asset', inplace=True)

    st.markdown(f"The portfolio allocation, optimized to meet the **target CAGR of {target_cagr_valid * 100:.2f}% while minimizing volatility**, based on historical data, is as follows:")
    # Display the weights table
    st.write(results_df.T)

    # Prepare cumulative returns for plotting
    cumprod_df = returns_benchmark.copy()
    cumprod_df.rename(columns={benchmark_index: benchmark_name}, inplace=True)

    if optimal_weights is not None and not np.any(np.isnan(optimal_weights)):
        pfolio_returns_at_optimal_weights = returns_assets.values @ optimal_weights
        cumprod_df['Optimized Portfolio'] = pfolio_returns_at_optimal_weights

    # Calculate cumulative returns
    cumprod_df = (1 + cumprod_df).cumprod()

    st.markdown(f"Here’s how the optimized portfolio performed over the past 10 years:")
    # Create two columns for side-by-side layout
    col1, col2 = st.columns([2, 6])  # Adjust column width ratio if needed

    with col1:
        # Display portfolio statistics using st.metric() with delta or custom formatting
        st.metric(label="CAGR", value=f"{optimal_portfolio_return * 100:.2f}%")
        st.metric(label="Average Annual Volatility", value=f"{optimal_portfolio_volatility * 100:.2f}%")
        st.metric(label="Sharpe Ratio", value=f"{optimal_sharpe_ratio:.4f}")

    with col2:
        # Display the cumulative returns chart
        st.line_chart(cumprod_df, use_container_width=True, x_label=f"Optimized Portfolio compared to {benchmark_name} over the past 10 years.")

    st.write('________________________________________________________________________')


import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

def visualize_analyses(pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, benchmark_index, no_of_iterations, assets_list, returns_assets_ann, returns_all_corr, returns_assets_cov, risk_free_rate):
    """
    Visualizes portfolio analysis results in a Streamlit app, including the efficient frontier, identified portfolios, and their relative daily return movements.
    """
    # Identify relevant portfolios
    min_volatility_idx = np.argmin(pfolio_volatility)
    max_return_idx = np.argmax(pfolio_return)
    max_sharpe_idx = np.argmax(sharpe_ratios)

    # Calculate daily portfolio returns
    pfolio_returns_at_max_sharpe_weights = returns_assets.values @ weights[max_sharpe_idx]
    pfolio_returns_at_min_vol_weights = returns_assets.values @ weights[min_volatility_idx]
    pfolio_returns_at_max_ret_weights = returns_assets.values @ weights[max_return_idx]


    # Outputs
    # Section Title
    st.markdown("<h2><u>Modern Portfolio Theory</u></h2>", unsafe_allow_html=True)

   
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["Efficient Frontier", "Portfolio Allocation","Performance Comparison", "Constituents"])

    # Efficient Frontier Analysis
    with tab1:
        # Efficient Frontier Plot
        st.subheader("Efficient Frontier")
        st.markdown("""
            **Modern Portfolio Theory (MPT)** identifies key portfolios on the efficient frontier, 
            including the **maximum return portfolio**, the **minimum volatility portfolio**, and the 
            **maximum Sharpe ratio portfolio**, each representing different trade-offs between 
            risk and return. MPT underscores the importance of asset diversification to 
            achieve the optimal balance between risk and return.
        """)
        st.markdown("""
            The following plot visualizes the efficient frontier generated by Monte Carlo simulations, showing the relationship between 
            portfolio return and volatility.
        """)
        # Display the number of Monte Carlo simulation runs
        st.write(f"**Number of Monte Carlo simulation runs:** {no_of_iterations}")
        fig = go.Figure()

        # Add all portfolios scatter plot
        fig.add_trace(go.Scatter(
            x=pfolio_volatility,
            y=pfolio_return,
            mode='markers',
            name='All Portfolios',
            marker=dict(
                color=sharpe_ratios,
                colorscale='Viridis',
                size=6,
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            )
        ))

        # Highlight Min Volatility Portfolio
        fig.add_trace(go.Scatter(
            x=[pfolio_volatility[min_volatility_idx]],
            y=[pfolio_return[min_volatility_idx]],
            mode='markers+text',
            name='Min Volatility',
            marker=dict(color='green', size=10),
            text='Min Volatility',
            textposition='top center'
        ))

        # Highlight Max Return Portfolio
        fig.add_trace(go.Scatter(
            x=[pfolio_volatility[max_return_idx]],
            y=[pfolio_return[max_return_idx]],
            mode='markers+text',
            name='Max Return',
            marker=dict(color='blue', size=10),
            text='Max Return',
            textposition='top center'
        ))

        # Highlight Max Sharpe Ratio Portfolio
        fig.add_trace(go.Scatter(
            x=[pfolio_volatility[max_sharpe_idx]],
            y=[pfolio_return[max_sharpe_idx]],
            mode='markers+text',
            name='Max Sharpe Ratio',
            marker=dict(color='red', size=10),
            text='Max Sharpe',
            textposition='top center'
        ))

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title='Volatility (%)',
            yaxis_title='Return (%)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )

        # Render Efficient Frontier plot
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Prepare data for Portfolio Weights and Metrics Table
        table2_data = {
            "Portfolio": [
                "Maximum Return Portfolio", 
                "Minimum Volatility Portfolio", 
                "Maximum Sharpe Ratio Portfolio"
            ],
            **{ticker: [] for ticker in assets_list},  # Initialize empty lists for each ticker
            "Portfolio Return": [],
            "Portfolio Volatility": [],
            "Portfolio Sharpe Ratio": [],
        }

        # Define portfolios and their corresponding weights
        portfolios = [
            ("Maximum Return Portfolio", weights[np.argmax(pfolio_return)]),
            ("Minimum Volatility Portfolio", weights[np.argmin(pfolio_volatility)]),
            ("Maximum Sharpe Ratio Portfolio", weights[np.argmax(sharpe_ratios)]),
        ]

        # Populate the table with data
        for name, portfolio_weights in portfolios:
            portfolio_return, portfolio_volatility, portfolio_sharpe = portfolio_stats(
                portfolio_weights, returns_assets_ann, returns_assets_cov, risk_free_rate
            )
            for i, ticker in enumerate(assets_list):
                table2_data[ticker].append(f"{portfolio_weights[i]:.2f}")  # Add weight for each ticker
            table2_data["Portfolio Return"].append(f"{portfolio_return * 100:.2f}%")
            table2_data["Portfolio Volatility"].append(f"{portfolio_volatility * 100:.2f}%")
            table2_data["Portfolio Sharpe Ratio"].append(f"{portfolio_sharpe:.4f}")

        # Display the table with improved styling
        st.subheader("Portfolio Allocation")

        st.markdown("""
            The table below presents three portfolios based on MPT, along with their 
            corresponding asset weights and historical performance metrics.
        """)


        # Convert the table data into a pandas DataFrame for better control over styling
        table2_df = pd.DataFrame(table2_data)

        # Display the table with index (Portfolio) as rows
        st.table(table2_df.set_index("Portfolio"))

    with tab3: 

        # Prepare cumulative product of returns for comparison
        cumprod_df = returns_benchmark.copy()
        cumprod_df.rename(columns={benchmark_index: benchmark_name}, inplace=True)
        cumprod_df['Maximum Sharpe Ratio Portfolio'] = pfolio_returns_at_max_sharpe_weights
        cumprod_df['Minimum Volatility Portfolio'] = pfolio_returns_at_min_vol_weights
        cumprod_df['Maximum Return Portfolio'] = pfolio_returns_at_max_ret_weights

        # Cumulative returns calculation
        cumprod_df = (1 + cumprod_df).cumprod() - 1

        # Cumulative Returns Plot
        st.subheader("Performance Comparison")
        st.markdown(f"""
            The following plot shows the cumulative performance of each of the portfolios identified, 
            compared to the {benchmark_name} index. 
        """)


        # Line Chart for cumulative returns comparison
        fig = px.line(
            cumprod_df,
            labels={"value": "Cumulative Return", "index": "Date"},
            line_shape='linear'
        )

        fig.update_traces(mode='lines', line=dict(width=2))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template='plotly_white',
            plot_bgcolor='rgba(245, 245, 245, 0.85)',
            hovermode='x unified',
            hoverlabel=dict(bgcolor="white", font_size=12),
            legend=dict(
                x=0,                 # Move to left side
                y=1,                 # Move to top
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)',  # Optional: background color
                bordercolor='Black', # Optional: border color
                borderwidth=1        # Optional: border width
            )
        )


        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
       st.markdown("The table and heatmap below shows the CAGR, Volatility, Sharpe Ratio as well as correlation between constituent assets of the portfolio:")
       generate_asset_and_portfolio_tables(
        assets_list, 
        returns_assets_ann, 
        returns_assets, 
        risk_free_rate, 
        optimal_weights, 
        weights, 
        pfolio_return, 
        pfolio_volatility, 
        sharpe_ratios, 
        returns_all_corr,
        returns_assets_cov
    ) 

import seaborn as sns
def generate_asset_and_portfolio_tables(
    assets_list, 
    returns_assets_ann, 
    returns_assets, 
    risk_free_rate, 
    optimal_weights, 
    weights, 
    pfolio_return, 
    pfolio_volatility, 
    sharpe_ratios, 
    returns_all_corr,
    returns_assets_cov
):
    # Prepare data for Table 1: Return, Volatility, Sharpe Ratio of Each Asset
    table1_data = {
        "Metric": ["Average Return", "Average Volatility", "Sharpe Ratio"]
    }
    
    for ticker in assets_list:
        asset_return = returns_assets_ann[ticker].mean()
        asset_volatility = returns_assets[ticker].std() * (250 ** 0.5)
        asset_sharpe = (asset_return - risk_free_rate) / asset_volatility

        table1_data[ticker] = [
            f"{asset_return * 100:.2f}%",  # Average Return in percentage
            f"{asset_volatility * 100:.2f}%",  # Average Volatility in percentage
            f"{asset_sharpe:.4f}",  # Sharpe Ratio
        ]

    # Display Table 1
    st.subheader("Asset Metrics")
    table1_df = pd.DataFrame(table1_data)
    st.table(table1_df.set_index("Metric"))

    # Correlation Heatmap
    st.subheader("Asset Correlation Heatmap:")
    fig = plt.figure(figsize=(15, 8))
    sns.heatmap(returns_all_corr, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)  # Display the heatmap in Streamlit






