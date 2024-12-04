import streamlit as st
import yfinance as yf
import numpy as np

# Inputs

# Function to generate random inputs
def generate_random_inputs():
    # Generate random stock tickers
    sample_assets = [
    "AAPL",  # Apple Inc. (Tech stock)
    "MSFT",  # Microsoft Corp. (Tech stock)
    "GOOGL",  # Alphabet Inc. (Tech stock)
    "AMZN",  # Amazon.com Inc. (Consumer Discretionary)
    "TSLA",  # Tesla Inc. (Consumer Discretionary)
    "NVDA",  # NVIDIA Corporation (Tech stock)
    "META",  # Meta Platforms Inc. (Tech stock)
    "NFLX",  # Netflix Inc. (Communication Services)
    "V",  # Visa Inc. (Financials)
    "JPM",  # JPMorgan Chase & Co. (Financials)
    "DIS",  # The Walt Disney Company (Communication Services)
    "KO",  # The Coca-Cola Company (Consumer Staples)
    "PEP",  # PepsiCo, Inc. (Consumer Staples)
    "PFE",  # Pfizer Inc. (Healthcare)
    "JNJ",  # Johnson & Johnson (Healthcare)
    "UNH",  # UnitedHealth Group (Healthcare)
    "WMT",  # Walmart Inc. (Consumer Staples)
    "XOM",  # Exxon Mobil Corporation (Energy)
    "CVX",  # Chevron Corporation (Energy)
    "BA",  # Boeing Company (Industrials)
    "CAT",  # Caterpillar Inc. (Industrials)
    "GM",  # General Motors Company (Consumer Discretionary)
    "IBM",  # International Business Machines (Tech stock)
    "CSCO",  # Cisco Systems, Inc. (Tech stock)
    "SPG",  # Simon Property Group (Real Estate)
    "SLB",  # Schlumberger Limited (Energy)
    "LMT",  # Lockheed Martin Corporation (Aerospace & Defense)
    "GS",  # Goldman Sachs Group, Inc. (Financials)
    "MRK",  # Merck & Co., Inc. (Healthcare)
    "BNS",  # Bank of Nova Scotia (Financials)
    "T",  # AT&T Inc. (Communication Services)
    "VZ",  # Verizon Communications Inc. (Communication Services)
]
    assets_list = np.random.choice(sample_assets, size=10, replace=False)  # Select 5 random tickers
    # Get names
    #asset_names = {'AAPL':'Apple Inc. (AAPL)', 'GOOG':'Alphabet Inc. (GOOG)', 'AMZN':'Amazon.com Inc. (AMZN)', 'MSFT':'Microsoft Corporation (MSFT)', 'TSLA':'Tesla Inc. (TSLA)'}
    #names = [asset_names[asset] for asset in assets_list]
    names = []

    return assets_list, names

def assets(user_input):
    st.markdown("##### Processing input tickers...")
    if not user_input:
        st.error("Input cannot be empty.")
        return None

    # Split and strip tickers, removing empty or whitespace-only entries
    input_split = [ticker.strip().replace("'","").upper() for ticker in user_input.split(',') if ticker.strip()]
    if not input_split:
        st.error("No valid tickers found in input.")
        return None

    invalid_tickers = []
    valid_assets = []
    stock_names = []

    for ticker in input_split:
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period="1mo")
            stock_name = stock.info.get("longName")
            if stock_data.empty:  # Check if data exists for the ticker
                invalid_tickers.append(ticker)
            else:
                valid_assets.append(ticker)
                stock_names.append(stock_name)
        except Exception:
            invalid_tickers.append(ticker)

    if invalid_tickers:
        st.error(f"Invalid ticker found: {', '.join(invalid_tickers)}")
        st.error("Try again with valid tickers!")
        return None
    elif valid_assets:
        st.success(f"Tickers validated: {', '.join(stock_names)}")
        return valid_assets

def run_button(user_input):
    # Validate that there are at least two tickers
    tickers = [ticker.strip() for ticker in user_input.split(',') if ticker.strip()]  
    if len(tickers) < 2:
        st.error("Please enter at least two stock tickers.")
        return None
    else:
        st.session_state.inputs_ready = True
        st.session_state.mode = "custom"
        assets_list = assets(user_input)
        return assets_list

def random_button():
    st.session_state.mode = "random"
    st.session_state.inputs_ready = True  # Skip Run button for default inputs
    assets_list, names = generate_random_inputs()
    # Logic for Random Inputs
    if st.session_state.inputs_ready:
        st.markdown("##### Preparing random tickers...")
        st.success(f"Assets selected: {', '.join(assets_list)}")
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
            st.error(f"No data available for asset: {t}")
            return None, None, None, None
    
    # Benchmark index data
    benchmark_df = pd.DataFrame()
    benchmark_df[benchmark_index] = yf.download(benchmark_index, start=start_date, end=end_date)['Adj Close']
    if benchmark_df.empty:
        st.error(f"No data available for benchmark index: {benchmark_index}")
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
    st.markdown("<h2 style='text-align:center;'><u>Analysis</u></h2>",
    unsafe_allow_html=True,)
    st.markdown(f'##### The following analysis was conducted on {no_of_years}Y daily adjusted closing price data from Yahoo Finance.')
    st.markdown(f'**Time period of analysis:**    {(start_date).strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    st.markdown(f'**Index used as benchmark:**    {benchmark_name}')
    st.markdown(f'**Risk free rate used:**        {risk_free_rate*100:.2f}%')
    st.markdown(f'**Assets analysed:**           {",  ".join(asset_names)}')
    st.markdown('________________________________________________________________________')
        
    # Display the dataframes using Streamlit
    # st.subheader("Adjusted Close Data for Assets and Benchmark:")
    # st.dataframe(combined_df)
        
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

def opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, min_return):
    """
    Optimizes portfolio allocation using Mean-Variance Optimization with convex optimization (cvxpy).
    Returns floats (or None if the optimization fails).
    """

    # Initialize variables
    n = len(returns_assets_ann.columns)  # Number of assets (columns in the dataframe)
    w = cp.Variable(n)  # Portfolio weights as a cvxpy variable

    #st.write(f"\nII. Portfolio Analysis: \n\nA. Mean-Variance Optimization (cvxpy):")

    # Check if the input min_return is feasible (it needs to be equal to or lower than the max average return of individual assets)
    max_expected_return = returns_assets_ann.mean().max()
    if min_return > max_expected_return:
        st.write(f"\nWarning: The specified minimum return of {min_return*100:.2f}% exceeds the maximum feasible return of {max_expected_return*100:.2f}%.")
        st.write("Adjusting the target return to the maximum feasible value to proceed with the optimization.")
        min_return_valid = max_expected_return  # Adjust min_return to the highest feasible value
    else:
        min_return_valid = min_return
    
    # Calculate expected Return and Risk
    ret = returns_assets_ann.mean().values @ w  # Expected return: dot product of weights and asset returns ("@" is the matrix multiplication operator)
    risk = cp.quad_form(w, returns_assets_cov.values)  # Portfolio risk: computes a quadratic form giving the variance
    
    # Define the objective: Minimize portfolio risk
    objective = cp.Minimize(risk)
    
    # Define the constraints: 
    constraints = [
        cp.sum(w) == 1,  # Sum of weights equals 1 (fully invested)
        ret >= min_return_valid,  # Target minimum return constraint
        w >= 0  # Non-negative weights (no short positions)
    ]
    
    # CVXPY solver
    prob = cp.Problem(objective, constraints)  # The optimization problem
    prob.solve()  
    optimal_weights = w.value
    
    # Check if the optimization was successful and the optimal weights are valid
    if prob.status != cp.OPTIMAL or optimal_weights is None or any(np.isnan(optimal_weights)):
        st.write(f"Optimization failed. Solver status: {prob.status} or Invalid portfolio weights.")
        return None, min_return_valid  

    return optimal_weights, min_return_valid


def opt_portfolio_results(optimal_weights, returns_assets_ann, returns_assets_cov, risk_free_rate, assets, min_return):
    """
    Displays the results of the portfolio optimization if the optimal weights are valid.
    """
    # Return if weights are invalid
    if optimal_weights is None or np.any(np.isnan(optimal_weights)):
        st.warning("Optimal weights are invalid. Please check your inputs.")
        return

    # Calculate stats of portfolio with optimal weights
    # Portfolio Return
    optimal_portfolio_return = np.dot(optimal_weights, returns_assets_ann.mean())
    
    # Portfolio Volatility
    optimal_portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns_assets_cov, optimal_weights)))
    
    # Portfolio Sharpe Ratio
    optimal_sharpe_ratio = (optimal_portfolio_return - risk_free_rate) / optimal_portfolio_volatility

    # Display the portfolio optimization results
    # st.subheader(f"Portfolio optimized for minimum volatility with a target return of {min_return * 100:.2f}%:")

    # # Create a DataFrame for weights
    # results_df = pd.DataFrame({
    #     "Asset": assets,
    #     "Weight": [f"{weight:.3f}" for weight in optimal_weights]
    # })

    # # Display the results table
    # st.table(results_df)

    # Display portfolio statistics
    # st.text(f"Expected Annual Return: {optimal_portfolio_return * 100:.2f}%")
    # st.text(f"Expected Volatility: {optimal_portfolio_volatility * 100:.2f}%")
    # st.text(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")


import plotly.graph_objects as go
import matplotlib.pyplot as plt

def visualize_analyses(pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, assets, benchmark_index, no_of_iterations):
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
    
    if optimal_weights is not None and not np.any(np.isnan(optimal_weights)):
        pfolio_returns_at_optimal_weights = returns_assets.values @ optimal_weights

    # Compare cumulative product of assets and benchmark
    cumprod_df = returns_benchmark.copy()
    cumprod_df.rename(columns={benchmark_index: benchmark_name}, inplace=True)
    cumprod_df['Maximum Sharpe Ratio Portfolio'] = pfolio_returns_at_max_sharpe_weights
    cumprod_df['Minimum Volatility Portfolio'] = pfolio_returns_at_min_vol_weights
    cumprod_df['Maximum Return Portfolio'] = pfolio_returns_at_max_ret_weights
    
    if optimal_weights is not None and not np.any(np.isnan(optimal_weights)):
        cumprod_df['Optimized Portfolio'] = pfolio_returns_at_optimal_weights

    cumprod_df = (1 + cumprod_df).cumprod() - 1

    # Outputs
    st.subheader("B. Markowitz Portfolio Analysis")
    st.write(f"Number of iterations: {no_of_iterations}")

    # Plot the efficient frontier
    st.subheader("Efficient Frontier with Key Portfolios")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pfolio_volatility,
        y=pfolio_return,
        mode='markers',
        name='All Portfolios',
        marker=dict(color=sharpe_ratios, colorscale='Viridis', size=6, showscale=True),
    ))
    fig.add_trace(go.Scatter(
        x=[pfolio_volatility[min_volatility_idx]],
        y=[pfolio_return[min_volatility_idx]],
        mode='markers+text',
        name='Min Volatility',
        marker=dict(color='green', size=10),
        text='Min Volatility',
        textposition='top center'
    ))
    fig.add_trace(go.Scatter(
        x=[pfolio_volatility[max_return_idx]],
        y=[pfolio_return[max_return_idx]],
        mode='markers+text',
        name='Max Return',
        marker=dict(color='blue', size=10),
        text='Max Return',
        textposition='top center'
    ))
    fig.add_trace(go.Scatter(
        x=[pfolio_volatility[max_sharpe_idx]],
        y=[pfolio_return[max_sharpe_idx]],
        mode='markers+text',
        name='Max Sharpe Ratio',
        marker=dict(color='red', size=10),
        text='Max Sharpe',
        textposition='top center'
    ))

    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (%)',
        yaxis_title='Return (%)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig)

    # Display portfolio stats
    # portfolios = [
    #     ("Minimum Volatility Portfolio", weights[min_volatility_idx], pfolio_return[min_volatility_idx], pfolio_volatility[min_volatility_idx], sharpe_ratios[min_volatility_idx]),
    #     ("Maximum Return Portfolio", weights[max_return_idx], pfolio_return[max_return_idx], pfolio_volatility[max_return_idx], sharpe_ratios[max_return_idx]),
    #     ("Maximum Sharpe Ratio Portfolio", weights[max_sharpe_idx], pfolio_return[max_sharpe_idx], pfolio_volatility[max_sharpe_idx], sharpe_ratios[max_sharpe_idx]),
    # ]
    # if optimal_weights is not None and not np.any(np.isnan(optimal_weights)):
    #     portfolios.append(("Optimized Portfolio", optimal_weights, None, None, None))

    # for name, weights, ret, vol, sharpe in portfolios:
    #     st.subheader(name)
    #     weights_df = pd.DataFrame({"Asset": assets, "Weight": weights})
    #     st.write(weights_df)
    #     if ret is not None and vol is not None and sharpe is not None:
    #         st.write(f"Portfolio Return: {ret:.2f}%")
    #         st.write(f"Portfolio Volatility: {vol:.2f}%")
    #         st.write(f"Sharpe Ratio: {sharpe:.4f}")

    # Plot cumulative returns
    st.subheader("Cumulative Returns Comparison")
    st.line_chart(cumprod_df)




