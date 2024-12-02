import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tabulate import tabulate
import yfinance as yf
from fredapi import Fred
import cvxpy as cp
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st
import utils

# Header section
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>Portfolio Optimization Model</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size: 18px;'>This is a Python-based application for analyzing and optimizing financial portfolios. "
    "It uses modern portfolio theory to identify optimal allocations and visualize performances.</p>", unsafe_allow_html=True
)

# Separator line 
st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)

# Instructional text
st.markdown(
    "<h3>1. Choose an option to proceed:</h3>", unsafe_allow_html=True
)

# Create buttons for random and custom inputs in three columns
col1, col3 = st.columns(2)

with col1:
    #st.markdown("<h4 style='text-align:center;'>Use Random Inputs</h4>", unsafe_allow_html=True)
    random_inputs = st.button("Test with Random Inputs")
    st.markdown("<p>Run a test with default inputs</p>", unsafe_allow_html=True)
    
#with col2:
    #st.markdown('<p style="text-align: center; font-size: 24px;">or</p>', unsafe_allow_html=True)

with col3:
    #st.markdown("<h4 style='text-align:center;'>Use Custom Inputs</h4>", unsafe_allow_html=True)
    custom_inputs = st.button("Use Custom Inputs")
    st.markdown("<p>Use customized inputs</p>", unsafe_allow_html=True)
    
# Button functionality
if random_inputs:
    assets_list, benchmark_index, risk_free_rate, min_return = utils.generate_random_inputs()
elif custom_inputs:
    assets_list, benchmark_index, risk_free_rate, min_return = utils.custom_inputs()

# def input_assets():
#     """
#     Prompts the user to input stock tickers, with an option to use default values, and validates the entered tickers. 
#     Returns a list of valid tickers and an integer indicating chosen input type (0 for default inputs, 1 for custom inputs).
#     """

#     # Option to test the model with default values
#     user_input = st.text_input("Press 'Enter' or 'Esc' to run the model with default inputs, or any other key to proceed with custom settings: ")

#     # test assets list to use if user chooses to use default inputs
#     test_assets = [
#         'AAPL',  # Apple (Technology, U.S. stock)
#         'VTI',   # Vanguard Total Stock Market ETF (Broad U.S. stock market exposure)
#         'VXUS',  # Vanguard Total International Stock ETF (International stocks)
#         'BND',   # Vanguard Total Bond Market ETF (U.S. Bonds)
#         'XLE',   # Energy Select Sector SPDR Fund (Energy sector)
#         'GLD',   # SPDR Gold Shares (Gold ETF)
#         'VNQ'    # Vanguard Real Estate ETF (Real Estate sector)
#     ]

#     if user_input == "" or user_input.lower() == "esc":
#         # Run model with default inputs
#         st.write("Running the model with default inputs:\n________________________________________________________________________\n")
#         assets = test_assets
#         st.write(f"Tickers entered: {'  '.join(assets)}", end="")
#         st.write("\n\nProcessing tickers......", end="")
#         st.write("100%")
#         default = 0
#         return assets, default
#     else:
#         # Proceed with custom settings
#         default = 1
#         # Prompt user input
#         user_input = st.text_input("Enter stock tickers (separate by commas): ")
#         input_split = user_input.split(',')
#         assets_list = [ticker.strip().upper() for ticker in input_split]
    
#     # Handle if empty or has invalid characters
#     try:
#         if not user_input.strip():
#             raise ValueError("No tickers were entered.")
#     except ValueError as e:
#         st.write(f"Critical Error:\n---------------\n{e}")
#         return [], default  # Return empty list instead of None

#     # Validate the tickers
#     st.write(f"Tickers entered: {', '.join(assets_list)}\n\nProcessing tickers......", end="")

#     assets = []
#     invalid_tickers = []

#     for ticker in assets_list:
#         try:
#             stock = yf.Ticker(ticker)
#             stock_data = stock.history(period="1mo")
#             if stock_data.empty:
#                 invalid_tickers.append(ticker)
#             else:
#                 assets.append(ticker)
#         except Exception:
#             invalid_tickers.append(ticker)

#     # If there are invalid tickers
#     if invalid_tickers:
#         st.write(f"\nCritical Error:\n---------------\nyfinance could not retrieve data for the following tickers: {', '.join(invalid_tickers)}.\n\nPlease check the tickers and try again.\ne.g., AAPL, WMT, GOOG")
#         return [], default  # Return empty list instead of None

#     st.write("100%")
#     return assets, default

# def select_benchmark():
#     """
#     Prompts the user to select a benchmark index from a list of valid indexes.
#     Returns the selected index ticker as a string (default is ^GSPC).
#     """

#     valid_indexes = ['^GSPC', '^DJI', '^IXIC']
    
#     benchmark_index = st.selectbox("Select index to benchmark the portfolio:", valid_indexes)

#     return benchmark_index

# def retrieve_risk_free_rate():
#     """
#     Prompts the user for a FRED API key to retrieve the latest 10-year Treasury yield as the risk-free rate (default is 4.1%). 
#     Returns a float representing the risk-free rate.
#     """
#     # Input prompt
#     api_key = st.text_input("Enter FRED API key (Press Enter to skip): ").strip()
    
#     # Handle empty input
#     if not api_key or api_key.lower() == "esc":
#         st.write("\nWarning:\n--------\nNo API key provided. Default risk-free rate of 4.1% applied.")
#         return 0.041
    
#     # Handle non-empty input
#     try:
#         # Initialize the FRED API with the provided API key
#         fred = Fred(api_key=api_key)
#         ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100

#         # Handle error
#         if ten_year_treasury_rate is None or ten_year_treasury_rate.empty:
#             st.write("\nWarning:\n--------\nCould not retrieve valid data. Default risk-free rate of 4.1% applied.")
#             return 0.041
        
#         return ten_year_treasury_rate.iloc[-1]
    
#     except Exception as e:
#         st.write(f"\nError:\n------\nAn error occurred while retrieving the risk-free rate.\nDetails: {e}")
#         st.write("\nDefault rate of 4.1% applied.")
#         return 0.041

# def target_return():
#     # Input prompt
#     return_input = st.slider(f"Enter desired target return for convex optimization (%):", min_value=2.0, max_value=100.0, value=5.0)

#     min_return_value = return_input

#     return min_return_value

def retrieve_data(assets, risk_free_rate, benchmark_index, no_of_years):
    """
    Retrieves historical adjusted closing prices for the specified assets and benchmark index from Yahoo Finance.
    Returns data as Pandas DataFrames along with the benchmark's name.
    """
    
    # Determine start_date and end_date based on no_of_years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(no_of_years*365))
    
    # Assets data
    st.write("\nDownloading data for the given tickers:")
    adj_close = pd.DataFrame()
    for t in assets:
        adj_close[t] = yf.download(t, start_date, end_date)['Adj Close']
        if adj_close[t].empty:
            raise ValueError(f"No data available for asset: {t}")
    
    # Benchmark index data
    benchmark_df = pd.DataFrame()
    benchmark_df[benchmark_index] = yf.download(benchmark_index, start_date, end_date)['Adj Close']
    if benchmark_df.empty:
        raise ValueError(f"No data available for benchmark index: {benchmark_index}")
    
    # Retrieve long names of assets 
    asset_names = []
    for t in assets:
        t = yf.Ticker(t)
        company_name = t.info['longName']
        asset_names.append(company_name)

    # Retrieve long name of benchmark index
    benchmark_ticker = yf.Ticker(benchmark_index)
    benchmark_name = benchmark_ticker.info['longName']

    # Combine asset and benchmark data on common index.
    combined_df = pd.merge(adj_close, benchmark_df, left_index=True, right_index=True)

# Outputs
    st.write('________________________________________________________________________')
    st.write(f'\n\nThe following analysis is based on {no_of_years}Y daily adjusted closing price data from Yahoo Finance.')
    st.write(f'\nTime period of analysis:    {(start_date).strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    st.write(f'Index used as benchmark:    {benchmark_name}')
    st.write(f'Risk free rate used:        {risk_free_rate*100:.2f}%')
    st.write(f'\nAssets analysed:            {"\n                            ".join(asset_names)}')
    st.write('________________________________________________________________________')
        
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
    
# Outputs
    st.write(f'\n\nI. Individual Asset Analysis:\n')
    
    tickers = assets + [benchmark_index]

    # CAGR table (Daily Compounding) 
    table1 = [(ticker, f"{mean*100:.2f}%") for ticker, mean in zip(tickers,returns_all_ann.mean())]
    st.write(tabulate(table1, headers=["Asset",f"{no_of_years}-Year CAGR (Daily Compounding)"]))
    st.write(f"Average (excluding {benchmark_index}): {ann_total_returns_mean.mean()*100:.2f}%")
    st.write()

    # Volatility table
    table2 = [(ticker, f"{(std * 100 * 250 ** 0.5):.2f}%") for ticker, std in zip(tickers, returns_all.std())]
    st.write(tabulate(table2, headers=["Asset", "Annualized Volatility"]))
    st.write(f"Average (excluding {benchmark_index}): {ann_volatility_mean.mean()*100:.2f}%")
    st.write()

    # Correlation heatmap
    sns.heatmap(returns_all_corr, annot = True)
    plt.rcParams['figure.figsize'] = (15,8)
    plt.show()
    
    return returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark

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
    sharpe_ratios = (pfolio_return - risk_free_rate) / pfolio_volatility
    
    return pfolio_volatility, pfolio_return, weights, sharpe_ratios

def opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, min_return):
    """
    Optimizes portfolio allocation using Mean-Variance Optimization with convex optimization (cvxpy).
    Returns floats (or None if the optimization fails).
    """

    # Initialize variables
    n = len(returns_assets_ann.columns)  # Number of assets (columns in the dataframe)
    w = cp.Variable(n)  # Portfolio weights as a cvxpy variable

    st.write(f"\nII. Portfolio Analysis: \n\nA. Mean-Variance Optimization (cvxpy):")

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
    Prints the results of the portfolio optimization if the optimal weights are valid.
    """
    # Return if weights are invalid
    if optimal_weights is None or np.any(np.isnan(optimal_weights)):
        return

    # Calculate stats of portfolio with optimal weights
    optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio = portfolio_stats(
        optimal_weights, 
        returns_assets_ann, 
        returns_assets_cov, 
        risk_free_rate
    )

    # Output the results of the portfolio optimization
    st.write(f"   Portfolio optimized for minimum volatility with a target return of {min_return*100:.2f}%:")

    results = [(ticker, f"{weight:.3f}") for ticker, weight in zip(assets, optimal_weights)]
    st.write(f"\n{tabulate(results, headers=["Asset", "Weight"])}")

    st.write(f"\n   Expected Annual Return: {optimal_portfolio_return * 100:.2f}%")
    st.write(f"   Expected Volatility: {optimal_portfolio_volatility * 100:.2f}%")
    st.write(f"   Sharpe Ratio: {optimal_sharpe_ratio:.4f}")


def visualize_analyses(pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, assets, benchmark_index, no_of_iterations):
    """
    Visualizes portfolio analysis results, including the efficient frontier, identified portfolios, and their relative daily return movements.
    """

# Identify relevant portfolios (Return and volatility values were already multiplied by 100; see eff_frontier()) 
    
    # 1. Identify minimum volatility portfolio
    min_volatility_idx = np.argmin(pfolio_volatility)     # Returns the indices of the lowest value in pfolio_volatility array
    min_volatility = pfolio_volatility[min_volatility_idx]     # Extract the value
    return_at_min_volatility = pfolio_return[min_volatility_idx]     # Extract corresponding return from pfolio_return array
    weights_at_min_volatility = weights[min_volatility_idx]     # Extract corresponding weights from weights array
    sharpe_ratio_at_min_volatility = sharpe_ratios[min_volatility_idx]     # Extract corresponding sharpe ratio from sharpe_ratios array
    
    # 2. Identify maximum return portfolio
    # Repeat the same steps as above
    max_return_idx = np.argmax(pfolio_return)     
    max_return = pfolio_return[max_return_idx]
    vol_at_max_return = pfolio_volatility[max_return_idx]
    weights_at_max_return = weights[max_return_idx]
    sharpe_ratio_at_max_return = sharpe_ratios[max_return_idx]
    
    # 3. Identify maximum sharpe ratio portfolio
    # Repeat the same steps as above
    max_sharpe_idx = np.argmax(sharpe_ratios)     
    max_sharpe = sharpe_ratios[max_sharpe_idx]
    return_at_max_sharpe = pfolio_return[max_sharpe_idx]
    volatility_at_max_sharpe = pfolio_volatility[max_sharpe_idx]
    weights_at_max_sharpe = weights[max_sharpe_idx]
    
    
# Calculate daily returns of each of the above portfolios to plot against benchmark_index:
    
    # 1. Use the weights to calculate daily returns of maximum sharpe ratio portfolio 
    pfolio_returns_at_max_sharpe_weights = returns_assets.values @ weights_at_max_sharpe   
    
    # 2. Use the weights to calculate daily returns of minimum volatility portfolio
    pfolio_returns_at_min_vol_weights = returns_assets @ weights_at_min_volatility
    
    # 3. Use the weights to calculate daily returns of maximum return portfolio
    pfolio_returns_at_max_ret_weights = returns_assets.values @ weights_at_max_return
    
    # 4. Use optimal weights to calculate daily returns of the optimized portfolio
    if optimal_weights is not None and not np.any(np.isnan(optimal_weights)):
        pfolio_returns_at_optimal_weights = returns_assets.values @ optimal_weights

# Compare cumulative product of assets and benchmark
    # 1. Create a dataframe in order to calculate and compare cumulative product of assets and benchmark
    cumprod_df = returns_benchmark.copy()
    cumprod_df.rename(columns={benchmark_index:benchmark_name}, inplace=True)              

    # 2. Insert each of the above portfolio return numpy arrays to this dataframe before calculating cumulative product.
    cumprod_df['Maximum Sharpe ratio Portfolio']  = pfolio_returns_at_max_sharpe_weights
    cumprod_df['Minimum Volatility Portfolio']  = pfolio_returns_at_min_vol_weights
    cumprod_df['Maximum Return Portfolio']  = pfolio_returns_at_max_ret_weights
    if optimal_weights is not None and not np.any(np.isnan(optimal_weights)):
        cumprod_df['Optimized Portfolio']  = pfolio_returns_at_optimal_weights
    
    # 3. Calculate cumulative product of this dataframe 
    cumprod_df = (1 + cumprod_df).cumprod() - 1
    

# Outputs
    st.write('\nB. Markowitz Portfolio Analysis:')
    st.write(f'   Number of iterations: {no_of_iterations}')
    
    # Create a Plotly scatter plot
    fig = go.Figure()

    # Add all portfolios
    fig.add_trace(go.Scatter(
        x=pfolio_volatility, 
        y=pfolio_return, 
        mode='markers', 
        name='All Portfolios',
        marker=dict(color=sharpe_ratios, colorscale='Viridis', size=6, showscale=True),
    ))

    # Annotate key portfolios
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

    # Add layout details
    fig.update_layout(
        title='Efficient Frontier with Key Portfolios',
        xaxis_title='Volatility (%)',
        yaxis_title='Return (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=1000,  
        height=600  
    )

    # Streamlit integration: Display the Plotly chart
    st.plotly_chart(fig)

    # 2. Print each portfolio identified and plot their daily returns against benchmark index 
    # Minimum volatility portfolio
    st.write(f'\nEfficient Frontier Portfolios:\n\n(a) Minimum volatility portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_min_volatility):
        print (f'      {ticker}: {weight:.3f}')
    st.write(f'  2. Portfolio Return: {return_at_min_volatility:.2f}%\n  3. Portfolio Volatility: {min_volatility:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_min_volatility:.4f}\n')

    # Maximum return portfolio
    st.write(f'(b) Maximum return portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_max_return):
        print (f'      {ticker}: {weight:.3f}')
    st.write(f'  2. Expected Annual Return: {max_return:.2f}%\n  3. Expected Volatility: {vol_at_max_return:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_max_return:.4f}\n')
    
    # Maximum sharpe ratio portfolio
    st.write(f'(c) Maximum Sharpe Ratio portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_max_sharpe):
        print (f'      {ticker}: {weight:.3f}')
    st.write(f'  2. Portfolio Return: {return_at_max_sharpe:.2f}%\n  3. Portfolio Volatility: {volatility_at_max_sharpe:.2f}%\n  4. Sharpe Ratio:{max_sharpe:.4f}\n')
    
    # 3. Plot relative daily return movements (cumulative product)
    fig = plt.figure(figsize=(10, 6))
    for column in cumprod_df.columns:
        plt.plot(cumprod_df.index, cumprod_df[column], label=column)

    plt.title(f'Relative Daily Return Movements: Portfolios vs {benchmark_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

def prepare_csv_data(assets, optimal_weights, pfolio_volatility, pfolio_return, sharpe_ratios):
    """
    Prepare DataFrames for optimized portfolio weights and efficient frontier data.
    Returns them as CSV strings.
    """
    # Prepare optimized portfolio weights DataFrame
    if optimal_weights is not None:
        weights_df = pd.DataFrame({
            'Asset': assets,
            'Weight': optimal_weights
        })
        weights_csv = weights_df.to_csv(index=False)
    else:
        weights_csv = None

    # Prepare efficient frontier data DataFrame
    ef_df = pd.DataFrame({
        'Volatility (%)': pfolio_volatility,
        'Return (%)': pfolio_return,
        'Sharpe Ratio': sharpe_ratios
    })
    ef_csv = ef_df.to_csv(index=False)

    return weights_csv, ef_csv

def save_results(assets, optimal_weights, pfolio_volatility, pfolio_return, sharpe_ratios):
    """
    Provides options to download portfolio analysis results as CSV files.
    """
    weights_csv, ef_csv = prepare_csv_data(assets, optimal_weights, pfolio_volatility, pfolio_return, sharpe_ratios)

    if weights_csv:
        st.download_button(
            label="Download Optimized Portfolio Weights CSV",
            data=weights_csv,
            file_name='optimized_portfolio_weights.csv',
            mime='text/csv'
        )

    st.download_button(
        label="Download Efficient Frontier Data CSV",
        data=ef_csv,
        file_name='efficient_frontier.csv',
        mime='text/csv'
    )

def main():
    """
    Main function. Let's go!
    """
# Inputs
    # Step 1: Input Assets
    assets, default = input_assets()

    # Step 2: Validate Assets
    if not assets:
        st.write("No assets entered. Exiting.")
        return  # Exit if no assets are provided

    # Step 3: Set Parameters based on Default or User Input
    if default == 0:                # User opts to use default inputs
        benchmark_index = '^GSPC'   # Default benchmark: S&P 500
        risk_free_rate = 0.041      # Default risk-free rate
        min_return = 0.08           # Default target return
   
    else:                           # User-provided input, set up accordingly                
        benchmark_index = select_benchmark()  # User selects benchmark
        risk_free_rate = retrieve_risk_free_rate()  # Fetch the current risk-free rate
        min_return = target_return()  # User sets the target return

    # Step 4: Set optional parameters for analysis
    no_of_years = 10  # Number of years of historical data to be used for analysis
    no_of_iterations = 1000  # Number of Monte Carlo simulations

# Processes
    # Step 5: Retrieve Historical Data
    adj_close, benchmark_df, combined_df, benchmark_name = retrieve_data(
        assets, risk_free_rate, benchmark_index, no_of_years
    )

    # Step 6: Calculate Returns and Risk Statistics
    returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark = return_stats(
        adj_close, benchmark_df, combined_df, assets, benchmark_index, no_of_years
    )

    # Step 7: Perform Efficient Frontier Analysis
    pfolio_volatility, pfolio_return, weights, sharpe_ratios = eff_frontier(
        assets, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations
    )

    # Step 8: Optimize Portfolio Using Mean-Variance Optimization
    optimal_weights, min_return_valid = opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, min_return)

# Outputs
    # Step 9: Output Results
    opt_portfolio_results(optimal_weights, returns_assets_ann, returns_assets_cov, risk_free_rate, assets, min_return_valid)
    
    # Step 10: Visualize Results
    plt.style.use('bmh')
    visualize_analyses(
        pfolio_volatility, pfolio_return, weights, sharpe_ratios, 
        returns_assets, optimal_weights, returns_benchmark, benchmark_name, 
        assets, benchmark_index, no_of_iterations
    )

# Save outputs (optional)
    save_results(assets, optimal_weights, pfolio_volatility, pfolio_return, sharpe_ratios)

# Run the main function
main()