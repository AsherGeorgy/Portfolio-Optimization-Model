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

def input_assets():
    """
    Prompts the user to input stock tickers, with an option to use default values, and validates the entered tickers. 
    Returns a list of valid tickers and an integer indicating chosen input type (0 for default inputs, 1 for custom inputs).
    """

    # Option to test the model with default values
    while True:
        user_input = input("Enter 'y' to customize the settings (skip to use default settings): ").lower().strip()
        if user_input in ['y','n','','esc']:
            break

    # test assets list to use if user chooses to use default inputs
    test_assets = [
        'AAPL',  # Apple (Technology, U.S. stock)
        'VTI',   # Vanguard Total Stock Market ETF (Broad U.S. stock market exposure)
        'VXUS',  # Vanguard Total International Stock ETF (International stocks)
        'BND',   # Vanguard Total Bond Market ETF (U.S. Bonds)
        'XLE',   # Energy Select Sector SPDR Fund (Energy sector)
        'GLD',   # SPDR Gold Shares (Gold ETF)
        'VNQ'    # Vanguard Real Estate ETF (Real Estate sector)
    ]

    if user_input == "" or user_input in ['n', "esc"]:
        # Run model with default inputs
        print("Running the model with default inputs:\n________________________________________________________________________\n")
        assets = test_assets
        print(f"Tickers entered: {'  '.join(assets)}", end="")
        print("\n\nProcessing tickers......", end="")
        print("100%")
        default = 0
        return assets, default
    elif user_input == 'y':
        # Proceed with custom settings
        default = 1
        # Prompt user input
        user_input = input("Enter stock tickers separate by commas (skip to abort): ").strip()
        input_split = user_input.split(',')
        assets_list = [ticker.upper() for ticker in input_split]
    
    # Handle if empty or has invalid characters
    try:
        if not user_input.strip():
            raise ValueError("No tickers were entered. Process aborted.")
    except ValueError as e:
        print(f"Critical Error:\n---------------\n{e}")
        return [], default  

    # Validate the tickers
    print(f"Tickers entered: {', '.join(assets_list)}\n\nProcessing tickers......", end="")

    assets = []
    invalid_tickers = []

    for ticker in assets_list:
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period="1mo")
            if stock_data.empty:
                invalid_tickers.append(ticker)
            else:
                assets.append(ticker)
        except Exception:
            invalid_tickers.append(ticker)

    # If there are invalid tickers
    if invalid_tickers:
        print(f"\nCritical Error:\n---------------\nyfinance could not retrieve data for the following tickers: {', '.join(invalid_tickers)}.\n\nPlease check the tickers and try again.\ne.g., AAPL, WMT, GOOG")
        return [], default  

    print("100%")
    return assets, default

def select_benchmark():
    """
    Prompts the user to select a benchmark index from a list of valid indexes.
    Returns the selected index ticker as a string (default is ^GSPC).
    """

    valid_indexes = ['^GSPC', '^DJI', '^IXIC']
    
    # Input prompt
    input_index = input("Select benchmark index: ^GSPC = 0; ^DJI = 1; ^IXIC = 2: ").strip()

    # Handle empty input
    if not input_index or input_index.lower() == "esc":
        print("Warning:\n--------\nNo input provided. Default benchmark (^GSPC) applied.")
        return valid_indexes[0]

    # Handle non-empty inputs
    try:
        input_index = int(input_index)
        if 0 <= input_index <= 2:
            return valid_indexes[input_index]
        else:
            print("Warning:\n--------\nInvalid selection. Default benchmark (^GSPC) applied.")
            return valid_indexes[0]
    except ValueError:
        print("Warning:\n--------\nInvalid input. Default benchmark (^GSPC) applied.")
        return valid_indexes[0]

def retrieve_risk_free_rate():
    """
    Prompts the user for a FRED API key to retrieve the latest 10-year Treasury yield as the risk-free rate (default is 4.1%). 
    Returns a float representing the risk-free rate.
    """
    # Input prompt
    api_key = input("Enter FRED API key (Press Enter to skip): ").strip()
    
    # Handle empty input
    if not api_key or api_key.lower() == "esc":
        print("\nWarning:\n--------\nNo API key provided. Default risk-free rate of 4.1% applied.")
        return 0.041
    
    # Handle non-empty input
    try:
        # Initialize the FRED API with the provided API key
        fred = Fred(api_key=api_key)
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100

        # Handle error
        if ten_year_treasury_rate is None or ten_year_treasury_rate.empty:
            print("\nWarning:\n--------\nCould not retrieve valid data. Default risk-free rate of 4.1% applied.")
            return 0.041
        
        return ten_year_treasury_rate.iloc[-1]
    
    except Exception as e:
        print(f"\nError:\n------\nAn error occurred while retrieving the risk-free rate.\nDetails: {e}")
        print("\nDefault rate of 4.1% applied.")
        return 0.041

def target_return():
    """
    Prompts the user for a desired target return for optimization (default is 5%). 
    Returns the value as a float.
    """

    default_return = 0.08

    # Input prompt
    return_input = input(f"Enter desired minimum return for convex optimization (default is {default_return * 100}%): ").strip()

    # Handle empty input
    if not return_input or return_input.lower() == "esc":
        print(f"Using default target return of {default_return * 100}%.")
        return default_return

    # Handle non-empty input
    def convert_return(value):      
        """
        Convert the input to a valid return value
        """
        try:
            if value.endswith('%'):
                return float(value[:-1]) / 100  # Remove '%' and convert to decimal
            return float(value)  
        except ValueError:
            return None  

    # Convert input to float
    min_return_value = convert_return(return_input)

    # Validate the result
    if min_return_value is None or min_return_value <= 0:
        print(f"Warning: Invalid or non-positive input. Using default target return of {default_return * 100}%.")
        return default_return

    return min_return_value

def retrieve_data(assets, risk_free_rate, benchmark_index, no_of_years):
    """
    Retrieves historical adjusted closing prices for the specified assets and benchmark index from Yahoo Finance.
    Returns data as Pandas DataFrames along with the benchmark's name.
    """
    
    # Determine start_date and end_date based on no_of_years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(no_of_years*365))
    
    # Assets data
    print("\nDownloading data for the given tickers:")
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
    print('________________________________________________________________________')
    print(f'\n\nThe following analysis is based on {no_of_years}Y daily adjusted closing price data from Yahoo Finance.')
    print(f'\nTime period of analysis:    {(start_date).strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    print(f'Index used as benchmark:    {benchmark_name}')
    print(f'Risk free rate used:        {risk_free_rate*100:.2f}%')
    print(f'\nAssets analysed:            {"\n                            ".join(asset_names)}')
    print('________________________________________________________________________')
        
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
    print(f'\n\nI. Individual Asset Analysis:\n')
    
    tickers = assets + [benchmark_index]

    # CAGR table (Daily Compounding) 
    table1 = [(ticker, f"{mean*100:.2f}%") for ticker, mean in zip(tickers,returns_all_ann.mean())]
    print(tabulate(table1, headers=["Asset",f"{no_of_years}-Year CAGR (Daily Compounding)"]))
    print(f"Average (excluding {benchmark_index}): {ann_total_returns_mean.mean()*100:.2f}%")
    print()

    # Volatility table
    table2 = [(ticker, f"{(std * 100 * 250 ** 0.5):.2f}%") for ticker, std in zip(tickers, returns_all.std())]
    print(tabulate(table2, headers=["Asset", "Annualized Volatility"]))
    print(f"Average (excluding {benchmark_index}): {ann_volatility_mean.mean()*100:.2f}%")
    print()

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

    print(f"\nII. Portfolio Analysis: \n\nA. Mean-Variance Optimization (cvxpy):")

    # Check if the input min_return is feasible (it needs to be equal to or lower than the max average return of individual assets)
    max_expected_return = returns_assets_ann.mean().max()
    if min_return > max_expected_return:
        print(f"\nWarning: The specified minimum return of {min_return*100:.2f}% exceeds the maximum feasible return of {max_expected_return*100:.2f}%.")
        print("Adjusting the target return to the maximum feasible value to proceed with the optimization.")
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
        print(f"Optimization failed. Solver status: {prob.status} or Invalid portfolio weights.")
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
    print(f"   Portfolio optimized for minimum volatility with a target return of {min_return*100:.2f}%:")

    results = [(ticker, f"{weight:.3f}") for ticker, weight in zip(assets, optimal_weights)]
    print(f"\n{tabulate(results, headers=["Asset", "Weight"])}")

    print(f"\n   Expected Annual Return: {optimal_portfolio_return * 100:.2f}%")
    print(f"   Expected Volatility: {optimal_portfolio_volatility * 100:.2f}%")
    print(f"   Sharpe Ratio: {optimal_sharpe_ratio:.4f}")


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
    print('\nB. Markowitz Portfolio Analysis:')
    print(f'   Number of iterations: {no_of_iterations}')
    
    # 1. Plot the efficient frontier curve:
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

    fig.show()

    # 2. Print each portfolio identified and plot their daily returns against benchmark index 
    # Minimum volatility portfolio
    print(f'\nEfficient Frontier Portfolios:\n\n(a) Minimum volatility portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_min_volatility):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Portfolio Return: {return_at_min_volatility:.2f}%\n  3. Portfolio Volatility: {min_volatility:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_min_volatility:.4f}\n')

    # Maximum return portfolio
    print(f'(b) Maximum return portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_max_return):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Expected Annual Return: {max_return:.2f}%\n  3. Expected Volatility: {vol_at_max_return:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_max_return:.4f}\n')
    
    # Maximum sharpe ratio portfolio
    print(f'(c) Maximum Sharpe Ratio portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_max_sharpe):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Portfolio Return: {return_at_max_sharpe:.2f}%\n  3. Portfolio Volatility: {volatility_at_max_sharpe:.2f}%\n  4. Sharpe Ratio:{max_sharpe:.4f}\n')
    
    # 3. Plot relative daily return movements (cumulative product)
    plt.figure(figsize=(10, 6))
    for column in cumprod_df.columns:
        plt.plot(cumprod_df.index, cumprod_df[column], label=column)

    plt.title(f'Relative Daily Return Movements: Portfolios vs {benchmark_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_csv(data, file_path):
    """
    Save a DataFrame or array to a CSV file.
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


def prepare_and_save_csvs(
    save_folder,
    assets,
    optimal_weights,
    pfolio_volatility,
    pfolio_return,
    sharpe_ratios
):
    """
    Save only the optimized portfolio weights and efficient frontier data to CSVs.
    """
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    try:
        # Save optimized portfolio weights
        if optimal_weights is not None:
            weights_df = pd.DataFrame({
                'Asset': assets,
                'Weight': optimal_weights
            })
            save_csv(weights_df, os.path.join(save_folder, "optimized_portfolio_weights.csv"))

        # Save efficient frontier data
        ef_df = pd.DataFrame({
            'Volatility (%)': pfolio_volatility,
            'Return (%)': pfolio_return,
            'Sharpe Ratio': sharpe_ratios
        })
        save_csv(ef_df, os.path.join(save_folder, "efficient_frontier.csv"))

        if save_folder == "./":
            print(f"CSV files saved successfully in root directory.")
        else:
            print(f"CSV files saved successfully in '{save_folder}'.")
    except Exception as e:
        print(f"Error during CSV save operation: {e}")

def save_results(assets, optimal_weights, pfolio_volatility, pfolio_return, sharpe_ratios): 
    """
    Prompts the user to save portfolio analysis results as CSV files and calls a helper function to save the data if confirmed.
    """
    save = input("\nDo you want to save the results as CSV files? (y/n): ").strip().lower()
    if save == 'y':
        save_path = input("Enter the folder path to save results (or press Enter to save to the current directory): ").strip()
        if save_path == "" or save_path.lower() == "esc":
            save_folder = "./"
        else:
            save_folder = save_path
        
        prepare_and_save_csvs(
            save_folder, assets, optimal_weights, pfolio_volatility,
            pfolio_return, sharpe_ratios
        )
    else:
        print("Results not saved.")

def main():
    """
    Main function. Let's go!
    """
# Inputs
    # Step 1: Input Assets
    assets, default = input_assets()

    # Step 2: Validate Assets
    if not assets:
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