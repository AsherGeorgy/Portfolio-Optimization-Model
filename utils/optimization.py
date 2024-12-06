# optimization.py

import streamlit as st
import numpy as np
import cvxpy as cp

def return_stats(adj_close, benchmark_df, combined_df):
    """
    Calculates return statistics (annualized returns, volatility, covariance, and correlation) for assets and a benchmark.
    Returns pd DataFrames.
    """
    # Calculate simple returns and covariance of the assets
    returns_assets = adj_close.pct_change().dropna()
    returns_assets_ann = returns_assets * 250   # annualised
    returns_assets_cov = returns_assets.cov() * 250
    
    # Calculate simple returns of the benchmark
    returns_benchmark = benchmark_df.pct_change().dropna()

    # Create a df with both assets and benchmark combined 
    returns_all = combined_df.pct_change().dropna()
    returns_all_corr = returns_all.corr()
    
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

    # Check if the input target_cagr is feasible (it needs to be lower than the max average return of individual assets and higher than the min average return)
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

