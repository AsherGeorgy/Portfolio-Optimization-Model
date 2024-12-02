import streamlit as st
import numpy as np

# Function to generate random inputs
def generate_random_inputs():
    # Generate random stock tickers
    sample_assets = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA']
    assets_list = np.random.choice(sample_assets, size=3, replace=False)  # Select 3 random tickers
    st.write("Running the model with random inputs:")
    st.write(f"Selected Tickers: {', '.join(assets_list)}")

    benchmark_index = '^GSPC'   # Default benchmark: S&P 500
    risk_free_rate = 0.041      # Default risk-free rate
    min_return = 0.08           # Default target return

    return assets_list, benchmark_index, risk_free_rate, min_return

# Function for custom inputs
def custom_inputs():
    # Proceed with custom settings
    default = 1
    user_input = st.("Enter stock tickers separate by commas (skip to abort): ").strip()
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




    user_input = st.text_input("Enter stock tickers (separate by commas): ", placeholder="AAPL, MSFT, GOOG")
    input_split = user_input.split(',')
    assets_list = [ticker.strip().upper() for ticker in input_split]
    
    # Handle if empty or has invalid characters
    try:
        if not user_input.strip():
            raise ValueError("No tickers were entered.")
    except ValueError as e:
        st.write(f"Critical Error:\n---------------\n{e}")
        return [] 

    # Validate the tickers
    st.write(f"Tickers entered: {', '.join(assets_list)}\n\nProcessing tickers......", end="")

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
        st.write(f"\nCritical Error:\n---------------\nyfinance could not retrieve data for the following tickers: {', '.join(invalid_tickers)}.\n\nPlease check the tickers and try again.\ne.g., AAPL, WMT, GOOG")
        return [], default  # Return empty list instead of None

    st.write("100%")
    return assets, default
