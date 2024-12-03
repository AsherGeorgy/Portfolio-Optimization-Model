import streamlit as st
import yfinance as yf
import numpy as np
from fredapi import Fred

# Function to generate random inputs
def generate_random_inputs():
    # Generate random stock tickers
    sample_assets = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA']
    assets_list = np.random.choice(sample_assets, size=3, replace=False)  # Select 3 random tickers
    # Get names
    asset_names = {'AAPL':'Apple Inc.', 'GOOG':'Alphabet Inc.', 'AMZN':'Amazon.com, Inc.', 'MSFT':'Microsoft Corporation', 'TSLA':'Tesla Inc.'}
    names = [asset_names[asset] for asset in assets_list]

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
        st.success(f"Assets validated: {', '.join(stock_names)}")
        return valid_assets


def risk_free_rate(api_key):
    if api_key == 'No':
        st.write("Default risk-free rate of 4.1% applied.")
        return 0.041
    
    # Handle non-empty input
    try:
        # Initialize the FRED API with the provided API key
        fred = Fred(api_key=api_key)
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100

        # Handle error
        if ten_year_treasury_rate is None or ten_year_treasury_rate.empty:
            st.write("\nWarning:\n--------\nCould not retrieve valid data from FRED. Default risk-free rate of 4.1% applied.")
            return 0.041
        
        return ten_year_treasury_rate.iloc[-1]
    
    except Exception as e:
        st.write(f"\nError:\n------\nAn error occurred while retrieving the risk-free rate.\nDetails: {e}")
        st.write("\nDefault rate of 4.1% applied.")
        return 0.041



