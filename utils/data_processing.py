# data_processing.py

import streamlit as st
import yfinance as yf
import numpy as np

# Inputs

# Function to generate random inputs
def generate_random_inputs():
    # Sample stock database
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

    # Select random tickers
    size = np.random.randint(3,11)
    assets_list = np.random.choice(list(sample_assets.keys()), size=size, replace=False)  
    
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

    # Check if data exists for the ticker
    for ticker in input_split:
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period="1mo")
            stock_name = stock.info.get("longName")
            if stock_data.empty:  
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
    st.markdown("<h2 style='text-align:center;'><u>Report</u></h2>",
    unsafe_allow_html=True,)
    st.markdown(f'##### The following analysis was conducted on {no_of_years}Y daily adjusted closing price data from Yahoo Finance.')
    st.markdown(f'**Time period of analysis:**    {(start_date).strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    st.markdown(f'**Index used as benchmark:**    {benchmark_name}')
    st.markdown(f'**Risk free rate used:**        {risk_free_rate*100:.2f}%')
    st.markdown(f'**Assets analysed:**           {",  ".join(asset_names)}')
    st.markdown(f'**Techniques used:**           Convex Optimization, Modern Portfolio Theory')
    st.markdown('________________________________________________________________________')
        
    return adj_close, benchmark_df, combined_df, benchmark_name