# Portfolio Optimization Model

## Overview
A Python-based application for analyzing and optimizing financial portfolios using **Modern Portfolio Theory** (MPT) and **convex optimization**. This tool allows users to configure their portfolio, analyze risk and returns, and visualize key metrics such as the Efficient Frontier, portfolio allocations, and performance comparisons.

## Features

- **Portfolio Configuration**: Input custom or random tickers, select benchmark indices, and define target metrics (CAGR, risk-free rate).
- **Data Processing**: Retrieves and validates historical data from Yahoo Finance, calculating key metrics like returns, covariance, and correlation.
- **Portfolio Optimization**: Uses convex optimization to allocate weights for target returns, identifying key portfolios (e.g., Maximum Sharpe Ratio, Minimum Volatility).
- **Visualization**: Plots Efficient Frontier, portfolio performance, cumulative return comparisons, and asset correlation heatmaps.
- **User Interface**: Intuitive, clean UI built with [Streamlit](https://streamlit.io/), featuring a sidebar for developer info and disclaimers.


## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/ashergeorgy/portfolio-optimization-tool.git
cd portfolio-optimization-tool
```
### 2. Install dependencies:
```bash
pip install -r requirements.txt
```
### 3. Run the app:
```bash
streamlit run app.py
```

## How to Use

### Configure Your Portfolio:
- Input stock tickers (`e.g., aapl, msft, tsla`) or use the "Test with random tickers" button.
- Adjust settings such as target CAGR, benchmark index, analysis period, and risk-free rate.

### Run Optimization:
- Click "Run" to perform portfolio analysis and optimization.
- View the results, including optimized weights, key portfolios, and performance charts.

### Explore Results:
- Analyze the Efficient Frontier, asset allocations, and correlation heatmaps.
- Compare the performance of your portfolio against the selected benchmark.

## Demo
![Interactive Eff Frontier](sample_outputs/Efficient%20Portfolios%20Backtest.png)

## Technologies Used
- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Sources:** Yahoo Finance (via yfinance)
- **Optimization Library:** CVXPY
- **Visualization:** Plotly, Seaborn, Matplotlib

## Project Structure
```bash
Portfolio-Optimization-Model/
│
├── .devcontainer/                # Dev container configuration 
│   └── devcontainer.json
│
├── .streamlit/                   # Streamlit-specific configuration
│   └── config.toml
│
├── app.py                        # Main application script 
│
├── legacy/                       # Legacy implementation
│   └── main.py
│
├── utils/                         # Utility scripts for data processing, optimization, and visualization
│   ├── data_processing.py
│   ├── optimization.py
│   └── visualization.py
│
├── sample_outputs/               # Folder for sample outputs and results (CSV, images)
│   ├── Constituent Asset Correlation.png
│   ├── Constituent Asset Metrics.csv
│   ├── Efficient Frontier.png
│   ├── Efficient Portfolios Backtest.png
│   ├── Optimized Portfolio Backtest.png
│   ├── Optimized Portfolio Weights.csv
│   └── Portfolio Metrics.csv
│
├── .gitignore                    # Specifies files to ignore in version control
├── LICENSE                       # License file 
├── README.md                     # Project documentation and instructions
├── requirements.txt              # Python dependencies for the project
└── README.md                     # Project documentation and instructions
```

## Disclaimers
This application is for educational and informational purposes only. The analysis and results provided are based on historical data and theoretical models. They do not predict future performance. Users should conduct their own due diligence and consult with a qualified financial advisor before making any investment decisions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
