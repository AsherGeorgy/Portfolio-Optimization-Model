# Portfolio Optimization Tool

## Overview
The **Portfolio Optimization Tool** is a Python-based application for analyzing and optimizing financial portfolios. Portfolio optimization is the process of selecting the best distribution of investments to maximize returns for a given level of risk, based on historical data. This tool uses modern portfolio theory to identify optimal allocations and visualize performance.

Key features include:
- Mean-Variance Optimization using convex programming.
- Monte Carlo simulations for efficient frontier analysis.
- Interactive visualizations with Plotly.

## Features
- **Asset Analysis**: Calculate returns, volatility, and correlations for selected assets.
- **Efficient Frontier**: Simulate thousands of portfolio combinations to identify optimal allocations.
- **Portfolio Optimization**: Optimize weights for minimum risk or target return.
- **Benchmark Comparison**: Analyze portfolio performance relative to a chosen benchmark.

## Requirements
- Python 3.7 or higher
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `plotly`
  - `seaborn`
  - `yfinance`
  - `fredapi`
  - `cvxpy`
  - `tabulate`

Install dependencies using:
```bash
pip install -r requirements.txt
```
## Installation
1. Clone the repository:
```bash
git clone https://github.com/ashergeorgy/portfolio-optimization-tool.git
cd portfolio-optimization-tool
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the tool with:

```bash
python main.py
```

## Steps:
1. **Input Assets**: Provide stock tickers or use the default portfolio.
2. **Set Benchmark**: Choose a benchmark index (e.g., S&P 500).
3. **Set Parameters**: Define the risk-free rate and target return.
4. **Analyze Results**: View detailed analysis, visualizations, and save outputs.

## Example:
For a portfolio of Apple, Google, and Microsoft with a target return of 8%:

```bash
Enter stock tickers (separate by commas): AAPL, GOOGL, MSFT
Select benchmark index: ^GSPC = 0; ^DJI = 1; ^IXIC = 2: 0
Enter FRED API key (Press Enter to skip): 
Enter desired minimum return for convex optimization (default is 8.0%): 8
```

## Outputs
- **Optimized Portfolio Weights**: Saved as ``optimized_portfolio_weights.csv``, containing asset weights in the optimized portfolio.
- **Efficient Frontier Data**: Saved as efficient_frontier.csv, containing volatility, returns, and Sharpe Ratio for simulations.
- **Visualizations**: Interactive plots for the efficient frontier and performance comparisons.

## Project Structure
```bash
portfolio-optimization-tool/
│
├── main.py                   # Main script for portfolio optimization
├── requirements.txt          # List of dependencies
├── LICENSE                   # License information 
├── README.md                 # Project documentation
├── .gitignore                # Files and directories to ignore in Git
└── sample_output/            # Folder containing sample outputs
    ├── optimized_portfolio_weights.csv
    ├── efficient_frontier.csv
    ├── efficient_frontier_plot.png
    └── performance_comparison.png   
```

## Troubleshooting
- **Missing data for some tickers**: Ensure you entered valid stock tickers and have a stable internet connection.
- **Dependency issues**: Run `pip install -r requirements.txt` to reinstall dependencies.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See LICENSE for details.