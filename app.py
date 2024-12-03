import streamlit as st
import utils

# Initialize session state for inputs and run button
if "inputs_ready" not in st.session_state:
    st.session_state.inputs_ready = False
if "mode" not in st.session_state:
    st.session_state.mode = "custom"  # Default to custom mode

# Header section
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>Portfolio Optimization Model</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size: 18px;'>This is a Python-based application for analyzing and optimizing financial portfolios. "
    "It uses modern portfolio theory to identify optimal allocations and visualize performances.</p>",
    unsafe_allow_html=True,
)

# Separator line
st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)

# Instructional text
#st.markdown("<h2>Choose an option to start:</h2>", unsafe_allow_html=True)

# Control input processing
process_inputs = True

# Custom input fields
st.markdown("### 1. Enter Portfolio Inputs:")
user_input = st.text_input(
    "Enter stock tickers separated by commas: ",
    placeholder="AAPL, MSFT, TSLA",
    key="user_input",
).strip()

benchmark_index = st.selectbox(
    "Select an index to benchmark the portfolio:",
    ["^GSPC", "^DJI", "^IXIC"],
    key="benchmark_index",
)

risk_free_rate = st.slider(
    "Risk free rate:",
    min_value=2.0,
    max_value=6.0,
    value=4.1,
    key="risk_free_rate",
)

min_return = st.slider(
    "Desired target return (%):",
    min_value=2.0,
    max_value=20.0,
    value=5.0,
    key="min_return",
)

# Button layout for Run and Random Inputs
col1, col2 = st.columns([1, 0.4])

with col1:
    # Run button for custom inputs
    run_button = st.button("Run")
with col2:
    # Random Button for testing with random tickers
    random_button = st.button("Test with random tickers")

# Logic for Custom Inputs after validation
if run_button:
    utils.run_button(user_input)

# Logic for random Inputs 
if random_button:
    utils.random_button()



