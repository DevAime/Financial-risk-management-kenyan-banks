"""
Financial Risk Management Dashboard for Kenyan Banks
VaR Analysis, GBM Simulation, and Black-Scholes Option Pricing

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Financial Risk Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        kcb_data = pd.read_csv('kcb_historical_prices.csv')
        equity_data = pd.read_csv('equity_historical_prices.csv')
        coop_data = pd.read_csv('coop_historical_prices.csv')
        interest_rates = pd.read_csv('central_bank_rates.csv')
        usd_kes_data = pd.read_csv('USD_KES_historical_data.csv')
        
        # Convert dates
        kcb_data['Date'] = pd.to_datetime(kcb_data['Date'])
        equity_data['Date'] = pd.to_datetime(equity_data['Date'])
        coop_data['Date'] = pd.to_datetime(coop_data['Date'])
        usd_kes_data['Date'] = pd.to_datetime(usd_kes_data['Date'])
        
        # Sort by date
        kcb_data = kcb_data.sort_values('Date').reset_index(drop=True)
        equity_data = equity_data.sort_values('Date').reset_index(drop=True)
        coop_data = coop_data.sort_values('Date').reset_index(drop=True)
        usd_kes_data = usd_kes_data.sort_values('Date').reset_index(drop=True)

        kcb_data.columns = kcb_data.columns.str.strip()
        equity_data.columns = equity_data.columns.str.strip()
        coop_data.columns = coop_data.columns.str.strip()
        
        # Handle 
        # missing values in interest rates
        for col in interest_rates.columns:
            if interest_rates[col].dtype in ['float64', 'int64']:
                interest_rates[col].fillna(interest_rates[col].median(), inplace=True)
        
        return kcb_data, equity_data, coop_data, interest_rates, usd_kes_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure all CSV files are in the same directory as this script.")
        return None, None, None, None, None
    


def calculate_returns(data):
    """Calculate daily returns"""
    data['Returns'] = data['Close'].pct_change()
    return data['Returns'].dropna()


def variance_covariance_var(returns, confidence_level, investment):
    """Calculate VaR using Variance-Covariance Method"""
    mean = returns.mean()
    std = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var = investment * (mean + z_score * std)
    return abs(var)


def historical_simulation_var(returns, confidence_level, investment):
    """Calculate VaR using Historical Simulation Method"""
    percentile = (1 - confidence_level) * 100
    var_return = np.percentile(returns, percentile)
    var = investment * abs(var_return)
    return var


def monte_carlo_var(returns, confidence_level, investment, num_simulations=10000):
    """Calculate VaR using Monte Carlo Simulation Method"""
    mean = returns.mean()
    std = returns.std()
    simulated_returns = np.random.normal(mean, std, num_simulations)
    percentile = (1 - confidence_level) * 100
    var_return = np.percentile(simulated_returns, percentile)
    var = investment * abs(var_return)
    return var, simulated_returns


def gbm_monte_carlo_var(S0, mu, sigma, T, confidence_level, investment, num_simulations=10000):
    """Calculate VaR using GBM-based Monte Carlo simulation"""
    final_prices = S0 * np.exp(
        (mu - 0.5 * sigma**2) * T + 
        sigma * np.sqrt(T) * np.random.normal(0, 1, num_simulations)
    )
    returns = (final_prices - S0) / S0
    percentile = (1 - confidence_level) * 100
    var_return = np.percentile(returns, percentile)
    var = investment * abs(var_return)
    return var, final_prices, returns


def gbm_simulation(S0, mu, sigma, T, dt, num_simulations=100):
    """Simulate stock prices using Geometric Brownian Motion"""
    num_steps = int(T / dt)
    random_shocks = np.random.normal(0, 1, (num_simulations, num_steps))
    prices = np.zeros((num_simulations, num_steps + 1))
    prices[:, 0] = S0
    
    for t in range(1, num_steps + 1):
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[:, t-1]
        )
    return prices


def black_scholes_put(S, K, T, r, sigma):
    """Calculate European Put Option Price using Black-Scholes"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def calculate_greeks(S, K, T, r, sigma):
    """Calculate option Greeks for put option"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             + r * K * np.exp(-r * T) * norm.cdf(-d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📊 Financial Risk Management Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Value-at-Risk Analysis for Kenyan Banking Sector</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        kcb_data, equity_data, coop_data, interest_rates, usd_kes_data = load_data()
    
    if kcb_data is None:
        st.stop()
    
    # Calculate returns
    kcb_returns = calculate_returns(kcb_data)
    equity_returns = calculate_returns(equity_data)
    coop_returns = calculate_returns(coop_data)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Investment amount
    investment = st.sidebar.number_input(
        "Portfolio Value (KES)",
        min_value=100000,
        max_value=100000000,
        value=1000000,
        step=100000,
        format="%d"
    )
    
    # Confidence level
    confidence_level = st.sidebar.select_slider(
        "Confidence Level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%"
    )
    
    # Number of simulations
    num_simulations = st.sidebar.slider(
        "Monte Carlo Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    st.sidebar.markdown("---")
    
    # Portfolio weights
    st.sidebar.subheader("Portfolio Composition")
    weight_kcb = st.sidebar.slider("KCB Weight", 0.0, 1.0, 0.40, 0.05)
    weight_equity = st.sidebar.slider("Equity Bank Weight", 0.0, 1.0, 0.30, 0.05)
    weight_coop = st.sidebar.slider("Co-op Bank Weight", 0.0, 1.0, 0.30, 0.05)
    
    # Normalize weights
    total_weight = weight_kcb + weight_equity + weight_coop
    if total_weight > 0:
        weight_kcb /= total_weight
        weight_equity /= total_weight
        weight_coop /= total_weight
    
    st.sidebar.info(f"Normalized: KCB {weight_kcb*100:.1f}%, Equity {weight_equity*100:.1f}%, Co-op {weight_coop*100:.1f}%")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🎯 VaR Analysis", 
        "🔮 GBM Simulation", 
        "💰 Options Pricing",
        "📊 Comparison"
    ])
    
    # =============================================================================
    # TAB 1: OVERVIEW
    # =============================================================================
    
    with tab1:
        st.markdown('<div class="sub-header">Market Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Data Period",
                f"{kcb_data['Date'].min().strftime('%b %Y')} - {kcb_data['Date'].max().strftime('%b %Y')}"
            )
        
        with col2:
            st.metric("Trading Days", len(kcb_returns))
        
        with col3:
            st.metric("Portfolio Value", f"KES {investment:,.0f}")
        
        with col4:
            st.metric("Confidence Level", f"{int(confidence_level*100)}%")
        
        st.markdown("---")
        
        # Stock prices chart
        st.subheader("Stock Price Trends")
        fig_prices = go.Figure()
        
        fig_prices.add_trace(go.Scatter(
            x=kcb_data['Date'], y=kcb_data['Close'],
            name='KCB', mode='lines', line=dict(color='blue', width=2)
        ))
        fig_prices.add_trace(go.Scatter(
            x=equity_data['Date'], y=equity_data['Close'],
            name='Equity Bank', mode='lines', line=dict(color='green', width=2)
        ))
        fig_prices.add_trace(go.Scatter(
            x=coop_data['Date'], y=coop_data['Close'],
            name='Co-op Bank', mode='lines', line=dict(color='red', width=2)
        ))
        
        fig_prices.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (KES)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig_prices, use_container_width=True)
        
        # Statistics table
        st.subheader("Return Statistics")
        
        stats_df = pd.DataFrame({
            'Bank': ['KCB', 'Equity Bank', 'Co-op Bank'],
            'Mean Daily Return (%)': [
                kcb_returns.mean() * 100,
                equity_returns.mean() * 100,
                coop_returns.mean() * 100
            ],
            'Daily Volatility (%)': [
                kcb_returns.std() * 100,
                equity_returns.std() * 100,
                coop_returns.std() * 100
            ],
            'Annual Return (%)': [
                kcb_returns.mean() * 252 * 100,
                equity_returns.mean() * 252 * 100,
                coop_returns.mean() * 252 * 100
            ],
            'Annual Volatility (%)': [
                kcb_returns.std() * np.sqrt(252) * 100,
                equity_returns.std() * np.sqrt(252) * 100,
                coop_returns.std() * np.sqrt(252) * 100
            ]
        })
        
        st.dataframe(stats_df.style.format({
            'Mean Daily Return (%)': '{:.4f}',
            'Daily Volatility (%)': '{:.4f}',
            'Annual Return (%)': '{:.2f}',
            'Annual Volatility (%)': '{:.2f}'
        }), use_container_width=True)
    
    # =============================================================================
    # TAB 2: VAR ANALYSIS
    # =============================================================================
    
    with tab2:
        st.markdown('<div class="sub-header">Value-at-Risk Analysis</div>', unsafe_allow_html=True)
        
        # Select bank
        selected_bank = st.selectbox(
            "Select Bank for Analysis",
            ["KCB", "Equity Bank", "Co-op Bank"]
        )
        
        # Get returns for selected bank
        if selected_bank == "KCB":
            returns = kcb_returns
            current_price = kcb_data['Close'].iloc[-1]
        elif selected_bank == "Equity Bank":
            returns = equity_returns
            current_price = equity_data['Close'].iloc[-1]
        else:
            returns = coop_returns
            current_price = coop_data['Close'].iloc[-1]
        
        # Calculate VaR using all methods
        with st.spinner("Calculating VaR..."):
            var_vc = variance_covariance_var(returns, confidence_level, investment)
            var_hs = historical_simulation_var(returns, confidence_level, investment)
            var_mc, simulated_returns = monte_carlo_var(returns, confidence_level, investment, num_simulations)
            
            # GBM VaR
            mu_annual = returns.mean() * 252
            sigma_annual = returns.std() * np.sqrt(252)
            var_gbm, _, returns_gbm = gbm_monte_carlo_var(
                current_price, mu_annual, sigma_annual, 1/252, confidence_level, investment, num_simulations
            )
        
        # Display VaR results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Variance-Covariance",
                f"KES {var_vc:,.0f}",
                delta=f"{(var_vc/investment)*100:.2f}% of portfolio",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Historical Simulation",
                f"KES {var_hs:,.0f}",
                delta=f"{(var_hs/investment)*100:.2f}% of portfolio",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Monte Carlo",
                f"KES {var_mc:,.0f}",
                delta=f"{(var_mc/investment)*100:.2f}% of portfolio",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "GBM Monte Carlo",
                f"KES {var_gbm:,.0f}",
                delta=f"{(var_gbm/investment)*100:.2f}% of portfolio",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Comparison chart
        st.subheader("VaR Methods Comparison")
        
        fig_var = go.Figure(data=[
            go.Bar(
                x=['Var-Cov', 'Historical', 'Monte Carlo', 'GBM-MC'],
                y=[var_vc, var_hs, var_mc, var_gbm],
                text=[f'KES {x:,.0f}' for x in [var_vc, var_hs, var_mc, var_gbm]],
                textposition='outside',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )
        ])
        
        fig_var.update_layout(
            xaxis_title='Method',
            yaxis_title='VaR (KES)',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Historical Returns Distribution")
            fig_hist = go.Figure(data=[go.Histogram(
                x=returns * 100,
                nbinsx=50,
                marker_color='lightblue',
                name='Returns'
            )])
            
            var_pct = (var_hs / investment) * 100
            fig_hist.add_vline(
                x=-var_pct,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{confidence_level*100}% VaR"
            )
            
            fig_hist.update_layout(
                xaxis_title='Daily Return (%)',
                yaxis_title='Frequency',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Monte Carlo Simulated Returns")
            fig_mc = go.Figure(data=[go.Histogram(
                x=simulated_returns * 100,
                nbinsx=50,
                marker_color='lightgreen',
                name='Simulated Returns'
            )])
            
            var_pct_mc = (var_mc / investment) * 100
            fig_mc.add_vline(
                x=-var_pct_mc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{confidence_level*100}% VaR"
            )
            
            fig_mc.update_layout(
                xaxis_title='Daily Return (%)',
                yaxis_title='Frequency',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)
    
    # =============================================================================
    # TAB 3: GBM SIMULATION
    # =============================================================================
    
    with tab3:
        st.markdown('<div class="sub-header">Geometric Brownian Motion Simulation</div>', unsafe_allow_html=True)
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            horizon_days = st.slider("Simulation Horizon (Trading Days)", 5, 60, 21, 1)
        
        with col2:
            num_paths = st.slider("Number of Paths to Display", 10, 100, 50, 10)
        
        # Select bank for GBM
        gbm_bank = st.selectbox(
            "Select Bank for GBM Simulation",
            ["KCB", "Equity Bank", "Co-op Bank"],
            key="gbm_bank"
        )
        
        if gbm_bank == "KCB":
            S0 = kcb_data['Close'].iloc[-1]
            returns_gbm_calc = kcb_returns
        elif gbm_bank == "Equity Bank":
            S0 = equity_data['Close'].iloc[-1]
            returns_gbm_calc = equity_returns
        else:
            S0 = coop_data['Close'].iloc[-1]
            returns_gbm_calc = coop_returns
        
        mu = returns_gbm_calc.mean() * 252
        sigma = returns_gbm_calc.std() * np.sqrt(252)
        
        # Display parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price (S₀)", f"KES {S0:.2f}")
        
        with col2:
            st.metric("Expected Annual Return (μ)", f"{mu*100:.2f}%")
        
        with col3:
            st.metric("Annual Volatility (σ)", f"{sigma*100:.2f}%")
        
        # Simulate prices
        with st.spinner("Simulating price paths..."):
            T = horizon_days / 252
            dt = 1 / 252
            prices_paths = gbm_simulation(S0, mu, sigma, T, dt, num_paths)
        
        # Plot simulation
        time_array = np.arange(0, horizon_days + 1)
        
        fig_gbm = go.Figure()
        
        # Plot sample paths
        for i in range(num_paths):
            fig_gbm.add_trace(go.Scatter(
                x=time_array,
                y=prices_paths[i, :],
                mode='lines',
                line=dict(width=0.5, color='lightblue'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add mean path
        mean_path = prices_paths.mean(axis=0)
        fig_gbm.add_trace(go.Scatter(
            x=time_array,
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(width=3, color='darkblue')
        ))
        
        # Add confidence intervals
        percentile_5 = np.percentile(prices_paths, 5, axis=0)
        percentile_95 = np.percentile(prices_paths, 95, axis=0)
        
        fig_gbm.add_trace(go.Scatter(
            x=time_array,
            y=percentile_95,
            mode='lines',
            name='95th Percentile',
            line=dict(width=2, color='green', dash='dash')
        ))
        
        fig_gbm.add_trace(go.Scatter(
            x=time_array,
            y=percentile_5,
            mode='lines',
            name='5th Percentile',
            line=dict(width=2, color='red', dash='dash')
        ))
        
        fig_gbm.update_layout(
            title=f'{gbm_bank} Stock Price Simulation ({horizon_days} Trading Days)',
            xaxis_title='Trading Days',
            yaxis_title='Price (KES)',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_gbm, use_container_width=True)
        
        # Final price distribution
        st.subheader("Final Price Distribution")
        
        final_prices = prices_paths[:, -1]
        
        fig_final = go.Figure(data=[go.Histogram(
            x=final_prices,
            nbinsx=50,
            marker_color='lightblue',
            name='Final Prices'
        )])
        
        fig_final.add_vline(
            x=S0,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Current: KES {S0:.2f}"
        )
        
        fig_final.add_vline(
            x=mean_path[-1],
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Mean: KES {mean_path[-1]:.2f}"
        )
        
        fig_final.update_layout(
            xaxis_title='Price (KES)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_final, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Final Price", f"KES {final_prices.mean():.2f}")
        
        with col2:
            st.metric("5th Percentile", f"KES {np.percentile(final_prices, 5):.2f}")
        
        with col3:
            st.metric("95th Percentile", f"KES {np.percentile(final_prices, 95):.2f}")
        
        with col4:
            expected_return = ((mean_path[-1] - S0) / S0) * 100
            st.metric(f"Expected Return ({horizon_days} days)", f"{expected_return:.2f}%")
    
    # =============================================================================
    # TAB 4: OPTIONS PRICING
    # =============================================================================
    
    with tab4:
        st.markdown('<div class="sub-header">Black-Scholes Option Pricing</div>', unsafe_allow_html=True)
        
        # Select bank for options
        option_bank = st.selectbox(
            "Select Bank",
            ["KCB", "Equity Bank", "Co-op Bank"],
            key="option_bank"
        )
        
        if option_bank == "KCB":
            S = kcb_data['Close'].iloc[-1]
            returns_opt = kcb_returns
        elif option_bank == "Equity Bank":
            S = equity_data['Close'].iloc[-1]
            returns_opt = equity_returns
        else:
            S = coop_data['Close'].iloc[-1]
            returns_opt = coop_returns
        
        sigma_opt = returns_opt.std() * np.sqrt(252)
        num_shares = investment / S
        
        # Risk-free rate
        r_annual = st.slider("Risk-Free Rate (Annual %)", 5.0, 20.0, 13.0, 0.5) / 100
        
        st.markdown("---")
        
        # Protection levels and time horizons
        protection_levels = [0.90, 0.95, 0.97]
        time_horizons = [30, 60, 90]
        
        # Calculate put prices
        put_data = []
        
        for protection in protection_levels:
            K = S * protection
            for days in time_horizons:
                T = days / 365
                put_price = black_scholes_put(S, K, T, r_annual, sigma_opt)
                total_cost = put_price * num_shares
                
                put_data.append({
                    'Protection Level': f'{int(protection*100)}%',
                    'Strike Price': K,
                    'Time (days)': days,
                    'Put Price': put_price,
                    'Total Cost': total_cost,
                    'Cost %': (total_cost / investment) * 100
                })
        
        put_df = pd.DataFrame(put_data)
        
        # Display table
        st.subheader("Protective Put Option Pricing")
        
        st.dataframe(put_df.style.format({
            'Strike Price': 'KES {:.2f}',
            'Put Price': 'KES {:.2f}',
            'Total Cost': 'KES {:.2f}',
            'Cost %': '{:.2f}%'
        }), use_container_width=True)
        
        # Heatmap
        st.subheader("Cost as % of Portfolio")
        
        pivot_table = put_df.pivot_table(
            values='Cost %',
            index='Protection Level',
            columns='Time (days)'
        )
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Blues',
            text=pivot_table.values,
            texttemplate='%{text:.2f}%',
            textfont={"size": 12},
            colorbar=dict(title="Cost %")
        ))
        
        fig_heat.update_layout(
            xaxis_title='Time Horizon (days)',
            yaxis_title='Protection Level',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Greeks analysis
        st.subheader("Option Greeks Analysis")
        
        K_selected = st.selectbox("Select Strike Price", 
                                 [f"{int(p*100)}% (KES {S*p:.2f})" for p in protection_levels])
        T_selected = st.selectbox("Select Time Horizon", 
                                 [f"{d} days" for d in time_horizons])
        
        # Extract values
        protection_pct = float(K_selected.split('%')[0]) / 100
        K_greeks = S * protection_pct
        T_greeks = int(T_selected.split()[0]) / 365
        
        greeks = calculate_greeks(S, K_greeks, T_greeks, r_annual, sigma_opt)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta", f"{greeks['delta']:.4f}")
            st.caption("Price sensitivity")
        
        with col2:
            st.metric("Gamma", f"{greeks['gamma']:.6f}")
            st.caption("Delta change rate")
        
        with col3:
            st.metric("Theta", f"{greeks['theta']:.4f}")
            st.caption("Time decay (per day)")
        
        with col4:
            st.metric("Vega", f"{greeks['vega']:.4f}")
            st.caption("Volatility sensitivity")
        
        with col5:
            st.metric("Rho", f"{greeks['rho']:.4f}")
            st.caption("Rate sensitivity")
    
    # =============================================================================
    # TAB 5: COMPARISON
    # =============================================================================
    
    with tab5:
        st.markdown('<div class="sub-header">Comprehensive Risk Analysis</div>', unsafe_allow_html=True)
        
        # Calculate VaR for all banks
        banks_dict = {
            'KCB': (kcb_returns, kcb_data['Close'].iloc[-1]),
            'Equity Bank': (equity_returns, equity_data['Close'].iloc[-1]),
            'Co-op Bank': (coop_returns, coop_data['Close'].iloc[-1])
        }
        
        all_results = []
        
        with st.spinner("Calculating comprehensive analysis..."):
            for bank_name, (returns, price) in banks_dict.items():
                var_vc = variance_covariance_var(returns, confidence_level, investment)
                var_hs = historical_simulation_var(returns, confidence_level, investment)
                var_mc, _ = monte_carlo_var(returns, confidence_level, investment, num_simulations)
                
                mu_annual = returns.mean() * 252
                sigma_annual = returns.std() * np.sqrt(252)
                var_gbm, _, _ = gbm_monte_carlo_var(
                    price, mu_annual, sigma_annual, 1/252, confidence_level, investment, num_simulations
                )
                
                all_results.append({
                    'Bank': bank_name,
                    'Var-Cov': var_vc,
                    'Historical': var_hs,
                    'Monte Carlo': var_mc,
                    'GBM-MC': var_gbm
                })
        
        results_df = pd.DataFrame(all_results)
        
        # Portfolio VaR
        st.subheader("Portfolio Analysis")
        
        # Align returns
        kcb_df = kcb_data[['Date', 'Returns']].rename(columns={'Returns': 'KCB_Returns'})
        equity_df = equity_data[['Date', 'Returns']].rename(columns={'Returns': 'Equity_Returns'})
        coop_df = coop_data[['Date', 'Returns']].rename(columns={'Returns': 'Coop_Returns'})
        
        portfolio_df = kcb_df.merge(equity_df, on='Date', how='inner')
        portfolio_df = portfolio_df.merge(coop_df, on='Date', how='inner')
        portfolio_df = portfolio_df.dropna()
        
        portfolio_df['Portfolio_Returns'] = (
            weight_kcb * portfolio_df['KCB_Returns'] +
            weight_equity * portfolio_df['Equity_Returns'] +
            weight_coop * portfolio_df['Coop_Returns']
        )
        
        portfolio_returns = portfolio_df['Portfolio_Returns']
        
        # Portfolio VaR
        port_var_vc = variance_covariance_var(portfolio_returns, confidence_level, investment)
        port_var_hs = historical_simulation_var(portfolio_returns, confidence_level, investment)
        port_var_mc, _ = monte_carlo_var(portfolio_returns, confidence_level, investment, num_simulations)
        
        # Add portfolio to results
        portfolio_row = {
            'Bank': 'Portfolio',
            'Var-Cov': port_var_vc,
            'Historical': port_var_hs,
            'Monte Carlo': port_var_mc,
            'GBM-MC': np.nan
        }
        
        results_df = pd.concat([results_df, pd.DataFrame([portfolio_row])], ignore_index=True)
        
        # Display results table
        st.dataframe(results_df.style.format({
            'Var-Cov': 'KES {:,.0f}',
            'Historical': 'KES {:,.0f}',
            'Monte Carlo': 'KES {:,.0f}',
            'GBM-MC': 'KES {:,.0f}'
        }), use_container_width=True)
        
        # Comparison chart
        st.subheader("VaR Comparison Across Banks")
        
        fig_comp = go.Figure()
        
        methods = ['Var-Cov', 'Historical', 'Monte Carlo']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, method in enumerate(methods):
            fig_comp.add_trace(go.Bar(
                name=method,
                x=results_df['Bank'],
                y=results_df[method],
                marker_color=colors[i],
                text=[f'KES {x:,.0f}' for x in results_df[method]],
                textposition='outside'
            ))
        
        fig_comp.update_layout(
            xaxis_title='Bank',
            yaxis_title='VaR (KES)',
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Diversification benefit
        st.subheader("Diversification Benefit")
        
        weighted_individual = (
            weight_kcb * results_df[results_df['Bank'] == 'KCB']['Monte Carlo'].values[0] +
            weight_equity * results_df[results_df['Bank'] == 'Equity Bank']['Monte Carlo'].values[0] +
            weight_coop * results_df[results_df['Bank'] == 'Co-op Bank']['Monte Carlo'].values[0]
        )
        
        portfolio_var = results_df[results_df['Bank'] == 'Portfolio']['Monte Carlo'].values[0]
        
        diversification_benefit = weighted_individual - portfolio_var
        benefit_pct = (diversification_benefit / weighted_individual) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Weighted Individual VaR",
                f"KES {weighted_individual:,.0f}"
            )
        
        with col2:
            st.metric(
                "Portfolio VaR",
                f"KES {portfolio_var:,.0f}",
                delta=f"-{benefit_pct:.2f}%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Diversification Benefit",
                f"KES {diversification_benefit:,.0f}",
                delta=f"{benefit_pct:.2f}% reduction"
            )
        
        st.success(f"✅ By holding a diversified portfolio, risk is reduced by {benefit_pct:.2f}%")
        
        # Risk vs Hedging Cost
        st.subheader("Risk Mitigation: VaR Loss vs Option Hedging Cost")
        
        # Use KCB as example
        kcb_var = results_df[results_df['Bank'] == 'KCB']['Monte Carlo'].values[0]
        
        # Calculate 30-day put at 95% protection
        S_kcb = kcb_data['Close'].iloc[-1]
        K_put = S_kcb * 0.95
        T_put = 30 / 365
        sigma_kcb = kcb_returns.std() * np.sqrt(252)
        r = 0.13
        
        put_cost = black_scholes_put(S_kcb, K_put, T_put, r, sigma_kcb) * (investment / S_kcb)
        
        fig_hedge = go.Figure()
        
        fig_hedge.add_trace(go.Bar(
            x=['VaR Loss (1-day)', '30-Day Put Cost'],
            y=[kcb_var, put_cost],
            marker_color=['red', 'green'],
            text=[f'KES {kcb_var:,.0f}', f'KES {put_cost:,.0f}'],
            textposition='outside'
        ))
        
        fig_hedge.update_layout(
            title='KCB: Potential Loss vs Hedging Cost',
            yaxis_title='Amount (KES)',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_hedge, use_container_width=True)
        
        cost_benefit_ratio = put_cost / kcb_var
        
        if cost_benefit_ratio < 1:
            st.success(f"✅ Hedging is cost-effective: Paying KES {put_cost:,.0f} to protect against potential KES {kcb_var:,.0f} loss (Ratio: {cost_benefit_ratio:.2f}x)")
        else:
            st.info(f"ℹ️ Hedging costs exceed 1-day VaR but may be suitable for risk-averse portfolios (Ratio: {cost_benefit_ratio:.2f}x)")
        
        # Download results
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download VaR Results as CSV",
            data=csv,
            file_name="var_analysis_results.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>Financial Risk Management Dashboard</strong></p>
            <p>VaR Analysis for Kenyan Banking Sector | Built with Streamlit</p>
            <p style="font-size: 0.8rem;">⚠️ For educational purposes only. Not financial advice.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()