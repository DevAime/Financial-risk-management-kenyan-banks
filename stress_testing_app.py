import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Kenyan Banks Stress Testing",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Define stress test scenarios
stress_scenarios = {
    'Base Case': {
        'kes_depreciation': 0.0,
        'inflation_shock': 0.0,
        'interest_rate_change': 0.0,
        'equity_price_decline': 0.0,
        'credit_loss_rate': 0.0,
        'description': 'No shocks - baseline scenario'
    },
    'Mild Depreciation': {
        'kes_depreciation': 0.05,
        'inflation_shock': 0.02,
        'interest_rate_change': 0.01,
        'equity_price_decline': 0.05,
        'credit_loss_rate': 0.01,
        'description': 'Mild currency and inflationary pressures'
    },
    'Moderate Depreciation': {
        'kes_depreciation': 0.10,
        'inflation_shock': 0.05,
        'interest_rate_change': 0.02,
        'equity_price_decline': 0.15,
        'credit_loss_rate': 0.03,
        'description': 'Moderate currency and economic stress'
    },
    'Severe Depreciation': {
        'kes_depreciation': 0.15,
        'inflation_shock': 0.08,
        'interest_rate_change': 0.03,
        'equity_price_decline': 0.25,
        'credit_loss_rate': 0.05,
        'description': 'Severe currency crisis scenario'
    },
    'Combined Crisis': {
        'kes_depreciation': 0.20,
        'inflation_shock': 0.10,
        'interest_rate_change': 0.04,
        'equity_price_decline': 0.35,
        'credit_loss_rate': 0.08,
        'description': 'Systemic crisis - combined macroeconomic shocks'
    }
}

# CBK Regulatory Requirements
cbk_requirements = {
    'Tier 1 Capital Ratio (%)': 10.5,
    'Total Capital Ratio (%)': 14.5,
    'Leverage Ratio (%)': 6.0
}

def load_bank_data():
    """Load and process bank data from CSV files"""
    try:
        # Load real bank price data
        kcb_data = pd.read_csv('kcb_historical_prices.csv')
        equity_data = pd.read_csv('equity_historical_prices.csv')
        coop_data = pd.read_csv('coop_historical_prices.csv')
        
        # Clean column names
        kcb_data.columns = kcb_data.columns.str.strip()
        equity_data.columns = equity_data.columns.str.strip()
        coop_data.columns = coop_data.columns.str.strip()
        
        # Calculate equity and assets proxies
        kcb_assets = kcb_data['Close'].sum() * 1e3
        equity_assets = equity_data['Close'].sum() * 1e3
        coop_assets = coop_data['Close'].sum() * 1e3
        
        kcb_equity = kcb_assets * 0.1
        equity_equity = equity_assets * 0.1
        coop_equity = coop_assets * 0.1
        
        # Construct DataFrame
        banks_df = pd.DataFrame({
            'Bank Name': ['KCB', 'Equity', 'Co-op'],
            'Total Assets (KES Billion)': [kcb_assets/1e9, equity_assets/1e9, coop_assets/1e9],
            'Equity (KES Billion)': [kcb_equity/1e9, equity_equity/1e9, coop_equity/1e9],
        })
        
        # Add capital structure fields
        banks_df['Tier 1 Capital (KES Billion)'] = banks_df['Equity (KES Billion)'] * 0.8
        banks_df['Tier 2 Capital (KES Billion)'] = banks_df['Equity (KES Billion)'] * 0.2
        banks_df['Risk-Weighted Assets (KES Billion)'] = banks_df['Total Assets (KES Billion)'] * 0.5
        banks_df['Total Loans (KES Billion)'] = banks_df['Total Assets (KES Billion)'] * 0.6
        banks_df['Non-Performing Loans Ratio'] = [0.04, 0.035, 0.045]
        banks_df['Foreign Currency Exposure (%)'] = [0.25, 0.20, 0.15]
        banks_df['Equity Portfolio Value (KES Billion)'] = banks_df['Equity (KES Billion)'] * 0.05
        
        # Calculate capital ratios
        banks_df['Tier 1 Capital Ratio (%)'] = (
            banks_df['Tier 1 Capital (KES Billion)'] / banks_df['Risk-Weighted Assets (KES Billion)'] * 100
        )
        banks_df['Total Capital Ratio (%)'] = (
            (banks_df['Tier 1 Capital (KES Billion)'] + banks_df['Tier 2 Capital (KES Billion)']) /
            banks_df['Risk-Weighted Assets (KES Billion)'] * 100
        )
        banks_df['Leverage Ratio (%)'] = (
            banks_df['Equity (KES Billion)'] / banks_df['Total Assets (KES Billion)'] * 100
        )
        
        return banks_df, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, False

def create_sample_data():
    """Create sample data if CSV files are not available"""
    banks_df = pd.DataFrame({
        'Bank Name': ['KCB', 'Equity', 'Co-op'],
        'Total Assets (KES Billion)': [1200.0, 950.0, 800.0],
        'Equity (KES Billion)': [120.0, 95.0, 80.0],
    })
    
    banks_df['Tier 1 Capital (KES Billion)'] = banks_df['Equity (KES Billion)'] * 0.8
    banks_df['Tier 2 Capital (KES Billion)'] = banks_df['Equity (KES Billion)'] * 0.2
    banks_df['Risk-Weighted Assets (KES Billion)'] = banks_df['Total Assets (KES Billion)'] * 0.5
    banks_df['Total Loans (KES Billion)'] = banks_df['Total Assets (KES Billion)'] * 0.6
    banks_df['Non-Performing Loans Ratio'] = [0.04, 0.035, 0.045]
    banks_df['Foreign Currency Exposure (%)'] = [0.25, 0.20, 0.15]
    banks_df['Equity Portfolio Value (KES Billion)'] = banks_df['Equity (KES Billion)'] * 0.05
    
    banks_df['Tier 1 Capital Ratio (%)'] = (
        banks_df['Tier 1 Capital (KES Billion)'] / banks_df['Risk-Weighted Assets (KES Billion)'] * 100
    )
    banks_df['Total Capital Ratio (%)'] = (
        (banks_df['Tier 1 Capital (KES Billion)'] + banks_df['Tier 2 Capital (KES Billion)']) /
        banks_df['Risk-Weighted Assets (KES Billion)'] * 100
    )
    banks_df['Leverage Ratio (%)'] = (
        banks_df['Equity (KES Billion)'] / banks_df['Total Assets (KES Billion)'] * 100
    )
    
    return banks_df

def apply_stress_scenario(bank_row, scenario_params):
    """Apply stress scenario parameters to a bank's capital position"""
    bank_stressed = bank_row.copy()
    
    fx_exposure = bank_row['Foreign Currency Exposure (%)']
    rwa_stress = bank_row['Risk-Weighted Assets (KES Billion)'] * (1 + scenario_params['kes_depreciation'] * fx_exposure)
    
    equity_loss = bank_row['Equity Portfolio Value (KES Billion)'] * scenario_params['equity_price_decline']
    fx_loss = bank_row['Total Assets (KES Billion)'] * fx_exposure * scenario_params['kes_depreciation']
    tier1_stressed = bank_row['Tier 1 Capital (KES Billion)'] - equity_loss - fx_loss
    
    npl_ratio = bank_row['Non-Performing Loans Ratio']
    credit_loss = bank_row['Total Loans (KES Billion)'] * (npl_ratio + scenario_params['credit_loss_rate'])
    tier2_stressed = bank_row['Tier 2 Capital (KES Billion)'] - credit_loss
    
    tier1_stressed = max(tier1_stressed, 0)
    tier2_stressed = max(tier2_stressed, 0)
    
    tier1_ratio_stressed = (tier1_stressed / rwa_stress) * 100
    total_capital_ratio_stressed = ((tier1_stressed + tier2_stressed) / rwa_stress) * 100
    
    return {
        'Tier 1 Capital (KES B)': tier1_stressed,
        'Tier 2 Capital (KES B)': tier2_stressed,
        'RWA (KES B)': rwa_stress,
        'Tier 1 Ratio (%)': tier1_ratio_stressed,
        'Total Capital Ratio (%)': total_capital_ratio_stressed,
        'Equity Loss (KES B)': equity_loss,
        'FX Loss (KES B)': fx_loss,
        'Credit Loss (KES B)': credit_loss
    }

def run_stress_tests(banks_df, stress_scenarios):
    """Run stress tests for all scenarios and banks"""
    stress_results = {}
    
    for scenario_name, scenario_params in stress_scenarios.items():
        stress_results[scenario_name] = {}
        
        for idx, bank_row in banks_df.iterrows():
            bank_name = bank_row['Bank Name']
            stressed_ratios = apply_stress_scenario(bank_row, scenario_params)
            stressed_ratios['Bank Name'] = bank_name
            stressed_ratios['Baseline Tier 1 (%)'] = bank_row['Tier 1 Capital Ratio (%)']
            stressed_ratios['Baseline Total Capital (%)'] = bank_row['Total Capital Ratio (%)']
            stress_results[scenario_name][bank_name] = stressed_ratios
    
    return stress_results

def find_max_depreciation_tolerance(bank_row, cbk_requirements):
    """Find maximum KES depreciation tolerance for a bank"""
    min_tier1_ratio = cbk_requirements['Tier 1 Capital Ratio (%)']
    
    low_depr = 0.0
    high_depr = 0.5
    tolerance = 1e-4
    
    while high_depr - low_depr > tolerance:
        mid_depr = (low_depr + high_depr) / 2
        
        test_scenario = {
            'kes_depreciation': mid_depr,
            'inflation_shock': 0.0,
            'interest_rate_change': 0.0,
            'equity_price_decline': 0.0,
            'credit_loss_rate': 0.0
        }
        
        stressed_ratios = apply_stress_scenario(bank_row, test_scenario)
        
        if stressed_ratios['Tier 1 Ratio (%)'] >= min_tier1_ratio:
            low_depr = mid_depr
        else:
            high_depr = mid_depr
    
    return low_depr

# Main App
st.markdown('<div class="main-header">🏦 Kenyan Banks Stress Testing & Scenario Analysis</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source:",
        ["Use Sample Data", "Load from CSV Files"],
        help="Choose whether to use sample data or load from your CSV files"
    )
    
    st.markdown("---")
    
    # Scenario selection
    st.subheader("📊 Select Scenarios to Analyze")
    selected_scenarios = st.multiselect(
        "Scenarios:",
        list(stress_scenarios.keys()),
        default=list(stress_scenarios.keys())
    )
    
    st.markdown("---")
    
    # Custom scenario builder
    st.subheader("🔧 Custom Scenario")
    if st.checkbox("Create Custom Scenario"):
        custom_name = st.text_input("Scenario Name:", "Custom Scenario")
        custom_kes = st.slider("KES Depreciation (%)", 0, 30, 10) / 100
        custom_inflation = st.slider("Inflation Shock (%)", 0, 15, 5) / 100
        custom_rate = st.slider("Interest Rate Change (%)", 0, 5, 2) / 100
        custom_equity = st.slider("Equity Price Decline (%)", 0, 50, 20) / 100
        custom_credit = st.slider("Credit Loss Rate (%)", 0, 10, 3) / 100
        
        if st.button("Add Custom Scenario"):
            stress_scenarios[custom_name] = {
                'kes_depreciation': custom_kes,
                'inflation_shock': custom_inflation,
                'interest_rate_change': custom_rate,
                'equity_price_decline': custom_equity,
                'credit_loss_rate': custom_credit,
                'description': 'Custom user-defined scenario'
            }
            st.success(f"✅ {custom_name} added!")

# Load data
if data_source == "Load from CSV Files":
    banks_df, success = load_bank_data()
    if not success:
        st.warning("⚠️ Could not load CSV files. Using sample data instead.")
        banks_df = create_sample_data()
else:
    banks_df = create_sample_data()

# Run stress tests
stress_results = run_stress_tests(banks_df, stress_scenarios)

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview", 
    "📊 Scenario Analysis", 
    "📈 Capital Ratios",
    "💰 Loss Decomposition",
    "🔍 Risk Assessment",
    "📑 Summary Report"
])

# Tab 1: Overview
with tab1:
    st.markdown('<div class="section-header">Baseline Bank Positions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total System Assets", f"KES {banks_df['Total Assets (KES Billion)'].sum():.1f}B")
    with col2:
        st.metric("Total System Equity", f"KES {banks_df['Equity (KES Billion)'].sum():.1f}B")
    with col3:
        avg_tier1 = banks_df['Tier 1 Capital Ratio (%)'].mean()
        st.metric("Avg Tier 1 Ratio", f"{avg_tier1:.2f}%")
    
    st.markdown("---")
    
    # Display baseline positions
    st.subheader("Bank Capital Positions")
    display_cols = ['Bank Name', 'Total Assets (KES Billion)', 'Equity (KES Billion)',
                    'Tier 1 Capital Ratio (%)', 'Total Capital Ratio (%)', 'Leverage Ratio (%)']
    st.dataframe(banks_df[display_cols].style.format({
        'Total Assets (KES Billion)': '{:.2f}',
        'Equity (KES Billion)': '{:.2f}',
        'Tier 1 Capital Ratio (%)': '{:.2f}',
        'Total Capital Ratio (%)': '{:.2f}',
        'Leverage Ratio (%)': '{:.2f}'
    }), use_container_width=True)
    
    st.markdown("---")
    
    # CBK Requirements
    st.subheader("🎯 CBK Regulatory Requirements")
    req_col1, req_col2, req_col3 = st.columns(3)
    with req_col1:
        st.info(f"**Tier 1 Capital Ratio:** {cbk_requirements['Tier 1 Capital Ratio (%)']}%")
    with req_col2:
        st.info(f"**Total Capital Ratio:** {cbk_requirements['Total Capital Ratio (%)']}%")
    with req_col3:
        st.info(f"**Leverage Ratio:** {cbk_requirements['Leverage Ratio (%)']}%")
    
    st.markdown("---")
    
    # Stress Scenarios
    st.subheader("📋 Defined Stress Test Scenarios")
    for scenario_name, params in stress_scenarios.items():
        with st.expander(f"{scenario_name}: {params['description']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**KES Depreciation:** {params['kes_depreciation']*100:.1f}%")
                st.write(f"**Inflation Shock:** {params['inflation_shock']*100:.1f}%")
                st.write(f"**Interest Rate Change:** {params['interest_rate_change']*100:.1f}%")
            with col2:
                st.write(f"**Equity Price Decline:** {params['equity_price_decline']*100:.1f}%")
                st.write(f"**Credit Loss Rate:** {params['credit_loss_rate']*100:.1f}%")

# Tab 2: Scenario Analysis
with tab2:
    st.markdown('<div class="section-header">Stress Test Results by Scenario</div>', unsafe_allow_html=True)
    
    scenario_choice = st.selectbox("Select Scenario to Analyze:", selected_scenarios)
    
    if scenario_choice:
        scenario_params = stress_scenarios[scenario_choice]
        
        st.info(f"**Description:** {scenario_params['description']}")
        
        # Create results dataframe for selected scenario
        scenario_data = []
        for idx, bank_row in banks_df.iterrows():
            bank_name = bank_row['Bank Name']
            stressed_ratios = stress_results[scenario_choice][bank_name]
            scenario_data.append(stressed_ratios)
        
        scenario_df = pd.DataFrame(scenario_data)
        
        # Display capital ratios
        st.subheader("Capital Ratio Impacts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tier 1 Capital Ratios**")
            tier1_df = scenario_df[['Bank Name', 'Baseline Tier 1 (%)', 'Tier 1 Ratio (%)']].copy()
            tier1_df['Change (pp)'] = tier1_df['Tier 1 Ratio (%)'] - tier1_df['Baseline Tier 1 (%)']
            st.dataframe(tier1_df.style.format({
                'Baseline Tier 1 (%)': '{:.2f}',
                'Tier 1 Ratio (%)': '{:.2f}',
                'Change (pp)': '{:.2f}'
            }), use_container_width=True)
        
        with col2:
            st.markdown("**Total Capital Ratios**")
            total_df = scenario_df[['Bank Name', 'Baseline Total Capital (%)', 'Total Capital Ratio (%)']].copy()
            total_df['Change (pp)'] = total_df['Total Capital Ratio (%)'] - total_df['Baseline Total Capital (%)']
            st.dataframe(total_df.style.format({
                'Baseline Total Capital (%)': '{:.2f}',
                'Total Capital Ratio (%)': '{:.2f}',
                'Change (pp)': '{:.2f}'
            }), use_container_width=True)
        
        st.markdown("---")
        
        # Display losses
        st.subheader("Total Losses by Component (KES Billions)")
        loss_df = scenario_df[['Bank Name', 'Equity Loss (KES B)', 'FX Loss (KES B)', 'Credit Loss (KES B)']].copy()
        loss_df['Total Losses'] = loss_df['Equity Loss (KES B)'] + loss_df['FX Loss (KES B)'] + loss_df['Credit Loss (KES B)']
        st.dataframe(loss_df.style.format({
            'Equity Loss (KES B)': '{:.2f}',
            'FX Loss (KES B)': '{:.2f}',
            'Credit Loss (KES B)': '{:.2f}',
            'Total Losses': '{:.2f}'
        }), use_container_width=True)
        
        st.markdown("---")
        
        # Compliance status
        st.subheader("Capital Adequacy Status")
        for idx, row in scenario_df.iterrows():
            bank_name = row['Bank Name']
            tier1_ok = row['Tier 1 Ratio (%)'] >= cbk_requirements['Tier 1 Capital Ratio (%)']
            total_ok = row['Total Capital Ratio (%)'] >= cbk_requirements['Total Capital Ratio (%)']
            
            col1, col2 = st.columns(2)
            with col1:
                if tier1_ok:
                    st.success(f"✅ {bank_name}: Tier 1 Ratio ADEQUATE ({row['Tier 1 Ratio (%)']:.2f}%)")
                else:
                    st.error(f"❌ {bank_name}: Tier 1 Ratio INADEQUATE ({row['Tier 1 Ratio (%)']:.2f}%)")
            with col2:
                if total_ok:
                    st.success(f"✅ {bank_name}: Total Capital ADEQUATE ({row['Total Capital Ratio (%)']:.2f}%)")
                else:
                    st.error(f"❌ {bank_name}: Total Capital INADEQUATE ({row['Total Capital Ratio (%)']:.2f}%)")

# Tab 3: Capital Ratios
with tab3:
    st.markdown('<div class="section-header">Capital Ratio Evolution Across Scenarios</div>', unsafe_allow_html=True)
    
    # Prepare data for visualization
    scenario_names = selected_scenarios if selected_scenarios else list(stress_scenarios.keys())
    tier1_by_bank = {bank: [] for bank in banks_df['Bank Name']}
    total_capital_by_bank = {bank: [] for bank in banks_df['Bank Name']}
    
    for scenario in scenario_names:
        for bank_name in banks_df['Bank Name']:
            tier1_by_bank[bank_name].append(stress_results[scenario][bank_name]['Tier 1 Ratio (%)'])
            total_capital_by_bank[bank_name].append(stress_results[scenario][bank_name]['Total Capital Ratio (%)'])
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tier 1 Capital Ratio Under Stress", "Total Capital Ratio Under Stress")
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Tier 1 Capital Ratio
    for idx, bank_name in enumerate(banks_df['Bank Name']):
        fig.add_trace(
            go.Scatter(
                x=scenario_names,
                y=tier1_by_bank[bank_name],
                mode='lines+markers',
                name=f'{bank_name}',
                line=dict(width=3, color=colors[idx]),
                marker=dict(size=10),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add CBK requirement line for Tier 1
    fig.add_hline(
        y=cbk_requirements['Tier 1 Capital Ratio (%)'],
        line_dash="dash",
        line_color="red",
        annotation_text="CBK Minimum",
        row=1, col=1
    )
    
    # Total Capital Ratio
    for idx, bank_name in enumerate(banks_df['Bank Name']):
        fig.add_trace(
            go.Scatter(
                x=scenario_names,
                y=total_capital_by_bank[bank_name],
                mode='lines+markers',
                name=f'{bank_name}',
                line=dict(width=3, color=colors[idx]),
                marker=dict(size=10),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Add CBK requirement line for Total Capital
    fig.add_hline(
        y=cbk_requirements['Total Capital Ratio (%)'],
        line_dash="dash",
        line_color="red",
        annotation_text="CBK Minimum",
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Stress Scenario", row=1, col=1)
    fig.update_xaxes(title_text="Stress Scenario", row=1, col=2)
    fig.update_yaxes(title_text="Capital Ratio (%)", row=1, col=1)
    fig.update_yaxes(title_text="Capital Ratio (%)", row=1, col=2)
    
    fig.update_layout(
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Loss Decomposition
with tab4:
    st.markdown('<div class="section-header">Loss Decomposition Analysis</div>', unsafe_allow_html=True)
    
    # Prepare loss component data
    loss_components = []
    for scenario_name in selected_scenarios if selected_scenarios else list(stress_scenarios.keys()):
        for bank_name in banks_df['Bank Name']:
            stressed_data = stress_results[scenario_name][bank_name]
            loss_components.append({
                'Bank': bank_name,
                'Scenario': scenario_name,
                'Equity Loss': stressed_data['Equity Loss (KES B)'],
                'FX Loss': stressed_data['FX Loss (KES B)'],
                'Credit Loss': stressed_data['Credit Loss (KES B)']
            })
    
    loss_comp_df = pd.DataFrame(loss_components)
    
    # Create stacked bar chart
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"{bank}" for bank in banks_df['Bank Name']],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    for col_idx, bank_name in enumerate(banks_df['Bank Name']):
        bank_losses = loss_comp_df[loss_comp_df['Bank'] == bank_name]
        
        fig.add_trace(
            go.Bar(
                x=bank_losses['Scenario'],
                y=bank_losses['Equity Loss'],
                name='Equity Loss',
                marker_color='#ff9999',
                showlegend=(col_idx == 0)
            ),
            row=1, col=col_idx+1
        )
        
        fig.add_trace(
            go.Bar(
                x=bank_losses['Scenario'],
                y=bank_losses['FX Loss'],
                name='FX Loss',
                marker_color='#66b3ff',
                showlegend=(col_idx == 0)
            ),
            row=1, col=col_idx+1
        )
        
        fig.add_trace(
            go.Bar(
                x=bank_losses['Scenario'],
                y=bank_losses['Credit Loss'],
                name='Credit Loss',
                marker_color='#99ff99',
                showlegend=(col_idx == 0)
            ),
            row=1, col=col_idx+1
        )
    
    fig.update_yaxes(title_text="Loss (KES Billions)", row=1, col=1)
    fig.update_layout(
        height=500,
        barmode='stack',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # System-wide losses
    st.subheader("System-Wide Loss Analysis")
    
    scenario_losses = []
    for scenario in selected_scenarios if selected_scenarios else list(stress_scenarios.keys()):
        total_loss = sum(stress_results[scenario][bank]['Equity Loss (KES B)'] + 
                        stress_results[scenario][bank]['FX Loss (KES B)'] + 
                        stress_results[scenario][bank]['Credit Loss (KES B)']
                        for bank in banks_df['Bank Name'])
        scenario_losses.append({'Scenario': scenario, 'Total System Losses (KES B)': total_loss})
    
    losses_df = pd.DataFrame(scenario_losses)
    
    fig = go.Figure(go.Bar(
        x=losses_df['Scenario'],
        y=losses_df['Total System Losses (KES B)'],
        marker_color='#ff7f0e',
        text=losses_df['Total System Losses (KES B)'].round(2),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Total System Losses by Scenario",
        xaxis_title="Scenario",
        yaxis_title="Total Losses (KES Billions)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Risk Assessment
with tab5:
    st.markdown('<div class="section-header">Risk Assessment & Vulnerability Analysis</div>', unsafe_allow_html=True)
    
    # Risk heatmap
    st.subheader("Risk Heatmap: Bank Vulnerability Matrix")
    
    heatmap_matrix = []
    scenario_labels = selected_scenarios if selected_scenarios else list(stress_scenarios.keys())
    
    for scenario in scenario_labels:
        row = []
        for bank in banks_df['Bank Name']:
            stressed_ratio = stress_results[scenario][bank]['Tier 1 Ratio (%)']
            row.append(stressed_ratio)
        heatmap_matrix.append(row)
    
    heatmap_matrix = np.array(heatmap_matrix).T
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=scenario_labels,
        y=banks_df['Bank Name'],
        colorscale='RdYlGn',
        text=np.round(heatmap_matrix, 2),
        texttemplate='%{text:.2f}%',
        textfont={"size": 12},
        colorbar=dict(title="Tier 1 Capital<br>Ratio (%)"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Tier 1 Ratio: %{z:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Risk Heatmap: Bank Vulnerability to Stress Scenarios<br><sub>Red = Below CBK Minimum (10.5%)</sub>",
        xaxis_title="Stress Scenario",
        yaxis_title="Bank",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Reverse stress testing
    st.subheader("🔄 Reverse Stress Testing: Maximum KES Depreciation Tolerance")
    
    tolerance_results = []
    
    for idx, bank_row in banks_df.iterrows():
        bank_name = bank_row['Bank Name']
        max_tolerance = find_max_depreciation_tolerance(bank_row, cbk_requirements)
        
        tolerance_results.append({
            'Bank': bank_name,
            'Max KES Depreciation Tolerance (%)': max_tolerance * 100
        })
    
    tolerance_df = pd.DataFrame(tolerance_results)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=tolerance_df['Bank'],
            y=tolerance_df['Max KES Depreciation Tolerance (%)'],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=tolerance_df['Max KES Depreciation Tolerance (%)'].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Maximum KES Depreciation Tolerance by Bank",
            xaxis_title="Bank",
            yaxis_title="KES Depreciation (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Interpretation:**")
        for _, row in tolerance_df.iterrows():
            st.write(f"**{row['Bank']}:** Can withstand up to {row['Max KES Depreciation Tolerance (%)']:.2f}% KES depreciation before breaching CBK Tier 1 minimum")

# Tab 6: Summary Report
with tab6:
    st.markdown('<div class="section-header">Comprehensive Stress Test Summary Report</div>', unsafe_allow_html=True)
    
    # System-wide summary
    st.subheader("📊 System-Wide Results Summary")
    
    summary_data = []
    for scenario in selected_scenarios if selected_scenarios else list(stress_scenarios.keys()):
        total_system_losses = 0
        compliant_banks = 0
        total_capital_decline = 0
        
        for bank_name in banks_df['Bank Name']:
            stressed_data = stress_results[scenario][bank_name]
            losses = (stressed_data['Equity Loss (KES B)'] + 
                     stressed_data['FX Loss (KES B)'] + 
                     stressed_data['Credit Loss (KES B)'])
            total_system_losses += losses
            
            if stressed_data['Tier 1 Ratio (%)'] >= cbk_requirements['Tier 1 Capital Ratio (%)']:
                compliant_banks += 1
            
            total_capital_decline += (stressed_data['Baseline Tier 1 (%)'] - 
                                     stressed_data['Tier 1 Ratio (%)'])
        
        severity = '🟢 Low' if scenario == 'Base Case' else \
                   '🟡 Medium' if 'Mild' in scenario or 'Moderate' in scenario else \
                   '🔴 High'
        
        summary_data.append({
            'Scenario': scenario,
            'Total Losses (KES B)': round(total_system_losses, 2),
            'Avg Tier 1 Decline (pp)': round(total_capital_decline / 3, 2),
            'Compliant Banks': f"{compliant_banks}/3",
            'Severity': severity
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    st.markdown("---")
    
    # Key findings
    st.subheader("🔍 Key Findings & Vulnerabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1️⃣ Currency Depreciation Risk**")
        st.warning("10% KES depreciation causes material impact on bank capital ratios. FX exposure is a critical vulnerability factor.")
        
        st.markdown("**2️⃣ Credit Risk Concentration**")
        st.warning("Credit loss represents the largest loss component in crisis scenarios. Banks with higher NPL ratios are more vulnerable.")
        
        st.markdown("**3️⃣ Capital Adequacy Compliance**")
        for scenario in selected_scenarios if selected_scenarios else list(stress_scenarios.keys()):
            compliant_count = sum(1 for bank_name in banks_df['Bank Name'] 
                                if stress_results[scenario][bank_name]['Tier 1 Ratio (%)'] >= cbk_requirements['Tier 1 Capital Ratio (%)'])
            if compliant_count < 3:
                st.error(f"{scenario}: Only {compliant_count}/3 banks remain compliant")
    
    with col2:
        st.markdown("**4️⃣ Equity Portfolio Vulnerability**")
        st.warning("35% equity decline creates largest mark-to-market losses. Banks with concentrated equity portfolios at higher risk.")
        
        st.markdown("**5️⃣ Interest Rate Risk**")
        st.warning("Rising rate scenarios compress net interest margins. Combined with credit losses, threatens profitability.")
        
        st.markdown("**6️⃣ Overall Risk Assessment**")
        st.error("**SYSTEM RISK RATING: MODERATE TO HIGH**")
        st.write("Rationale: Baseline capital positions adequate but limited buffer. Vulnerability to combined macro shocks.")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    st.markdown("""
    1. **Capital Buffer Enhancement**: Banks should maintain capital buffers above minimum requirements to absorb potential shocks
    2. **FX Risk Management**: Strengthen hedging strategies to mitigate currency exposure risks
    3. **Credit Quality Monitoring**: Enhanced monitoring of loan portfolios, especially in stressed sectors
    4. **Stress Testing Frequency**: Conduct regular stress tests with updated macroeconomic assumptions
    5. **Contingency Planning**: Develop and maintain robust capital contingency plans
    6. **Portfolio Diversification**: Reduce concentration risks in equity and FX exposures
    """)
    
    st.markdown("---")
    
    # Download report
    st.subheader("📥 Export Results")
    
    if st.button("Generate Downloadable Report"):
        # Compile all results
        report_data = {
            'Bank Baseline': banks_df.to_dict(),
            'Stress Results': {scenario: {bank: stress_results[scenario][bank] 
                                         for bank in banks_df['Bank Name']} 
                              for scenario in stress_scenarios.keys()},
            'Summary': summary_df.to_dict()
        }
        
        st.success("✅ Report generated! (In production, this would trigger a download)")
        st.json(report_data)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Kenyan Banks Stress Testing Application</strong></p>
        <p>Built with Streamlit | Data source: Historical bank prices & CBK Financial Stability Reports</p>
        <p>⚠️ For educational and analytical purposes only</p>
    </div>
""", unsafe_allow_html=True)