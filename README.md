# Financial Risk Management for Kenyan Banks

## Project Overview

This project presents a comprehensive financial risk management analysis for major Kenyan commercial banks (Kenya Commercial Bank, Equity Bank, and Co-operative Bank). The analysis is divided into two main components:

1. **Value-at-Risk (VaR) Analysis** - Quantifying market risk exposure using multiple methodologies
2. **Stress Testing & Scenario Analysis** - Evaluating bank resilience under adverse macroeconomic conditions

The project demonstrates practical application of risk management techniques mandated by the Central Bank of Kenya (CBK) and Basel III regulatory framework.

>  **Live Demo**: [Interactive Dashboard](https://financial-risk-management-var-git-nbfkhttvyoqzkwauu6h5.streamlit.app/)
---

## Table of Contents

- [Data Sources](#data-sources)
- [Part 1: Value-at-Risk Analysis](#part-1-value-at-risk-analysis)
- [Part 2: Stress Testing & Scenario Analysis](#part-2-stress-testing--scenario-analysis)
- [Interactive Dashboards](#interactive-dashboards)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [References](#references)

---

## Data Sources

### Stock Price Data
- **Source**: Wall Street Journal (WSJ) financial database via Investing.com
- **Frequency**: Daily closing prices
- **Period**: August 26, 2020 - November 27, 2025
- **Format**: CSV files with Date and Close columns

### Interest Rate Data
- **Source**: Central Bank of Kenya (CBK)
- **Variables**: 91-day T-Bill rate, Central Bank Rate (CBR), interbank rates
- **Frequency**: Monthly observations
- **Period**: January 2010 - August 2025

### Exchange Rate Data
- **Source**: CBK foreign exchange statistics
- **Pair**: KES/USD
- **Frequency**: Daily rates
- **Period**: November 2020 - November 2025

### Bank Financial Data
- **Source**: Derived from stock market capitalizations and historical data
- **Methodology**: Assets estimated from cumulative price data; equity calculated as 10% of total assets (typical banking leverage)
- **Note**: Actual bank balance sheets from annual reports would provide more precise capital structure data

---

## Part 1: Value-at-Risk Analysis

### Overview

Value-at-Risk (VaR) measures the maximum potential loss in portfolio value over a defined time period at a given confidence level. This analysis implements three industry-standard VaR calculation methods to quantify market risk for Kenyan bank equity portfolios.

### Methodologies Implemented

**1. Variance-Covariance Method (Parametric)**
- Assumes normal distribution of returns
- Formula: VaR = Portfolio Value × Z-score × Standard Deviation
- Fastest computation, suitable for daily risk monitoring

**2. Historical Simulation (Non-Parametric)**
- Uses actual historical return distribution
- No distributional assumptions
- Captures real market behavior including tail events

**3. Monte Carlo Simulation**
- Generates 10,000 random scenarios based on historical statistics
- Flexible approach for complex portfolios
- Industry standard for regulatory reporting

### Advanced Modeling

**Geometric Brownian Motion (GBM)**
- Stochastic process modeling stock price evolution
- Ensures prices remain positive (log-normal distribution)
- Validates simpler VaR methods with sophisticated mathematical framework

**Black-Scholes Option Pricing**
- Calculates cost of portfolio insurance using protective put options
- Demonstrates hedging strategies to mitigate downside risk
- Cost-benefit analysis: hedging vs. accepting VaR losses

### Dataset

- **Banks Analyzed**: KCB, Equity Bank, Co-operative Bank
- **Time Period**: August 2020 - November 2025 (1,305 trading days)
- **Data Source**: Wall Street Journal financial database
- **Portfolio Size**: KES 1,000,000 (standardized for comparison)

### Key Results

**Individual Bank VaR (95% Confidence, 1-Day Horizon)**


| Bank         | Variance-Covariance | Historical Simulation | Monte Carlo |
|--------------|--------------------:|----------------------:|------------:|
| KCB          | 27,591 KES          | 22,604 KES            | 27,715 KES  |
| Equity Bank  | 23,252 KES          | 21,086 KES            | 23,169 KES  |
| Co-op Bank   | 23,744 KES          | 17,971 KES            | 23,843 KES  |

**Interpretation**: KCB exhibits the highest market risk across all methods, with potential 1-day losses of approximately 2.8% of portfolio value at 95% confidence. The convergence of Variance-Covariance and Monte Carlo methods (within 0.5%) validates the normal distribution assumption for these assets.

<img width="1394" height="500" alt="image" src="https://github.com/user-attachments/assets/d59a398f-cb01-446e-a143-c236cf68800b" />

**Portfolio Diversification Analysis**

- **Portfolio Composition**: 40% KCB, 30% Equity Bank, 30% Co-op Bank
- **Diversified Portfolio VaR**: 17,543 KES
- **Weighted Individual VaR**: 25,190 KES
- **Risk Reduction**: 30.35%

**Interpretation**: Diversification across three banks reduces portfolio VaR by nearly one-third compared to concentrated positions. This demonstrates Modern Portfolio Theory in practice - imperfect correlation between banks (0.68-0.73) enables significant risk reduction without sacrificing expected returns (15.67% annualized).

<img width="1065" height="500" alt="newplot (4)" src="https://github.com/user-attachments/assets/a2807bbc-8bbc-45a4-b5d4-683590968fdf" />

**Hedging Strategy: Black-Scholes Put Options**

30-Day Protective Put Pricing (95% Strike):
- **Put Option Cost**: 9,181 KES (0.92% of portfolio)
- **95% VaR Protection**: 27,715 KES potential loss
- **Cost/Loss Ratio**: 0.33x
- **Potential Savings**: 66.9%

**Interpretation**: Purchasing protective puts at 95% strike price costs less than 1% of portfolio value but protects against nearly 3% downside risk. This 0.33 cost-to-protection ratio makes hedging highly cost-effective during volatile periods (elections, currency instability, global crises).

<img width="1065" height="400" alt="newplot (5)" src="https://github.com/user-attachments/assets/143384e5-882f-45f1-9277-809a0864a9b8" />

### Files

- `risk_management_var.ipynb` - Complete VaR analysis with visualizations
- `var_app.py` - Streamlit interactive dashboard
- `kcb_historical_prices.csv`, `equity_historical_prices.csv`, `coop_historical_prices.csv` - Stock price data

---

## Part 2: Stress Testing & Scenario Analysis

### Overview

Stress testing evaluates how bank capital adequacy ratios respond to severe but plausible adverse scenarios. This analysis implements CBK-aligned stress scenarios to assess systemic risk and identify vulnerabilities in the Kenyan banking sector.

### Stress Test Scenarios

Five scenarios ranging from baseline to systemic crisis:

**1. Base Case** - No shocks (baseline)

**2. Mild Depreciation**
- KES depreciation: 5%
- Inflation shock: 2%
- Interest rate increase: 1%
- Equity decline: 5%
- Credit loss rate: 1%

**3. Moderate Depreciation**
- KES depreciation: 10%
- Inflation shock: 5%
- Interest rate increase: 2%
- Equity decline: 15%
- Credit loss rate: 3%

**4. Severe Depreciation**
- KES depreciation: 15%
- Inflation shock: 8%
- Interest rate increase: 3%
- Equity decline: 25%
- Credit loss rate: 5%

**5. Combined Crisis**
- KES depreciation: 20%
- Inflation shock: 10%
- Interest rate increase: 4%
- Equity decline: 35%
- Credit loss rate: 8%

### Methodology

**Impact Channels**:
1. **Foreign Exchange Risk**: Currency depreciation increases risk-weighted assets for FX-exposed portfolios
2. **Market Risk**: Equity price declines create mark-to-market losses on trading portfolios
3. **Credit Risk**: Rising NPL ratios deplete Tier 2 capital through provisioning requirements

**Capital Ratio Calculation**:
```
Tier 1 Capital Ratio (%) = Tier 1 Capital / Risk-Weighted Assets × 100
Total Capital Ratio (%) = (Tier 1 + Tier 2 Capital) / Risk-Weighted Assets × 100
```

**CBK Regulatory Minimums**:
- Tier 1 Capital Ratio: 10.5%
- Total Capital Ratio: 14.5%
- Leverage Ratio: 6.0%

### Key Results

**Baseline Capital Position (Derived from Stock Price Data)**


| Bank         | Total Assets (KES B) | Equity (KES B) | Tier 1 Ratio (%) | Total Capital Ratio (%) | Leverage Ratio (%) |
|--------------|---------------------:|---------------:|-----------------:|------------------------:|-------------------:|
| KCB          | 87.5                 | 8.8            | 16.0             | 20.0                    | 10.0               |
| Equity Bank  | 75.2                 | 7.5            | 16.0             | 20.0                    | 10.0               |
| Co-op Bank   | 68.9                 | 6.9            | 16.0             | 20.0                    | 10.0               |

**Stress Test Impact on Tier 1 Capital Ratios**

<img width="672" height="605" alt="image" src="https://github.com/user-attachments/assets/d76a53b9-1d7c-4c1d-aaef-22bfa0264a0e" />

**Combined Crisis Scenario Results**:
- **KCB**: Tier 1 ratio falls from 16.0% to 11.2% (4.8 percentage point decline)
- **Equity Bank**: Tier 1 ratio falls from 16.0% to 11.5% (4.5 percentage point decline)
- **Co-op Bank**: Tier 1 ratio falls from 16.0% to 12.1% (3.9 percentage point decline)

**Interpretation**: All three banks maintain capital ratios above CBK minimums even under the most severe combined crisis scenario. However, capital buffers compress significantly - KCB's cushion above the 10.5% minimum reduces from 5.5 percentage points to just 0.7 percentage points. This limited buffer indicates vulnerability to prolonged crisis conditions.

**Loss Decomposition Analysis**

<img width="1065" height="500" alt="newplot (6)" src="https://github.com/user-attachments/assets/96b2dc0b-9577-4444-8701-77a3570125c9" />

**Combined Crisis Total Losses**:
- **KCB**: 4.2 billion KES (Equity: 1.1B, FX: 2.1B, Credit: 1.0B)
- **Equity Bank**: 3.5 billion KES (Equity: 0.9B, FX: 1.5B, Credit: 1.1B)
- **Co-op Bank**: 2.8 billion KES (Equity: 0.7B, FX: 1.0B, Credit: 1.1B)

**Interpretation**: Foreign exchange losses constitute the largest component (approximately 50%) in severe scenarios, highlighting FX exposure as the primary vulnerability. KCB's higher FX exposure (25% vs 20% for Equity, 15% for Co-op) explains its greater stress test losses. Credit losses become material in severe scenarios, emphasizing the importance of loan portfolio quality.

**Compliance Status Under Stress**

<img width="1065" height="500" alt="newplot (7)" src="https://github.com/user-attachments/assets/c190fdb5-defc-4409-a33a-e54638960332" />

- **Base Case**: 3/3 banks compliant
- **Mild Depreciation**: 3/3 banks compliant
- **Moderate Depreciation**: 3/3 banks compliant
- **Severe Depreciation**: 3/3 banks compliant
- **Combined Crisis**: 3/3 banks compliant (but with reduced buffers)

**Reverse Stress Testing: Maximum KES Depreciation Tolerance**

<img width="702" height="400" alt="newplot (8)" src="https://github.com/user-attachments/assets/03b2846d-2500-4438-b082-2ef2cc900127" />

- **KCB**: 22.5% maximum KES depreciation before breaching Tier 1 minimum
- **Equity Bank**: 24.1% maximum tolerance
- **Co-op Bank**: 26.8% maximum tolerance

**Interpretation**: Co-op Bank demonstrates highest resilience to currency shocks due to lower FX exposure. KCB's 22.5% tolerance means a depreciation beyond this threshold would push its Tier 1 ratio below the 10.5% regulatory minimum, potentially requiring capital injections or regulatory intervention.

### System-Wide Risk Assessment

**Aggregate Results**:
- **Total System Losses (Combined Crisis)**: 10.5 billion KES
- **Average Tier 1 Ratio Decline**: 4.4 percentage points
- **Banks Meeting CBK Requirements**: 3/3 (all scenarios)

**Risk Rating**: MODERATE TO HIGH

**Rationale**:
1. Baseline capital positions are adequate but provide limited buffers
2. Vulnerability to combined macroeconomic shocks (currency + credit)
3. FX exposure creates concentration risk requiring active management
4. Increasing probability of adverse scenarios given recent economic headwinds

### Files

- `stress_testing.ipynb` - Complete stress testing analysis
- `stress_testing_app.py` - Streamlit interactive dashboard

---

## Interactive Dashboards

### VaR Analysis Dashboard
**URL**: https://financial-risk-management-var-git-nbfkhttvyoqzkwauu6h5.streamlit.app/

<img width="1903" height="875" alt="image" src="https://github.com/user-attachments/assets/fad02cab-5d5d-42c8-a377-34b9871dea6e" />


**Features**:
- Interactive VaR calculations with adjustable parameters (portfolio value, confidence level, simulation count)
- Real-time portfolio weight customization with automatic normalization
- Comparative analysis across all three banks and methods
- GBM price simulation with adjustable time horizons
- Black-Scholes option pricing calculator with Greeks analysis
- Downloadable CSV results for reporting

**Navigation**:
1. Overview - Stock price trends and descriptive statistics
2. VaR Analysis - Calculate and compare VaR across methods
3. GBM Simulation - Interactive price path simulations
4. Options Pricing - Hedging cost calculator
5. Comparison - Portfolio analysis and diversification benefits

### Stress Testing Dashboard
**URL**: https://financial-risk-management-stress-testing-yqgciqycxhxoesmf5bbem8.streamlit.app/

<img width="1919" height="870" alt="image" src="https://github.com/user-attachments/assets/a84c5123-ada7-4258-bcc6-4d8681945c1d" />


**Features**:
- Five predefined stress scenarios based on CBK framework
- Custom scenario builder for user-defined stress parameters
- Real-time capital ratio calculations and compliance checking
- Interactive visualizations of loss components
- Risk heatmaps showing bank-scenario vulnerability
- Reverse stress testing for maximum shock tolerance
- System-wide risk assessment and scenario probability weighting

**Navigation**:
1. Scenario Overview - Definitions and parameter specifications
2. Capital Impact - Bank-by-bank stress test results
3. Loss Analysis - Decomposition of losses by component
4. Risk Heatmap - Vulnerability assessment matrix
5. Reverse Stress Test - Maximum tolerance calculations
6. System Summary - Aggregate findings and recommendations


---

## Installation & Setup

### Prerequisites
```
Python 3.8+
pip package manager
```

### Required Libraries
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- pandas - Data manipulation
- numpy - Numerical computing
- scipy - Statistical functions
- plotly - Interactive visualizations
- streamlit - Web dashboard framework
- matplotlib - Static plotting

### Running Jupyter Notebooks

**VaR Analysis**:
```bash
jupyter notebook risk_management_var.ipynb
```

**Stress Testing**:
```bash
jupyter notebook stress_testing.ipynb
```

### Running Streamlit Apps Locally

**VaR Dashboard**:
```bash
streamlit run var_app.py
```

**Stress Testing Dashboard**:
```bash
streamlit run stress_testing_app.py
```

Apps will open in your default browser at `http://localhost:8501`

---

## Project Structure

```
financial-risk-management/
│
├── risk_management_var.ipynb          # VaR analysis notebook
├── stress_testing.ipynb               # Stress testing notebook
│
├── var_app.py                         # VaR Streamlit dashboard
├── stress_testing_app.py              # Stress testing Streamlit dashboard
│
├── kcb_historical_prices.csv          # KCB stock data
├── equity_historical_prices.csv       # Equity Bank stock data
├── coop_historical_prices.csv         # Co-op Bank stock data
├── USD_KES_historical_data.csv        # Exchange rate data
├── central_bank_rates.csv             # Interest rate data
│
├── requirements.txt                   # Python dependencies
└── README.md                          
```

---

## Key Findings

### VaR Analysis Conclusions

1. **Methodology Validation**: Convergence of Variance-Covariance and Monte Carlo methods (within 1.4%) confirms normal distribution assumption is reasonable for Kenyan bank stocks.

2. **Risk Hierarchy**: KCB demonstrates 15-20% higher VaR than Equity Bank and Co-op Bank, indicating greater volatility. This suggests KCB's larger market capitalization and broader business activities create higher market risk exposure.

3. **Diversification Effectiveness**: 30.35% risk reduction through portfolio diversification demonstrates powerful risk mitigation without sacrificing returns. The Sharpe ratio of 0.91 for the diversified portfolio indicates acceptable risk-adjusted performance.

4. **Hedging Cost-Effectiveness**: Protective puts offer affordable insurance with cost-to-protection ratios as low as 0.33. This makes option-based hedging particularly attractive during:
   - Election periods (political uncertainty)
   - Currency instability (KES depreciation expectations)
   - Global market volatility (financial crises)
   - Regulatory changes affecting banking sector

5. **GBM Validation**: Advanced modeling confirms simpler VaR methods are statistically sound for Kenyan banking stocks, providing confidence in regulatory reporting using these techniques.

### Stress Testing Conclusions

1. **Baseline Resilience**: All three banks maintain adequate capital buffers above CBK requirements under baseline conditions, with Tier 1 ratios of 16% (5.5 percentage points above the 10.5% minimum).

2. **Currency Risk Dominance**: Foreign exchange exposure represents the primary systemic vulnerability. In the Combined Crisis scenario, FX losses constitute approximately 50% of total losses across all banks. This highlights the importance of:
   - Active FX risk management and hedging programs
   - Diversification of foreign currency funding sources
   - Limits on unhedged FX positions
   - Stress testing FX exposures regularly

3. **Differential Vulnerability**: Co-op Bank demonstrates greatest resilience (26.8% depreciation tolerance) due to lower FX exposure, while KCB shows highest vulnerability (22.5% tolerance) due to greater international operations and FX-denominated lending.

4. **Limited Capital Buffers**: While all banks remain compliant even under severe stress, capital cushions compress significantly. In the Combined Crisis scenario, buffers shrink to 0.7-1.6 percentage points above minimums - insufficient to absorb prolonged crises or multiple consecutive shocks.

5. **Credit Quality Critical**: NPL ratios and credit loss provisioning become material stress factors in severe scenarios. Banks with higher baseline NPL ratios (Co-op: 4.5%) face greater capital erosion during economic downturns.

### Integrated Risk Management Recommendations

**For Banks**:
1. Implement multiple VaR methodologies (don't rely on single approach)
2. Conduct weekly VaR calculations; monthly stress testing
3. Establish hedging protocols for FX and equity exposures
4. Maintain diversified portfolios across sectors and geographies
5. Build capital buffers beyond regulatory minimums (target 2-3 percentage points above requirements)
6. Develop contingency capital plans for crisis scenarios

**For Central Bank of Kenya (CBK)**:
1. Mandate quarterly stress testing with standardized scenarios
2. Require banks to report VaR alongside traditional capital ratios
3. Develop macroprudential policies targeting systemic FX risk
4. Promote derivatives market development for hedging instruments
5. Implement countercyclical capital buffers during economic expansions
6. Conduct annual system-wide stress tests with coordinated scenarios

**For Investors and Stakeholders**:
1. Monitor bank capital ratios relative to stress-tested minimums
2. Assess FX exposure as key vulnerability indicator
3. Favor banks with diversified portfolios and strong credit quality
4. Consider hedging equity positions during high-volatility periods
5. Demand transparency in risk reporting and stress test methodologies

---

## Limitations

### VaR Analysis Limitations
- Focus on equity portfolios only (excludes loan books, bonds, real estate)
- Assumes liquid markets (may not hold during crises)
- Historical data may not predict future tail events (e.g., COVID-19)
- Black-Scholes assumptions (constant volatility, no dividends) are approximations
- Does not account for transaction costs or bid-ask spreads

### Stress Testing Limitations
- Simplified balance sheet structure based on stock price proxies
- Does not model second-round effects (e.g., contagion, liquidity spirals)
- Scenario probabilities are subjective estimates
- Assumes static bank behavior (in reality, banks would adjust strategies under stress)
- Limited to five predefined scenarios (infinite possible combinations exist)

---

## Future Research Directions

1. **Credit VaR Models**: Extend analysis to loan portfolios using default probabilities and loss-given-default estimates

2. **Liquidity Risk**: Incorporate funding liquidity stress scenarios (deposit runs, interbank freezes)

3. **Contagion Modeling**: Network analysis of interbank exposures and systemic risk propagation

4. **Machine Learning**: Apply LSTM networks for volatility forecasting; GARCH models for time-varying risk

5. **ESG Risk Integration**: Incorporate climate risk scenarios (drought, flooding) and social factors (political instability)

6. **Regulatory Reporting Automation**: Develop end-to-end pipeline from data ingestion to CBK report generation

7. **Real-Time Monitoring**: Implement intraday VaR tracking for trading desks

---

## References

### Academic and Regulatory Sources

1. Basel Committee on Banking Supervision (2019). "Minimum Capital Requirements for Market Risk." Bank for International Settlements.

2. Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk." 3rd Edition, McGraw-Hill.

3. Central Bank of Kenya (2023). "Bank Supervision Annual Report 2023." CBK Financial Stability Department.

4. Central Bank of Kenya (2024). "Financial Stability Report - First Half 2024." CBK Publications.

5. Hull, J. C. (2018). "Options, Futures, and Other Derivatives." 10th Edition, Pearson.

6. Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77-91.

7. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." Journal of Political Economy, 81(3), 637-654.

### Data Sources

8. Wall Street Journal (2025). "Market Data Center - Nairobi Securities Exchange." WSJ Financial Database.

9. Investing.com (2025). "Kenya Stock Market Historical Data." Retrieved from https://www.investing.com

10. Central Bank of Kenya (2025). "Statistical Bulletin - Interest Rates and Exchange Rates." CBK Statistics Department.

11. Nairobi Securities Exchange (2025). "Market Statistics and Historical Data." NSE Official Website.

### Industry Reports

12. Kenya Institute for Public Policy Research and Analysis (2023). "Financial Sector Stability and Economic Growth in Kenya." KIPPRA Working Paper Series.

13. Capital Markets Authority (2024). "Statistical Bulletin Q2 2024." CMA Kenya Publications.

---

## Contributing

This project was developed as part of an academic research initiative on financial risk management in East African banking. Contributions, suggestions, and feedback are welcome.

**Contact**: For questions or collaboration inquiries, please open an issue in the repository or contact the project maintainers:
Part 1: https://github.com/DevAime
Part 2: https://github.com/ArnoldBophine

---


**Last Updated**: November 2025
