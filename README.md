# Gas Temperature Correlation Analysis - Ciudad Juárez, Chihuahua

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-orange)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-yellow)

An analytical study investigating the relationship between natural gas prices and temperature trends in Ciudad Juárez, Chihuahua. This project demonstrates time series analysis, correlation studies, and environmental economics using Python's data science stack.

## 🔥 Project Overview

This project analyzes how temperature variations affect natural gas prices in a border city with extreme climate conditions. The analysis includes:

- **Time Series Analysis**: Decomposition of gas price and temperature data into trend, seasonal, and residual components
- **Correlation Studies**: Statistical correlation between temperature anomalies and gas price fluctuations
- **Regression Modeling**: Predictive models to estimate gas price changes based on temperature patterns
- **Seasonal Analysis**: Examination of winter heating demand versus summer cooling effects
- **Climate Impact Assessment**: Quantifying the economic impact of temperature changes on energy costs

## 📈 Key Research Questions

1. **Does temperature significantly correlate with natural gas prices in Ciudad Juárez?**
2. **What is the time lag between temperature changes and price adjustments?**
3. **How does seasonal variation affect the price-temperature relationship?**
4. **Can temperature data predict short-term gas price movements?**

## 🏗️ Project Structure

```
gas-temperature-analysis/
│
├── analysis.ipynb          # Main Jupyter notebook with complete analysis
├── gas_temperature_analysis.py  # Python script with reusable analysis functions
├── requirements.txt        # Dependencies for reproducing the environment
├── README.md               # Project documentation
├── environment.yml         # Conda environment configuration
└── .gitignore              # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gas-temperature-analysis.git
   cd gas-temperature-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

**Option 1: Jupyter Notebook (Interactive Exploration)**
```bash
jupyter notebook analysis.ipynb
```

**Option 2: Python Script (Automated Analysis)**
```bash
python gas_temperature_analysis.py
```

## 📊 Methodology

### Data Generation
Synthetic datasets are generated to simulate realistic scenarios:
- **Gas Prices**: Historical natural gas price patterns with seasonal winter peaks
- **Temperature Data**: Daily temperature records with climate change trend
- **Economic Factors**: Simulated demand variables and market conditions

### Analytical Techniques
1. **Time Series Decomposition**: Separating trend, seasonal, and residual components
2. **Cross-Correlation Analysis**: Identifying lag relationships between variables
3. **Multiple Regression**: Modeling gas price as function of temperature and other factors
4. **Granger Causality Tests**: Investigating predictive relationships
5. **Rolling Correlation Windows**: Analyzing how relationships change over time

## 🌡️ Expected Findings

### Temperature-Price Relationship
- **Negative Correlation**: Expected inverse relationship (colder temperatures → higher gas demand → higher prices)
- **Seasonal Lag**: Price adjustments may lag temperature changes by 1-2 weeks
- **Threshold Effects**: Temperature thresholds where price sensitivity increases significantly
- **Climate Change Impact**: Long-term warming trend may reduce winter heating demand

### Economic Implications
- **Heating Degree Days**: Correlation between HDD (Heating Degree Days) and gas consumption
- **Price Elasticity**: How sensitive gas prices are to temperature changes
- **Forecast Accuracy**: Temperature-based price prediction models

## 📈 Visualization Portfolio

The project includes the following key visualizations:

1. **Dual-Axis Time Series**: Gas prices and temperature on shared timeline
2. **Scatter Plot with Regression Line**: Temperature vs. gas price relationship
3. **Cross-Correlation Plot**: Lag analysis between variables
4. **Seasonal Decomposition**: Separate trend, seasonal, and residual components
5. **Rolling Correlation Heatmap**: How correlation changes over time
6. **Residual Analysis**: Model diagnostics and validation

## 🔬 Technical Implementation

### Data Processing Pipeline
1. **Data Generation**: Synthetic time series with realistic statistical properties
2. **Feature Engineering**: Creating derived features (moving averages, lags, differences)
3. **Statistical Testing**: Hypothesis tests for stationarity, correlation significance
4. **Model Validation**: Cross-validation and out-of-sample testing

### Advanced Analytics
- **ARIMA Modeling**: Time series forecasting for gas prices
- **Vector Autoregression (VAR)**: Multivariate time series analysis
- **Cointegration Tests**: Long-term equilibrium relationships
- **Breakpoint Analysis**: Detecting structural changes in relationships

## 🎯 Skills Demonstrated

This project showcases the following data analysis skills:

- **Time Series Analysis**: ARIMA, decomposition, stationarity testing
- **Statistical Modeling**: Regression, correlation, hypothesis testing
- **Environmental Economics**: Energy price analysis in climate context
- **Data Visualization**: Time series plots, heatmaps, diagnostic charts
- **Geospatial Analysis**: Location-specific climate impacts
- **Research Methodology**: Hypothesis formulation and testing

## 📚 Dataset Details

The analysis uses synthetic datasets with the following characteristics:

### Natural Gas Price Data
- **Time Period**: 5 years of daily prices (2019-2024)
- **Features**: Price per MMBtu, price changes, volatility measures
- **Seasonality**: Winter peaks (December-February), summer troughs
- **Trend**: Long-term price trajectory with market fluctuations

### Temperature Data (Ciudad Juárez)
- **Climate Profile**: Semi-arid with extreme temperatures (hot summers, cold winters)
- **Features**: Daily average temperature, min/max, heating/cooling degree days
- **Trend**: Climate change warming trend (+0.5°C over 5 years)
- **Variability**: Seasonal patterns and extreme weather events

### Derived Features
- **Heating Degree Days (HDD)**: Measure of heating demand
- **Cooling Degree Days (CDD)**: Measure of cooling demand
- **Price Volatility**: Rolling standard deviation of prices
- **Temperature Anomalies**: Deviations from seasonal norms

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add some improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/gas-temperature-analysis](https://github.com/yourusername/gas-temperature-analysis)

---

*This project was created as a data analysis portfolio piece. The analysis uses synthetic data to demonstrate analytical techniques while protecting proprietary information. Real-world applications would require access to actual gas price and temperature datasets.*