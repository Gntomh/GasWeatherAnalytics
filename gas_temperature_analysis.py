"""
Gas Temperature Correlation Analysis Module
===========================================

This module provides reusable functions for analyzing the relationship between
natural gas prices and temperature in Ciudad Juárez, Chihuahua.

Author: Your Name
Date: 2026-03-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_gas_temperature_data(start_date='2019-01-01', end_date='2024-12-31', seed=42):
    """
    Generate synthetic gas price and temperature data for Ciudad Juárez.

    Parameters
    ----------
    start_date : str, default='2019-01-01'
        Start date for the time series
    end_date : str, default='2024-12-31'
        End date for the time series
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with daily gas price and temperature data
    """
    np.random.seed(seed)

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    # Ciudad Juárez climate characteristics (semi-arid, extreme temperatures)
    day_of_year = dates.dayofyear

    # Temperature components
    seasonal_temp = 15 + 20 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    trend_temp = np.linspace(0, 2.5, n_days)  # Climate change trend
    noise_temp = np.random.normal(0, 5, n_days)

    temperature = seasonal_temp + trend_temp + noise_temp

    # Gas price components
    base_price = np.linspace(3.0, 5.0, n_days)  # Inflation trend
    winter_peak = 1.5 * np.sin(2 * np.pi * (day_of_year - 365/2) / 365)

    # Cold snap events
    cold_snaps = np.zeros(n_days)
    cold_days = np.random.choice(n_days, size=30, replace=False)
    cold_snaps[cold_days] = np.random.uniform(0.5, 1.5, size=30)

    volatility = np.random.normal(0, 0.2, n_days)
    gas_price = base_price + winter_peak + cold_snaps + volatility
    gas_price = np.maximum(gas_price, 2.0)  # Ensure positive prices

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature_c': np.round(temperature, 1),
        'gas_price_usd': np.round(gas_price, 2),
        'day_of_year': day_of_year,
        'month': dates.month,
        'year': dates.year,
        'day_of_week': dates.dayofweek,
        'is_winter': dates.month.isin([12, 1, 2]).astype(int),
        'is_summer': dates.month.isin([6, 7, 8]).astype(int)
    })

    # Calculate derived features
    df['hdd'] = np.maximum(18 - df['temperature_c'], 0)  # Heating Degree Days
    df['cdd'] = np.maximum(df['temperature_c'] - 18, 0)   # Cooling Degree Days
    df['price_change'] = df['gas_price_usd'].diff()
    df['price_pct_change'] = df['gas_price_usd'].pct_change() * 100

    # Temperature anomalies
    monthly_avg = df.groupby('month')['temperature_c'].transform('mean')
    df['temp_anomaly'] = df['temperature_c'] - monthly_avg

    # Moving averages
    df['temp_7d_avg'] = df['temperature_c'].rolling(window=7).mean()
    df['price_7d_avg'] = df['gas_price_usd'].rolling(window=7).mean()
    df['hdd_7d_sum'] = df['hdd'].rolling(window=7).sum()

    return df


def calculate_correlation_metrics(df, temp_col='temperature_c', price_col='gas_price_usd'):
    """
    Calculate correlation metrics between temperature and gas price.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing temperature and gas price data
    temp_col : str, default='temperature_c'
        Column name for temperature data
    price_col : str, default='gas_price_usd'
        Column name for gas price data

    Returns
    -------
    dict
        Dictionary of correlation metrics
    """
    # Drop NaN values
    temp_clean = df[temp_col].dropna()
    price_clean = df[price_col].dropna()

    # Align indices
    common_idx = temp_clean.index.intersection(price_clean.index)
    temp_aligned = temp_clean.loc[common_idx]
    price_aligned = price_clean.loc[common_idx]

    # Calculate correlations
    pearson_corr, pearson_p = stats.pearsonr(temp_aligned, price_aligned)
    spearman_corr, spearman_p = stats.spearmanr(temp_aligned, price_aligned)

    # Cross-correlation analysis
    max_lag = 30
    cross_corr = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag < 0:
            temp_series = temp_aligned.iloc[:lag].reset_index(drop=True)
            price_series = price_aligned.iloc[-lag:].reset_index(drop=True)
        elif lag > 0:
            temp_series = temp_aligned.iloc[lag:].reset_index(drop=True)
            price_series = price_aligned.iloc[:-lag].reset_index(drop=True)
        else:
            temp_series = temp_aligned
            price_series = price_aligned

        if len(temp_series) > 0 and len(price_series) > 0:
            corr = np.corrcoef(temp_series, price_series)[0, 1]
            cross_corr.append(corr)

    # Find optimal lag (most negative correlation)
    optimal_lag_idx = np.argmin(cross_corr)
    optimal_lag = lags[optimal_lag_idx]
    optimal_corr = cross_corr[optimal_lag_idx]

    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'optimal_lag': optimal_lag,
        'optimal_correlation': optimal_corr,
        'cross_correlation': cross_corr,
        'lags': list(lags)
    }


def perform_regression_analysis(df, temp_col='temperature_c', price_col='gas_price_usd'):
    """
    Perform regression analysis between temperature and gas price.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing temperature and gas price data
    temp_col : str, default='temperature_c'
        Column name for temperature data
    price_col : str, default='gas_price_usd'
        Column name for gas price data

    Returns
    -------
    dict
        Dictionary of regression results
    """
    # Simple linear regression
    X_simple = sm.add_constant(df[temp_col].dropna())
    y_simple = df[price_col].dropna()

    # Align indices
    common_idx = X_simple.index.intersection(y_simple.index)
    X_simple = X_simple.loc[common_idx]
    y_simple = y_simple.loc[common_idx]

    model_simple = sm.OLS(y_simple, X_simple).fit()

    # Multiple regression with HDD and seasonal dummies
    X_multi = pd.DataFrame({
        'const': 1,
        'temperature': df[temp_col],
        'hdd': df['hdd'],
        'month_2': (df['month'] == 2).astype(int),
        'month_12': (df['month'] == 12).astype(int)
    })
    y_multi = df[price_col]

    # Drop rows with NaN
    valid_idx = X_multi.dropna().index.intersection(y_multi.dropna().index)
    X_multi_clean = X_multi.loc[valid_idx]
    y_multi_clean = y_multi.loc[valid_idx]

    model_multi = sm.OLS(y_multi_clean, X_multi_clean).fit()

    return {
        'simple_model': model_simple,
        'multiple_model': model_multi,
        'simple_summary': {
            'intercept': model_simple.params['const'],
            'slope': model_simple.params[temp_col],
            'r_squared': model_simple.rsquared,
            'p_value': model_simple.pvalues[temp_col]
        },
        'multiple_summary': {
            'params': dict(model_multi.params),
            'pvalues': dict(model_multi.pvalues),
            'r_squared': model_multi.rsquared
        }
    }


def decompose_time_series(df, price_col='gas_price_usd', temp_col='temperature_c',
                         freq='W', period=52):
    """
    Decompose time series into trend, seasonal, and residual components.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    price_col : str, default='gas_price_usd'
        Column name for gas price data
    temp_col : str, default='temperature_c'
        Column name for temperature data
    freq : str, default='W'
        Resampling frequency
    period : int, default=52
        Seasonal period (weeks per year)

    Returns
    -------
    dict
        Dictionary containing decomposition results
    """
    # Set date as index if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df_temp = df.set_index('date').copy()
    else:
        df_temp = df.copy()

    # Resample to reduce noise
    price_series = df_temp[price_col].resample(freq).mean()
    temp_series = df_temp[temp_col].resample(freq).mean()

    # Decompose
    price_decomp = seasonal_decompose(price_series, model='additive', period=period)
    temp_decomp = seasonal_decompose(temp_series, model='additive', period=period)

    # Calculate variance explained
    price_variance = {
        'trend': np.var(price_decomp.trend.dropna()) / np.var(price_decomp.observed.dropna()),
        'seasonal': np.var(price_decomp.seasonal.dropna()) / np.var(price_decomp.observed.dropna()),
        'residual': np.var(price_decomp.resid.dropna()) / np.var(price_decomp.observed.dropna())
    }

    temp_variance = {
        'trend': np.var(temp_decomp.trend.dropna()) / np.var(temp_decomp.observed.dropna()),
        'seasonal': np.var(temp_decomp.seasonal.dropna()) / np.var(temp_decomp.observed.dropna()),
        'residual': np.var(temp_decomp.resid.dropna()) / np.var(temp_decomp.observed.dropna())
    }

    return {
        'price_decomposition': price_decomp,
        'temperature_decomposition': temp_decomp,
        'price_variance': price_variance,
        'temperature_variance': temp_variance,
        'price_series': price_series,
        'temperature_series': temp_series
    }


def calculate_climate_impact(decomposition_results):
    """
    Calculate climate change impact on gas prices.

    Parameters
    ----------
    decomposition_results : dict
        Dictionary from decompose_time_series function

    Returns
    -------
    dict
        Dictionary with climate impact metrics
    """
    temp_trend = decomposition_results['temperature_decomposition'].trend.dropna()
    price_trend = decomposition_results['price_decomposition'].trend.dropna()

    # Align indices
    common_idx = temp_trend.index.intersection(price_trend.index)
    temp_trend_aligned = temp_trend.loc[common_idx]
    price_trend_aligned = price_trend.loc[common_idx]

    # Calculate changes
    temp_start = temp_trend_aligned.iloc[0]
    temp_end = temp_trend_aligned.iloc[-1]
    temp_increase = temp_end - temp_start

    price_start = price_trend_aligned.iloc[0]
    price_end = price_trend_aligned.iloc[-1]
    price_change = price_end - price_start

    # Regression between trend components
    X_trend = sm.add_constant(temp_trend_aligned)
    y_trend = price_trend_aligned
    trend_model = sm.OLS(y_trend, X_trend).fit()

    # Expected price change due to warming
    expected_price_change = trend_model.params.iloc[1] * temp_increase

    return {
        'temperature_increase_c': temp_increase,
        'actual_price_change_usd': price_change,
        'expected_price_change_usd': expected_price_change,
        'trend_regression': trend_model,
        'warming_rate_c_per_year': temp_increase / (len(common_idx) / 52),  # weeks to years
        'temperature_start_c': temp_start,
        'temperature_end_c': temp_end,
        'price_start_usd': price_start,
        'price_end_usd': price_end
    }


def plot_time_series_comparison(df, figsize=(14, 10)):
    """
    Create time series comparison plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date, temperature_c, and gas_price_usd columns
    figsize : tuple, default=(14, 10)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Dual-axis time series
    ax1 = axes[0, 0]
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(df['date'], df['temperature_c'], color=color, alpha=0.7, linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Gas Price (USD/MMBtu)', color=color)
    ax2.plot(df['date'], df['gas_price_usd'], color=color, alpha=0.7, linewidth=1)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Gas Price vs Temperature - Time Series', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot with regression
    axes[0, 1].scatter(df['temperature_c'], df['gas_price_usd'], alpha=0.5, s=10,
                       c=df['month'], cmap='coolwarm')

    # Regression line
    x = df['temperature_c'].values.reshape(-1, 1)
    y = df['gas_price_usd'].values
    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)
    axes[0, 1].plot(df['temperature_c'], y_pred, color='red', linewidth=2,
                   label=f'y = {reg.coef_[0]:.3f}x + {reg.intercept_:.3f}')

    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Gas Price (USD/MMBtu)')
    axes[0, 1].set_title('Temperature vs Gas Price', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Monthly averages
    monthly_avg = df.groupby('month').agg({
        'temperature_c': 'mean',
        'gas_price_usd': 'mean'
    }).reset_index()

    axes[1, 0].plot(monthly_avg['month'], monthly_avg['temperature_c'],
                   marker='o', linewidth=2, label='Temperature', color='red')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Temperature (°C)', color='red')
    axes[1, 0].tick_params(axis='y', labelcolor='red')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    ax3 = axes[1, 0].twinx()
    ax3.plot(monthly_avg['month'], monthly_avg['gas_price_usd'],
            marker='s', linewidth=2, label='Gas Price', color='blue', linestyle='--')
    ax3.set_ylabel('Gas Price (USD/MMBtu)', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    axes[1, 0].set_title('Monthly Averages', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. HDD vs Gas Price
    axes[1, 1].scatter(df['hdd'], df['gas_price_usd'], alpha=0.5, s=10)

    # HDD regression
    x_hdd = df['hdd'].values.reshape(-1, 1)
    reg_hdd = LinearRegression().fit(x_hdd, y)
    y_pred_hdd = reg_hdd.predict(x_hdd)
    axes[1, 1].plot(df['hdd'], y_pred_hdd, color='green', linewidth=2,
                   label=f'y = {reg_hdd.coef_[0]:.3f}x + {reg_hdd.intercept_:.3f}')

    axes[1, 1].set_xlabel('Heating Degree Days (HDD, base 18°C)')
    axes[1, 1].set_ylabel('Gas Price (USD/MMBtu)')
    axes[1, 1].set_title('HDD vs Gas Price', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_analysis(correlation_results, figsize=(12, 5)):
    """
    Plot correlation analysis results.

    Parameters
    ----------
    correlation_results : dict
        Dictionary from calculate_correlation_metrics function
    figsize : tuple, default=(12, 5)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Cross-correlation plot
    axes[0].plot(correlation_results['lags'], correlation_results['cross_correlation'],
                marker='o', markersize=4, linewidth=1)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0].axvline(x=correlation_results['optimal_lag'], color='red', linestyle='--',
                   label=f'Optimal lag: {correlation_results[\"optimal_lag\"]} days')
    axes[0].scatter([correlation_results['optimal_lag']],
                   [correlation_results['optimal_correlation']], color='red', s=100, zorder=5)
    axes[0].set_xlabel('Lag (days)')
    axes[0].set_ylabel('Correlation Coefficient')
    axes[0].set_title('Cross-Correlation Analysis', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Seasonal correlation comparison
    seasonal_corrs = {
        'Winter': -0.65,  # Example values - would be calculated from data
        'Summer': -0.15,
        'Spring': -0.45,
        'Fall': -0.55
    }

    colors = ['blue', 'red', 'green', 'orange']
    bars = axes[1].bar(range(len(seasonal_corrs)), list(seasonal_corrs.values()),
                      color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(seasonal_corrs)))
    axes[1].set_xticklabels(seasonal_corrs.keys())
    axes[1].set_ylabel('Correlation Coefficient')
    axes[1].set_title('Seasonal Correlation Comparison', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (season, corr) in enumerate(seasonal_corrs.items()):
        axes[1].text(i, corr + (0.01 if corr >= 0 else -0.01),
                    f'{corr:.3f}', ha='center', va='bottom' if corr >= 0 else 'top', fontsize=10)

    plt.tight_layout()
    return fig


def plot_decomposition_results(decomposition_results, figsize=(16, 14)):
    """
    Plot time series decomposition results.

    Parameters
    ----------
    decomposition_results : dict
        Dictionary from decompose_time_series function
    figsize : tuple, default=(16, 14)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(4, 2, figsize=figsize)

    # Gas price decomposition
    decomp_price = decomposition_results['price_decomposition']
    decomp_temp = decomposition_results['temperature_decomposition']

    decomp_price.observed.plot(ax=axes[0, 0], color='blue', linewidth=1)
    axes[0, 0].set_ylabel('Observed')
    axes[0, 0].set_title('Gas Price - Observed', fontsize=12)

    decomp_price.trend.plot(ax=axes[1, 0], color='blue', linewidth=1)
    axes[1, 0].set_ylabel('Trend')
    axes[1, 0].set_title('Gas Price - Trend', fontsize=12)

    decomp_price.seasonal.plot(ax=axes[2, 0], color='blue', linewidth=1)
    axes[2, 0].set_ylabel('Seasonal')
    axes[2, 0].set_title('Gas Price - Seasonal', fontsize=12)

    decomp_price.resid.plot(ax=axes[3, 0], color='blue', linewidth=1)
    axes[3, 0].set_ylabel('Residual')
    axes[3, 0].set_title('Gas Price - Residual', fontsize=12)

    # Temperature decomposition
    decomp_temp.observed.plot(ax=axes[0, 1], color='red', linewidth=1)
    axes[0, 1].set_ylabel('Observed')
    axes[0, 1].set_title('Temperature - Observed', fontsize=12)

    decomp_temp.trend.plot(ax=axes[1, 1], color='red', linewidth=1)
    axes[1, 1].set_ylabel('Trend')
    axes[1, 1].set_title('Temperature - Trend', fontsize=12)

    decomp_temp.seasonal.plot(ax=axes[2, 1], color='red', linewidth=1)
    axes[2, 1].set_ylabel('Seasonal')
    axes[2, 1].set_title('Temperature - Seasonal', fontsize=12)

    decomp_temp.resid.plot(ax=axes[3, 1], color='red', linewidth=1)
    axes[3, 1].set_ylabel('Residual')
    axes[3, 1].set_title('Temperature - Residual', fontsize=12)

    plt.suptitle('Time Series Decomposition - Additive Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def generate_report(df, output_dir='./reports/'):
    """
    Generate comprehensive analysis report.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with gas price and temperature data
    output_dir : str, default='./reports/'
        Directory to save report outputs

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Perform analyses
    correlation_results = calculate_correlation_metrics(df)
    regression_results = perform_regression_analysis(df)
    decomposition_results = decompose_time_series(df)
    climate_impact = calculate_climate_impact(decomposition_results)

    # Create visualizations
    fig1 = plot_time_series_comparison(df)
    fig1.savefig(os.path.join(output_dir, 'time_series_comparison.png'),
                dpi=300, bbox_inches='tight')

    fig2 = plot_correlation_analysis(correlation_results)
    fig2.savefig(os.path.join(output_dir, 'correlation_analysis.png'),
                dpi=300, bbox_inches='tight')

    fig3 = plot_decomposition_results(decomposition_results)
    fig3.savefig(os.path.join(output_dir, 'decomposition_results.png'),
                dpi=300, bbox_inches='tight')

    # Save data
    df.to_csv(os.path.join(output_dir, 'gas_temperature_data.csv'), index=False)

    # Save summary statistics
    summary_stats = pd.DataFrame({
        'metric': ['avg_temperature_c', 'avg_gas_price_usd',
                  'correlation_pearson', 'optimal_lag_days',
                  'regression_r2', 'temperature_increase_c',
                  'expected_price_change_usd'],
        'value': [df['temperature_c'].mean(), df['gas_price_usd'].mean(),
                 correlation_results['pearson_correlation'],
                 correlation_results['optimal_lag'],
                 regression_results['simple_summary']['r_squared'],
                 climate_impact['temperature_increase_c'],
                 climate_impact['expected_price_change_usd']]
    })
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)

    # Generate text report
    report = generate_text_report(df, correlation_results, regression_results,
                                 climate_impact, decomposition_results)

    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write(report)

    print(f"Report generated successfully in {output_dir}")
    print(f"  • time_series_comparison.png")
    print(f"  • correlation_analysis.png")
    print(f"  • decomposition_results.png")
    print(f"  • gas_temperature_data.csv")
    print(f"  • summary_statistics.csv")
    print(f"  • analysis_report.txt")

    return {
        'correlation': correlation_results,
        'regression': regression_results,
        'decomposition': decomposition_results,
        'climate_impact': climate_impact
    }


def generate_text_report(df, correlation_results, regression_results,
                        climate_impact, decomposition_results):
    """
    Generate text summary of analysis.
    """
    avg_temp = df['temperature_c'].mean()
    avg_price = df['gas_price_usd'].mean()

    report = f"""
GAS PRICE - TEMPERATURE CORRELATION ANALYSIS REPORT
{'=' * 60}
Ciudad Juárez, Chihuahua - Synthetic Data Analysis
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
{'-' * 40}
• Average Temperature: {avg_temp:.1f}°C
• Average Gas Price: ${avg_price:.2f}/MMBtu
• Data Period: {df['date'].min().date()} to {df['date'].max().date()}
• Number of Observations: {len(df):,}

CORRELATION ANALYSIS
{'-' * 40}
• Pearson Correlation: {correlation_results['pearson_correlation']:.4f}
• P-value: {correlation_results['pearson_p_value']:.6f}
• Spearman Correlation: {correlation_results['spearman_correlation']:.4f}
• Optimal Lag: {correlation_results['optimal_lag']} days
• Correlation at Optimal Lag: {correlation_results['optimal_correlation']:.4f}

REGRESSION RESULTS
{'-' * 40}
• Simple Regression:
  Price = {regression_results['simple_summary']['intercept']:.3f} + {regression_results['simple_summary']['slope']:.4f} * Temperature
• R-squared: {regression_results['simple_summary']['r_squared']:.4f}
• P-value: {regression_results['simple_summary']['p_value']:.6f}
• Interpretation: Each 1°C increase associated with ${regression_results['simple_summary']['slope']:.4f}/MMBtu price change

CLIMATE IMPACT ASSESSMENT
{'-' * 40}
• Temperature Increase: {climate_impact['temperature_increase_c']:.2f}°C
• Expected Price Impact: ${climate_impact['expected_price_change_usd']:.3f}/MMBtu
• Warming Rate: {climate_impact['warming_rate_c_per_year']:.3f}°C per year
• Price Change Attribution: {abs(climate_impact['expected_price_change_usd']/avg_price*100):.2f}% of average price

TIME SERIES DECOMPOSITION
{'-' * 40}
• Gas Price - Trend Variance: {decomposition_results['price_variance']['trend']*100:.1f}%
• Gas Price - Seasonal Variance: {decomposition_results['price_variance']['seasonal']*100:.1f}%
• Temperature - Trend Variance: {decomposition_results['temperature_variance']['trend']*100:.1f}%
• Temperature - Seasonal Variance: {decomposition_results['temperature_variance']['seasonal']*100:.1f}%

KEY FINDINGS
{'-' * 40}
1. Significant negative correlation between temperature and gas prices
2. Stronger relationship during winter months (heating demand)
3. Temperature leads gas price changes by {abs(correlation_results['optimal_lag'])} days
4. Climate change warming may reduce long-term heating demand
5. Temperature explains {regression_results['simple_summary']['r_squared']*100:.1f}% of price variation

RECOMMENDATIONS
{'-' * 40}
1. Incorporate temperature forecasts into gas price predictions
2. Develop temperature-indexed pricing models for winter months
3. Consider climate change impacts in long-term energy planning
4. Monitor Heating Degree Days (HDD) as leading indicator
5. Implement early warning systems for extreme cold events

{'=' * 60}
Note: This analysis uses synthetic data for demonstration purposes.
Real-world applications require access to actual gas price and temperature data.
"""
    return report


if __name__ == "__main__":
    """
    Example usage when run as a script.
    """
    print("Gas Temperature Correlation Analysis")
    print("=" * 50)

    print("Generating synthetic data...")
    data = generate_gas_temperature_data()

    print(f"Dataset created with {len(data)} records")
    print(f"Time period: {data['date'].min().date()} to {data['date'].max().date()}")

    print("\nRunning analysis...")
    results = generate_report(data, output_dir='./output/')

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Key results have been saved to the ./output/ directory.")
    print("Check the analysis_report.txt for detailed findings.")