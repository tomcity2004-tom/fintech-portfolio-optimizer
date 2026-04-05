# utils.py
import numpy as np
import pandas as pd

def monte_carlo_simulation(returns: pd.DataFrame, weights: np.ndarray, 
                          num_simulations: int = 10000, days: int = 252):
    """進行 Monte Carlo 未來報酬模擬"""
    np.random.seed(42)  # 確保結果可重現
    
    daily_returns = returns.mean()
    daily_vol = returns.std()
    
    portfolio_daily_return = np.dot(weights, daily_returns)
    portfolio_daily_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov().values, weights)))
    
    # 生成模擬報酬
    simulated_daily_returns = np.random.normal(portfolio_daily_return, 
                                               portfolio_daily_vol, 
                                               (days, num_simulations))
    
    # 累積報酬
    cumulative_returns = np.cumprod(1 + simulated_daily_returns, axis=0)
    return cumulative_returns

def calculate_var_cvar(simulation_results: np.ndarray, confidence: float = 0.95):
    """計算 Value at Risk 和 Conditional VaR"""
    final_returns = simulation_results[-1, :]
    var = np.percentile(final_returns, (1 - confidence) * 100)
    cvar = final_returns[final_returns <= var].mean()
    return var, cvar