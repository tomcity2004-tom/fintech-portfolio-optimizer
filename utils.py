# utils.py
import numpy as np

def monte_carlo_simulation(returns: pd.DataFrame, weights: np.ndarray, 
                          num_simulations: int = 10000, days: int = 252):
    np.random.seed(42)
    daily_ret = returns.mean()
    daily_vol = returns.std()
    
    port_ret = np.dot(weights, daily_ret)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov().values, weights)))
    
    sim_returns = np.random.normal(port_ret, port_vol, (days, num_simulations))
    cum_returns = np.cumprod(1 + sim_returns, axis=0)
    return cum_returns

def calculate_var_cvar(simulation_results: np.ndarray, confidence: float = 0.95):
    """計算 95% VaR 和 CVaR（正確版本）"""
    final_returns = simulation_results[-1, :]           # 最後一天的累積報酬
    
    # 計算報酬率變化（相對於初始1.0）
    returns_change = final_returns - 1.0
    
    # VaR：95%信心水準下的最大損失（應為負數）
    var = np.percentile(returns_change, (1 - confidence) * 100)
    
    # CVaR：超過VaR的平均損失
    cvar = returns_change[returns_change <= var].mean()
    
    return var, cvar
