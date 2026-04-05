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
    final_returns = simulation_results[-1, :]
    var = np.percentile(final_returns, (1 - confidence) * 100)
    cvar = final_returns[final_returns <= var].mean()
    return var, cvar