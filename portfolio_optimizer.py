# portfolio_optimizer.py
import numpy as np
import cvxpy as cp
import pandas as pd

def calculate_portfolio_metrics(returns: pd.DataFrame):
    """計算平均報酬與共變異數矩陣"""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0, 
                      target_return=None, max_weight=0.35):
    n = len(mean_returns)
    weights = cp.Variable(n)
    
    portfolio_return = mean_returns.values @ weights
    portfolio_risk = cp.quad_form(weights, cov_matrix.values)
    
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight
    ]
    
    if target_return is not None:
        constraints.append(portfolio_return >= target_return)
    
    objective = cp.Minimize(portfolio_risk) if target_return is None else \
                cp.Maximize(portfolio_return / cp.sqrt(portfolio_risk))
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    
    if weights.value is None:
        raise ValueError("優化失敗，請降低目標報酬率或調整資產組合")
    
    return weights.value, np.sqrt(portfolio_risk.value), portfolio_return.value