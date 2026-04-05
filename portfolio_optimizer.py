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
                      target_return=None, max_weight=0.50):   # 預設放寬到 50%
    n = len(mean_returns)
    weights = cp.Variable(n)
    
    portfolio_return = mean_returns.values @ weights
    portfolio_risk = cp.quad_form(weights, cov_matrix.values)
    
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight
    ]
    
    # 當目標報酬率為 0 或 None 時，不強加 >= target_return
    if target_return is not None and target_return > 0:
        constraints.append(portfolio_return >= target_return)
    
    objective = cp.Minimize(portfolio_risk)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)   # 或 cp.ECOS
    
    if weights.value is None:
        raise ValueError("優化失敗，請降低目標報酬率、增加高成長資產，或放寬單一資產權重")
    
    return weights.value, np.sqrt(portfolio_risk.value), portfolio_return.value