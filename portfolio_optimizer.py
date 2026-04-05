# portfolio_optimizer.py
import numpy as np
import cvxpy as cp
import pandas as pd

def calculate_portfolio_metrics(returns: pd.DataFrame):
    """計算平均報酬與共變異數矩陣"""
    if returns.empty:
        raise ValueError("報酬率資料為空")
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0, 
                      target_return=None, max_weight=0.35):
    """
    使用 cvxpy 進行 Markowitz 投資組合優化
    """
    n = len(mean_returns)
    weights = cp.Variable(n)
    
    # 預期報酬與風險
    portfolio_return = mean_returns.values @ weights
    portfolio_risk = cp.quad_form(weights, cov_matrix.values)
    
    # 限制條件
    constraints = [
        cp.sum(weights) == 1,      # 權重總和為 1
        weights >= 0,              # 不允許賣空
        weights <= max_weight      # 單一資產最高權重限制
    ]
    
    if target_return is not None:
        constraints.append(portfolio_return >= target_return)
    
    # 目標函數：最小化風險（預設）或最大化夏普比率
    if target_return is None:
        objective = cp.Minimize(portfolio_risk)
    else:
        objective = cp.Maximize(portfolio_return / cp.sqrt(portfolio_risk))
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    
    if weights.value is None:
        raise ValueError("優化失敗，請調整目標報酬率或資產組合")
    
    return weights.value, np.sqrt(portfolio_risk.value), portfolio_return.value