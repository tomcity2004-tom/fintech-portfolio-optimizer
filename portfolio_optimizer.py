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
    """
    Markowitz 投資組合優化 - DCP 相容版本
    """
    n = len(mean_returns)
    weights = cp.Variable(n)
    
    portfolio_return = mean_returns.values @ weights
    portfolio_risk = cp.quad_form(weights, cov_matrix.values)   # 這是 convex
    
    # 基本限制條件
    constraints = [
        cp.sum(weights) == 1,      # 權重總和 = 1
        weights >= 0,              # 不賣空
        weights <= max_weight      # 單一資產最高權重
    ]
    
    # ==================== 關鍵修改 ====================
    if target_return is not None and target_return > 0:
        # 有設定目標報酬率 → 在達到目標報酬的前提下，最小化風險（最穩定）
        constraints.append(portfolio_return >= target_return)
        objective = cp.Minimize(portfolio_risk)
    else:
        # 沒有目標報酬率 → 直接最小化風險
        objective = cp.Minimize(portfolio_risk)
    
    prob = cp.Problem(objective, constraints)
    
    # 使用 Clarabel 或 ECOS（你已安裝 ecos）
    prob.solve(solver=cp.CLARABEL, verbose=False)
    # 如果 Clarabel 有問題，可改成： prob.solve(solver=cp.ECOS, verbose=False)
    
    if weights.value is None:
        raise ValueError("優化失敗，請降低目標報酬率、減少資產數量，或放寬單一資產權重限制")
    
    return weights.value, np.sqrt(portfolio_risk.value), portfolio_return.value