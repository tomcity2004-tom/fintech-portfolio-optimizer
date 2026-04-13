import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from data_fetcher import fetch_historical_data, get_risk_free_rate
from portfolio_optimizer import optimize_portfolio, calculate_portfolio_metrics
from utils import monte_carlo_simulation, calculate_var_cvar

# ====================== 登入系統 ======================
# Hardcode 的帳號密碼（你可以自行修改）
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"   # 建議正式使用時改成更安全的密碼

# 初始化 session_state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ====================== Login 頁面 ======================
def show_login_page():
    st.set_page_config(page_title="登入 - 投資組合優化工具", layout="centered")
    st.title("🔐 智慧型投資組合優化工具")
    st.subheader("請先登入以繼續使用")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("使用者名稱", placeholder="輸入 username")
        password = st.text_input("密碼", type="password", placeholder="輸入 password")

        if st.button("登入", type="primary", use_container_width=True):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"登入成功！歡迎，{username}")
                st.rerun()  # 重新執行，切換到主畫面
            else:
                st.error("使用者名稱或密碼錯誤，請再試一次")

        st.caption("預設帳號：**admin** / 密碼：**password123**")

# ====================== 主應用程式（原本的內容） ======================
def show_main_app():
    st.set_page_config(page_title="智慧型投資組合優化器", layout="wide")
    
    # 頂部顯示已登入資訊 + Logout 按鈕
    col_title, col_logout = st.columns([4, 1])
    with col_title:
        st.title("📈 智慧型個人化投資組合優化決策支援工具")
    with col_logout:
        if st.button("登出", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    st.markdown(f"**目前登入使用者：{st.session_state.username}**")

    # ==================== 側邊欄輸入（原本的內容） ====================
    st.sidebar.header("🎯 您的投資偏好")

    tickers = st.sidebar.multiselect(
        "選擇資產 (可複選)",
        options=["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "BTC-USD", "^HSI"],
        default=["NVDA", "AAPL", "MSFT", "TSLA"]
    )

    investment_amount = st.sidebar.number_input("投資總金額 (USD)", 
                                              min_value=10000, 
                                              value=100000, 
                                              step=10000)

    max_single_weight = st.sidebar.slider("單一資產最高權重 (%)", 20, 90, 60) / 100

    # ==================== 主畫面 ====================
    if st.sidebar.button("🚀 開始優化投資組合", type="primary"):
        if len(tickers) < 2:
            st.error("請至少選擇 2 個資產進行優化")
        else:
            with st.spinner("正在從 Yahoo Finance 抓取最新資料並進行優化..."):
                try:
                    prices = fetch_historical_data(tickers, period="2y")
                    returns = prices.pct_change().dropna()
                    
                    mean_returns, cov_matrix = calculate_portfolio_metrics(returns)
                    rf_rate = get_risk_free_rate()
                    
                    # 呼叫優化（target_return=None 表示最小風險優化）
                    weights, risk, expected_return = optimize_portfolio(
                        mean_returns, 
                        cov_matrix, 
                        risk_free_rate=rf_rate,
                        target_return=None,        # 已移除目標報酬率
                        max_weight=max_single_weight
                    )
                    
                    # 顯示結果（使用你之前修正過的安全版本）
                    exp_annual = float(expected_return * 252 * 100)
                    risk_annual = float(risk * np.sqrt(252) * 100)
                    try:
                        sharpe = float((expected_return*252 - rf_rate*252) / (risk * np.sqrt(252)))
                        sharpe = max(0.0, (expected_return*252 - rf_rate*252) / (risk * np.sqrt(252)))
                    except:
                       sharpe = 0.0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("預期年化報酬率", f"{exp_annual:.2f}%")
                    with col2: st.metric("年化風險 (波動率)", f"{risk_annual:.2f}%")
                    with col3: st.metric("夏普比率", f"{sharpe:.2f}")
                    with col4: st.metric("投資金額", f"${investment_amount:,.0f}")
                    
                    # 資產配置
                    st.subheader("📊 最佳投資組合權重配置")
                    weight_df = pd.DataFrame({
                        "資產": tickers,
                        "權重 (%)": np.round(np.array(weights) * 100, 2)
                    })
                    
                    col_a, col_b = st.columns([3, 2])
                    with col_a:
                        fig_pie = px.pie(weight_df, values="權重 (%)", names="資產", title="投資組合配置比例")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col_b:
                        st.dataframe(weight_df, use_container_width=True, hide_index=True)
                    
                    # Monte Carlo
                    st.subheader("🔮 未來1年報酬分布模擬 (10,000 次)")
                    sim_results = monte_carlo_simulation(returns, weights)
                    final_returns = sim_results[-1, :]
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(x=final_returns*100, nbinsx=50))
                    fig_mc.update_layout(title="1年後累積報酬率分布")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    var, cvar = calculate_var_cvar(sim_results)
                    st.info(f"**95% VaR**：{var*100:.2f}%　　**95% CVaR**：{cvar*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"發生錯誤：{str(e)}")
                    st.info("💡 建議：減少資產數量、放寬單一資產最高權重，或檢查網路連線")
    else:
        st.info("👈 請在左側側邊欄選擇資產並調整參數，然後點擊「開始優化投資組合」")

# ====================== 主程式入口 ======================
if st.session_state.logged_in:
    show_main_app()
else:
    show_login_page()
