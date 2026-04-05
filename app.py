# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_fetcher import fetch_historical_data, get_risk_free_rate
from portfolio_optimizer import optimize_portfolio, calculate_portfolio_metrics
from utils import monte_carlo_simulation, calculate_var_cvar

# 頁面設定
st.set_page_config(page_title="智慧投資組合優化器", layout="wide")
st.title("📈 智慧型個人化投資組合優化決策支援工具")
st.markdown("**金融科技期末專案** | 使用 Markowitz 模型 + Monte Carlo 模擬")

# ==================== 側邊欄輸入 ====================
st.sidebar.header("🎯 您的投資偏好")

tickers = st.sidebar.multiselect(
    "選擇資產 (可複選)",
    options=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "^TWII", "BTC-USD", "GC=F"],
    default=["AAPL", "MSFT", "GOOGL", "NVDA"]
)

investment_amount = st.sidebar.number_input("投資總金額 (USD)", 
                                          min_value=10000, 
                                          value=100000, 
                                          step=10000)

target_annual_return = st.sidebar.slider("目標年化報酬率 (%)", 
                                       min_value=5, 
                                       max_value=30, 
                                       value=12) / 100

risk_preference = st.sidebar.selectbox(
    "風險偏好等級",
    options=["保守型", "平衡型", "積極型"]
)

max_single_weight = st.sidebar.slider("單一資產最高權重 (%)", 20, 50, 35) / 100

# ==================== 主畫面 ====================
if st.sidebar.button("🚀 開始優化投資組合", type="primary"):
    if len(tickers) < 2:
        st.error("請至少選擇 2 個資產進行優化")
    else:
        with st.spinner("正在從 Yahoo Finance 抓取最新資料並進行優化..."):
            try:
                # 抓取資料
                prices = fetch_historical_data(tickers, period="2y")
                returns = prices.pct_change().dropna()
                
                mean_returns, cov_matrix = calculate_portfolio_metrics(returns)
                rf_rate = get_risk_free_rate()
                
                # 執行優化
                weights, risk, expected_return = optimize_portfolio(
                    mean_returns, 
                    cov_matrix, 
                    risk_free_rate=rf_rate,
                    target_return=target_annual_return,
                    max_weight=max_single_weight
                )
                
                # ==================== 顯示結果 ====================
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("預期年化報酬率", f"{expected_return*252*100:.2f}%")
                with col2:
                    st.metric("年化風險 (波動率)", f"{risk*np.sqrt(252)*100:.2f}%")
                with col3:
                    sharpe = (expected_return*252 - rf_rate*252) / (risk * np.sqrt(252))
                    st.metric("夏普比率 (Sharpe)", f"{sharpe:.2f}")
                with col4:
                    st.metric("投資金額", f"${investment_amount:,.0f}")
                
                # 資產配置餅圖
                st.subheader("📊 最佳投資組合權重配置")
                weight_df = pd.DataFrame({
                    "資產": tickers,
                    "權重 (%)": weights * 100
                }).round(2)
                
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fig_pie = px.pie(weight_df, values="權重 (%)", names="資產", 
                                   title="投資組合配置比例")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_b:
                    st.dataframe(weight_df, use_container_width=True, hide_index=True)
                
                # Monte Carlo 模擬
                st.subheader("🔮 未來1年報酬分布模擬 (10,000 次)")
                sim_results = monte_carlo_simulation(returns, weights)
                
                final_returns = sim_results[-1, :]
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(x=final_returns*100, nbinsx=50, 
                                            name="報酬分布"))
                fig_mc.update_layout(title="1年後累積報酬率分布")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # VaR 與 CVaR
                var, cvar = calculate_var_cvar(sim_results)
                st.info(f"**95% Value at Risk (VaR)**: {var*100:.2f}%　　"
                       f"**95% Conditional VaR (CVaR)**: {cvar*100:.2f}%")
                
            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
                st.info("建議：減少目標報酬率、減少資產數量，或檢查網路連線")

else:
    st.info("👈 請在左側側邊欄調整參數，然後點擊「開始優化投資組合」按鈕")