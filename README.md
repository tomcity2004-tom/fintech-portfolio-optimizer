# 智慧型個人化投資組合優化決策支援工具

金融科技期末專案 - 使用 Python + Streamlit 開發

## 功能特色
- Markowitz 現代投資組合理論優化
- Monte Carlo 未來報酬模擬
- VaR / CVaR 風險指標
- 即時 Yahoo Finance 資料抓取
- 互動式 Streamlit 網頁介面

## 如何執行

```bash
pip install -r requirements.txt
streamlit run app.py

##專案結構

```bash
fintech-portfolio-optimizer/
├── app.py                          ← 主程式入口（Login + 主要介面）
├── data_fetcher.py                 ← 資料抓取模組
├── portfolio_optimizer.py          ← 核心優化模型
├── utils.py                        ← 輔助工具函數（Monte Carlo & 風險指標）
├── requirements.txt                ← 套件清單
├── .streamlit/config.toml          ← Streamlit 介面美化設定
└── README.md
