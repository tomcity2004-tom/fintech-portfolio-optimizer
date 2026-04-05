# data_fetcher.py
import yfinance as yf
import pandas as pd

def fetch_historical_data(tickers: list, period: str = "2y"):
    """從 Yahoo Finance 下載歷史調整後價格資料"""
    if not tickers:
        raise ValueError("請至少選擇一個資產")
    
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    
    # 處理單一或多資產的情況，取出 Close（已自動調整）
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        # 單一資產時轉成 DataFrame
        if len(tickers) == 1:
            prices = data[['Close']].rename(columns={'Close': tickers[0]})
        else:
            prices = data
    
    prices = prices.dropna(how='all')
    
    if prices.empty:
        raise ValueError(f"無法取得 {tickers} 的資料，請檢查網路或資產代碼")
    
    return prices

def get_risk_free_rate():
    """取得無風險利率（美國3個月公債）"""
    try:
        rf = yf.download("^IRX", period="5d", progress=False)['Close'].iloc[-1] / 100
        return rf / 252
    except:
        return 0.0001  # 失敗時使用預設值