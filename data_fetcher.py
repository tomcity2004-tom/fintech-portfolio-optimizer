# data_fetcher.py
import yfinance as yf
import pandas as pd

def fetch_historical_data(tickers: list, period: str = "2y"):
    """
    從 Yahoo Finance 下載歷史資料（使用調整後的 Close 價格）
    """
    if not tickers:
        raise ValueError("請至少選擇一個資產")
    
    # 關鍵修改：明確設定 auto_adjust=True，並只取 Close 欄位
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    
    # 如果是多個資產，data 是 MultiIndex，需要取出 Close
    if len(tickers) > 1:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']          # 現在 Close 就是調整後價格
        else:
            prices = data
    else:
        prices = data['Close'].to_frame(name=tickers[0])
    
    prices = prices.dropna()  # 移除有缺失值的日期
    
    if prices.empty:
        raise ValueError("無法取得資料，請檢查網路或資產代碼是否正確")
    
    return prices