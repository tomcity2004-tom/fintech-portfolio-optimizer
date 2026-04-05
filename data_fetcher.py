# data_fetcher.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_historical_data(tickers: list, period: str = "2y"):
    """
    從 Yahoo Finance 下載歷史調整收盤價
    """
    if not tickers:
        raise ValueError("請至少選擇一個資產")
    
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Adj Close']
    data = data.dropna()  # 移除有缺失值的日期
    
    if data.empty:
        raise ValueError("無法取得資料，請檢查網路或資產代碼")
    
    return data

def get_risk_free_rate():
    """取得美國3個月公債收益率作為無風險利率"""
    try:
        rf_data = yf.download("^IRX", period="5d", progress=False)['Adj Close']
        rf = rf_data.iloc[-1] / 100  # 轉成小數
        return rf / 252  # 轉成日化無風險利率
    except:
        return 0.0001  # 若失敗，使用預設小值