import yfinance as yf
import talib


def collect_nasdaq_data(data_config, indicators=True):
    tickers = data_config["tickers"]
    start_date = data_config["start_date"]
    end_date = data_config["end_date"]

    # download data
    data = yf.download(tickers, start=start_date, end=end_date , interval="1d")

    # select columns
    selected_columns = ["Open", "High", "Low", "Close", "Volume"]
    df = data[selected_columns]

    # === calculate technical indicators ===
    if indicators:
        for ticker in tickers:
            df[("SMA_50", ticker)] = talib.SMA(df[("Close", ticker)], timeperiod=50)  # 50 days SMA
            df[("EMA_20", ticker)] = talib.EMA(df[("Close", ticker)], timeperiod=30)  # 20 days EMA
            df[("RSI", ticker)] = talib.RSI(df[("Close", ticker)], timeperiod=30)  # RSI
            df[("MACD", ticker)], df[("MACD_signal", ticker)], _ = talib.MACD(df[("Close", ticker)])  # MACD
            df[("ATR", ticker)] = talib.ATR(df[("High", ticker)], df[("Low", ticker)], df[("Close", ticker)], timeperiod=30)  # ATR
    
    # drop NaN
    df.dropna(inplace=True)
    # rename columns
    df.columns = [f"{stock}-{indicator}" for indicator, stock in df.columns]

    ixic_close = df[[col for col in df.columns if col.startswith("^IXIC-Close")]]

    # drop other IXIC columns, keep FAANG indicators
    df = df.drop(columns=[col for col in df.columns if col.startswith("^IXIC-") and col not in ixic_close.columns])
    
    # save to csv
    df.to_csv("faang_nasdaq_extended_data.csv")
    return df