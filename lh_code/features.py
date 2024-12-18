import pandas as pd
import os
import pandas as pd
import talib

def generate_features(processed_path, output_path="data/features/"):
    """Generate features from processed data."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    processed_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]

    for file in processed_files:
        df = pd.read_csv(os.path.join(processed_path, file))

        # Generate moving averages
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()

        # Generate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Save features
        output_file = os.path.join(output_path, f"features_{file}")
        df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")


def generate_technical_indicators(data):
    """
    Generate technical indicators using TA-Lib and add them to the DataFrame.
    """
    # Ensure the data is sorted by date
    data = data.sort_values(by="trade_date")

    # Simple Moving Averages (SMA)
    data["sma_10"] = talib.SMA(data["close"], timeperiod=10)
    data["sma_30"] = talib.SMA(data["close"], timeperiod=30)

    # Relative Strength Index (RSI)
    data["rsi_14"] = talib.RSI(data["close"], timeperiod=14)

    # Bollinger Bands
    data["upper_band"], data["middle_band"], data["lower_band"] = talib.BBANDS(
        data["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    # Exponential Moving Average (EMA)
    data["ema_10"] = talib.EMA(data["close"], timeperiod=10)

    # Average True Range (ATR)
    data["atr_14"] = talib.ATR(data["high"], data["low"], data["close"], timeperiod=14)

    # Momentum
    data["momentum"] = talib.MOM(data["close"], timeperiod=10)

    return data
def add_time_features(data):
    """
    Add time-related features to the DataFrame.
    """
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["day_of_week"] = data["trade_date"].dt.dayofweek  # Monday=0, Sunday=6
    data["month"] = data["trade_date"].dt.month
    data["is_month_start"] = data["trade_date"].dt.is_month_start.astype(int)
    data["is_month_end"] = data["trade_date"].dt.is_month_end.astype(int)
    return data
def calculate_future_return(data, n=5):
    """
    Calculate the future return over the next n days.
    """
    data["future_return"] = (data["close"].shift(-n) - data["close"]) / data["close"]
    return data


def enrich_features(input_path="data/processed/labeled.csv", output_path="data/processed/featured.csv"):
    """
    Enrich labeled data with technical indicators, time features, and future returns.
    """
    # Load labeled data
    data = pd.read_csv(input_path)

    #generate_features()
    # Generate technical indicators
    data = generate_technical_indicators(data)

    # Add time-related features
    data = add_time_features(data)

    # Calculate future returns
    data = calculate_future_return(data, n=5)

    # Save the enriched data
    data.to_csv(output_path, index=False)
    print(f"Featured data saved to {output_path}")
