import chinadata.ca_data as ts
import pandas as pd
import os

from lh_code.features import generate_features, enrich_features
from lh_code.labeling import generate_labeled_features
from lh_code.preprocess import preprocess_data


def initialize_tushare(api_token):
    """Initialize the Tushare API with the provided token."""
    ts.set_token(api_token)
    return ts.pro_api(api_token)


def fetch_daily_data(api, stock_list, start_date, end_date, save_path="data/daily/"):
    """Fetch daily stock data for a list of stocks."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for stock in stock_list:
        print(f"Fetching daily data for {stock}...")
        try:
            df = api.daily(ts_code=stock, start_date=start_date, end_date=end_date)
            df.to_csv(f"{save_path}{stock}_daily.csv", index=False)
            print(f"Saved {stock} data to {save_path}{stock}_daily.csv")
        except Exception as e:
            print(f"Failed to fetch data for {stock}: {e}")


def fetch_financial_data(api, stock_list, fields, save_path="data/financial/"):
    """Fetch static financial data for a list of stocks."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for stock in stock_list:
        print(f"Fetching financial data for {stock}...")
        try:
            df = api.fina_indicator(ts_code=stock, fields=fields)
            df.to_csv(f"{save_path}{stock}_financial.csv", index=False)
            print(f"Saved {stock} data to {save_path}{stock}_financial.csv")
        except Exception as e:
            print(f"Failed to fetch data for {stock}: {e}")






if __name__ == "__main__":
    DAILY_DATA_PATH = "data/daily/"
    FINANCIAL_DATA_PATH = "data/financial/"
    PROCESSED_DATA_PATH = "data/processed/"
    FEATURES_DATA_PATH = "data/features/"
    LABELED_DATA_PATH = "data/labeled/"
    # User-specific configurations
    API_TOKEN = "n9e84ed87f29cf43fdac84cdbb14d306777"  # Replace with your Tushare API token
    STOCK_LIST = ["000001.SZ", "600000.SH"]  # Add stock codes here
    START_DATE = "20150101"
    END_DATE = "20231231"

    # Fields to extract from financial data
    FINANCIAL_FIELDS = "ts_code,ann_date,end_date,roe,roa,grossprofit_margin,net_profit_margin,net_profit_yr,yoy_net_profit,pe,pb"

    # Initialize Tushare API
    api = initialize_tushare(API_TOKEN)

    # Fetch daily stock data
    fetch_daily_data(api, STOCK_LIST, START_DATE, END_DATE)

    # Fetch financial data
    fetch_financial_data(api, STOCK_LIST, FINANCIAL_FIELDS)
  # Step 3: 数据预处理
    print("Step 3: Preprocessing data...")
    preprocess_data(DAILY_DATA_PATH, FINANCIAL_DATA_PATH, PROCESSED_DATA_PATH)

    # Step 4: 特征工程
    print("Step 4: Generating features...")
   # generate_features(PROCESSED_DATA_PATH, FEATURES_DATA_PATH)

    enrich_features(input_path=PROCESSED_DATA_PATH, output_path=FEATURES_DATA_PATH)

    # Step 5: 生成标签
    print("Step 5: Generating labels...")
    generate_labeled_features(FEATURES_DATA_PATH, LABELED_DATA_PATH)

    print("Pipeline completed successfully!")


