import pandas as pd
import os

def preprocess_data(daily_path, financial_path, output_path="data/processed/"):
    """Preprocess daily and financial data."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load daily data
    daily_files = [f for f in os.listdir(daily_path) if f.endswith('.csv')]
    financial_files = [f for f in os.listdir(financial_path) if f.endswith('.csv')]

    for daily_file, financial_file in zip(daily_files, financial_files):
        daily_df = pd.read_csv(os.path.join(daily_path, daily_file))
        financial_df = pd.read_csv(os.path.join(financial_path, financial_file))

        # Merge financial data into daily data
        daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'], format='%Y%m%d')
        financial_df['end_date'] = pd.to_datetime(financial_df['end_date'], format='%Y%m%d')

        # Fill financial data forward for daily dates
        merged_df = pd.merge_asof(
            daily_df.sort_values('trade_date'),
            financial_df.sort_values('end_date'),
            left_on='trade_date',
            right_on='end_date',
            by='ts_code',
            direction='backward'
        )

        # Handle missing values (example: forward fill)
        merged_df.fillna(method='ffill', inplace=True)

        # Save processed data
        output_file = os.path.join(output_path, f"processed_{daily_file}")
        merged_df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
