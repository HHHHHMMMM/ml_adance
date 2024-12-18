import pandas as pd
import os


def generate_labeled_features(data_path, save_path):
    """Generate labeled features for model training."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_path, file))

            # 添加日期字段
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')

            # 生成 action 标签
            df['action'] = 0  # 默认观望
            df.loc[df['close'].shift(-1) > df['close'], 'action'] = 1  # 买入
            df.loc[df['close'].shift(-1) < df['close'], 'action'] = 2  # 卖出

            # 保存带有标签的数据
            save_file = os.path.join(save_path, file.replace("_features.csv", "_labeled.csv"))
            df.to_csv(save_file, index=False)
            print(f"Labeled features saved to {save_file}")

