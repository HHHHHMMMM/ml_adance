import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_xgboost_model(data_path):
    # 读取数据
    df = pd.read_csv(data_path)

    # 删除无用字段（如 ts_code, trade_date 等）
    df = df.drop(columns=['ts_code', 'trade_date', 'end_date'], errors='ignore')

    # 分离特征和标签
    X = df.drop(columns=['action', 'date'])  # 删除非数值特征
    y = df['action']

    # 确保所有特征为数值类型
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为 DMatrix 格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 设置模型参数
    params = {
        'objective': 'multi:softmax',  # 多分类任务
        'num_class': 3,  # action 有 3 类：0=观望, 1=买入, 2=卖出
        'eta': 0.1,
        'max_depth': 6,
        'eval_metric': 'merror'
    }

    # 训练模型
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10)

    # 模型评估
    y_pred = model.predict(dtest)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 保存模型
    model.save_model("xgboost_stock_model.json")
    print("Model saved as xgboost_stock_model.json")


if __name__ == "__main__":
    data_path = "data/labeled/features_processed_600000.SH_daily.csv"  # 替换为你的数据路径
    train_xgboost_model(data_path)
