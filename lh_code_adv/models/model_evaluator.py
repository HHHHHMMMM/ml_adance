# models/model_evaluator.py
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging


class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}

    def evaluate_model(self, model, test_data, test_labels):
        """评估模型性能"""
        try:
            # 获取预测结果
            predictions = model.predict(test_data)
            pred_labels = np.argmax(predictions, axis=1)

            # 计算评估指标
            metrics = {
                'accuracy': accuracy_score(test_labels, pred_labels),
                'precision': precision_score(test_labels, pred_labels, average='weighted'),
                'recall': recall_score(test_labels, pred_labels, average='weighted'),
                'f1': f1_score(test_labels, pred_labels, average='weighted')
            }

            # 保存评估结果
            self.evaluation_results = {
                'metrics': metrics,
                'predictions': predictions,
                'true_labels': test_labels,
                'timestamp': datetime.now()
            }

            return metrics

        except Exception as e:
            logging.error(f"模型评估失败: {str(e)}")
            return None

    def plot_learning_curves(self, train_losses, valid_losses):
        """绘制学习曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curves.png')
        plt.close()

    def plot_confusion_matrix(self, true_labels, pred_labels):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(true_labels, pred_labels, normalize='index'),
                    annot=True, fmt='.2%', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def analyze_predictions(self):
        """分析预测结果"""
        if not self.evaluation_results:
            return None

        analysis = {
            'prediction_distribution': np.bincount(
                np.argmax(self.evaluation_results['predictions'], axis=1)
            ),
            'confidence_scores': {
                'mean': np.mean(np.max(self.evaluation_results['predictions'], axis=1)),
                'std': np.std(np.max(self.evaluation_results['predictions'], axis=1))
            }
        }

        return analysis

    def generate_report(self):
        """生成评估报告"""
        if not self.evaluation_results:
            return None

        report = {
            'metrics': self.evaluation_results['metrics'],
            'analysis': self.analyze_predictions(),
            'timestamp': self.evaluation_results['timestamp']
        }

        # 保存报告
        report_path = f'model_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)

        return report