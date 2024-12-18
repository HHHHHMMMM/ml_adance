# models/parameter_optimizer.py

import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid
import optuna
import logging
from datetime import datetime
import json


class ParameterOptimizer:
    def __init__(self, model_engine, data_engine, evaluator):
        self.model_engine = model_engine
        self.data_engine = data_engine
        self.evaluator = evaluator
        self.best_params = None
        self.optimization_history = []

    def grid_search(self, param_grid, train_data, valid_data):
        """
        网格搜索优化

        Parameters:
        param_grid (dict): 参数网格
        train_data: 训练数据
        valid_data: 验证数据
        """
        try:
            best_score = float('-inf')
            best_params = None

            # 创建参数组合
            param_combinations = ParameterGrid(param_grid)
            total_combinations = len(param_combinations)

            logging.info(f"开始网格搜索，共{total_combinations}种参数组合")

            for i, params in enumerate(param_combinations, 1):
                logging.info(f"测试参数组合 {i}/{total_combinations}: {params}")

                # 训练模型
                self.model_engine.train(train_data, valid_data, model_params=params)

                # 评估模型
                metrics = self.evaluator.evaluate_model(
                    self.model_engine.model,
                    valid_data.features,
                    valid_data.labels
                )

                # 记录结果
                result = {
                    'params': params,
                    'metrics': metrics,
                    'timestamp': datetime.now()
                }
                self.optimization_history.append(result)

                # 更新最佳参数
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    best_params = params

            self.best_params = best_params
            self._save_optimization_results()

            return best_params

        except Exception as e:
            logging.error(f"网格搜索失败: {str(e)}")
            return None

    def bayesian_optimization(self, param_space, train_data, valid_data, n_trials=100):
        """
        贝叶斯优化

        Parameters:
        param_space (dict): 参数空间
        train_data: 训练数据
        valid_data: 验证数据
        n_trials (int): 优化迭代次数
        """

        def objective(trial):
            # 从参数空间采样
            params = {
                name: trial._suggest(name, spec)
                for name, spec in param_space.items()
            }

            # 训练模型
            self.model_engine.train(train_data, valid_data, model_params=params)

            # 评估模型
            metrics = self.evaluator.evaluate_model(
                self.model_engine.model,
                valid_data.features,
                valid_data.labels
            )

            # 记录结果
            result = {
                'params': params,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            self.optimization_history.append(result)

            return metrics['f1']

        try:
            # 创建优化研究
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.best_params = study.best_params
            self._save_optimization_results()

            return study.best_params

        except Exception as e:
            logging.error(f"贝叶斯优化失败: {str(e)}")
            return None

    def _save_optimization_results(self):
        """保存优化结果"""
        results = {
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }

        filename = f'parameter_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4, default=str)

    def plot_optimization_history(self):
        """绘制优化历史"""
        if not self.optimization_history:
            return

        plt.figure(figsize=(12, 6))

        # 提取性能指标
        scores = [result['metrics']['f1'] for result in self.optimization_history]
        trials = range(1, len(scores) + 1)

        # 绘制优化过程
        plt.plot(trials, scores, 'b-', label='F1 Score')
        plt.plot(trials, np.maximum.accumulate(scores), 'r--', label='Best F1 Score')

        plt.xlabel('Trial')
        plt.ylabel('F1 Score')
        plt.title('Parameter Optimization History')
        plt.legend()
        plt.grid(True)

        plt.savefig('optimization_history.png')
        plt.close()