# data/feature_selector.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from boruta import BorutaPy
import logging


class FeatureSelector:
    def __init__(self):
        self.logger = logging.getLogger('feature_selector')
        self.selected_features = None
        self.importance_scores = None

    def select_features(self, X, y, method='random_forest', n_features=20):
        """
        特征选择

        Parameters:
        -----------
        X : DataFrame
            特征数据
        y : Series
            标签
        method : str
            特征选择方法 ('boruta', 'mutual_info', 'random_forest')
        n_features : int
            选择的特征数量，仅在mutual_info和random_forest方法中使用

        Returns:
        --------
        list : 选中的特征名列表
        """
        try:
            self.logger.info(f"开始特征选择，方法：{method}")

            if method == 'boruta':
                selected = self._boruta_selection(X, y)
            elif method == 'mutual_info':
                selected = self._mutual_info_selection(X, y, n_features)
            elif method == 'random_forest':
                selected = self._random_forest_selection(X, y, n_features)
            else:
                raise ValueError(f"不支持的特征选择方法: {method}")

            self.selected_features = selected
            self.logger.info(f"特征选择完成，选择了 {len(selected)} 个特征")

            return selected

        except Exception as e:
            self.logger.error(f"特征选择失败: {str(e)}")
            # 如果特征选择失败，返回所有特征
            return list(X.columns)

    def _boruta_selection(self, X, y):
        """Boruta特征选择"""
        try:
            rf = RandomForestClassifier(
                n_jobs=-1,
                class_weight='balanced',
                max_depth=5
            )

            boruta = BorutaPy(
                rf,
                n_estimators='auto',
                verbose=0,  # 减少输出
                random_state=42
            )

            # 确保数据类型正确
            X_values = X.values.astype(np.float32)
            y_values = y.values.astype(np.float32)

            boruta.fit(X_values, y_values)

            selected_features = X.columns[boruta.support_].tolist()
            self.importance_scores = pd.Series(
                boruta.ranking_,
                index=X.columns
            )

            return selected_features

        except Exception as e:
            self.logger.error(f"Boruta特征选择失败: {str(e)}")
            return list(X.columns)

    def _mutual_info_selection(self, X, y, n_features):
        """互信息特征选择"""
        try:
            if n_features is None:
                n_features = len(X.columns) // 2  # 默认选择一半特征

            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(n_features, len(X.columns))
            )

            selector.fit(X, y)

            selected_features = X.columns[selector.get_support()].tolist()
            self.importance_scores = pd.Series(
                selector.scores_,
                index=X.columns
            )

            return selected_features

        except Exception as e:
            self.logger.error(f"互信息特征选择失败: {str(e)}")
            return list(X.columns)

    def _random_forest_selection(self, X, y, n_features):
        """随机森林特征选择"""
        try:
            if n_features is None:
                n_features = len(X.columns) // 2  # 默认选择一半特征

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X, y)

            importance_scores = pd.Series(
                rf.feature_importances_,
                index=X.columns
            )

            selected_features = importance_scores.nlargest(
                min(n_features, len(X.columns))
            ).index.tolist()

            self.importance_scores = importance_scores

            return selected_features

        except Exception as e:
            self.logger.error(f"随机森林特征选择失败: {str(e)}")
            return list(X.columns)

    def get_feature_importance(self):
        """获取特征重要性得分"""
        if self.importance_scores is None:
            raise ValueError("请先运行特征选择")
        return self.importance_scores.sort_values(ascending=False)