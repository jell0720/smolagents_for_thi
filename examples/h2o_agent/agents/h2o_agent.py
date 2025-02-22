"""
此模組提供 H2O‑3 高階操作接口，
包含資料載入、AutoML 與手動模型訓練、預測以及模型解釋（部分內建解釋功能）。
"""

import h2o
import logging
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator
from h2o.automl import H2OAutoML

class H2OAgent:
    def __init__(self):
        # 初始化 H2O 群集與 logger
        h2o.init()
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_data(self, data_path: str):
        """
        載入 CSV 檔案並回傳 H2OFrame
        """
        try:
            data = h2o.import_file(data_path)
            return data
        except Exception as e:
            self.logger.error("載入資料失敗: %s", e)
            raise

    def train_model_auto(self, data_path: str, target: str, max_runtime_secs: int = 3600, problem_type: str = "classification"):
        """
        使用 H2OAutoML 進行自動模型訓練

        回傳:
            leader: 表現最好的模型
            automl: AutoML 物件，可查詢排行榜等資訊
        """
        data = self.load_data(data_path)
        automl = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1, balance_classes=True)
        automl.train(y=target, training_frame=data)
        self.logger.info("AutoML 訓練完成")
        return automl.leader, automl

    def train_model_manual(self, data_path: str, target: str, algorithm: str = "GBM", params: dict = {}, problem_type: str = "classification"):
        """
        手動訓練模型，預設使用 GBM，如需其他算法請自行擴充

        回傳:
            訓練完成的模型
        """
        data = self.load_data(data_path)
        features = [col for col in data.col_names if col != target]
        if algorithm.upper() == "GBM":
            model = H2OGradientBoostingEstimator(**params)
        else:
            # 其他算法使用隨機森林作為範例
            model = H2ORandomForestEstimator(**params)
        model.train(x=features, y=target, training_frame=data)
        self.logger.info("手動訓練完成，使用算法：%s", algorithm)
        return model

    def predict_model(self, data_path: str, model):
        """
        針對新資料進行預測，並回傳 pandas.DataFrame 格式的結果
        """
        new_data = self.load_data(data_path)
        predictions = model.predict(new_data)
        self.logger.info("預測完成")
        return predictions.as_data_frame()

    def explain_model(self, model, data_path: str):
        """
        取得模型說明（例如特徵重要性），若模型不支援則回傳錯誤訊息
        """
        data = self.load_data(data_path)
        try:
            # 以模型變數重要性作為示範
            vi = model.varimp(use_pandas=True)
            self.logger.info("解釋結果已取得")
            return vi
        except Exception as e:
            self.logger.error("模型解釋失敗: %s", e)
            return {"error": str(e)} 