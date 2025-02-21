import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
from smolagents import Tool, HfApiModel, ToolCallingAgent
import pandas as pd
import logging

class H2OAgent:
    """
    封裝 H2O 相關操作的代理類別，包含資料導入、模型訓練、預測及解釋。
    """
    def __init__(self):
        # 初始化 H2O 並設定 logger
        h2o.init()
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @staticmethod
    def import_dataset(data_path, target_col=None, as_factor=False):
        """
        讀取 CSV 檔案並（選擇性）轉換目標欄位為類別型。
        """
        data = h2o.import_file(data_path)
        if target_col and target_col in data.columns:
            if as_factor:
                data[target_col] = data[target_col].asfactor()
        return data

    def auto_train_model(self, data_path, target_col, max_runtime_secs=3600, problem_type="classification", **kwargs):
        """
        使用 H2O AutoML 訓練模型。
        參數:
            data_path (str): 訓練資料 CSV 路徑。
            target_col (str): 目標欄位名稱。
            max_runtime_secs (int): 訓練時間上限（秒）。
            problem_type (str): "classification" 或 "regression"。
            **kwargs: 其他傳送至 H2OAutoML 的參數。
        回傳:
            dict: 包含 best_model_id 與 leaderboard (或 error 訊息)。
        """
        try:
            as_factor = (problem_type == "classification")
            data = self.import_dataset(data_path, target_col, as_factor)

            features = data.columns
            features.remove(target_col)
            aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1, **kwargs)
            aml.train(x=features, y=target_col, training_frame=data)

            leaderboard_df = aml.leaderboard.as_data_frame()
            result = {
                "best_model_id": aml.leader.model_id,
                "leaderboard": leaderboard_df.to_dict(),
            }
            self.logger.info("AutoML 訓練成功")
            return result
        except Exception as e:
            error_msg = f"AutoML 訓練失敗: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def manual_train_model(self, data_path, target_col, algorithm="GBM", params=None, problem_type="classification"):
        """
        手動訓練模型，僅支援 GBM 演算法。
        參數:
            data_path (str): 訓練資料路徑。
            target_col (str): 目標欄位名稱。
            algorithm (str): 預設 "GBM" 可支援未來擴充其他演算法。
            params (dict): 演算法參數。
            problem_type (str): "classification" 或 "regression"。
        回傳:
            dict: 包含 model_id (或 error 訊息)。
        """
        if params is None:
            params = {}

        try:
            as_factor = (problem_type == "classification")
            data = self.import_dataset(data_path, target_col, as_factor)

            features = data.columns
            features.remove(target_col)

            if algorithm.upper() == "GBM":
                model = H2OGradientBoostingEstimator(**params)
                model.train(x=features, y=target_col, training_frame=data)
            else:
                error_msg = f"目前不支援此演算法: {algorithm}"
                self.logger.error(error_msg)
                return {"error": error_msg}

            self.logger.info("手動訓練成功")
            return {"model_id": model.model_id}
        except Exception as e:
            error_msg = f"手動訓練失敗: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def predict_model(self, data_path, model_id):
        """
        使用已訓練好的模型進行預測。
        參數:
            data_path (str): 預測資料路徑。
            model_id (str): 已訓練好的模型 ID。
        回傳:
            dict: 包含 predictions 列表 (或 error 訊息)。
        """
        try:
            model = h2o.get_model(model_id)
            if not model:
                error_msg = f"無此模型: {model_id}"
                self.logger.error(error_msg)
                return {"error": error_msg}

            test_data = h2o.import_file(data_path)
            preds = model.predict(test_data).as_data_frame()["predict"].tolist()
            self.logger.info("預測成功")
            return {"predictions": preds}
        except Exception as e:
            error_msg = f"預測失敗: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def explain_model(self, model_id, data_path):
        """
        對模型進行特徵重要性解釋。
        參數:
            model_id (str): 模型 ID。
            data_path (str): 解釋用的資料路徑（可選）。
        回傳:
            dict: 包含 importance 字典、提示訊息或 error 訊息。
        """
        try:
            model = h2o.get_model(model_id)
            if not model:
                error_msg = f"無此模型: {model_id}"
                self.logger.error(error_msg)
                return {"error": error_msg}

            # 載入資料以便需要時進行解釋（可選）
            _ = h2o.import_file(data_path)

            varimp = model.varimp(use_pandas=True)
            if varimp is None:
                info_msg = "此模型不支援特徵重要性"
                self.logger.info(info_msg)
                return {"info": info_msg}
            else:
                self.logger.info("成功取得特徵重要性")
                return {"importance": varimp.to_dict()}
        except Exception as e:
            error_msg = f"解釋失敗: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}


# 建立 H2OAgent 實例
h2o_agent = H2OAgent()

# ===== 將函式包裝成 Tool =====
h2o_automl_tool = Tool(
    name="h2o_automl",
    func=h2o_agent.auto_train_model,
    description="使用 H2O AutoML 訓練模型"
)

h2o_train_tool = Tool(
    name="h2o_train",
    func=h2o_agent.manual_train_model,
    description="手動設定參數訓練模型"
)

h2o_predict_tool = Tool(
    name="h2o_predict",
    func=h2o_agent.predict_model,
    description="使用訓練好的模型進行預測"
)

h2o_explain_tool = Tool(
    name="h2o_explain",
    func=h2o_agent.explain_model,
    description="解釋模型的特徵重要性"
)

# ===== 系統提示 =====
system_prompt = """
你是一個專門處理 H2O-3 模型訓練與預測解釋的 AI 助手。
接到指令時，你會根據需求自動選擇或詢問適合的工具。
"""

# ===== 配置模型 =====
model = HfApiModel(model_name="deepseek-ai/DeepSeek-R1")

# ===== 使用 ToolCallingAgent 封裝所有工具 =====
agent = ToolCallingAgent(
    system_prompt=system_prompt,
    model=model,
    tools=[
        h2o_automl_tool,
        h2o_train_tool,
        h2o_predict_tool,
        h2o_explain_tool
    ]
)

# ===== 測試範例 =====
task = """
用AutoML訓練 path/to/train.csv，目標欄位為 target，分類問題。
時間限制 3600秒。之後在 path/to/test.csv 做預測。再幫我做模型解釋。
"""
result = agent.run(task)
print("結果:", result)