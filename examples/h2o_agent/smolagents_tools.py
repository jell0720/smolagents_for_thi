"""
此模組整合了 H2O‑3 與 DeepSeek‑R1 的功能，
並藉由 smolagents 封裝為工具介面，
提供下列功能：
  - h2o_automl_tool: 使用 H2OAgent 執行 AutoML 模型訓練
  - manual_train_tool: 使用 H2OAgent 執行手動設定的模型訓練
  - predict_tool: 根據訓練好的模型進行預測
  - explain_tool: 利用 H2OAgent 取得模型特徵重要性解釋
  - deepseek_explain_tool: 利用 DeepSeek‑R1 模型針對預測結果生成自然語言解釋
"""

import json
import logging
import warnings

from smolagents import tool  # 使用 smolagents 提供的工具裝飾器
from h2o_agent import H2OAgent

from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.warn(
    "smolagents_tools 模組已被棄用，請改用 alternative_module（或其他替代方案）進行更新。",
    DeprecationWarning,
    stacklevel=2
)

# -------------------------------
# 全域初始化：H2OAgent 與 DeepSeek‑R1 模型
# -------------------------------
agent = H2OAgent()

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
deepseek_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# -------------------------------
# smolagents 工具函式定義
# -------------------------------

@tool
def h2o_automl_tool(data_path: str, target: str, max_runtime_secs: int = 3600, problem_type: str = "classification") -> dict:
    """
    使用 H2OAgent 的 AutoML 功能訓練模型。

    參數:
        data_path (str): 訓練資料 CSV 檔案路徑
        target (str): 目標欄位名稱
        max_runtime_secs (int): 訓練時間上限（秒）
        problem_type (str): 問題類型（例如 "classification" 或 "regression"）

    回傳:
        dict: 包含最佳模型 ID 與排行榜，或錯誤訊息
    """
    try:
        result = agent.auto_train_model(
            data_path=data_path,
            target_col=target,
            max_runtime_secs=max_runtime_secs,
            problem_type=problem_type
        )
        return result
    except Exception as e:
        return {"error": f"AutoML 訓練失敗: {e}"}

@tool
def manual_train_tool(data_path: str, target: str, params: str = "{}", algorithm: str = "GBM", problem_type: str = "classification") -> dict:
    """
    使用 H2OAgent 的手動訓練功能建立模型（預設支援 GBM）。

    參數:
        data_path (str): 訓練資料 CSV 檔案路徑
        target (str): 目標欄位名稱
        params (str): 演算法參數，格式為 JSON 字串（例如 '{"ntrees": 100}'）
        algorithm (str): 使用的算法（預設 "GBM"）
        problem_type (str): 問題類型

    回傳:
        dict: 包含模型 ID 或錯誤訊息
    """
    try:
        params_dict = json.loads(params)
    except Exception as e:
        return {"error": f"解析參數失敗: {e}"}
    try:
        result = agent.manual_train_model(
            data_path=data_path,
            target_col=target,
            algorithm=algorithm,
            params=params_dict,
            problem_type=problem_type
        )
        return result
    except Exception as e:
        return {"error": f"手動訓練失敗: {e}"}

@tool
def predict_tool(data_path: str, model_id: str) -> dict:
    """
    使用已訓練好的模型進行預測。

    參數:
        data_path (str): 預測資料 CSV 檔案路徑
        model_id (str): 模型 ID

    回傳:
        dict: 預測結果（包含 predictions 列表）或錯誤訊息
    """
    try:
        result = agent.predict_model(
            data_path=data_path,
            model_id=model_id
        )
        return result
    except Exception as e:
        return {"error": f"預測失敗: {e}"}

@tool
def explain_tool(model_id: str, data_path: str) -> dict:
    """
    利用 H2OAgent 針對模型進行特徵重要性解釋。

    參數:
        model_id (str): 模型 ID
        data_path (str): 用於解釋的資料 CSV 檔案路徑（可選，但需提供以載入資料）

    回傳:
        dict: 特徵重要性或提示訊息、錯誤資訊
    """
    try:
        result = agent.explain_model(
            model_id=model_id,
            data_path=data_path
        )
        return result
    except Exception as e:
        return {"error": f"解釋失敗: {e}"}

@tool
def deepseek_explain_tool(prediction: str, max_tokens: int = 256) -> str:
    """
    利用 DeepSeek‑R1 模型根據 H2O 預測結果生成詳細自然語言解釋。

    參數:
        prediction (str): H2O 預測結果（可為第一筆預測值）
        max_tokens (int): DeepSeek‑R1 生成解釋時的最大 token 數（預設 256）

    回傳:
        str: 生成的自然語言解釋，或錯誤訊息
    """
    prompt = f"根據 H2O 預測結果 '{prediction}'，請詳細說明可能的原因與影響。"
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = deepseek_model.generate(**inputs, max_new_tokens=max_tokens)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    except Exception as e:
        return f"DeepSeek 解釋失敗: {e}" 