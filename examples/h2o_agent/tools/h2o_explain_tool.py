"""
此模組以 smolagents 工具形式，提供利用輕量型語言模型（LiteLLMModel）
為 H2O‑3 預測結果生成詳細解釋的接口。

已棄用 deepseek.py，改用 litellm 作為生成引擎。
"""

from smolagents import tool
from litellm import LiteLLMModel
import os

@tool
def h2o_explain_tool(prediction: str, max_tokens: int = 300) -> str:
    """
    根據 H2O‑3 的預測結果生成詳細解釋，改為使用 LiteLLMModel 進行生成。

    參數:
        prediction (str): 預測結果字串
        max_tokens (int): 最大生成 token 數（預設 300）

    回傳:
        生成的自然語言解釋
    """
    prompt = f"請根據預測結果 '{prediction}'，提供詳細的原因分析與解釋。"

    # 從環境變數讀取 LLM 設定，或使用預設值
    llm_api_base = os.getenv("LLM_API_BASE")
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model_id = os.getenv("LLM_MODEL_ID", "gpt-3.5-turbo")

    # 初始化 LiteLLMModel，並設定最大生成 token 數量
    model = LiteLLMModel(llm_model_id, llm_api_base, llm_api_key, max_completion_tokens=max_tokens)
    explanation = model.complete(prompt)
    return explanation 