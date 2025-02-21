"""
此模組以 smolagents 工具形式，提供利用 DeepSeek‑R1 為 H2O‑3 預測結果生成解釋的接口。
"""

from smolagents import tool
from tools.deepseek import DeepSeek

# 初始化 DeepSeek‑R1 實例
deepseek = DeepSeek()

@tool
def h2o_explain_tool(prediction: str, max_tokens: int = 300) -> str:
    """
    根據 H2O‑3 的預測結果生成詳細解釋

    參數:
        prediction (str): 預測結果字串
        max_tokens (int): 最大生成 token 數（預設 300）

    回傳:
        生成的自然語言解釋
    """
    prompt = f"請根據預測結果 '{prediction}'，提供詳細的原因分析與解釋。"
    explanation = deepseek.inference(prompt, max_tokens=max_tokens)
    return explanation 