"""
本模組示範如何整合 H2O‑3、DeepSeek‑R1 與 smolagents：
1. 使用 H2O‑3 執行資料載入、模型訓練與預測。
2. 利用 DeepSeek‑R1 模型根據預測結果生成詳細解釋。
3. 透過 smolagents 的工具與 CodeAgent 將上述流程串接成自動化任務。

請根據專案需求調整參數、路徑與預設資料格式。
"""

import argparse
import os
import json
import logging
from dotenv import load_dotenv
import h2o

# 載入 transformers 與 smolagents 相關模組
from transformers import AutoModelForCausalLM, AutoTokenizer
from smolagents import tool, CodeAgent, HfApiModel

# 載入自訂的 H2O 代理人 (請確保 h2o_agent.py 與 h2o_tool.py 均已建立)
from agents.h2o_agent import H2OAgent

# 載入環境變數 (例如 .env 中可能設定 hf_token 等參數)
load_dotenv(override=True)

# 設定 logging 格式與等級
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_full_integration")

# -----------------------
# DeepSeek‑R1 模型初始化
# -----------------------
model_name = "deepseek-ai/DeepSeek-R1-Distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(model_name)

def deepseek_inference(prompt, max_tokens=256):
    """
    使用 DeepSeek‑R1 模型進行推理，根據輸入的 prompt 生成自然語言解釋。
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = deepseek_model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# 利用 smolagents 的 @tool 裝飾器封裝一個工具函式，
# 用於接收 H2O‑3 預測結果並產生詳細解釋。
@tool
def h2o_explain_tool(prediction: str) -> str:
    """
    利用 DeepSeek‑R1 依據 H2O‑3 預測結果提供詳細的原因分析與影響解釋。
    """
    prompt = f"根據 H2O 預測結果 '{prediction}'，請詳細說明可能的原因與影響。"
    explanation = deepseek_inference(prompt, max_tokens=300)
    return explanation

# -----------------------
# 參數解析
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="整合 H2O‑3、DeepSeek‑R1 與 smolagents 進行預測與解釋"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="資料檔案路徑 (例如: datasets/data.csv)"
    )
    parser.add_argument(
        "target",
        type=str,
        help="目標欄位名稱"
    )
    parser.add_argument(
        "--new-data",
        type=str,
        default=None,
        help="用於預測的新資料，格式需為 JSON 字串，例如：'{\"feature1\":[1.0, 2.0], \"feature2\":[3.0, 4.0]}'"
    )
    return parser.parse_args()

# -----------------------
# 主流程
# -----------------------
def main():
    args = parse_args()

    # 解析 --new-data 參數，若未指定則使用預設資料
    if args.new_data:
        try:
            new_data = json.loads(args.new_data)
        except json.JSONDecodeError as e:
            logger.error("解析 new_data JSON 失敗：%s", e)
            return
    else:
        new_data = {
            "feature1": [5.1, 6.2],
            "feature2": [3.5, 3.4],
            "feature3": [1.4, 5.5]
        }
    
    # 使用 H2OAgent 進行 H2O‑3 流程，取得預測結果
    agent = H2OAgent()
    try:
        prediction_df = agent.run(args.data_path, args.target, new_data)
        # 假設預測結果以 DataFrame 形式回傳，取第一筆預測值
        prediction_val = prediction_df.iloc[0]["predict"]
        logger.info("H2O 預測結果： %s", prediction_val)
    except Exception as e:
        logger.error("執行 H2OAgent 過程中發生錯誤：%s", e)
        return

    # 建立 CodeAgent 將 h2o_explain_tool 整合進多步驟工具流程中
    model_agent = HfApiModel()
    agent_tools = CodeAgent(
        tools=[h2o_explain_tool],
        model=model_agent
    )
    
    # 將 H2O 預測結果送入下一步驟，請求詳細解釋
    task_prompt = f"H2O 預測結果為：{prediction_val}，請提供詳細的原因分析與可能影響。"
    try:
        explanation = agent_tools.run(task_prompt)
        logger.info("最終解釋：\n%s", explanation)
    except Exception as e:
        logger.error("使用 smolagents 執行工具時發生錯誤：%s", e)
    finally:
        # 程式結束前關閉 H2O 群集，釋放資源
        h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main() 