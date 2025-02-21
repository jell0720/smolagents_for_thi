#!/usr/bin/env python3
# coding=utf-8
"""
依據 README 指引轉寫的程式碼，整合 H2O‑3、DeepSeek‑R1 與 smolagents。

主要流程：
1. 使用 H2O‑3 載入資料、進行訓練與預測。
2. 透過 DeepSeek‑R1 模型對預測結果生成詳細解釋。
3. 將解釋功能封裝成 smolagents 工具，藉由 CodeAgent 自動化流程。
"""

# --- H2O‑3 模型訓練與預測 ---
import h2o
from h2o.automl import H2OAutoML

# --- DeepSeek‑R1 本地推理相關套件 ---
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- smolagents 封裝工具與代理 ---
from smolagents import tool, CodeAgent, HfApiModel

# 載入 DeepSeek‑R1 模型與 Tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(model_name)

def deepseek_inference(prompt, max_tokens=256):
    """
    利用 DeepSeek‑R1 模型進行推理產生結果。
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = deepseek_model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

@tool
def h2o_explain_tool(prediction: str) -> str:
    """
    將 H2O‑3 的預測結果透過 DeepSeek‑R1 生成詳細解釋。
    """
    explanation_prompt = f"請根據預測結果 '{prediction}'，提供詳細的原因分析與解釋。"
    explanation = deepseek_inference(explanation_prompt, max_tokens=300)
    return explanation

def main():
    # 初始化 H2O‑3
    h2o.init()
    
    # 載入資料（請確認 your_data.csv 路徑正確）
    data = h2o.import_file("your_data.csv")
    
    # 分割資料為訓練與測試集
    train, test = data.split_frame(ratios=[0.8], seed=123)
    
    # 設定特徵與目標欄位
    x = data.columns
    y = "target_column"  # 請根據實際資料修改目標欄位名稱
    x.remove(y)
    
    # 使用 H2O AutoML 進行模型訓練
    aml = H2OAutoML(max_runtime_secs=300, seed=1)
    aml.train(x=x, y=y, training_frame=train)
    
    # 取得最佳模型對測試集進行預測
    best_model = aml.leader
    predictions = best_model.predict(test)
    pred_df = predictions.as_data_frame()
    if pred_df.empty:
        print("預測結果為空，請檢查資料內容。")
        return
    pred_result = pred_df.iloc[0]["predict"]
    
    # 利用 smolagents 整合工具建立代理
    model_agent = HfApiModel()
    agent = CodeAgent(
        tools=[h2o_explain_tool],
        model=model_agent,
        additional_authorized_imports=["requests"]
    )
    
    # 定義任務提示，結合 H2O 預測結果讓代理生成詳細解釋
    task_prompt = f"H2O 預測結果為：{pred_result}，請提供詳細的解釋與可能的影響分析。"
    final_explanation = agent.run(task_prompt)
    print("最終解釋：", final_explanation)

if __name__ == "__main__":
    main() 