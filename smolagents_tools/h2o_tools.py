from transformers import AutoModelForCausalLM, AutoTokenizer
from smolagents import tool

# 初始化 DeepSeek‑R1 模型與 Tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(model_name)

def deepseek_inference(prompt: str, max_tokens: int = 256) -> str:
    """
    使用 DeepSeek‑R1 模型根據提示進行推理並生成結果。
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = deepseek_model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

@tool
def h2o_explain_tool(prediction: str) -> str:
    """
    利用 DeepSeek‑R1 模型對 H2O‑3 的預測結果進行推理解釋，
    並返回詳細的原因分析與影響說明。

    參數:
      prediction: H2O‑3 預測結果字串。

    回傳:
      解釋的結果字串。
    """
    explanation_prompt = f"請根據預測結果 '{prediction}'，提供詳細的原因分析與解釋。"
    explanation = deepseek_inference(explanation_prompt, max_tokens=300)
    return explanation 