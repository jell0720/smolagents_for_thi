"""
此模組封裝 DeepSeek‑R1 的推理功能，
包含模型與 Tokenizer 的初始化以及推理操作，
便於其他工具模組重複使用。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

class DeepSeek:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def inference(self, prompt: str, max_tokens: int = 256) -> str:
        """
        根據輸入的 prompt 使用 DeepSeek‑R1 模型生成自然語言解釋

        參數:
            prompt (str): 輸入的提示文字
            max_tokens (int): 生成文字時的最大 token 數量

        回傳:
            生成的文字結果
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result 