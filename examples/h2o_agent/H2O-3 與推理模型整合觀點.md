           # 端到端自動化 AI 工作流程
## 整合 H2O‑3、DeepSeek‑R1 與 smolagents

## 1. 環境設定

* **建立虛擬環境**：建議使用 `venv` 或 `conda` 來隔離專案依賴。
* **安裝必要套件：**

```bash
pip install h2o smolagents transformers huggingface_hub
```

* **套件說明：**
  - `h2o`：分散式模型訓練與預測。
  - `transformers`：載入 DeepSeek‑R1 本地推理模型。
  - `smolagents`：封裝 Agent 以實現多步驟自動化流程。

* **取得 Hugging Face API Token**：若使用 Hugging Face API，請設定環境變數 `HF_TOKEN`。

---

## 2. H2O‑3 模型訓練與預測

### 初始化 H2O 與讀取資料

```python
import h2o
h2o.init()

data = h2o.import_file("your_data.csv")
train, test = data.split_frame(ratios=[0.8], seed=123)
```

### 使用 H2O AutoML 進行訓練

```python
from h2o.automl import H2OAutoML

x = data.columns
y = "target_column"  # 根據實際情況修改
x.remove(y)

aml = H2OAutoML(max_runtime_secs=300, seed=1)
aml.train(x=x, y=y, training_frame=train)

best_model = aml.leader
predictions = best_model.predict(test)
```

### 取得預測結果

```python
pred_result = predictions.as_data_frame().iloc[0]["predict"]
```

---

## 3. 撰寫 DeepSeek‑R1 本地推理函式

### 載入模型與 Tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 定義推理函式

```python
def deepseek_inference(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = deepseek_model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
```

---

## 4. 利用 smolagents 封裝代理

### 建立自訂工具整合 H2O 預測結果與 DeepSeek 推理

```python
from smolagents import tool

@tool
def h2o_explain_tool(prediction: str) -> str:
    """
    利用 DeepSeek‑R1 對 H2O‑3 的預測結果進行推理解釋。
    """
    explanation_prompt = f"請根據預測結果 '{prediction}'，提供詳細的原因分析與解釋。"
    explanation = deepseek_inference(explanation_prompt, max_tokens=300)
    return explanation
```

### 建立 CodeAgent 並整合工具

```python
from smolagents import CodeAgent, HfApiModel

model_agent = HfApiModel()

agent = CodeAgent(
    tools=[h2o_explain_tool],
    model=model_agent,
    additional_authorized_imports=["requests"]
)

task_prompt = f"H2O 預測結果為：{pred_result}，請提供詳細的解釋與可能的影響分析。"
final_explanation = agent.run(task_prompt)
print("最終解釋：", final_explanation)
```

---

## 5. 工作流程總結

* **環境設定**：安裝並初始化 H2O‑3、DeepSeek‑R1 與 smolagents。
* **H2O‑3 模型訓練與預測**：進行資料處理、模型訓練與預測，獲得初步結果。
* **DeepSeek‑R1 推理**：根據預測結果生成深度解釋或補充報告。
* **smolagents 封裝代理**：整合流程，使其具備自動化與多步驟推理能力。

---

## 參考資源

* [H2O‑3 官方文件](https://docs.h2o.ai/)
* [HuggingFace smolagents 官方文件](https://huggingface.co/docs/smolagents)
* [DeepSeek‑R1 模型 GitHub 倉庫](https://github.com/deepseek-ai)

---

透過上述步驟與程式碼範例，即可開始嘗試整合 H2O‑3、DeepSeek‑R1 與 smolagents，建立端到端自動化 AI 系統。

