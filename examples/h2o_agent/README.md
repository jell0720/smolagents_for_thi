
# H2O Agent 與 DeepSeek‑R1 整合

## 簡介
本專案整合了 H2O‑3 與 DeepSeek‑R1，提供一個端到端機器學習流程，包含以下模式：
- **full** : 完整流程（訓練 → 預測 → （可選）深度解釋）
- **train** : 僅進行模型訓練
- **predict** : 僅進行預測
- **explain** : 僅對預測結果產生深度解釋

使用者可以藉由命令列參數靈活設定流程、資料路徑、目標欄位以及訓練參數，並可選擇自動化（AutoML）或手動模式執行模型訓練。

## 目錄結構

```
examples/h2o_agent/
├── main.py                  # 主入口程式，根據模式執行訓練、預測、或解釋任務
├── agents/
│   └── h2o_agent.py         # H2O‑3 操作接口：資料載入、模型訓練、預測與解釋
└── tools/
    ├── h2o_explain_tool.py  # DeepSeek‑R1 預測解釋工具包裝
    └── deepseek.py          # 封裝 DeepSeek‑R1 推理功能（基於 Hugging Face transformers）
```

## 環境要求
- Python 3.7 或更新版本
- 必要模組：`h2o`、`python-dotenv`、`transformers`、`smolagents`

## 安裝與設定
1. **建立與啟動虛擬環境**

   ```bash
   python -m venv venv
   # Linux / macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

2. **安裝依賴套件**

   ```bash
   pip install h2o python-dotenv transformers smolagents
   ```

3. **設定環境變數（可選）**  
   如有需要，可在專案根目錄新增 `.env` 檔案，放入相應的環境設定。

## 使用方法
本程式透過命令列參數決定執行模式與相關設定。主要參數說明如下：

- `--mode`：選擇執行模式：
  - `full`：完整流程（訓練 → 預測 → （可選）深度解釋）。
  - `train`：僅進行模型訓練。
  - `predict`：僅進行預測。
  - `explain`：僅產生模型預測解釋。

- `--method`：定義訓練模式：
  - `automl`：使用 H2OAutoML 自動化訓練。
  - `manual`：使用手動參數設定，預設使用 GBM 模型（可擴充其他算法）。

- `--train_data`：訓練資料 CSV 檔案路徑。
- `--test_data`：測試資料 CSV 檔案路徑。
- `--target`：目標欄位名稱。
- `--params`：手動訓練參數（JSON 格式，例如 `{"ntrees": 100}`）。
- `--max_runtime`：AutoML 模型最大執行時間（秒）。
- `--model_id`：模型識別碼，用於預測或解釋模式時指定模型。
- `--explain`：若加上此參數，則在預測後進行深度解釋。

### 使用範例

#### 範例一：完整流程（訓練 → 預測 → 深度解釋）

```bash
python examples/h2o_agent/main.py --mode full --method automl --train_data path/to/train.csv --test_data path/to/test.csv --target target_column --explain
```

#### 範例二：僅進行模型訓練（手動模式）

```bash
python examples/h2o_agent/main.py --mode train --method manual --train_data path/to/train.csv --target target_column --params '{"ntrees": 100}'
```

#### 範例三：僅進行預測，並自動生成深度解釋

```bash
python examples/h2o_agent/main.py --mode predict --test_data path/to/test.csv --model_id your_model_id --explain
```

#### 範例四：僅生成深度解釋

```bash
python examples/h2o_agent/main.py --mode explain --test_data path/to/test.csv --model_id your_model_id
```

## 模組說明
### H2OAgent
位於 `agents/h2o_agent.py` 的 `H2OAgent` 類別封裝了 H2O‑3 的操作，包括：   
- 載入 CSV 檔案並轉換為 H2OFrame。   
- 利用 H2OAutoML 進行自動模型訓練。   
- 使用 H2OGradientBoostingEstimator 或 H2ORandomForestEstimator 進行手動模型訓練。   
- 根据新資料進行預測並產生 pandas.DataFrame 格式的結果。   
- 提供基礎模型解釋（如特徵重要性）。   

### `h2o_explain_tool` 與 DeepSeek
- **`h2o_explain_tool`**  
  位於 `tools/h2o_explain_tool.py`，此工具透過 DeepSeek‑R1 生成針對模型預測結果的詳細原因解釋，並使用 `@tool` 裝飾器（來自 smolagents）作為接口。

- **DeepSeek**  
  位於 `tools/deepseek.py`，封裝了 DeepSeek‑R1 的推理功能，使用 Hugging Face 的 transformers 完成預測解釋文字的生成。

## 注意事項
- 執行結束後，程式會自動關閉 H2O 群集以釋放資源。
- 請確保所提供的 CSV 資料路徑正確，並檢查資料格式是否符合 H2O 的要求。
- 若使用 DeepSeek‑R1 生成解釋，請確認網路連線狀況良好，並已正確下載所需模型。

## 常見問題
1. **資料載入失敗**  
   請確認資料檔案路徑正確，且 CSV 格式無誤。

2. **模型訓練或預測出現錯誤**  
   請檢查命令列參數是否齊全，並參考日誌訊息進行除錯。

3. **深度解釋結果不如預期**  
   可調整 `--explain` 模式下的 `max_tokens` 參數，或確認 DeepSeek‑R1 模型是否正確下載與初始化。

## 授權條款
本專案遵循 MIT 授權條款，詳情請參閱 LICENSE 檔案。

---

如有任何問題或建議，歡迎提出 Issue 或聯絡專案維護者。
```
