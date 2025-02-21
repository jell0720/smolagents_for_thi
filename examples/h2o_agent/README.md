# H2O & DeepSeek-R1 端到端整合

## 簡介
本專案提供了一個端到端的機器學習解決方案，結合了 **H2O-3**（AutoML 或手動訓練）與 **DeepSeek-R1**（自然語言解釋），可用於數據建模、預測以及模型結果解釋。

## 功能
- **完整流程 (full)**：執行模型訓練 → 預測 → 結果解釋
- **模型訓練 (train)**：使用 AutoML 或手動設定參數訓練模型
- **模型預測 (predict)**：使用已訓練的模型進行預測
- **預測結果解釋 (explain)**：利用 DeepSeek-R1 模型為預測結果生成自然語言解釋

## 環境設定
### 1. 安裝 Python 依賴套件
請確保您的環境安裝了 **Python 3.8+**，並使用以下指令安裝必要的套件：

```bash
pip install -r requirements.txt
```

> 若無 `requirements.txt`，請手動安裝以下套件：
>
```bash
pip install h2o transformers dotenv argparse json logging
```

### 2. 設定環境變數

在專案根目錄建立 `.env` 檔案，並設定 Hugging Face Token（如有需要）：
```ini
HF_TOKEN=your_huggingface_token
```

## 使用方式
### 1. 完整流程（訓練 → 預測 → 解釋）
```bash
python main.py --mode full --train_data train.csv --test_data test.csv --target label --method automl --max_runtime 3600 --explain
```

### 2. 只進行模型訓練
```bash
python main.py --mode train --train_data train.csv --target label --method manual --params '{"ntrees": 100, "max_depth": 5}'
```

### 3. 只進行預測
```bash
python main.py --mode predict --model_id model_12345 --test_data test.csv
```

### 4. 只進行解釋
```bash
python main.py --mode explain --model_id model_12345 --test_data test.csv
```

## 參數說明
| 參數 | 說明 | 必要性 |
|------|------|--------|
| `--mode` | 指定執行模式 (`full`, `train`, `predict`, `explain`) | 必填 |
| `--train_data` | 訓練數據 CSV 檔案 | `train` 或 `full` 必填 |
| `--test_data` | 預測數據 CSV 檔案 | `predict`, `explain`, `full` 必填 |
| `--target` | 目標欄位名稱 | `train` 或 `full` 必填 |
| `--method` | 訓練方法 (`automl` 或 `manual`) | `train` 或 `full` 適用 |
| `--params` | 手動訓練參數（JSON 格式） | `manual` 訓練模式時適用 |
| `--max_runtime` | AutoML 訓練時間上限（秒） | `automl` 訓練模式時適用 |
| `--model_id` | 指定已訓練模型的 ID | `predict` 或 `explain` 模式必填 |
| `--explain` | 是否進行預測結果解釋 | `full` 模式可選 |
| `--max_tokens` | 解釋時最大 token 數 | 預設 256 |

## H2O 與 DeepSeek-R1
### H2O-3
H2O 是一個開源的機器學習平台，支援 AutoML 以及手動設定模型參數。

### DeepSeek-R1
DeepSeek-R1 是一款自然語言處理 (NLP) 模型，能夠基於模型輸出生成人類可讀的解釋。

## 注意事項
- **H2O-3 需要 Java 環境**，請確保安裝了 Java 8 或以上版本。
- **DeepSeek-R1 需要 Hugging Face Token**，如未設定可能無法下載模型。
- 訓練數據與測試數據需為 **CSV 格式**，並確保目標欄位名稱一致。

## 參考資料
- [H2O 官方文件](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/)
- [DeepSeek AI](https://huggingface.co/deepseek-ai)

---
本專案旨在提供高效且易用的機器學習流程，適合資料科學家與開發者快速部署模型並進行解釋分析。

