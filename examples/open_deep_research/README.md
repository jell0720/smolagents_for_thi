# 自動化 AI 流程示範——以梅賽德斯·索薩的錄音室專輯為例

## 簡介

本專案展示了一個自動化的 AI 流程，旨在回答以下問題：**「梅賽德斯·索薩在 2007 年之前發行了多少張錄音室專輯？」**。

此專案結合了語言模型互動、網頁瀏覽和程式碼執行，透過代理人協作來收集、處理並回答該問題。

## 功能特色

- **命令列介面 (CLI)**：允許使用者輸入問題，並選擇 API 參數。
- **自動網頁搜尋**：使用 `BeautifulSoup` 解析維基百科或其他來源，擷取歌手的錄音室專輯資料。
- **模型驅動分析**：透過 `LiteLLMModel` 分析問題，並自動選擇最佳搜尋方法。
- **可擴展代理人架構**：
  - `ToolCallingAgent`：負責網頁搜尋與資訊擷取。
  - `CodeAgent`：負責執行最終的分析與結果輸出。
- **多執行緒安全性**：使用 `threading.Lock()` 確保並行執行時不發生衝突。
- **環境變數支援**：使用 `.env` 來儲存 API 金鑰，確保安全性與便利性。

## 系統需求

- **Python 3.7 以上版本**
- 需安裝以下套件 (可透過 `pip` 安裝)：
  ```bash
  pip install -r requirements.txt
  ```
  主要依賴：
  - `argparse`
  - `os`
  - `threading`
  - `litellm`
  - `python-dotenv`
  - `huggingface_hub`
  - `requests`
  - `beautifulsoup4`
  - 其他自訂模組 (位於 `scripts/` 與 `smolagents/`)

## 安裝與設定

1. **複製此專案**
   ```bash
   git clone https://github.com/yourusername/yourrepository.git](https://github.com/jell0720/smolagents_for_thi
   cd yourrepository
   ```

2. **安裝相依套件**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 使用 venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **設置環境變數**
   在專案根目錄建立 `.env` 檔案，內容如下：
   ```dotenv
   HF_TOKEN=your_huggingface_token_here
   SERPAPI_API_KEY=your_serpapi_key_here
   ```
   *請將 `your_huggingface_token_here` 和 `your_serpapi_key_here` 替換為實際的 API 金鑰。*

## 使用方法

使用者可以透過以下指令執行腳本，輸入問題與可選參數：

```bash
python run_0219v1.py "梅賽德斯·索薩在 2007 年前發行了多少張錄音室專輯？" --api-base <API_BASE_URL> --api-key <API_KEY> --model-id <MODEL_ID>
```

### 參數說明
- **必要參數：**
  - `question`：要查詢的問題。
- **可選參數：**
  - `--api-base`：API 服務的基本 URL。
  - `--api-key`：存取 API 的金鑰。
  - `--model-id`：選擇要使用的模型 ID。

## 程式架構

```
├── run_0219v1.py  # 主執行腳本
├── chat_template.jinja  # 聊天模板
├── scripts/
│   ├── text_inspector_tool.py  # 文字分析工具
│   ├── text_web_browser.py  # 網頁瀏覽工具
│   └── visual_qa.py  # 視覺化問答模組
├── smolagents/
│   ├── code_agent.py  # 程式碼代理人
│   ├── tool_calling_agent.py  # 工具調用代理人
│   └── lite_llm_model.py  # 輕量模型封裝
├── .env  # 環境變數設定 (請自行建立)
├── requirements.txt  # 相依套件列表
└── README.md  # 使用說明
```

## 代理人架構說明

本專案採用 **ToolCallingAgent** 與 **CodeAgent** 兩種代理人來協作處理問題。

### **ToolCallingAgent**
- **功能**：負責從網路上搜尋資訊，支援多種工具，如：
  - `SearchInformationTool`：關鍵字搜尋。
  - `VisitTool`：瀏覽網頁。
  - `FinderTool`：在頁面內尋找指定內容。
  - `PageUpTool` / `PageDownTool`：頁面滾動。
  - `ArchiveSearchTool`：搜尋歷史資料庫。

### **CodeAgent**
- **功能**：負責最終結果的分析與輸出。
- **使用工具**：
  - `visualizer`：可視化工具。
  - `TextInspectorTool`：文件內容分析。
- **限制**：最多 12 個步驟，每次最多處理 8192 個 token。

## 範例輸出

在一個完整執行流程中，系統將：
1. 查詢維基百科等網站，收集梅賽德斯·索薩的專輯列表。
2. 過濾掉 2007 年以後發行的專輯。
3. 計算錄音室專輯的總數。
4. 回傳結果。

### 執行範例
```bash
python run_0219v1.py "梅賽德斯·索薩在 2007 年前發行了多少張錄音室專輯？"
```

### 可能的輸出結果
```
Got this answer: 37
```

## 記錄檔案

- **`run_0219v1.log`**：記錄完整執行過程與代理人互動。

## 訊息擷取與驗證

- **正確性檢查**：
  - 系統將比對多個資料來源，如維基百科、Discogs、AllMusic。
  - 透過 `TextInspectorTool` 進行內容驗證。
- **異常處理**：
  - 若遇到不一致的資訊，將交叉比對並標記可能的錯誤。

## 未來改進方向

- 增強多來源驗證，提升答案準確性。
- 新增 GUI 介面，使操作更直覺。
- 支援更多音樂資料庫，如 Spotify API。

## 版權與貢獻

本專案基於開源協議開發，歡迎社群成員貢獻程式碼與改進建議。

若有任何問題或改進建議，請在 GitHub 提交 issue 或 pull request。

---

**聯絡方式：jell@thi.com.tw**
