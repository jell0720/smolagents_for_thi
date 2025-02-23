import argparse  # 用來解析命令列參數
import os  # 處理作業系統相關功能
import threading  # 用於線程鎖，避免多線程時資料衝突
import json  # 添加 json 模組引入

import litellm  # 輕量型語言模型工具
from dotenv import load_dotenv  # 載入環境變數檔案
from huggingface_hub import login  # 用於 Huggingface Hub 登入
from scripts.text_inspector_tool import TextInspectorTool  # 文字檢查工具
from scripts.text_web_browser import (
    ArchiveSearchTool,  # 檔案搜尋工具
    FinderTool,         # 尋找工具
    FindNextTool,       # 尋找下一項工具
    PageDownTool,       # 向下翻頁工具
    PageUpTool,         # 向上翻頁工具
    SearchInformationTool,  # 資訊搜尋工具
    SimpleTextBrowser,  # 簡易文字瀏覽器
    VisitTool,          # 網頁訪問工具
)
from scripts.visual_qa import visualizer  # 視覺問答工具

from smolagents import (
    CodeAgent,         # 程式碼代理人
    LiteLLMModel,      # 輕量型語言模型代理
    ToolCallingAgent,  # 工具呼叫代理人
    OpenAIServerModel,
)

# 指定模板檔案路徑
template_path = os.path.join(os.path.dirname(__file__), "chat_template.jinja")

# 讀取模板檔案內容
with open(template_path, "r", encoding="utf-8") as template_file:
    chat_template_content = template_file.read()

# 將模板內容放入配置字典中（後續可傳給 Processor 使用）
config = {"chat_template": chat_template_content}

# 授權導入的模組清單
AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]

# 載入 .env 環境變數，允許覆蓋現有設定
load_dotenv(override=True)
# 利用環境變數中的 HF_TOKEN 進行 Huggingface Hub 登入
login(os.getenv("HF_TOKEN"))

# 建立線程鎖，避免多線程時資料衝突
append_answer_lock = threading.Lock()

# 在主程式開始處添加這行，告訴 litellm 忽略不支援的參數
litellm.drop_params = True

def parse_args():
    parser = argparse.ArgumentParser()
    # 必填參數：使用者問題，例如：「Mercedes Sosa 在2007年前發行了多少張錄音室專輯？」
    parser.add_argument(
        "question", type=str, help="例如：'Mercedes Sosa 在2007年前發行了多少張錄音室專輯？'"
    )
    # 可選參數：API 基本網址
    parser.add_argument("--api-base", type=str, default=None)
    # 可選參數：API 金鑰
    parser.add_argument("--api-key", type=str, default=None)
    # 可選參數：模型 ID（預設為 None，可自行設定）
    parser.add_argument("--model-id", type=str, default=None)
    return parser.parse_args()

# 自訂角色轉換字典，用於調整模型對話中的角色名稱
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

# 定義瀏覽器用戶代理字串，用以模擬瀏覽器行為
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

# 瀏覽器設定配置
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,  # 設定視窗大小
    "downloads_folder": "downloads_folder",  # 下載檔案存放資料夾名稱
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},  # HTTP 請求標頭
        "timeout": 300,  # 請求逾時時間（秒）
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),  # 從環境變數中讀取 SERP API 金鑰
}

# 若下載資料夾不存在，則建立之
os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

def validate_tool_call(tool_call):
    """驗證工具呼叫的格式是否正確"""
    if not isinstance(tool_call, dict):
        return False
    required_fields = ['name', 'arguments']
    return all(field in tool_call for field in required_fields)

def parse_json_tool_call(output: str) -> dict:
    """解析輸出中的 JSON 格式工具呼叫
    
    Args:
        output (str): 包含工具呼叫的輸出字串
        
    Returns:
        dict: 解析後的工具呼叫字典，若解析失敗則返回 None
    """
    try:
        # 尋找 JSON 格式的工具呼叫
        start_idx = output.find('{')
        end_idx = output.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return None
            
        json_str = output[start_idx:end_idx + 1]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def custom_tool_parser(output: str) -> dict:
    """自定義工具呼叫解析器"""
    try:
        # 解析輸出中的工具呼叫
        tool_call = parse_json_tool_call(output)
        if validate_tool_call(tool_call):
            return tool_call
        return None
    except Exception as e:
        print(f"工具呼叫解析錯誤: {str(e)}")
        return None

def main():
    args = parse_args()
    text_limit = 100000  # 文字內容上限限制
    litellm.log_raw_request_response = True
    # 若需要調試，可啟用下行
    # litellm._turn_on_debug()

    model = LiteLLMModel(
        args.model_id,
        args.api_base,
        args.api_key,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        num_ctx=4096,
        temperature=0.6,  # 降低溫度以獲得更確定的輸出
        stop=["```", "Observation:", "<end_code>"],  # 添加明確的停止序列
    )
    tool_model = LiteLLMModel(
        model_id= "openai/Meta-Llama-3-1-405B-Instruct-FP8", # this is 'openai/gpt-4o' in my testing
        api_base= "https://chatapi.akash.network/api/v1/",
        api_key= "",
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=16000,
    )

    document_inspection_tool = TextInspectorTool(model, text_limit)
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    text_webbrowser_agent = ToolCallingAgent(
        model=tool_model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
        Ask him for all your questions that require browsing the web.
        Provide him as much context as possible, in particular if you need to search on a specific timeframe!
        And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
        Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
        """,
        provide_run_summary=True,
    )
    # 確保 'task' 欄位已初始化
    if text_webbrowser_agent.prompt_templates["managed_agent"].get("task") is None:
        text_webbrowser_agent.prompt_templates["managed_agent"]["task"] = ""

    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        Additionally, if after some searching you find out that you need more information to answer the question, you can use final_answer with your request for clarification as argument to request for more information."""

    # 在管理代理人的描述中，強調最終必須呼叫 final_answer 返回結果
    manager_description = (
        "你是一個協調管理者，負責指導其他子代理人從網路上蒐集資訊、分析數據，並在所有工具操作完成後生成最終答案。"
        "在過程中，請仔細檢查每個步驟的回應，確保資訊正確且完整，並在發現不一致或有疑慮時及時請求更多資料。"
        "你的目標是最後匯整出一個詳盡、可信的最終答案。"
    )

    print("Managed agent task template:", text_webbrowser_agent.prompt_templates.get("managed_agent"))
    print("manager_description:", manager_description)

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, document_inspection_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
        description=manager_description,
    )

    # 將自定義解析器添加到 manager_agent
    manager_agent.tool_parser = custom_tool_parser

    if args.question is None:
        args.question = "請提供一個問題"
    answer = manager_agent.run(args.question)
    print(f"Got this answer: {answer}")

if __name__ == "__main__":
    main()