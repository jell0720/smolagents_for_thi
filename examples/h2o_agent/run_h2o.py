"""
這個模組用來展示如何使用 H2OAgent 整合 H2O-3，
執行資料載入、模型訓練以及針對新資料進行預測的流程。
"""

import argparse
import os
import logging
import json
from dotenv import load_dotenv
import h2o

from agents.h2o_agent import H2OAgent

# 載入環境變數 (若 .env 中有相關設定)
load_dotenv(override=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 H2O-3 進行資料載入、模型訓練與預測"
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
        help="用於預測的新資料，必須為 JSON 格式字串，例如：'{\"feature1\":[1.0, 2.0], \"feature2\":[3.0, 4.0]}'"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # 設定 logging 格式與等級
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("run_h2o")
    
    # 解析 --new-data 參數，如果未提供則採用預設值
    if args.new_data:
        try:
            new_data = json.loads(args.new_data)
        except json.JSONDecodeError as e:
            logger.error("解析 new_data JSON 字串失敗：%s", e)
            return
    else:
        new_data = {
            "feature1": [5.1, 6.2],
            "feature2": [3.5, 3.4],
            "feature3": [1.4, 5.5]
        }
    
    # 初始化 H2OAgent 代理人
    agent = H2OAgent()
    
    try:
        # 執行代理人的流程：載入資料、訓練模型、進行預測
        result = agent.run(args.data_path, args.target, new_data)
        logger.info("預測結果：\n%s", result)
    except Exception as e:
        logger.error("執行 H2OAgent 過程中發生錯誤：%s", e)
    finally:
        # 當程式結束時，關閉 H2O 群集以釋放資源
        h2o.shutdown(prompt=False)
    
if __name__ == "__main__":
    main() 