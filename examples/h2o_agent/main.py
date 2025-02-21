"""
這個 main.py 為整合案例，結合了 run_full_integration.py、run_h2o_automl.py 與 run_h2o.py 的優點，
提供一個端到端的流程，包括模型訓練（自動化或手動）、預測以及利用 DeepSeek‑R1 模型生成自然語言解釋，
並支援命令列介面，可根據需求選擇不同模式：
  - full    : 完整流程 (訓練 → 預測 → 解釋)
  - train   : 僅進行模型訓練
  - predict : 僅進行預測
  - explain : 僅進行解釋
"""

import argparse
import os
import json
import logging
from dotenv import load_dotenv
import h2o

from h2o_agent import H2OAgent  # 假設此模組中定義了 H2OAgent 類別，參照 run_h2o_automl.py

from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# 初始化 DeepSeek‑R1 模型
# -------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
deepseek_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def deepseek_inference(prompt, max_tokens=256):
    """
    利用 DeepSeek‑R1 模型，根據輸入的 prompt 生成自然語言解釋。

    參數:
        prompt (str): 輸入的提示文字
        max_tokens (int): 最大生成 token 數量，預設 256

    回傳:
        str: 生成的解釋文字
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = deepseek_model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def main():
    # 載入環境變數（例如 .env 中可能設定 hf_token 等參數）
    load_dotenv(override=True)
    
    # 設定 logging 格式與等級
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("main")
    
    parser = argparse.ArgumentParser(
        description="整合 H2O‑3 與 DeepSeek‑R1 的端到端流程，進行模型訓練、預測與解釋"
    )
    
    # 定義共用參數
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "train", "predict", "explain"],
        default="full",
        help="執行模式：full (完整流程: 訓練 + 預測 + 解釋), train (僅訓練), predict (僅預測), explain (僅解釋)"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="訓練資料 CSV 檔案路徑"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        help="預測資料 CSV 檔案路徑，用於預測或解釋"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="目標欄位名稱（訓練時必須提供）"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["automl", "manual"],
        default="automl",
        help="訓練方法：automl (自動化) 或 manual (手動)，僅在訓練或 full 模式下使用"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help="手動訓練時的參數 (JSON 格式字串)，僅在 method 為 manual 時使用"
    )
    parser.add_argument(
        "--max_runtime",
        type=int,
        default=3600,
        help="自動化訓練時的運行時間上限（秒），僅在 method 為 automl 時使用"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="已訓練模型的 ID，用於預測或解釋"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="是否根據預測結果使用 DeepSeek‑R1 進行解釋（僅在 full 模式下生效）"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="DeepSeek‑R1 生成解釋時的最大 token 數"
    )
    
    args = parser.parse_args()
    
    # 建立 H2OAgent 實例
    agent = H2OAgent()
    
    try:
        if args.mode == "train":
            # 僅進行模型訓練
            if not args.train_data or not args.target:
                logger.error("訓練模式必須提供 --train_data 與 --target")
                return
            
            if args.method == "automl":
                result = agent.auto_train_model(
                    data_path=args.train_data,
                    target_col=args.target,
                    max_runtime_secs=args.max_runtime,
                    problem_type="classification"  # 根據需求可調整（例如 regression）
                )
            else:
                try:
                    manual_params = json.loads(args.params)
                except json.JSONDecodeError as e:
                    logger.error("解析 --params 參數失敗：%s", e)
                    return
                result = agent.manual_train_model(
                    data_path=args.train_data,
                    target_col=args.target,
                    algorithm="GBM",  # 預設使用 GBM
                    params=manual_params,
                    problem_type="classification"
                )
            logger.info("訓練結果： %s", result)
        
        elif args.mode == "predict":
            # 僅進行預測
            if not args.model_id or not args.test_data:
                logger.error("預測模式必須提供 --model_id 與 --test_data")
                return
            
            result = agent.predict_model(
                data_path=args.test_data,
                model_id=args.model_id
            )
            logger.info("預測結果： %s", result)
        
        elif args.mode == "explain":
            # 僅進行解釋
            if not args.model_id or not args.test_data:
                logger.error("解釋模式必須提供 --model_id 與 --test_data")
                return
            
            result = agent.explain_model(
                model_id=args.model_id,
                data_path=args.test_data
            )
            logger.info("模型解釋結果： %s", result)
        
        elif args.mode == "full":
            # 完整流程：訓練 -> 預測 -> (可選) 解釋
            if not args.train_data or not args.target or not args.test_data:
                logger.error("完整流程模式必須提供 --train_data、--target 與 --test_data")
                return
            
            # 模型訓練
            if args.method == "automl":
                train_result = agent.auto_train_model(
                    data_path=args.train_data,
                    target_col=args.target,
                    max_runtime_secs=args.max_runtime,
                    problem_type="classification"
                )
            else:
                try:
                    manual_params = json.loads(args.params)
                except json.JSONDecodeError as e:
                    logger.error("解析 --params 參數失敗：%s", e)
                    return
                train_result = agent.manual_train_model(
                    data_path=args.train_data,
                    target_col=args.target,
                    algorithm="GBM",
                    params=manual_params,
                    problem_type="classification"
                )
            logger.info("訓練結果： %s", train_result)
            
            # 取得模型 ID
            model_id = train_result.get("best_model_id") or train_result.get("model_id")
            if not model_id:
                logger.error("未取得有效模型 ID，流程中止")
                return
            
            # 進行預測
            predict_result = agent.predict_model(
                data_path=args.test_data,
                model_id=model_id
            )
            logger.info("預測結果： %s", predict_result)
            
            # 若指定 --explain 旗標，則使用 DeepSeek‑R1 為第一筆預測結果生成解釋
            if args.explain:
                predictions = predict_result.get("predictions")
                if predictions and len(predictions) > 0:
                    first_pred = predictions[0]
                    prompt = f"根據 H2O 預測結果 '{first_pred}'，請詳細說明可能的原因與影響。"
                    explanation = deepseek_inference(prompt, max_tokens=args.max_tokens)
                    logger.info("解釋結果：\n%s", explanation)
                else:
                    logger.error("預測結果中無有效資料，無法進行解釋")
        
        else:
            logger.error("未知的執行模式")
    
    except Exception as e:
        logger.error("執行過程中發生錯誤：%s", e)
    
    finally:
        # 程式結束前關閉 H2O 群集以釋放資源
        h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main()