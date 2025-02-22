"""
本模組整合了 agents/h2o_agent.py 和 tools/h2o_explain_tool.py，
提供一個端到端的流程，支援下列模式：
  - full    : 完整流程（訓練 → 預測 → (可選)深度解釋）
  - train   : 僅進行模型訓練
  - predict : 僅進行預測
  - explain : 僅對現有預測結果進行深度解釋

使用者可藉由命令列參數指定訓練資料、測試資料、目標欄位，
訓練模式（自動化或手動）、模型 ID 以及是否進行深度解釋等。
"""

import os
import argparse
import json
import logging
from dotenv import load_dotenv
import h2o

from agents.h2o_agent import H2OAgent
os.environ["JAVA_HOME"] = "/Volumes/Predator/Users/jell/.sdkman/candidates/java/17.0.7-tem"

def get_first_prediction(predict_df):
    """
    從預測結果 DataFrame 中取得第一筆預測值
    """
    if "predict" in predict_df.columns:
        return predict_df.iloc[0]["predict"]
    else:
        return str(predict_df.iloc[0].values[0])


def perform_prediction(agent, test_data, model):
    """
    封裝預測流程，回傳預測 DataFrame 與第一筆預測值
    """
    predict_df = agent.predict_model(data_path=test_data, model=model)
    first_pred = get_first_prediction(predict_df)
    return predict_df, first_pred


def perform_explanation(prediction, max_tokens=300):
    """
    調用 deepseek_explain_tool 生成解釋
    """
    from tools.h2o_explain_tool import h2o_explain_tool
    explanation = h2o_explain_tool(prediction=str(prediction), max_tokens=max_tokens)
    return explanation


def main():
    # 載入環境變數（例如 .env 中的設定）
    load_dotenv(override=True)

    # 設定 logging 格式與等級
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("main")

    parser = argparse.ArgumentParser(
        description="整合 H2O‑3 與 DeepSeek‑R1 的端到端流程"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "train", "predict", "explain"],
        default="full",
        help="執行模式：full (完整流程)、train (僅訓練)、predict (僅預測)、explain (僅解釋)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["automl", "manual"],
        default="automl",
        help="訓練模式：automl (自動化訓練) 或 manual (手動設定訓練參數)"
    )
    parser.add_argument("--train_data", type=str, help="訓練資料 CSV 檔案路徑")
    parser.add_argument("--test_data", type=str, help="測試資料 CSV 檔案路徑")
    parser.add_argument("--target", type=str, help="目標欄位名稱")
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help="手動訓練參數（JSON 格式），例如 '{\"ntrees\": 100}'"
    )
    parser.add_argument("--max_runtime", type=int, default=3600, help="AutoML 最大執行秒數")
    parser.add_argument("--model_id", type=str, help="模型識別碼（用於預測或解釋模式）")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="是否針對預測結果進行深度解釋（使用 DeepSeek‑R1）"
    )

    args = parser.parse_args()
    agent = H2OAgent()

    try:
        if args.mode == "train":
            # 僅進行模型訓練
            if not args.train_data or not args.target:
                logger.error("訓練模式必須提供 --train_data 與 --target")
                return

            if args.method == "automl":
                leader, automl_details = agent.train_model_auto(
                    data_path=args.train_data,
                    target=args.target,
                    max_runtime_secs=args.max_runtime
                )
                logger.info("AutoML 模型訓練完成，模型 ID：%s", leader.model_id)
            else:
                try:
                    params_dict = json.loads(args.params)
                except Exception as e:
                    logger.error("解析 --params 參數失敗：%s", e)
                    return
                model = agent.train_model_manual(
                    data_path=args.train_data,
                    target=args.target,
                    algorithm="GBM",
                    params=params_dict
                )
                logger.info("手動模型訓練完成，模型 ID：%s", model.model_id)

        elif args.mode == "predict":
            # 僅進行預測
            if not args.test_data or not args.model_id:
                logger.error("預測模式必須提供 --test_data 與 --model_id")
                return

            model = h2o.get_model(args.model_id)
            predict_df, first_pred = perform_prediction(agent, args.test_data, model)
            logger.info("預測結果：\n%s", predict_df)
            if args.explain:
                explanation = perform_explanation(first_pred)
                logger.info("深度解釋結果：\n%s", explanation)

        elif args.mode == "explain":
            # 僅進行解釋：依據預測結果產生深度解釋
            if not args.test_data or not args.model_id:
                logger.error("解釋模式必須提供 --test_data 與 --model_id")
                return

            model = h2o.get_model(args.model_id)
            predict_df, first_pred = perform_prediction(agent, args.test_data, model)
            logger.info("預測結果：\n%s", predict_df)
            explanation = perform_explanation(first_pred)
            logger.info("深度解釋結果：\n%s", explanation)

        elif args.mode == "full":
            # 完整流程：訓練 -> 預測 -> (可選) 深度解釋
            if not args.train_data or not args.test_data or not args.target:
                logger.error("完整流程模式必須提供 --train_data、--test_data 與 --target")
                return

            # 訓練模型
            if args.method == "automl":
                leader, automl_details = agent.train_model_auto(
                    data_path=args.train_data,
                    target=args.target,
                    max_runtime_secs=args.max_runtime
                )
                model = leader
                logger.info("AutoML 模型訓練完成，模型 ID：%s", model.model_id)
            else:
                try:
                    params_dict = json.loads(args.params)
                except Exception as e:
                    logger.error("解析 --params 參數失敗：%s", e)
                    return
                model = agent.train_model_manual(
                    data_path=args.train_data,
                    target=args.target,
                    algorithm="GBM",
                    params=params_dict
                )
                logger.info("手動模型訓練完成，模型 ID：%s", model.model_id)

            # 進行預測
            predict_df, first_pred = perform_prediction(agent, args.test_data, model)
            logger.info("預測結果：\n%s", predict_df)

            # (可選) 深度解釋
            if args.explain:
                explanation = perform_explanation(first_pred)
                logger.info("深度解釋結果：\n%s", explanation)
        else:
            logger.error("未知的執行模式: %s", args.mode)
    except Exception as e:
        logger.error("執行過程中發生錯誤：%s", e)
    finally:
        # 確保在程式結束前關閉 H2O 群集以釋放資源
        h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()