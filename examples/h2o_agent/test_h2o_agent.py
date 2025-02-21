from h2o_agent import H2OAgent

def main():
    # 初始化 H2OAgent
    agent = H2OAgent()
    # 請調整以下參數符合你的資料與需求
    data_path = "your_dataset.csv"  # 資料檔案路徑
    target_column = "target"        # 目標欄位名稱
    new_data = {
        "feature1": [5.1, 6.2],
        "feature2": [3.5, 3.4],
        "feature3": [1.4, 5.5]
    }
    # 執行代理人的流程
    result = agent.run(data_path, target_column, new_data)
    print(result)  # 查看預測結果

if __name__ == "__main__":
    main() 