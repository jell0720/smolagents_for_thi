from smolagents import BaseAgent  # 此為假設的基底代理人類別
from h2o_tool import H2OTool

class H2OAgent(BaseAgent):
    """
    使用 H2O-3 進行模型訓練與預測的代理人，能與其他代理人協作處理複雜任務。
    """
    def __init__(self):
        super().__init__()
        self.h2o_tool = H2OTool()

    def run(self, data_path: str, target: str, new_data: dict):
        """
        執行 H2O-3 的流程：
        1. 載入資料
        2. 訓練模型
        3. 針對新資料進行預測
        
        :param data_path: 資料檔案路徑
        :param target: 目標欄位名稱
        :param new_data: 預測用新資料（dict 格式）
        :return: 預測結果（H2OFrame 物件或轉為其他格式）
        """
        data = self.h2o_tool.load_data(data_path)
        model = self.h2o_tool.train_model(data, target)
        predictions = self.h2o_tool.predict(model, new_data)
        return predictions.as_data_frame()  # 轉為 pandas.DataFrame 方便後續處理 