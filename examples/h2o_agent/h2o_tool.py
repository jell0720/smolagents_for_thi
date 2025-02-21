from h2o import init, import_file, H2OFrame
from h2o.estimators import H2ORandomForestEstimator

class H2OTool:
    """
    封裝 H2O-3 功能，用於資料載入、模型訓練與預測
    """
    def __init__(self):
        # 初始化 H2O 群集，這邊可設定 IP 與 port，預設為 localhost
        init()

    def load_data(self, data_path: str) -> H2OFrame:
        """
        載入指定路徑的資料檔案，並回傳 H2OFrame 物件。
        """
        return import_file(data_path)

    def train_model(self, data: H2OFrame, target: str) -> H2ORandomForestEstimator:
        """
        使用隨機森林建立模型，並回傳訓練完成的模型。

        :param data: H2OFrame 資料集
        :param target: 目標欄位名稱
        """
        features = [col for col in data.columns if col != target]
        # 分割訓練集與測試集，可依需求調整比例與隨機種子
        train, _ = data.split_frame(ratios=[0.8], seed=1234)
        model = H2ORandomForestEstimator(ntrees=50)
        model.train(x=features, y=target, training_frame=train)
        return model

    def predict(self, model: H2ORandomForestEstimator, new_data: dict) -> H2OFrame:
        """
        根據傳入的新資料（dict 格式），使用訓練好的模型進行預測。

        :param model: 已訓練好的模型
        :param new_data: 要預測的新資料（鍵值對格式）
        """
        # 將 dict 轉為 H2OFrame
        test_frame = H2OFrame(new_data)
        return model.predict(test_frame) 