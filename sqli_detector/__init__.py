"""SQL 注入检测框架 sqli_detector。

基于 scikit-learn 的经典机器学习方案：
  文本 -> TF-IDF(字符 n-gram + 词 n-gram) -> 分类器 -> 注入/正常 概率

核心流程：
  dataset.load_synthetic() / load_external_csv()
  -> models.build_pipeline(name)
  -> train.run(dataframe)    生成 artifacts/model.joblib 等
  -> predict.load()          加载模型实时判定
"""
from . import dataset, features, models, predict, train  # noqa: F401
from .config import (
    ARTIFACT_DIR,
    DEFAULT_MODEL,
    DATASET_PATH,
    MODEL_FILE,
    THRESHOLD,
    VECTORIZER_FILE,
)

__version__ = "0.1.0"
__all__ = [
    "dataset",
    "features",
    "models",
    "predict",
    "train",
    "ARTIFACT_DIR",
    "DEFAULT_MODEL",
    "DATASET_PATH",
    "MODEL_FILE",
    "THRESHOLD",
    "VECTORIZER_FILE",
]
