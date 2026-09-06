"""项目路径与可调参数。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "synthetic_dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = ARTIFACT_DIR / "model.joblib"
VECTORIZER_FILE = ARTIFACT_DIR / "vectorizer.joblib"
METRICS_FILE = ARTIFACT_DIR / "metrics.json"

DEFAULT_MODEL = "logistic"
MODEL_CHOICES = ("logistic", "random_forest", "svm")

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
THRESHOLD = 0.5  # 判定为注入的正例概率阈值

TEXT_COL = "text"
LABEL_COL = "label"
POSITIVE_LABEL = 1
