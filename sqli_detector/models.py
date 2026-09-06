"""模型定义：Pipeline(向量化 + 分类器)。

提供三种可切换的经典分类器:
  - logistic     : 逻辑回归(基准，推荐，快且稳定)
  - random_forest: 随机森林(非线性，可捕获更复杂模式)
  - svm          : 线性 SVM(LinearSVC + Platt 校准，输出带概率)
"""
from __future__ import annotations

import sklearn.pipeline as skp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE
from .features import build_char_vectorizer, build_vectorizer

_MODEL_NAMES = ("logistic", "random_forest", "svm")


def _make_clf(name: str):
    if name == "logistic":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "svm":
        estimator = LinearSVC(C=1.0, max_iter=3000, dual="auto", random_state=RANDOM_STATE)
        return CalibratedClassifierCV(estimator=estimator, cv=3, method="sigmoid")
    raise ValueError(f"未知模型名 {name!r}，可选: {list(_MODEL_NAMES)}")


def build_pipeline(name: str = "logistic") -> Pipeline:
    """组装 文本 -> TF-IDF(词+字符) -> 分类器 的完整 Pipeline。"""
    if name not in _MODEL_NAMES:
        raise ValueError(f"未知模型名 {name!r}，可选: {list(_MODEL_NAMES)}")

    features = skp.FeatureUnion(
        transformer_list=[
            ("word", build_vectorizer()),
            ("char", build_char_vectorizer()),
        ]
    )
    return Pipeline([("features", features), ("class", _make_clf(name))])


def available_models() -> list[str]:
    return list(_MODEL_NAMES)
