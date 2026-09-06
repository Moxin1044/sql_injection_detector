"""预测：加载已保存模型对文本做实时判定。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from .config import MODEL_FILE, THRESHOLD

_INSTANCE: dict[str, object] = {}


def load(model_file: str | Path = MODEL_FILE):
    """加载已训练好的 Pipeline(向量化器+分类器已固化其中)，结果按路径缓存。"""
    model_file = Path(model_file).resolve()
    key = str(model_file)
    if key in _INSTANCE:
        return _INSTANCE[key]
    if not model_file.exists():
        raise FileNotFoundError(
            f"未找到模型文件 {model_file}。请先运行训练脚本生成 artifacts/model.joblib。"
        )
    model = joblib.load(model_file)
    _INSTANCE[key] = model
    return model


def _prepare(texts: str | Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def predict_proba(
    model, texts: str | Iterable[str]
) -> np.ndarray:
    """返回每个文本属于注入(1)的概率。"""
    return model.predict_proba(_prepare(texts))[:, 1]


def predict(
    model, texts: str | Iterable[str], threshold: float = THRESHOLD
) -> np.ndarray:
    """按阈值判定(1=注入, 0=正常)。"""
    proba = predict_proba(model, texts)
    return (proba >= threshold).astype(int)


def predict_one(text: str, threshold: float = THRESHOLD) -> tuple[int, float]:
    """便捷入口：判定单条文本，返回 (标签, 注入概率)。"""
    model = load()
    proba = float(predict_proba(model, [text])[0])
    return (1 if proba >= threshold else 0), proba
