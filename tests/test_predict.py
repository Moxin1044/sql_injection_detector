"""预测逻辑测试：拟合后对已知样本做正确判定。"""
import numpy as np
import pytest

from sqli_detector import predict
from sqli_detector.models import build_pipeline
from sqli_detector.dataset import load_synthetic

# 强特征样本: 显著偏向某类的文本
ATTACKS = [
    "' OR 1=1--",
    "1' AND SLEEP(5)--",
    "' UNION SELECT username,password FROM users--",
    "'; DROP TABLE users;--",
    "1 AND extractvalue(rand(),concat(0x3a,database()))",
    "0x61646D696E",
]
BENIGN = [
    "SELECT id, name FROM users WHERE id = 1",
    "The quick brown fox jumps over the lazy dog",
    "?page=2&sort=asc",
    "q=python tutorial",
    "SELECT * FROM users WHERE id = ?",
]


@pytest.fixture(scope="module")
def fitted():
    df = load_synthetic()
    pipe = build_pipeline("logistic")
    pipe.fit(df["text"], df["label"])
    return pipe


def test_injections_classified_positive(fitted):
    proba = predict.predict_proba(fitted, ATTACKS)
    labels = predict.predict(fitted, ATTACKS, threshold=0.5)
    assert np.all(proba >= 0.5)
    assert np.all(labels == 1)


def test_benign_classified_negative(fitted):
    proba = predict.predict_proba(fitted, BENIGN)
    labels = predict.predict(fitted, BENIGN, threshold=0.5)
    assert np.all(proba < 0.5)
    assert np.all(labels == 0)


def test_proba_in_range_and_monotonic(fitted):
    proba = predict.predict_proba(fitted, ["' or 1=1--", "hello world"])
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba[0] >= proba[1]


def test_threshold_effect(fitted):
    """提高阈值会降低攻击被召回数(至少不增加)。"""
    texts = ["' or 1=1--", "select 1", "1' and 1=1", "admin"]
    low = predict.predict(fitted, texts, threshold=0.1)
    high = predict.predict(fitted, texts, threshold=0.9)
    assert int(low.sum()) >= int(high.sum())


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load(tmp_path / "nope.joblib")
