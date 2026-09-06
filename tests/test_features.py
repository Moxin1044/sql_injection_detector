"""特征工程与模型管道测试。"""
import pytest

from sqli_detector import models
from sqli_detector.config import RANDOM_STATE
from sklearn.model_selection import train_test_split

from sqli_detector.dataset import load_synthetic

TINY = [
    ("select * from users", 0),
    ("admin", 0),
    ("hello world this is a normal query", 0),
    ("' or 1=1--", 1),
    ("' union select user,pass from users--", 1),
    ("1 and sleep(5)", 1),
    ("0x61646d696e", 1),
    ("normal sentence without any sql", 0),
]


def _make_df():
    import pandas as pd
    return pd.DataFrame(TINY, columns=["text", "label"])


@pytest.mark.parametrize("name", models.available_models())
def test_pipeline_fit_predict_shape(name):
    pipe = models.build_pipeline(name)
    df = _make_df()
    pipe.fit(df["text"], df["label"])
    proba = pipe.predict_proba(df["text"])
    assert proba.shape == (len(df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    pred = pipe.predict(df["text"])
    assert set(pred) <= {0, 1}


@pytest.mark.parametrize("name", models.available_models())
def test_pipeline_class_weight_balance_random_state(name):
    """相同数据/种子下训练结果应可复现。"""
    a = models.build_pipeline(name)
    b = models.build_pipeline(name)
    df = _make_df()
    a.fit(df["text"], df["label"])
    b.fit(df["text"], df["label"])
    assert (a.predict(df["text"]) == b.predict(df["text"])).all()


def test_invalid_model_name():
    with pytest.raises(ValueError):
        models.build_pipeline("does_not_exist")


def test_unknown_vectorizer_reuse():
    """同一向量化器拟合后应能transform新文本(feature 命名空间固定)。"""
    from sqli_detector.features import build_vectorizer
    vec = build_vectorizer()
    df = _make_df()
    X = vec.fit_transform(df["text"])
    assert X.shape[0] == len(df)
    # 用内置真实数据扩充字典后对新文本可 transform
    real = load_synthetic()
    vec.fit(real["text"])
    X2 = vec.transform(["admin", "' or 1=1--", "hello"])
    assert X2.shape == (3, len(vec.get_feature_names_out()))


def test_full_data_train_test_split_stratified():
    df = load_synthetic()
    tr, te = train_test_split(df, test_size=0.25, stratify=df["label"],
                              random_state=RANDOM_STATE)
    for s in (tr, te):
        assert (s["label"] == 0).any() and (s["label"] == 1).any()
