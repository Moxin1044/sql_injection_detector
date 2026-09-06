"""端到端测试：训练 -> 保存 -> 重载 -> 预测 全链路。"""
import json
import pytest

from sqli_detector import dataset, predict, train
from sqli_detector.config import METRICS_FILE, MODEL_FILE, VECTORIZER_FILE


def _assert_reasonable_metrics(metrics: dict):
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert set(cm) == {"tn", "fp", "fn", "tp"}


def test_train_save_reload_predict(tmp_path):
    df = dataset.load_synthetic()

    result = train.train(df, model_name="logistic", artifact_dir=tmp_path)
    metrics = result["metrics"]
    _assert_reasonable_metrics(metrics)
    assert metrics["n_train"] + metrics["n_test"] == len(df)

    # 产物落盘
    assert (tmp_path / MODEL_FILE.name).exists()
    assert (tmp_path / VECTORIZER_FILE.name).exists()
    assert (tmp_path / METRICS_FILE.name).exists()
    with (tmp_path / METRICS_FILE.name).open(encoding="utf-8") as f:
        assert json.load(f)["model"] == "logistic"

    # 落盘模型可重载并预测(新增未见过的文本也能被向量化)
    loaded = predict.load(tmp_path / MODEL_FILE.name)
    proba = predict.predict_proba(loaded, ["1' OR 1=1--", "完全正常的文本内容哦"])
    assert len(proba) == 2


def test_train_default_artifacts_created():
    """默认训练应把模型写入 artifacts/(若 artifacts 为空则生成)。"""
    if not MODEL_FILE.exists():
        df = dataset.load_synthetic()
        train.train(df, model_name="logistic")
        assert MODEL_FILE.exists()
        assert METRICS_FILE.exists()


def test_cross_validate_all_models_run():
    df = dataset.load_synthetic()
    results = train.compare_models(df)
    names = {r["model"] for r in results}
    assert names == set(train_available())
    for r in results:
        assert 0.0 <= r["cv_f1_mean"] <= 1.0


def train_available():
    from sqli_detector.models import available_models
    return available_models()


def test_reproducibility_same_seed():
    df = dataset.load_synthetic()
    a = train.train(df, model_name="logistic", save=False)
    b = train.train(df, model_name="logistic", save=False)
    assert a["test_predictions"] == b["test_predictions"]
