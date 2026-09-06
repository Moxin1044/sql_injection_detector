"""数据集加载与完整性测试。"""
import pytest

from sqli_detector import dataset
from sqli_detector.config import DATASET_PATH

SAMPLES = 123  # build_synthetic_dataset 生成的总数


def test_synthetic_dataset_exists():
    assert DATASET_PATH.exists(), "请先运行 python data/build_synthetic_dataset.py"


def test_load_synthetic_shape_and_columns():
    df = dataset.load_synthetic()
    assert list(df.columns) == ["text", "label"]
    assert len(df) == SAMPLES
    assert df["text"].isna().sum() == 0
    assert set(df["label"].unique()) <= {0, 1}


def test_balanced_both_classes_present():
    df = dataset.load_synthetic()
    assert (df["label"] == 1).any() and (df["label"] == 0).any()
    bal = dataset.class_balance(df)
    assert bal["positive"] + bal["negative"] == len(df)
    assert 0.0 < bal["positive_ratio"] < 1.0


def test_no_empty_texts():
    df = dataset.load_synthetic()
    assert df["text"].str.strip().eq("").sum() == 0


def test_injection_samples_cover_categories():
    """注入样本应覆盖不同攻击类别(关键字抽查)。"""
    df = dataset.load_synthetic()
    injections = df.loc[df["label"] == 1, "text"].str.lower().tolist()
    keywords = ["union select", "sleep(", "waitfor", "0x", "1=1", "drop table", "char("]
    found = [k for k in keywords if any(k in s for s in injections)]
    assert len(found) >= 5, f"攻击类别覆盖不足, 仅命中: {found}"


@pytest.fixture
def external_csv(tmp_path):
    p = tmp_path / "external.csv"
    p.write_text(
        "payload,is_sqli\n"
        "' or 1=1--,yes\n"
        "select name from users where id=1,no\n"
        "a,maybe\n",  # maybe 行应被丢弃
        encoding="utf-8",
    )
    return p


def test_load_external_csv_mapping(external_csv):
    df = dataset.load_external_csv(
        external_csv, text_col="payload", label_col="is_sqli",
        positive_value="yes", negative_value="no",
    )
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 2  # 'maybe' 行被丢弃
    assert set(df["label"]) == {0, 1}


def test_load_external_csv_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dataset.load_external_csv(p, text_col="payload", label_col="is_sqli")


def test_load_external_csv_only_one_class(tmp_path):
    p = tmp_path / "one_class.csv"
    p.write_text("text,label\nhello,0\nworld,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dataset.load_external_csv(p)
