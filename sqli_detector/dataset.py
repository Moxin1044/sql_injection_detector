"""数据加载：内置合成样本 + 外部 CSV 接口。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATASET_PATH, LABEL_COL, POSITIVE_LABEL, TEXT_COL


def load_synthetic(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """加载随项目附带的内置合成数据集(无需联网)。

    返回含 ``text`` 与 ``label`` 两列的 DataFrame，
    ``label``: 1 = SQL 注入, 0 = 正常。
    """
    df = pd.read_csv(path, encoding="utf-8")
    _validate(df)
    return df


def load_external_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    positive_value: object = 1,
    negative_value: object = 0,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """加载外部 CSV 数据集并规范化为两列 DataFrame。

    要求 CSV 含一列文本(默认 ``text``)与一列标签(默认 ``label``)。
    若你的 CSV 标签列不是 ``1``/``0``，可指定 ``positive_value``/``negative_value``
    将被映射为 ``1``/``0``，其余取值视为缺失而被丢弃。

    示例::

        df = load_external_csv(
            "my_data.csv",
            text_col="payload", label_col="is_sqli", positive_value="yes", negative_value="no",
        )
    """
    raw = pd.read_csv(path, encoding=encoding)
    if text_col not in raw.columns or label_col not in raw.columns:
        raise ValueError(
            f"CSV 缺少必需列。需要文本列 '{text_col}' 与标签列 '{label_col}'，"
            f"实际列: {list(raw.columns)}"
        )

    df = pd.DataFrame(
        {
            TEXT_COL: raw[text_col].astype(str),
            LABEL_COL: raw[label_col],
        }
    )
    df[LABEL_COL] = df[LABEL_COL].map(
        {positive_value: POSITIVE_LABEL, negative_value: 0}
    )
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("数据集为空。")
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"数据须包含列 {TEXT_COL!r} 与 {LABEL_COL!r}。")
    labels = set(df[LABEL_COL].unique())
    if not labels.issubset({0, POSITIVE_LABEL}):
        raise ValueError(f"标签列取值须为 {{0, {POSITIVE_LABEL}}}，实际: {labels}")
    if 0 not in labels or POSITIVE_LABEL not in labels:
        raise ValueError(f"数据须同时包含正例({POSITIVE_LABEL})与负例(0)。")


def class_balance(df: pd.DataFrame) -> dict[str, float]:
    """返回正负样本数量与正例占比。"""
    total = len(df)
    pos = int((df[LABEL_COL] == POSITIVE_LABEL).sum())
    neg = total - pos
    return {"positive": pos, "negative": neg, "positive_ratio": round(pos / total, 4)}


def to_records(df: pd.DataFrame) -> Iterable[tuple[str, int]]:
    """迭代 (text, label) 记录。"""
    return zip(df[TEXT_COL].tolist(), df[LABEL_COL].tolist())
