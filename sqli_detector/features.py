"""特征工程：TF-IDF 向量化。

SQL 注入载荷常被大小写/注释/URL 编码/拼接等手段混淆，
因此采用「字符 n-gram + 词 n-gram」混合特征比纯词袋更鲁棒。
"""
from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

CHAR_NGRAM_RANGE = (2, 4)  # 抓取 ' or', '1=1', "un'", 'union' 等子串
WORD_NGRAM_RANGE = (1, 2)
MAX_FEATURES = 20_000
MIN_DF = 1

# 无需词性/停用词清洗：注入特征大量依赖标点符号，必须保留。


def build_vectorizer(**kwargs: Any) -> TfidfVectorizer:
    """构建 TF-IDF 特征提取器(字符 + 词双通道由 Pipeline 的 FeatureUnion 组合)。"""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"\b\w+\b",
        **kwargs,
    )


def build_char_vectorizer(**kwargs: Any) -> TfidfVectorizer:
    """字符级 TF-IDF，对编码/注释混淆更敏感。"""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        **kwargs,
    )
