"""训练：划分数据集、交叉验证、训练多种模型并保存最佳结果。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import dataset, models
from .config import (
    ARTIFACT_DIR,
    CV_FOLDS,
    DEFAULT_MODEL,
    LABEL_COL,
    METRICS_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COL,
    VECTORIZER_FILE,
)


def _fit_one(pipeline, df: pd.DataFrame):
    X = df[TEXT_COL]
    y = df[LABEL_COL]
    return pipeline.fit(X, y)


def _report(y_true, y_pred, y_proba, threshold=0.5) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cross_validate(df: pd.DataFrame, name: str, cv: int = CV_FOLDS) -> dict[str, float]:
    """对单个模型做分层 k 折交叉验证，返回平均 F1/准确率/召回/精确率。"""
    X = df[TEXT_COL]
    y = df[LABEL_COL]
    pipe = models.build_pipeline(name)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    return {
        "model": name,
        "cv_f1_mean": round(float(cross_val_score(pipe, X, y, cv=skf, scoring="f1").mean()), 4),
        "cv_accuracy_mean": round(
            float(cross_val_score(pipe, X, y, cv=skf, scoring="accuracy").mean()), 4
        ),
        "cv_precision_mean": round(
            float(cross_val_score(pipe, X, y, cv=skf, scoring="precision").mean()), 4
        ),
        "cv_recall_mean": round(
            float(cross_val_score(pipe, X, y, cv=skf, scoring="recall").mean()), 4
        ),
    }


def compare_models(df: pd.DataFrame) -> list[dict[str, float]]:
    """对全部可用模型做交叉验证对比，供选择最佳模型。"""
    return [cross_validate(df, m) for m in models.available_models()]


def train(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    test_size: float = TEST_SIZE,
    artifact_dir: str | Path = ARTIFACT_DIR,
    save: bool = True,
) -> dict[str, Any]:
    """训练模型并保存产物。

    参数:
        df          : 含 text/label 两列的 DataFrame
        model_name  : logistic / random_forest / svm
        test_size   : 测试集比例(分层抽样)
        artifact_dir: 保存目录
        save        : 是否写盘(joblib + metrics.json)

    返回包含模型、向量化器、测试集及评估指标的字典。
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[LABEL_COL], random_state=RANDOM_STATE
    )
    pipeline = models.build_pipeline(model_name)
    _fit_one(pipeline, train_df)

    y_true = test_df[LABEL_COL].tolist()
    y_proba = pipeline.predict_proba(test_df[TEXT_COL])[:, 1]
    y_pred = pipeline.predict(test_df[TEXT_COL])

    metrics = _report(y_true, y_pred, y_proba)
    metrics.update(
        {
            "model": model_name,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "class_balance": dataset.class_balance(df),
        }
    )

    result = {
        "model_name": model_name,
        "pipeline": pipeline,
        "vectorizer": pipeline.named_steps["features"],
        "metrics": metrics,
        "test_texts": test_df[TEXT_COL].tolist(),
        "test_labels": y_true,
        "test_predictions": y_pred.tolist(),
        "test_proba": [round(float(p), 6) for p in y_proba],
    }

    if save:
        joblib.dump(pipeline, artifact_dir / MODEL_FILE.name)
        joblib.dump(
            pipeline.named_steps["features"], artifact_dir / VECTORIZER_FILE.name
        )
        with (artifact_dir / METRICS_FILE.name).open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[train] 模型已保存: {artifact_dir / MODEL_FILE.name}")
        print(f"[train] 特征提取器已保存: {artifact_dir / VECTORIZER_FILE.name}")
        print(f"[train] 指标已保存: {artifact_dir / METRICS_FILE.name}")

    return result


def print_report(result: dict[str, Any], detail: bool = True) -> None:
    """打印训练结果(分类报告 + 混淆矩阵)。"""
    m = result["metrics"]
    print("=" * 60)
    print(f"模型          : {result['model_name']}")
    print(f"训练/测试样本 : {m['n_train']} / {m['n_test']}")
    print("-" * 60)
    print("测试集指标(label=1 为 SQL 注入):")
    print(f"  准确率(accuracy) : {m['accuracy']:.4f}")
    print(f"  精确率(precision): {m['precision']:.4f}")
    print(f"  召回率(recall)   : {m['recall']:.4f}")
    print(f"  F1               : {m['f1']:.4f}")
    print(f"  混淆矩阵         : TN={m['confusion_matrix']['tn']} "
          f"FP={m['confusion_matrix']['fp']} "
          f"FN={m['confusion_matrix']['fn']} TP={m['confusion_matrix']['tp']}")
    if detail:
        print("-" * 60)
        print(classification_report(result["test_labels"], result["test_predictions"],
                                    target_names=["正常(0)", "注入(1)"], zero_division=0))
