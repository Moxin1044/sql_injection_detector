"""评测与交互式检测入口。

用法:
    # 训练全部模型并做交叉验证对比(推荐先跑)
    python train.py

    # 读取已保存模型的测试指标与分类报告
    python evaluate.py --report

    # 对内置数据集整体打分(精确率/召回率/F1/混淆矩阵)
    python evaluate.py --score

    # 进入交互式: 输入一行文本, 实时输出注入概率与判定
    python evaluate.py --interactive

    # 单条文本判定
    python evaluate.py --text "' OR 1=1--"
"""
from __future__ import annotations

import argparse
import json

from sklearn.metrics import classification_report

from sqli_detector import dataset, predict, train
from sqli_detector.config import METRICS_FILE, THRESHOLD


def _print_verdict(text: str, proba: float, threshold: float) -> None:
    label = "SQL 注入!" if proba >= threshold else "正常"
    bar_len = 30
    filled = int(round(proba * bar_len))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"  [{bar}] {proba:.3f}  ->  {label}")


def run_report() -> None:
    """读取训练时保存的评测摘要并打印。"""
    if not METRICS_FILE.exists():
        print(f"[evaluate] 找不到 {METRICS_FILE}，请先运行: python train.py")
        return
    with METRICS_FILE.open(encoding="utf-8") as f:
        metrics = json.load(f)
    train.print_report({"model_name": metrics["model"], "metrics": metrics}, detail=False)
    print(f"\n提示: 完整分类报告见 python train.py 的训练输出; "
          f"快速全量自检可运行 python evaluate.py --score")


def run_score() -> None:
    """用已保存模型对全部内置样本打分(含训练样本，仅作快速自检)。"""
    model = predict.load()
    df = dataset.load_synthetic()
    y_true = df[dataset.LABEL_COL]
    y_pred = predict.predict(model, df[dataset.TEXT_COL], threshold=THRESHOLD)
    print(classification_report(y_true, y_pred, target_names=["正常(0)", "注入(1)"],
                                zero_division=0))
    print("(注: 该评分含训练样本, 仅用于快速自检; 严谨评估请看 python train.py 的输出)")


def run_interactive(threshold: float) -> None:
    model = predict.load()
    print("交互式 SQL 注入检测(输入 quit 退出)")
    print("-" * 40)
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit", "q"}:
            break
        if not text.strip():
            continue
        proba = float(predict.predict_proba(model, [text])[0])
        _print_verdict(text, proba, threshold)


def run_text(text: str, threshold: float) -> None:
    model = predict.load()
    proba = float(predict.predict_proba(model, [text])[0])
    _print_verdict(text, proba, threshold)


def main() -> None:
    ap = argparse.ArgumentParser(description="SQL 注入检测评估工具")
    ap.add_argument("--report", action="store_true", help="打印已保存模型的评测摘要")
    ap.add_argument("--score", action="store_true", help="对内置样本快速打分(含训练样本)")
    ap.add_argument("--interactive", "-i", action="store_true", help="交互式实时判定")
    ap.add_argument("--text", "-t", help="判定单条文本")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"判定阈值(默认 {THRESHOLD})")
    args = ap.parse_args()

    if not (args.report or args.score or args.interactive or args.text):
        ap.print_help()
        return

    if args.report:
        run_report()
    if args.score:
        run_score()
    if args.interactive:
        run_interactive(args.threshold)
    if args.text:
        run_text(args.text, args.threshold)


if __name__ == "__main__":
    main()
