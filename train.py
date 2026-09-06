"""训练入口：训练/对比模型并保存到 artifacts/。

用法:
    python train.py                 # 对比所有模型(交叉验证)后自动用最优者训练保存
    python train.py --model logistic
    python train.py --model svm
    python train.py --model random_forest
    python train.py --no-save       # 只训练评估，不写盘
    python train.py --data 你的.csv  # 改用外部数据集(需含 text/label 两列)
"""
from __future__ import annotations

import argparse

from sqli_detector import dataset, models, predict, train
from sqli_detector.config import DEFAULT_MODEL


def _pick_best(cv_results: list[dict[str, float]]) -> str:
    best = max(cv_results, key=lambda r: r["cv_f1_mean"])
    print("\n交叉验证对比结果(值越高越好):")
    header = ["模型", "F1", "精确率", "召回率", "准确率"]
    print(f"{header[0]:<16}{header[1]:>8}{header[2]:>10}{header[3]:>10}{header[4]:>10}")
    for r in cv_results:
        print(f"{r['model']:<16}{r['cv_f1_mean']:>8.4f}{r['cv_precision_mean']:>10.4f}"
              f"{r['cv_recall_mean']:>10.4f}{r['cv_accuracy_mean']:>10.4f}")
    print(f"-> 选取交叉验证 F1 最优模型: {best['model']}")
    return best["model"]


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 SQL 注入检测模型")
    ap.add_argument("--model", "-m", choices=models.available_models() + ["auto"],
                    default="auto", help="指定模型(auto: 对比后自动选最优)")
    ap.add_argument("--compare", action="store_true", help="强制先跑所有模型交叉验证对比")
    ap.add_argument("--data", "-d", default=None, help="外部 CSV(需 text/label 两列)")
    ap.add_argument("--no-save", action="store_true", help="不保存模型到磁盘")
    args = ap.parse_args()

    if args.data:
        df = dataset.load_external_csv(args.data)
        print(f"[train] 已加载外部数据: {args.data} 共 {len(df)} 条")
    else:
        df = dataset.load_synthetic()
        print(f"[train] 已加载内置合成数据: 共 {len(df)} 条")

    print(f"[train] 类别分布: {dataset.class_balance(df)}")

    auto = args.model == "auto"
    do_compare = args.compare or auto
    model_name = args.model

    if do_compare:
        results = train.compare_models(df)
        if auto:
            model_name = _pick_best(results)
        else:
            print("交叉验证结果:")
            for r in results:
                print(r)

    print(f"\n>>> 使用模型 [{model_name}] 训练(hold-out 评估) ...")
    result = train.train(df, model_name=model_name, save=not args.no_save)
    train.print_report(result)


if __name__ == "__main__":
    main()
