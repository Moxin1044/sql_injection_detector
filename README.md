# SQL 注入检测（scikit-learn 机器学习框架）

基于**经典机器学习**检测 SQL 注入的完整小框架：内置合成数据集、特征工程、
三种模型训练与对比、评测报告、交互式实时判定，以及 pytest 自动化测试。

- 特征：**词 TF-IDF + 字符 n-gram TF-IDF** 双通道合并。
  字符级 n-gram 对 `OR 1=1`、注释混淆 `/**/`、大小写/URL 编码等注入特征敏感。
- 模型：`logistic`(逻辑回归) / `random_forest`(随机森林) / `svm`(LinearSVC+概率校准)。
- 判定：注入概率 ≥ 阈值(默认 0.5) 判为 SQL 注入。

## 目录结构

```
sql_injection_detector\
├── data\
│   ├── synthetic_dataset.csv      # 内置标注数据 (text,label)
│   └── build_synthetic_dataset.py # 重新生成数据集
├── sqli_detector\                 # 框架核心包
│   ├── dataset.py                 # 加载内置/外部数据
│   ├── features.py                # TF-IDF 特征提取器
│   ├── models.py                  # Pipeline 与三种分类器
│   ├── train.py                   # 训练/交叉验证/保存/报告
│   ├── predict.py                 # 加载模型与预测
│   └── config.py                  # 路径与参数
├── tests\                         # pytest 自动化测试(26 项)
├── train.py                       # 训练入口(可交互选模型/外部数据)
├── evaluate.py                    # 评测/交互检测入口
└── artifacts\                     # 训练产物 (model.joblib / metrics.json)
```

## 环境准备

```bash
python -m pip install -r requirements.txt
```

## 快速开始

```bash
# 1) 训练：自动对比三种模型(5折交叉验证)并选择 F1 最优者保存
python train.py
#    指定模型：python train.py --model logistic|random_forest|svm
#    用外部数据：python train.py --data your.csv   (需含 text,label 两列)

# 2) 查看已保存模型的评测摘要
python evaluate.py --report

# 3) 交互式实时检测(输入一行按回车，quit 退出)
python evaluate.py --interactive

# 4) 单条文本判定
python evaluate.py --text "' OR 1=1--"

# 5) 运行全部自动化测试
python -m pytest tests -q
```

## 使用外部数据集

```python
from sqli_detector import dataset
df = dataset.load_external_csv(
    "my_data.csv",
    text_col="payload",        # 你的文本列名
    label_col="is_sqli",       # 你的标签列名
    positive_value="yes",      # 映射为正例(注入)的取值
    negative_value="no",       # 映射为负例(正常)的取值
)
```

## 在自己代码里调用

```python
from sqli_detector import dataset, train, predict

df = dataset.load_synthetic()                      # 或 load_external_csv(...)
result = train.train(df, model_name="logistic")    # 训练并保存到 artifacts/
train.print_report(result)                         # 打印分类报告

label, proba = predict.predict_one("admin'--")     # 实时判定(需已训练)
print(label, proba)   # 1 0.97
```

## 当前模型参考效果(内置 123 条样本，仅作演示)

| 模型 | 5折CV F1 | 5折CV 精确率 | 5折CV 召回率 |
|---|---|---|---|
| logistic | 0.92 | 0.89 | 0.96 |
| random_forest | 0.92 | 0.88 | 0.97 |
| svm | 0.92 | 0.90 | 0.95 |

注：内置数据为**教学用合成样本**，请勿直接用于生产防护。
真实场景请用大量真实流量样本(正负均衡)重新训练，并持续迭代更新。
