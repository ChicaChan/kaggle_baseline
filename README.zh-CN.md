# Kaggle 表格赛冲榜 Baseline

[English README](README.md)

这是一个面向 Kaggle 表格赛（Tabular）的可复用冲榜起步方案，核心为三模型集成：

- LightGBM
- XGBoost
- CatBoost

支持 OOF 验证、加权融合、多种子平均、自动生成提交文件。

## 包含内容

- `tabular_boosting_ensemble.py`：核心训练脚本（可配置项最全）
- `tabular_boosting_conservative.py`：稳分/快速迭代预设
- `tabular_boosting_aggressive.py`：冲榜预设（更高复杂度）
- `requirements.txt`：依赖列表

## 快速开始

### 1）安装依赖

```bash
pip install -r requirements.txt
```

### 2）准备数据

将比赛数据放在同一目录（通常是项目根目录）：

- `train.csv`
- `test.csv`
- `sample_submission.csv`（推荐）

### 3）选择一个配置运行

基础版（平衡配置）：

```bash
python tabular_boosting_ensemble.py --data-dir . --target target --id-column id --output submission.csv
```

稳分版：

```bash
python tabular_boosting_conservative.py --data-dir . --target target --id-column id
```

冲榜版：

```bash
python tabular_boosting_aggressive.py --data-dir . --target target --id-column id
```

## 预设对比

| 预设 | 目标 | CV 折数 | 种子数 | 复杂度 |
|---|---|---:|---:|---|
| Conservative | 稳定得分、迭代更快 | 5-fold | 1 | 较低 |
| Aggressive | 提升榜单上限 | 10-fold | 3 | 较高 |
| Base | 可自定义的中间方案 | 5-fold（默认） | 1（默认） | 中等 |

## 输出文件

每次运行会生成：

- 提交文件（`--output` 指定）
- 指标报告（`<output>.metrics.json`），包含：
  - 各折各模型分数
  - 各模型与融合后的整体 OOF 分数
  - 使用的种子与关键参数

## 常用参数（核心脚本）

```bash
python tabular_boosting_ensemble.py \
  --data-dir . \
  --target target \
  --id-column id \
  --n-splits 5 \
  --seed 42 \
  --seed-list 42,2024,3407 \
  --weights 0.4,0.35,0.25 \
  --learning-rate 0.02 \
  --n-estimators 5000 \
  --early-stopping-rounds 250 \
  --output submission.csv
```

说明：

- 包装脚本（`conservative` / `aggressive`）会先带默认参数，再拼接你传入的参数，所以你传入的值会覆盖默认值。
- 如果不传 `--target` 或 `--id-column`，脚本会尝试自动推断。

## 推荐调参路径

1. 先跑稳分版，确认 CV 稳定性。
2. 再跑冲榜版，观察是否有稳定提升。
3. 根据 OOF 与 LB 差距，微调融合权重和树模型复杂度。
4. 加入特征工程和更贴近赛题的 CV（如 Group/Time Split）。

## 环境建议

- Python 3.10+
- 中大规模数据建议 16GB+ 内存
- 仅 CPU 可运行；有 GPU 可进一步提速
