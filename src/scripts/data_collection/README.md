# Data Cleaning Pipeline

## 文件说明

```
run_pipeline.py          ← 一键运行所有步骤
step1_merge.py           ← Step 1: 合并原始数据
step2_parse_numerics.py  ← Step 2: 解析数值字段
step3_filter_solvents.py ← Step 3: 过滤有机溶剂
step4_fill_defaults.py   ← Step 4: 填充缺失条件
step5_vanthoff.py        ← Step 5: 温度校正到25°C
step6_average_pairs.py   ← Step 6: 相同pair取平均
step7_outliers.py        ← Step 7: 离群值筛除
utils.py                 ← 共享工具函数
cleaned/                 ← 各步骤的中间/最终输出
```

---

## 每一步做了什么

### Step 1 — 合并 (step1_merge.py)
把 `cb7_full.csv`（630条）和 `all_interactions.csv`（4418条）合并去重，
按 `interaction_id` 去重，输出 `01_merged.csv`（5048条）。

---

### Step 2 — 解析数值 (step2_parse_numerics.py)

原始数据里的 Ka、温度、pH 都是字符串，格式混乱：

| 原始字符串 | 解析结果 |
|-----------|---------|
| `"380.0"` | 380.0 |
| `"7.76⋅10⁴"` | 77600.0 |
| `"1.12⋅10⁷M-1"` | 1.12×10⁷ |
| `"25.0°C"` | 25.0 |
| `"7.4"` | 7.4 |

新增列：`ka_numeric`, `logka_numeric`, `t_numeric`, `ph_numeric`

---

### Step 3 — 有机溶剂过滤 (step3_filter_solvents.py)

**原则**：有机溶剂中测的 Ka 和水相环境不可比，直接丢弃。

识别逻辑：检查 `solvent` 和 `solvents` 字段中是否含有机溶剂关键词：
- 丢弃：methanol, acetonitrile, DMSO, chloroform, DCM, toluene, acetone, THF...
- 保留：water, buffer, D₂O, complex（通常是缓冲体系）

结果：约 **500–600 条**有机溶剂记录被移到 `03_dropped_organic.csv`。

---

### Step 4 — 填充缺失条件 (step4_fill_defaults.py)

数据库有大量缺失的 T 和 pH（很多文献只报告"standard conditions"但不写出来）：
- **T 缺失**（约2238条）→ 假设 **25 °C**（最常见，标准条件）
- **pH 缺失**（约3358条）→ 假设 **pH 7.0**（中性，最常见值）

同时记录 `t_assumed=True/False` 和 `ph_assumed=True/False`，便于追溯。

---

### Step 5 — van't Hoff 温度校正 (step5_vanthoff.py)

**目标**：把所有 Ka 统一校正到 25 °C（298.15 K），便于对比。

**公式**：
```
ln(Ka₂₅/Ka_T) = -ΔH°/R × (1/T₂₅ - 1/T_meas)
```

**问题**：Suprabank 数据库的 ΔH 字段几乎全部为空或无效（"="）。

**解决方案**：对常见 host 使用文献平均 ΔH：

| Host | ΔH° (kJ/mol) | 来源 |
|------|-------------|------|
| CB7 | −40 | Kaifer et al., Assaf & Nau reviews |
| CB8 | −35 | 文献平均 |
| β-CD | −20 | 文献平均 |
| β-CD | −20 | 文献平均 |
| 其他 | — | 标记为 `not_possible`，保留原始值 |

`t_correction` 列标注每行校正状态：
- `none` — 本来就是25°C
- `vanthoff` — 用van't Hoff校正了
- `assumed_25c` — T字段为空，已假设25°C
- `not_possible` — T≠25°C且无可用ΔH，保留原值（建议后续审查）

---

### Step 6 — 相同 pair 取平均 (step6_average_pairs.py)

**"相同 pair"的定义**：`(molecule, host, pH_bin, solvent_class)`

pH 以 0.5 为单位离散化（如 pH 6.8–7.2 都归入 pH_bin=7.0），原因：
- 同一分子在 pH 6.9 和 7.1 测的结果应该来自同一质子化态，可以平均。
- pH 6.0 和 7.0 则可能对应不同质子化态，**不**合并（你的第5点）。

**平均方法**：
- `Ka_avg` = **几何平均**（等价于 log 空间的算术平均）
- `logKa_avg` = `log10(Ka_avg)`
- 同时记录 `n_measurements`（测量次数）和 `techniques`（使用了哪些方法）

---

### Step 7 — 离群值筛除 (step7_outliers.py)

**方法**：Tukey IQR 法（最常用的统计异常值检测）

对每个 `(molecule, host)` 组的 `logka_25c` 值：
1. 计算 Q1（25%分位）和 Q3（75%分位）
2. IQR = Q3 - Q1
3. 超出 `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` 的点标为 outlier

**限制**：组内 < 4 个测量值时不做筛除（样本太少无法判断）。

被筛除的记录保存到 `07_outliers.csv`，可手动审查。

---

## 最终输出

| 文件 | 内容 |
|------|------|
| `cleaned/final_clean.csv` | **推荐使用**：已去重/校正/取平均/去离群值 |
| `cleaned/06_averaged.csv` | 未去离群值的平均结果 |
| `cleaned/07_outliers.csv` | 被删除的离群值（可检查） |
| `cleaned/03_dropped_organic.csv` | 有机溶剂数据（可检查） |

## 运行方法

```bash
cd data_cleaning
python3 run_pipeline.py
```
