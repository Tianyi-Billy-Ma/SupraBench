# 数据清理分析报告

基于对 5048 条 Suprabank 数据的实际探索，逐项分析应该怎么做，以及为什么。

---

## 数据概况

| 字段 | 有效率 | 备注 |
|------|--------|------|
| Ka (数值) | 99.4% (5016/5048) | 31条是 M⁻²单位（2:1化学计量） |
| 温度 T | 55.7% (2810/5048) | 缺失 2238 条 |
| pH | 33.5% (1690/5048) | 缺失 3358 条 |
| 检测技术 | 55.7% (2810/5048) | 与 T 完全同步（同一批数据） |

**关键发现：缺失是结构性的**
- 有T → 必然有technique（2810条全部如此）
- 无T → 必然无technique、无pH（2238条全部如此）
- 说明数据库里有两类数据来源：详细报告条件的新文献 vs. 只报告Ka的老文献

---

## 1. 同一 pair 多测量值 → 取平均

### 数据现状
- 唯一 (molecule, host) pair：3310 个
- 有重复测量的 pair：837 个（25%）
- 最多重复：43 次（Pyrene | Macrobicyclic cyclophane）

### logKa spread 分析
重复测量间的差异非常大：

| Pair | n | logKa range | spread |
|------|---|-------------|--------|
| TMPyP4 + CB7 | 2 | [4.91, 19.65] | **14.74** |
| Acridine Yellow G + CB7 | 3 | [4.10, 10.16] | 6.06 |
| Vecuronium + CB7 | 3 | [5.34, 10.68] | 5.34 |

**结论：**
- 大 spread 的根本原因是不同 pH、温度、技术下测量的值混在一起
- "同一 pair" 的定义必须包含条件，否则平均毫无意义
- **推荐定义**：`(molecule, host, pH_bin, solvent_class)`
- pH 用 ±0.5 unit 分桶（见第5项分析）

---

## 2. 测量方法 → 是否平均？

### 数据现状
- 有技术数据：2810 条（ITC/Fluorescence/NMR/Potentiometry/Absorbance）
- 同一 pair 有多技术数据：122 pairs

### 跨技术 logKa 差异
```
Median spread:  0.38 log units
Mean spread:    0.94 log units
< 0.5 log:     71/122 (58%)  ← 技术间一致，可平均
0.5–1.0 log:   17/122 (14%)  ← 边界情况
1.0–2.0 log:   15/122 (12%)  ← 不一致，需审查
> 2.0 log:     19/122 (16%)  ← 显著不一致，不应平均
```

### 典型不一致案例
- Acridine Yellow G + CB7：ITC=4.78, Fluorescence=7.13，**Δ=6.06**
  - 原因：fluorescence 竞争法用了一个 indicator，Ka 值依赖 indicator 的精度
- Vecuronium + CB7：NMR=8.01, ITC=5.67，**Δ=5.34**
  - 原因：NMR 在 D₂O 中测，ITC 在 H₂O 中测

**结论：**
- 对于 spread < 1 log unit 的 pair：可以混合取平均（几何平均）
- 对于 spread ≥ 1 log unit 的 pair：**不应直接平均**，需要标记并拆分保留，或仅用 ITC（被认为是金标准）
- **推荐**：平均前先做 outlier 去除（Step 7），这样会自动处理大部分异常；同时加 `n_techniques` 和 `tech_spread` 列供后续审查

---

## 3. 温度校正（van't Hoff）

### 数据现状
- T ≠ 25°C 且 T 已知：508 条（10.1%）
- 主要偏离温度：30°C (141), 22°C (43), 23°C (42), 20°C (42)

### 最大问题：ΔH 数据缺失
数据库中 `δh` 字段 **全部为空**——无法从数据本身获得 ΔH。

### van't Hoff 方程
```
ln(Ka₂₅/Ka_T) = -ΔH°/R × (1/T₂₅ - 1/T_meas)
```

以 CB7（ΔH ≈ -40 kJ/mol）为例，温度偏差的影响：

| T_meas | ΔlogKa（对 logKa=5 的典型体系） |
|--------|-------------------------------|
| 30°C   | -0.07（可忽略）|
| 22°C   | +0.05（可忽略）|
| 20°C   | +0.08（可忽略）|
| 37°C   | -0.15（边界）|
| 45°C   | -0.28（显著）|
| 55°C   | -0.47（显著）|

**结论：**
- 对于大多数 host（ΔH ~ -20 to -50 kJ/mol），±10°C 带来的 logKa 误差 < 0.1 log units
- ΔlogKa 0.1 远小于 ITC 测量精度（0.05–0.2）和跨实验室重现性（0.2–0.5）
- **推荐**：对 |T-25| ≤ 10°C 的数据直接接受，不做校正（误差在实验精度范围内）
- 对 |T-25| > 10°C（主要是 30°C 的 141 条、45°C 的 22 条、55°C 的 12 条）：
  - CB7/CB8/β-CD 等常见 host：用文献 ΔH 默认值做校正，同时标记 `t_correction=vanthoff_default`
  - 其他 host：标记 `t_correction=not_possible`，保留原值，不强行校正

---

## 4. 溶剂/离子强度过滤

### 数据现状
- 有机溶剂行：670 条（13.3%）
- 主要有机溶剂：DMSO-d6 (109), methanol (94), acetonitrile (76), chloroform-D (75), chloroform (67)
- D₂O（氘代水）：428 条

### 有机溶剂的问题
超分子结合的驱动力（疏水效应、偶极-偶极）在有机溶剂中与水溶液完全不同：
- 有机溶剂中没有疏水效应，Ka 通常低 2–5 个数量级
- 不可与水溶液数据比较

**结论：**
- **丢弃**：纯有机溶剂（DMSO、MeOH、ACN、CHCl₃、DCM、acetone、toluene 等）
- **保留**：D₂O（同位素效应对 Ka 影响 < 0.2 log units，在实验精度范围内）
- **保留**：buffer、complex（通常是水-缓冲液混合体系）
- **需注意**：`solvent=complex` 里有 103 条实际含有机溶剂（通过 `solvents` 列识别）→ 需要检查 `solvents` 列来确认

### 离子强度
Suprabank 没有直接的离子强度字段，离子强度信息分散在 `additives` 列（如 "Sodium chloride", "Disodium hydrogen phosphate"）。由于无法标准化，**不做离子强度校正，保留所有水相数据**（与 AI 建议一致）。

---

## 5. pH / 质子化态

### 数据现状
- pH 缺失：3358/5048（66.5%）
- pH 分布：绝大部分集中在 pH 6–8（生理/近中性条件）
- 同一 pair 不同 pH 的：107 个 pair，ΔpH 最大达 10.0 units

### 核心问题
pH 不同 = 分子质子化态不同 = 本质上是不同的 guest。例如：
- 胺类分子在 pH 4 是 -NH₃⁺，在 pH 10 是 -NH₂，CB7 对两者的 Ka 可相差 >4 个数量级
- 数据中 `Cadaverine (fully protonated) + CB6`：NMR pH=6.39 时 Ka=10^6.39，Fluorescence pH... logKa=10.49，Δ=4.10

**结论：**
- pH 缺失时 **不能假设 pH = 7.0** 然后合并——这会把不同质子化态的测量混在一起
- **推荐方案**：
  1. pH 已知的数据：以 0.5 pH 单位分桶作为 grouping key
  2. pH 未知的数据：单独一个分组 `ph_bin="unknown"`，不与有 pH 的数据合并
  3. 最终 flag 列标注 `ph_source="measured"/"assumed_unknown"`
- 这样可以保留所有数据，同时清楚区分条件

---

## 6. 缺失数据

### 数据现状
缺失是完全结构性的（2238 行全部同时缺 T、pH、technique）。

这些数据来自哪些 host？主要是 CB8（347）、CB6（172）、Calix[4]arene（143）、β-CD（110）——都是重要体系。

### 是否假设"标准条件"？
**不完全推荐**，原因：
- T 缺失 → 假设 25°C 合理（实验室最常见温度，且温度偏差影响小）
- pH 缺失 → **不假设 pH 7.0 后合并**（原因见第5项）。应单独分组。
- technique 缺失 → 无法评估数据质量，但仍可使用 Ka 值

**推荐：**
- `t_final`：有就用测量值，缺失用 25°C，标记 `t_assumed=True`
- `ph_final`：有就用测量值，缺失标记 `ph_known=False`，**不填默认值，不与有 pH 数据合并**

---

## 7. Outliers

### 数据现状
- logKa 范围：-3.54 到 23.48（极值可疑）
- 全局 median=4.00，stdev=2.23
- **31 条 M⁻² 单位**（2:1化学计量，不是标准 1:1 Ka，必须单独处理）
- IQR k=1.5 检出：50 条
- IQR k=3.0 检出：25 条
- 组内 ≥4 个测量值的组：151 个（outlier 检测才有统计意义）

### M-2 单位（2:1 计量比）
```
Ka=1.00⋅10⁴ M⁻²  CB6 + 1-Ethyl-3-methylimidazolium
Ka=3.24⋅10¹³ M⁻²  CB8 + Pyronine
Ka=2.50⋅10¹⁶ M⁻²  CB8 + Thionin
```
这些是 2:1（host:guest 或 guest²:host）的 K₂ 常数，单位 M⁻²，不能与 1:1 Ka（M⁻¹）混用。

### Outlier 检测方法对比
| 方法 | 优点 | 缺点 | 适用情况 |
|------|------|------|---------|
| IQR k=1.5 | 简单标准 | 小样本下过于激进 | n≥8 |
| IQR k=3.0 | 保守 | 只去极端值 | n≥4 |
| Grubbs test | 统计严格 | 假设正态分布 | n≥3 |
| 修正Z-score | 对非正态鲁棒 | 稍复杂 | n≥5 |

**推荐方案：**
1. 首先**剔除 M⁻² 单位的条目**（31条，非1:1 Ka）
2. 剔除明显不合理值：logKa < -1 或 logKa > 20（物理上不合理的极端值）
3. 组内 n≥4：用 **IQR k=1.5** 检测，被标记的行保留在单独文件供审查
4. 组内 n<4：不做自动 outlier 检测
5. 被检测到的 outlier 不直接删除，先检查是否有 pH/technique 差异解释

---

## 总结：推荐清理流程

```
原始 5048 条
    ↓ 1. 剔除 M-2 单位（31条）
    ↓ 2. 剔除有机溶剂（~600条，精确识别 solvents 列）
    ↓ 3. 解析 Ka/T 为数值
    ↓ 4. 温度标记：T已知 → 记录；T缺失 → t_assumed=True，用25°C
    ↓ 5. 对 |T-25|>10°C 的常见 host：van't Hoff 默认ΔH校正；其他：标记
    ↓ 6. pH 分组：已知 → 0.5单位分桶；未知 → 单独分组，不合并
    ↓ 7. Outlier 标记（IQR 1.5, n≥4）→ 保存供审查
    ↓ 8. 每个 (molecule, host, ph_bin, solvent_class) 组内几何平均
    ↓
最终数据集（预计 ~2500–3000 unique pairs）
```

**最终数据集会有的标注列：**
- `logka_avg`：最终 logKa 值
- `n_measurements`：平均了几个测量值
- `t_assumed`：温度是否假设
- `ph_known`：pH 是否有测量值
- `t_correction`：温度校正状态
- `has_outlier_removed`：该 pair 是否有测量值被去除
- `tech_spread`：跨技术的 logKa 散布（方便审查）
