# CB7 数据分析报告

基于 630 条 CB7 interaction 数据的实际探索。

---

## 数据概况

| 字段 | 完整率 | 备注 |
|------|--------|------|
| Ka (字符串) | 630/630 (100%) | 无 M⁻² 单位条目 |
| 温度 T | 616/630 (98%) | 14条缺失 |
| pH | 335/630 (53%) | 295条缺失 |
| 技术方法 | 616/630 (98%) | 与T完全同步 |
| 溶剂 | 630/630 (100%) | |
| ΔH | 0/630 (0%) | 全缺失，无法直接用 |

CB7 数据质量远高于全库平均（全库 technique 仅55%）。

---

## 1. Ka 数值解析

**结果：630/630 全部可解析，无 M⁻² 单位。**

logKa 分布（n=630）：
- 范围：-3.0 到 19.65
- 众数区间：4–7（占 ~50%）
- 极端值：20 条 logKa > 12 或 < 0

CB7 物理上可合理的 Ka 范围参考：
- 上限：最强已知 CB7 客体（ferrocenemethyl trimethylammonium 等）logKa ≈ 15–16（ITC直测）
- 下限：极弱结合 logKa ≈ 0–1（几乎不结合）
- **logKa = 19.65（TMPyP4）可疑**，需专项核查

**做法：** 正常解析，不删极端值，但标记 logKa > 17 或 < 0 的条目供审查。

---

## 2. 重复测量值

**120个 molecule 有重复测量（共630条中的 ~240条）**

### spread 分布（logKa 最大 - 最小）
| spread 范围 | 分子数 |
|-------------|--------|
| < 0.5 | 53 (44%) ← 一致，可取平均 |
| 0.5 – 1.0 | 22 (18%) ← 可接受范围 |
| 1.0 – 2.0 | 18 (15%) ← 有差异，需检查原因 |
| > 2.0 | 27 (23%) ← 显著差异，不应盲目平均 |

### 大 spread 的典型原因

**① pH 不同 → 质子化态不同**（不是测量误差，是真实的物理差异）
```
L-Lysine：  pH 2.0 → logKa=2.32    pH 7.0 → logKa=5.49   Δ=3.17
L-Arginine：pH 2.0 → logKa=2.49    pH 6.0 → logKa=5.15   Δ=2.66
Thiabendazole：pH 3.4→logKa=2.18   pH 10.5→logKa=6.26     Δ=4.08
```
这些**不是 outlier**，是不同质子化态的 CB7-guest 对应的真实 Ka。

**② 竞争测定法（Competitive Assay）中 indicator 不同 → 系统误差**
```
Mercury(2+) + CB7：
  Hoechst 33342 作 indicator → logKa=3.73
  Berberine chloride 作 indicator → logKa=5.59   Δ=1.86
  Rhodamine B 作 indicator → logKa=5.48

1,2-Phenylenediamine + CB7：
  最高 indicator 组 → logKa=6.15
  最低 indicator 组 → logKa=4.37                 Δ=1.78
```
原因：竞争法的准确度依赖于 indicator 自身 Ka 的精度。Hoechst 33342 的 Ka 在不同文献里有争议，导致衍生出的客体 Ka 系统偏低。

**③ 直接测量 vs 竞争测量对同一分子的系统差异**
```
Acridine Yellow G：ITC直接=4.78    荧光竞争=7.13    Δ=2.35
Vecuronium：NMR竞争=8.01           ITC直接=5.67     Δ=2.34
Pancuronium：ITC直接=5.77          NMR竞争=7.67     Δ=1.90
```

**④ 真正的数据错误**
```
TMPyP4：
  doi=10.1039/D3SC03865A → logKa=4.91
  doi=10.1039/D3SC03865A → logKa=19.65 (同一篇论文！)
```
两条来自同一篇文章，logKa 差 14.74——几乎可以确定其中一条录入有误。

### 结论
"同一 pair 取平均"必须分两层处理：
1. **pH 分层**：不同 pH → 不同行，绝不合并（pH_bin = 0.5 单位）
2. **方法内平均**：同 pH、同 technique（ITC/荧光/NMR）的重复 → 取几何平均
3. **方法间不自动平均**：直接测量 vs 竞争测量差异 > 1 log unit 时，标记而不合并
4. **Outlier 先处理**：TMPyP4 19.65 这类先去掉再平均

---

## 3. 温度处理（van't Hoff）

### 温度分布
| 温度 | 行数 | 方法 |
|------|------|------|
| 25°C | 481 (78%) | ITC + Fluorescence + NMR |
| 23°C | 26 | 全部 Fluorescence（Nau课题组系列文章）|
| 27°C | 22 | 全部 ITC |
| 30°C | 11 | ITC 为主 |
| 37°C | 11 | 全部 Fluorescence |
| 45°C | 10 | ITC + Fluorescence |
| 55°C | 10 | ITC + Fluorescence |

### van't Hoff 可行性分析
公式：`ln(Ka₂₅/Ka_T) = -ΔH°/R × (1/T₂₅ - 1/T_meas)`

**问题：ΔH 全缺失，无法逐分子校正。**

CB7 的 ΔH°文献值因 guest 而异：
- 铵/胺类客体：ΔH ≈ -20 to -50 kJ/mol（放热主导）
- 芳香类客体：ΔH ≈ 0 to -30 kJ/mol（熵/焓混合）

以 ΔH = -30 kJ/mol（CB7 典型中间值）估算温度偏差影响：

| T_meas | ΔlogKa（校正量） |
|--------|----------------|
| 23°C | +0.02 | ← 可忽略，< 实验精度 |
| 27°C | -0.02 | ← 可忽略 |
| 30°C | -0.05 | ← 可忽略 |
| 37°C | -0.12 | ← 边界 |
| 45°C | -0.24 | ← 有意义，值得校正 |
| 55°C | -0.42 | ← 显著，必须校正 |
| 5°C  | +0.20 | ← 有意义 |

**结论：**
- |T - 25| ≤ 5°C（包含 23°C 的 26 条、27°C 的 22 条）：直接接受，不校正（误差 < 0.05，在 ITC 测量精度 ~0.1 以内）
- |T - 25| > 5°C（45°C/55°C 各10条，5°C 的8条）：用 CB7 文献 ΔH 默认值做近似校正，标记 `t_correction=vanthoff_default`
- 所有 van't Hoff 校正均标记，后续可以选择只用 25°C 数据做严格分析

---

## 4. 溶剂过滤

**CB7 几乎没有有机溶剂问题：**
- 有机：2 条（DMSO-d6 × 1，methanol × 1）→ 直接丢弃
- D₂O：22 条 → 保留（H₂O 与 D₂O 中 CB7 结合常数差异通常 < 0.1 log units，文献 Rekharsky & Inoue 1998 等有记录）
- water/buffer/complex：606 条 → 全部保留

---

## 5. pH / 质子化态

### pH 分布（335 条有 pH 数据）
```
pH 7.0：  140 条 (42%)  ← 主流
pH 6.0：   44 条 (13%)
pH 4.5：   39 条 (12%)  ← 荧光竞争法常用缓冲液（acetate buffer）
pH 4.3：   部分（同上）
pH 1–3：  ~43 条        ← 研究质子化态系列
pH 10–14： ~17 条        ← 研究去质子化态系列
```

### pH 缺失的 295 条怎么办

检查缺失 pH 的数据特征：
- 几乎全是 T≠25°C 的温度依赖性实验，**或**来自早期文献
- 这 295 条技术以 ITC（直接法）为主，在中性水溶液中测量
- 绝大多数文章描述条件为"water"或"50 mM phosphate buffer"，pH 6-7

**推荐：**
- 不假设具体 pH 值，但将 pH-missing 的条目单独分组 `ph_bin="unknown"`
- 在最终分析时可以选择：
  a. 只用有 pH 数据的条目（保守）
  b. 把 pH-unknown 当作近似中性（pH~7）做一次补充分析（需标注假设）

### pH 与 Ka 关系
对有 pKa 的 guest（胺类、氨基酸、碱性药物），pH 严重影响 Ka：
- 在 pKa 附近每 1 个 pH 单位，质子化比例变10倍，对应 Ka 变化 ~1 log unit
- **这些不同 pH 下的测量值必须分开保存，不能合并平均**

---

## 6. 缺失数据

**CB7 数据集的缺失模式简单：**
- 有 technique → 必有 T（2:1对应，因为这两个字段来自同一信息源）
- 14 条 T/technique 全缺失 → 这14条来自较早的报告，只报告了 Ka

这14条：
- Ka 有效、溶剂有记录、pH 全缺失
- **处理**：保留，T 标记 `t_assumed=True`（25°C），pH 标记 `ph_known=False`

---

## 7. Outlier 检测

### 明确的数据错误
```
TMPyP4 + CB7：
  doi=10.1039/D3SC03865A：logKa=4.91（合理）
  doi=10.1039/D3SC03865A：logKa=19.65（不合理）← 同一篇文章，录入错误
```
logKa=19.65 对应 Ka = 4.5×10¹⁹ M⁻¹，这超过了物理上任何已知非共价结合的上限（CB7最强已知结合 ~10¹⁷ M⁻¹，且是在极低离子强度下）。**直接删除。**

### 统计方法的局限
CB7 数据里：
- 有 ≥ 4 条测量值的 molecule：只有约 30-40 个
- 大部分重复只有 2-3 条，IQR 方法无效

**推荐用修正 Z-score（median-based）代替 IQR：**
```
modified_z = 0.6745 × (x - median) / MAD
|modified_z| > 3.5 → 标记为 outlier
```
MAD（Median Absolute Deviation）比 IQR 对小样本更鲁棒。

但要注意：对于只有 2 条测量值的 pair（最多），如果 2 条数值相差很大，**无法统计判断哪条是对的**，需要人工或文献核查。

### 极端值审查（logKa > 14）
这些不全是错误——CB7 对金刚烷胺类的 Ka 确实可达 10¹⁴ 量级（有高质量 ITC 文献支持）：
```
logKa=15.52  bis(Trimethylamminomethyl)ferrocene  ITC直接法  3次重复一致 ✓
logKa=15.28  Diamantane diammonium-6              NMR竞争法
logKa=14.89  coumarin 衍生物                      ITC直接法  doi验证了 ✓
logKa=14.30  bicyclo[2.2.2]octane-1,4-diyl        ITC直接法  ✓
```
**做法：** 极端值不自动删除，而是检查：
1. 是否有多次重复一致？（bis-TMAFc 3次均~15.5 ✓）
2. 用的是直接法(ITC)还是竞争法（竞争法误差更大）？
3. 与其他类似结构的分子的 logKa 是否相符？

---

## 清理决策总结（CB7 专项）

| 步骤 | 具体做法 | 依据 |
|------|---------|------|
| 剔除 | 2条有机溶剂（DMSO/MeOH）| 溶剂不可比 |
| 剔除 | TMPyP4 logKa=19.65 | 同DOI录入错误，超物理上限 |
| 保留 | D₂O 22条 | H₂O/D₂O 差异 < 0.1 log |
| 保留 | logKa 14–16的极端值 | ITC直接法多次一致，有文献支持 |
| 温度 | \|T-25\| ≤ 5°C：不校正 | 误差 < ITC 精度 |
| 温度 | \|T-25\| > 5°C：van't Hoff（ΔH=-30 kJ/mol默认值）| CB7中间典型值 |
| 温度 | 14条缺失：标记t_assumed=True | |
| pH | 按 0.5 pH 单位分桶分组 | 不同质子化态是不同guest |
| pH | 295条缺失：ph_bin="unknown"，不与有pH数据合并 | |
| 平均 | 方法内（同technique）重复：几何平均 | |
| 平均 | 方法间（ITC vs 荧光）spread > 1 log：标记，不合并 | 系统误差，非随机误差 |
| Outlier | 修正Z-score（MAD-based），阈值3.5，仅n≥3 | 比IQR对小样本更鲁棒 |
| 标记列 | logKa_flag（suspect/ok）、t_correction、ph_known、tech_spread | 便于后续筛选 |

---

## 预期最终数据集规模

- 原始：630 条
- 删除有机溶剂：-2
- 删除确认录入错误（TMPyP4 19.65）：-1
- 按(molecule, pH_bin, technique_group)分组后去重：~350–400 unique pairs（pH分层后有些分子会拆分成多行）
