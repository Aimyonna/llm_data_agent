# Olist 大盘异动归因分析

一套对 Olist 电商月度 GMV 环比异动进行**定位 → 分解 → 解释**的分析系统。Python 负责可验证、可复算的量化计算并落盘结构化结果；分析判断由人工（Claude 作为分析师）基于这些结果给出。LLM 只做定性假设生成，不做统计替代。

## Language

### 分析阶段

**异动定位 (Anomaly Localization)**:
在月度时间序列上找出环比变化最显著的月份。只回答"哪个月出了问题"，不回答为什么。
_Avoid_: 异动归因（定位 ≠ 归因）

**归因 (Attribution)**:
对已定位的异动给出解释，分三层，严格区分强度（见下）。默认指第一层。
_Avoid_: 把定性聚类直接称为"归因"

### 归因的三层

**因素分解 (Factor Decomposition)**:
将 ΔGMV 按会计恒等式拆成价格、销量、结构等因子的贡献，数学上穷尽、可验证、可复算。**本项目的量化骨架，由 Python 计算。** 回答"跌掉的钱来自哪一块"。

**因果根因 (Causal Root-Cause)**:
回答"什么事件/动作**导致**了下跌"，需要反事实或对照组。Olist 公开数据不支撑强因果主张，**本项目显式不做**（见 ADR-0001），避免过度主张。

**定性假设 (Qualitative Hypothesis)**:
从差评文本与量化结果归纳出的业务解释，作为待验证的假设而非结论。回答"那块为什么可能差"。强度低于分解，必须标注为假设。由 LLM 生成（见 ADR-0002）。

### 量化分析单元

**GMV**:
成交总额 = Σ `order_items.price`，**不含运费 `freight_value`**。所有贡献分析与 PVM 的被分解对象。
_Avoid_: 把 freight 计入 GMV

**基期 (Baseline)**:
异动月的上一个完整月。②PVM 与 ③④⑤ 贡献分析统一以基期为对照（环比口径）。
_Avoid_: 同比、全期均值作为基期

**PVM 三因子**:
品类级加法桥，精确对账 Σ 三因子 = ΔGMV：
- **销量效应** = Δqty × 基期价
- **价格效应** = Δprice × 基期量
- **结构效应** = ΔGMV − 销量效应 − 价格效应（即 Δqty×Δprice 交互残差）
_Avoid_: 把结构效应等同于"品类结构迁移"——后者由品类贡献表(③)单独呈现

**品类贡献 (Category Contribution)**:
各品类 ΔGMV（异动月 vs 基期）的排序，定位哪些品类拖动大盘。

**地区贡献 (Region Contribution)**:
按买家所在州 `customer_state` 聚合的 ΔGMV，需求侧视角。
_Avoid_: seller_state（供给侧，备用，非默认）

**Seller 贡献 (Seller Contribution)**:
按 `seller_id` 聚合的 ΔGMV，Top-N + 长尾聚合。

**评论主题抽取 (Review Theme Extraction)**:
对异动月低分评论做评分分布 + 葡语 TF-IDF 高频主题词，纯 Python 可复算。**是主题词抽取，非真聚类**；为 LLM 假设提供结构化输入。
_Avoid_: 称之为"评论聚类"导致与统计聚类混淆

### 角色分工

**量化引擎 (Quantitative Engine)**:
`auto_analyzer.py` 中由 Python 确定性计算的部分——清洗、定位、PVM、各维度贡献、评论主题。产出结构化中间数据，可审计、可复算。

**分析师 (Analyst)**:
基于量化引擎的产出做判断与解释的角色（本会话中为 Claude）。定性假设由 LLM 辅助生成、分析师解读，但不由脚本自动断言为根因。
