# 基于 LLM 终端智能体（Agent）的电商全链路业务环比异动与负面评论全自动诊断系统

## 1. 项目背景与业务痛点 (Business Background)
在跨境电商（以巴西最大零售平台 Olist 为例）的日常经营中，面对 **11.4万笔全链路星型数仓数据** 与数十万条非结构化多语种用户评论，传统数据分析链路存在严重断层：
- 传统大盘 BI 仅能通过 Pandas/SQL 矩阵聚合下钻到“结果指标”暴跌（例如大盘 GMV 突发月环比暴跌）。
- 无法秒级渗透入底层非结构化文本的因果链（到底用户在骂什么？是物流瘫痪、商家缺件还是商品破损？）。
- 跨语言（葡萄牙语）的情感归因极度依赖人工翻译和文本聚类，策略产出耗时通常长达 **3天以上**。

为了实现真正的智能化降维提效，本项目基于 **Claude Code CLI 终端智能体环境** 托管本地雪花数仓，设计并打通了一条**“原始脏数据清洗 $\rightarrow$ 时序异动探针 $\rightarrow$ LLM 溢出鲁棒性防御 $\rightarrow$ 高管级商业策略自动交付”** 的端到端真·全自动 BI 数据管道（Pipeline）。

---

## 2. 企业级架构与数仓目录规范 (Architecture)
项目严格遵循工业级生产环境代码规范，通过前置数据阻击线隔离隐私密钥与海量高维 CSV 大文件：

```text
llm_data_agent/
│
├── data/                      # 原始数仓大文件区（已加入 .gitignore，引导用户自去 Kaggle 下载）
│   ├── olist_orders_dataset.csv
│   └── olist_order_reviews_dataset.csv
│
├── output/                    # 全自动流水线交付产物目录
│   ├── analysis_intermediate.json      # Pipeline 阶段性下钻量化指标缓存
│   └── 大盘业务环比异动归因诊断报告.md # 最终大模型混合模态生成的 CEO 级策略报告
│
├── auto_analyzer.py           # 核心 Pipeline：承载 Pandas 清洗、大模型调用及防御算子
└── .gitignore                 # 安全铁闸：屏蔽 CSV/ZIP/Token，防止生产泄漏与 Git 大文件崩溃
```
---

## 3. 工业级数据清洗与底池守卫 (Data Pipeline Defense)
脏数据是算法模型的死敌，在 Pipeline 运行前，脚本强制执行了预处理黄金预检防线：

DtypeWarning 强行拦截：在 pd.read_csv 阶段，针对多表联结时由于空值频繁爆发的 Pandas 混合类型冲突地雷，强制指定 keep_default_na=False 与 low_memory=False 规避类型混淆。

底池真空过滤：前置剔除 order_id 或 product_id 为空的残缺异常行，通过 how='left' 联结多表，守住流失率与大盘变动的分母底池，防止数据膨胀与指标失真。

文本脱水去噪：针对捞出的低分负面评论，通过正则过滤大段纯符号或留空文本，为后续喂给 LLM 进行高效 Token 脱水。

---

