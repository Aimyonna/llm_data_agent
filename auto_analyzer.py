#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  Olist 电商大盘 GMV 环比异动归因分析 —— 量化为主、LLM 为辅
  auto_analyzer.py
===============================================================================
  链路（详见 CONTEXT.md / docs/adr/）:
    Step 1  脏数据清洗与底池锁定             (Python)
    Step 2  ① 异动定位                      (Python)
    Step 3  ② PVM 因素分解                  (Python, 可对账)
    Step 4  ③ 品类贡献分析                  (Python)
    Step 5  ④ 地区贡献分析                  (Python)
    Step 6  ⑤ Seller 贡献分析               (Python)
    Step 7  ⑥ 评论主题抽取                  (Python, 可复算)
    Step 8  ⑦ LLM 定性假设生成              (LLM, 标注为假设)
    Step 9  ⑧ 自动商业报告落盘              (Python)

  原则: Python 承担所有可验证、可复算的量化计算; LLM 只把量化结果翻译成
        业务语言假设, 不做统计替代, 不单独断言根因。本项目不做因果推断
        (见 ADR-0001)。

  作者 : Data Warehouse Expert Agent
===============================================================================
"""

import os
import sys
import json
import re
import textwrap
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_PATH = OUTPUT_DIR / "大盘业务环比异动归因诊断报告.md"
INTERMEDIATE_JSON = OUTPUT_DIR / "analysis_intermediate.json"

ORDERS_CSV = DATA_DIR / "olist_orders_dataset.csv"
ITEMS_CSV = DATA_DIR / "olist_order_items_dataset.csv"
REVIEWS_CSV = DATA_DIR / "olist_order_reviews_dataset.csv"
CUSTOMERS_CSV = DATA_DIR / "olist_customers_dataset.csv"
PRODUCTS_CSV = DATA_DIR / "olist_products_dataset.csv"
CATEGORY_CSV = DATA_DIR / "product_category_name_translation.csv"

# --- 常量 ---
NEGATIVE_SCORE_THRESHOLD = 2        # review_score <= 2 视为恶评
SAMPLE_SIZE_FOR_LLM = 20            # (保留) 送入大模型的恶评抽样条数
MOM_CHANGE_THRESHOLD_PCT = -5.0     # 环比跌幅阈值(百分点), 辅助高亮
TOP_N_CONTRIBUTION = 10             # 贡献分析 Top-N
TOP_N_SELLERS = 20                  # Seller 贡献 Top-N
TOP_N_REVIEW_TERMS = 20             # 评论主题词 Top-N

# 葡语停用词表 (主题词抽取用, 手维护, 可审计)
PT_STOPWORDS = {
    "a", "o", "e", "é", "de", "do", "da", "dos", "das", "no", "na", "nos",
    "nas", "em", "um", "uma", "uns", "umas", "que", "com", "para", "por",
    "não", "mas", "se", "as", "os", "ao", "à", "às", "ou", "só", "já",
    "meu", "minha", "meus", "minhas", "me", "muito", "muita", "min", "mais",
    "menos", "como", "isso", "isto", "este", "essa", "esse", "este", "aquela",
    "the", "and", "to", "it", "is", "i", "of", "my", "in", "was", "que",
    "para", "com", "por", "uma", "recebi", "produto", "não", "ainda",
    "comprei", "antes", "depois", "q", "pra", "tá", "ta", "to", "mto",
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def print_stage(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}\n")


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """安全读取 CSV, 强制 keep_default_na=False 以避免混合类型推断警告。"""
    print(f"  📂 读取 {path.name} ...")
    defaults = {
        "keep_default_na": False,
        "dtype": str,
        "encoding": "utf-8",
    }
    defaults.update(kwargs)
    df = pd.read_csv(str(path), **defaults)
    print(f"     ✓ 读取完成: {len(df):,} 行 × {len(df.columns)} 列")
    return df


def clean_column_name(col: str) -> str:
    return col.strip()


def sanitize_for_json(obj: Any) -> Any:
    """递归将 NaN / Inf 替换为 None, 产出合法 JSON (RFC 8259)。"""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    if obj is pd.NaT:
        return None
    return obj


def add_year_month(df: pd.DataFrame, ts_col: str = "order_purchase_timestamp") -> pd.Series:
    """从时间戳列派生 year_month 字符串 (YYYY-MM)。"""
    dt = pd.to_datetime(df[ts_col], errors="coerce")
    return dt.dt.to_period("M").astype(str)


def fmt_money(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"R$ {float(x):,.2f}"


def fmt_pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):+.2f}%"


# ============================================================================
#  Step 1: 脏数据清洗与底池锁定
# ============================================================================

def step1_load_and_clean() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取六张 CSV, 清洗, 构建:
      - df_items: item 级底池 (orders ⋈ items ⋈ products ⋈ customers),
                  用于 GMV/PVM/品类/地区/Seller 贡献分析。
      - df_reviews: review 级表 (reviews ⋈ orders), 用于评论主题抽取。
    reviews 不并入 item 底池, 以免一个订单的多件商品把同一条评价重复计数。
    """
    print_stage("Step 1  脏数据清洗与底池锁定")

    df_orders = safe_read_csv(ORDERS_CSV)
    df_items = safe_read_csv(ITEMS_CSV)
    df_reviews = safe_read_csv(REVIEWS_CSV)
    df_customers = safe_read_csv(CUSTOMERS_CSV)
    df_products = safe_read_csv(PRODUCTS_CSV)
    df_cat = safe_read_csv(CATEGORY_CSV)

    for df_ in (df_orders, df_items, df_reviews, df_customers, df_products, df_cat):
        df_.columns = [clean_column_name(c) for c in df_.columns]

    # 清洗: 剔除主键残缺行
    df_orders = df_orders[df_orders["order_id"].str.strip() != ""].copy()
    df_items = df_items[
        (df_items["order_id"].str.strip() != "") &
        (df_items["product_id"].str.strip() != "")
    ].copy()
    df_reviews = df_reviews[df_reviews["order_id"].str.strip() != ""].copy()
    df_products = df_products[df_products["product_id"].str.strip() != ""].copy()

    # 数值列
    df_items["price_num"] = pd.to_numeric(df_items["price"], errors="coerce").fillna(0.0)
    df_items["freight_num"] = pd.to_numeric(df_items["freight_value"], errors="coerce").fillna(0.0)
    df_reviews["review_score_num"] = pd.to_numeric(df_reviews["review_score"], errors="coerce")

    # 类目英文名 (缺失则保留葡语原名)
    df_products = df_products.merge(
        df_cat, on="product_category_name", how="left"
    )
    df_products["category"] = df_products["product_category_name_english"].fillna(
        df_products["product_category_name"]
    ).replace("", np.nan).fillna(df_products["product_category_name"])

    # item 级底池: orders ⋈ items ⋈ products ⋈ customers
    print("\n  🔗 构建 item 级底池 (orders ⋈ items ⋈ products ⋈ customers) ...")
    keep_orders = ["order_id", "customer_id", "order_purchase_timestamp"]
    keep_items = ["order_id", "product_id", "seller_id", "price_num"]
    keep_products = ["product_id", "category"]
    keep_customers = ["customer_id", "customer_state"]

    df_items_base = (
        df_orders[keep_orders]
        .merge(df_items[keep_items], on="order_id", how="inner")
        .merge(df_products[keep_products], on="product_id", how="left")
        .merge(df_customers[keep_customers], on="customer_id", how="left")
    )
    df_items_base["category"] = (
        df_items_base["category"].fillna("unknown").replace("", "unknown")
    )
    df_items_base["customer_state"] = df_items_base["customer_state"].fillna("??")
    df_items_base["year_month"] = add_year_month(df_items_base)
    df_items_base = df_items_base[df_items_base["year_month"] != "NaT"].copy()

    print(f"     ✓ item 级底池: {len(df_items_base):,} 行")

    # review 级表: reviews ⋈ orders (取时间戳)
    df_reviews_m = df_reviews.merge(
        df_orders[["order_id", "order_purchase_timestamp"]], on="order_id", how="left"
    )
    df_reviews_m["year_month"] = add_year_month(df_reviews_m)
    df_reviews_m = df_reviews_m[df_reviews_m["year_month"] != "NaT"].copy()
    print(f"     ✓ review 级表: {len(df_reviews_m):,} 行")

    return df_items_base, df_reviews_m


# ============================================================================
#  Step 2: ① 异动定位
# ============================================================================

def step2_anomaly_detection(df_items: pd.DataFrame) -> Dict[str, Any]:
    """
    按月聚合 GMV (Σ price), 计算环比, 排除碎片月, 定位跌幅最深的异动月。
    返回异动月及其基期(上月)。
    """
    print_stage("Step 2  ① 异动定位")

    monthly = (
        df_items.groupby("year_month", observed=False)["price_num"]
        .sum()
        .sort_index()
        .reset_index()
    )
    monthly.columns = ["year_month", "gmv"]
    monthly["prev_gmv"] = monthly["gmv"].shift(1)
    monthly["mom_abs"] = monthly["gmv"] - monthly["prev_gmv"]
    monthly["mom_pct"] = (
        (monthly["gmv"] - monthly["prev_gmv"])
        / monthly["prev_gmv"].replace(0, np.nan) * 100
    )

    # 碎片月防御: 剔除 GMV 低于中位数 5% 的月份
    median_gmv = monthly["gmv"].median()
    sales_floor = median_gmv * 0.05
    monthly["valid"] = monthly["gmv"] >= sales_floor
    n_excluded = int((~monthly["valid"]).sum())
    if n_excluded:
        excl = monthly.loc[~monthly["valid"], "year_month"].tolist()
        print(f"  ⚠ 排除 {n_excluded} 个碎片月: {excl}")

    mom_valid = monthly[monthly["valid"]].dropna(subset=["mom_pct"]).copy()
    if len(mom_valid) == 0:
        raise RuntimeError("排除碎片月后无可用的环比数据。")

    worst_row = mom_valid.loc[mom_valid["mom_pct"].idxmin()]
    worst_month = worst_row["year_month"]
    prev_month = worst_row["prev_gmv"]  # 仅为占位, 真正 prev_month 取 year_month

    # 取基期 year_month: 异动月在排序表中的上一行
    worst_idx = monthly.index[monthly["year_month"] == worst_month][0]
    prev_month = monthly.loc[worst_idx - 1, "year_month"] if worst_idx > 0 else None

    print(f"  🔴 异动月: {worst_month}  环比 {worst_row['mom_pct']:+.2f}%  "
          f"(GMV {fmt_money(worst_row['gmv'])} ← {fmt_money(worst_row['prev_gmv'])})")
    print(f"  ◽ 基期:   {prev_month}")

    # 月度总览
    print(f"\n  📋 月度 GMV 总览:")
    print(f"     {'月份':<10} {'GMV(R$)':>16} {'环比':>10} {'有效':>4}")
    print(f"     {'-'*46}")
    for _, r in monthly.iterrows():
        pct = r["mom_pct"]
        pct_s = "  (首月)" if pd.isna(pct) else f"{pct:+.2f}%"
        print(f"     {r['year_month']:<10} {r['gmv']:>16,.2f} {pct_s:>10} "
              f"{'✓' if r['valid'] else '✗':>4}")

    return {
        "monthly": monthly.to_dict(orient="records"),
        "worst_month": worst_month,
        "prev_month": prev_month,
        "worst_pct": round(float(worst_row["mom_pct"]), 2),
        "worst_abs": round(float(worst_row["mom_abs"]), 2),
        "worst_gmv": round(float(worst_row["gmv"]), 2),
        "prev_gmv": round(float(worst_row["prev_gmv"]), 2),
    }


# ============================================================================
#  Step 3: ② PVM 因素分解
# ============================================================================

def _monthly_category_metrics(df_items: pd.DataFrame, month: str) -> pd.DataFrame:
    """单月品类级: qty / gmv / avg_price。"""
    m = df_items[df_items["year_month"] == month]
    g = (
        m.groupby("category", observed=False)
        .agg(qty=("price_num", "size"), gmv=("price_num", "sum"))
        .reset_index()
    )
    g["avg_price"] = np.where(g["qty"] > 0, g["gmv"] / g["qty"], 0.0)
    return g


def step3_pvm_decomposition(df_items: pd.DataFrame, worst: str, prev: str) -> Dict[str, Any]:
    """
    品类级 PVM 加法桥 (异动月 vs 基期):
      销量效应 = Δqty × 基期价
      价格效应 = Δprice × 基期量
      结构效应 = ΔGMV − 销量效应 − 价格效应   (= Δqty × Δprice, 交互残差)
    Σ 三因子 = ΔGMV, 可对账。
    """
    print_stage("Step 3  ② PVM 因素分解")

    g_prev = _monthly_category_metrics(df_items, prev).rename(
        columns={"qty": "qty_prev", "gmv": "gmv_prev", "avg_price": "price_prev"}
    )
    g_worst = _monthly_category_metrics(df_items, worst).rename(
        columns={"qty": "qty_w", "gmv": "gmv_w", "avg_price": "price_w"}
    )
    g = g_prev.merge(g_worst, on="category", how="outer").fillna(0.0)

    g["d_qty"] = g["qty_w"] - g["qty_prev"]
    g["d_price"] = g["price_w"] - g["price_prev"]
    g["d_gmv"] = g["gmv_w"] - g["gmv_prev"]

    g["volume_effect"] = g["d_qty"] * g["price_prev"]
    g["price_effect"] = g["d_price"] * g["qty_prev"]
    g["mix_effect"] = g["d_gmv"] - g["volume_effect"] - g["price_effect"]

    # 对账
    tot_delta = float(g["d_gmv"].sum())
    tot_vol = float(g["volume_effect"].sum())
    tot_prc = float(g["price_effect"].sum())
    tot_mix = float(g["mix_effect"].sum())
    reconcile = tot_vol + tot_prc + tot_mix
    diff = reconcile - tot_delta

    print(f"  基期 GMV: {fmt_money(g['gmv_prev'].sum())}  →  "
          f"异动月 GMV: {fmt_money(g['gmv_w'].sum())}  ΔGMV: {fmt_money(tot_delta)}")
    print(f"  ├─ 销量效应: {fmt_money(tot_vol)}")
    print(f"  ├─ 价格效应: {fmt_money(tot_prc)}")
    print(f"  ├─ 结构效应: {fmt_money(tot_mix)}")
    print(f"  └─ 对账: Σ三因子 {fmt_money(reconcile)} vs ΔGMV {fmt_money(tot_delta)} "
          f"(误差 {diff:+.4f})")

    per_cat = g.sort_values("d_gmv").copy()
    per_cat_records = []
    for _, r in per_cat.iterrows():
        per_cat_records.append({
            "category": r["category"],
            "gmv_prev": round(float(r["gmv_prev"]), 2),
            "gmv_worst": round(float(r["gmv_w"]), 2),
            "d_gmv": round(float(r["d_gmv"]), 2),
            "volume_effect": round(float(r["volume_effect"]), 2),
            "price_effect": round(float(r["price_effect"]), 2),
            "mix_effect": round(float(r["mix_effect"]), 2),
        })

    return {
        "prev_gmv": round(float(g["gmv_prev"].sum()), 2),
        "worst_gmv": round(float(g["gmv_w"].sum()), 2),
        "delta_gmv": round(tot_delta, 2),
        "volume_effect": round(tot_vol, 2),
        "price_effect": round(tot_prc, 2),
        "mix_effect": round(tot_mix, 2),
        "reconcile_diff": round(float(diff), 4),
        "per_category": per_cat_records,
    }


# ============================================================================
#  Step 4: ③ 品类贡献分析
# ============================================================================

def step4_category_contribution(pvm_result: Dict[str, Any]) -> Dict[str, Any]:
    """品类 ΔGMV 排序: 跌得最多 + 涨得最多。"""
    print_stage("Step 4  ③ 品类贡献分析")

    per = sorted(pvm_result["per_category"], key=lambda x: x["d_gmv"])
    losers = per[:TOP_N_CONTRIBUTION]
    gainers = list(reversed([c for c in per if c["d_gmv"] > 0]))[:TOP_N_CONTRIBUTION]

    print(f"  📉 跌幅 Top {TOP_N_CONTRIBUTION} 品类:")
    for c in losers:
        print(f"     - {c['category']:<30} ΔGMV {fmt_money(c['d_gmv'])}")
    print(f"  📈 涨幅 Top 品类:")
    for c in gainers[:5]:
        print(f"     + {c['category']:<30} ΔGMV {fmt_money(c['d_gmv'])}")

    return {"losers": losers, "gainers": gainers, "all": per}


# ============================================================================
#  Step 5: ④ 地区贡献分析
# ============================================================================

def _dim_contribution(df_items: pd.DataFrame, dim: str, worst: str, prev: str) -> pd.DataFrame:
    """按维度聚合两月 GMV, 算 Δ, 排序。"""
    g_prev = (
        df_items[df_items["year_month"] == prev]
        .groupby(dim, observed=False)["price_num"].sum()
        .rename("gmv_prev")
    )
    g_worst = (
        df_items[df_items["year_month"] == worst]
        .groupby(dim, observed=False)["price_num"].sum()
        .rename("gmv_w")
    )
    g = pd.concat([g_prev, g_worst], axis=1).fillna(0.0).reset_index()
    g["d_gmv"] = g["gmv_w"] - g["gmv_prev"]
    g["d_pct"] = np.where(g["gmv_prev"] > 0, g["d_gmv"] / g["gmv_prev"] * 100, np.nan)
    return g.sort_values("d_gmv")


def step5_region_contribution(df_items: pd.DataFrame, worst: str, prev: str) -> Dict[str, Any]:
    print_stage("Step 5  ④ 地区贡献分析 (customer_state)")

    g = _dim_contribution(df_items, "customer_state", worst, prev)
    losers = g.head(TOP_N_CONTRIBUTION)
    gainers = g[g["d_gmv"] > 0].sort_values("d_gmv", ascending=False).head(TOP_N_CONTRIBUTION)

    print(f"  📉 跌幅 Top {TOP_N_CONTRIBUTION} 州:")
    for _, r in losers.iterrows():
        print(f"     - {r['customer_state']:<4} ΔGMV {fmt_money(r['d_gmv'])} "
              f"({fmt_pct(r['d_pct'])})")

    return {
        "losers": _dim_records(losers, "customer_state"),
        "gainers": _dim_records(gainers, "customer_state"),
    }


def _dim_records(df: pd.DataFrame, dim: str) -> List[Dict[str, Any]]:
    out = []
    for _, r in df.iterrows():
        out.append({
            dim: r[dim],
            "gmv_prev": round(float(r["gmv_prev"]), 2),
            "gmv_worst": round(float(r["gmv_w"]), 2),
            "d_gmv": round(float(r["d_gmv"]), 2),
            "d_pct": None if pd.isna(r["d_pct"]) else round(float(r["d_pct"]), 2),
        })
    return out


# ============================================================================
#  Step 6: ⑤ Seller 贡献分析
# ============================================================================

def step6_seller_contribution(df_items: pd.DataFrame, worst: str, prev: str) -> Dict[str, Any]:
    print_stage("Step 6  ⑤ Seller 贡献分析")

    g = _dim_contribution(df_items, "seller_id", worst, prev)
    losers = g.head(TOP_N_SELLERS)
    gainers = g[g["d_gmv"] > 0].sort_values("d_gmv", ascending=False).head(TOP_N_CONTRIBUTION)

    # 长尾聚合 (排除已列入 losers 的)
    losers_ids = set(losers["seller_id"])
    tail = g[~g["seller_id"].isin(losers_ids)]
    tail_sum = float(tail["d_gmv"].sum())
    print(f"  📉 跌幅 Top {TOP_N_SELLERS} Seller (合计 ΔGMV {fmt_money(losers['d_gmv'].sum())})")
    for _, r in losers.iterrows():
        print(f"     - {r['seller_id']:<36} ΔGMV {fmt_money(r['d_gmv'])}")
    print(f"  ◽ 其余 {len(tail)} 个 seller 合计 ΔGMV {fmt_money(tail_sum)}")

    return {
        "losers": _dim_records(losers, "seller_id"),
        "gainers": _dim_records(gainers, "seller_id"),
        "long_tail_count": int(len(tail)),
        "long_tail_d_gmv": round(tail_sum, 2),
    }


# ============================================================================
#  Step 7: ⑥ 评论主题抽取 (纯 Python, 可复算)
# ============================================================================

def _tokenize_pt(text: str) -> List[str]:
    """葡语分词: 小写化, 取字母(含重音) token, 去停用词, 长度>=3。"""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"[a-zà-ÿ]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in PT_STOPWORDS]


def step7_review_themes(df_reviews: pd.DataFrame, worst: str, prev: str) -> Dict[str, Any]:
    """
    对异动月低分评论做: (a) 评分分布 vs 基期; (b) 葡语 TF 主题词 + 相对基期的 salience。
    无 LLM, 可复算。
    """
    print_stage("Step 7  ⑥ 评论主题抽取")

    df = df_reviews.copy()
    df["has_text"] = df["review_comment_message"].fillna("").str.strip().astype(bool)

    def neg_count(month: str) -> Dict[str, Any]:
        sub = df[(df["year_month"] == month) & (df["has_text"])]
        neg = sub[sub["review_score_num"] <= NEGATIVE_SCORE_THRESHOLD]
        return {
            "total_reviews": int(len(sub)),
            "neg_reviews": int(len(neg)),
            "neg_texts": int(neg["has_text"].sum()),
            "score_1": int((sub["review_score_num"] == 1).sum()),
            "score_2": int((sub["review_score_num"] == 2).sum()),
        }

    stat_w = neg_count(worst)
    stat_p = neg_count(prev)
    print(f"  异动月 {worst}: 低分评论 {stat_w['neg_reviews']} 条 (1★ {stat_w['score_1']}, "
          f"2★ {stat_w['score_2']})")
    print(f"  基期   {prev}: 低分评论 {stat_p['neg_reviews']} 条 (1★ {stat_p['score_1']}, "
          f"2★ {stat_p['score_2']})")

    # TF: 异动月低分评论词频
    neg_w = df[(df["year_month"] == worst) & (df["review_score_num"] <= NEGATIVE_SCORE_THRESHOLD)]
    neg_p = df[(df["year_month"] == prev) & (df["review_score_num"] <= NEGATIVE_SCORE_THRESHOLD)]

    def tf(corpus) -> Counter:
        c = Counter()
        for txt in corpus["review_comment_message"].fillna(""):
            c.update(_tokenize_pt(txt))
        return c

    tf_w = tf(neg_w)
    tf_p = tf(neg_p)
    total_w = sum(tf_w.values()) or 1
    total_p = sum(tf_p.values()) or 1

    # salience = 异动月词频份额 − 基期词频份额 (正值=该词在异动月更突出)
    salience = []
    for term, fw in tf_w.items():
        share_w = fw / total_w
        share_p = tf_p.get(term, 0) / total_p
        salience.append((term, fw, share_w - share_p))
    salience.sort(key=lambda x: x[1], reverse=True)

    top_by_freq = [{"term": t, "freq": f, "salience": round(s, 4)} for t, f, s in salience[:TOP_N_REVIEW_TERMS]]
    salience_sorted = sorted(salience, key=lambda x: x[2], reverse=True)
    top_by_salience = [{"term": t, "freq": f, "salience": round(s, 4)}
                       for t, f, s in salience_sorted[:15] if s > 0]

    print(f"  🔑 异动月高频主题词: {[t['term'] for t in top_by_freq[:10]]}")
    print(f"  ⬆ 相对基期上升词: {[t['term'] for t in top_by_salience[:10]]}")

    return {
        "worst_stat": stat_w,
        "prev_stat": stat_p,
        "top_terms_by_freq": top_by_freq,
        "top_terms_by_salience": top_by_salience,
    }


# ============================================================================
#  Step 8: ⑦ LLM 定性假设生成
# ============================================================================

def step8_llm_hypothesis(quant: Dict[str, Any]) -> Dict[str, Any]:
    """
    把量化结果(②③④⑤⑥)喂给 LLM, 产出业务语言假设 + 运营建议。
    明确标注为假设, 不做因果断言, 不发明数字。
    """
    print_stage("Step 8  ⑦ LLM 定性假设生成")

    evidence = {
        "anomaly": {
            "month": quant["anomaly"]["worst_month"],
            "baseline_month": quant["anomaly"]["prev_month"],
            "mom_pct": quant["anomaly"]["worst_pct"],
            "delta_gmv": quant["anomaly"]["worst_abs"],
        },
        "pvm": {
            "volume_effect": quant["pvm"]["volume_effect"],
            "price_effect": quant["pvm"]["price_effect"],
            "mix_effect": quant["pvm"]["mix_effect"],
            "delta_gmv": quant["pvm"]["delta_gmv"],
        },
        "top_loser_categories": [c["category"] for c in quant["category"]["losers"][:5]],
        "top_loser_regions": [(r["customer_state"], r["d_gmv"]) for r in quant["region"]["losers"][:5]],
        "top_loser_sellers_count": len(quant["seller"]["losers"]),
        "review_themes": [t["term"] for t in quant["review"]["top_terms_by_freq"][:15]],
    }

    system_prompt = textwrap.dedent("""
        你是资深电商数据分析师。你会收到一份 GMV 环比异动的【量化分析结果】(JSON)。
        你的任务: 把这些数字翻译成业务语言的【假设性解释】与运营建议。

        铁律:
        1. 只能用提供的数字, 禁止编造任何未给出的数字或比例。
        2. 这是观察性数据, 不得做因果断言。用"可能与/假设/疑似"等措辞, 禁用"导致/因为"。
        3. 每条假设必须指向具体的量化证据 (PVM 因子 / 品类 / 地区 / 主题词)。
        4. 明确区分: 哪些结论有量化支撑 (因素分解/贡献分析), 哪些只是定性假设。

        严格输出纯 JSON (无 markdown 标记):
        {
            "executive_hypothesis": "200字内总括: 结合PVM主因子与贡献, 给出本次异动最可能的业务解释(标注为假设)",
            "hypotheses": [
                {"id": 1, "hypothesis": "...", "evidence": "引用的具体量化证据", "confidence": "中/低"}
            ],
            "recommendations": [
                {"priority": "P0/P1/P2", "action": "...", "rationale": "..."}
            ],
            "evidence_boundary": "一句话声明: 本分析为因素分解+定性假设, 非因果结论"
        }
        所有中文输出用简体中文。
    """).strip()

    user_prompt = (
        "【量化分析结果】(异动月 vs 基期, 环比口径):\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
        + "\n\n请基于以上证据生成定性假设与建议, 输出纯 JSON。"
    )

    llm_result = call_llm(system_prompt, user_prompt)
    if llm_result is None:
        print("  ❌ LLM 不可用, 使用结构化降级摘要 (无定性假设)。")
        return fallback_hypothesis(evidence)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "llm_raw_response.txt").write_text(llm_result, encoding="utf-8")

    parsed = parse_llm_json(llm_result)
    if parsed is None:
        parsed = extract_json_from_response(llm_result)
    if parsed is None:
        print("  ⚠ LLM 响应解析失败, 使用结构化降级摘要。")
        return fallback_hypothesis(evidence)

    print("  ✅ LLM 假设生成完成。")
    return {"llm_success": True, "evidence": evidence, **parsed}


def fallback_hypothesis(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 不可用时的降级: 只回放量化事实, 不编造定性假设。"""
    pvm = evidence["pvm"]
    # 找主导因子
    effects = {"销量": pvm["volume_effect"], "价格": pvm["price_effect"], "结构": pvm["mix_effect"]}
    dominant = max(effects, key=lambda k: abs(effects[k]))
    return {
        "llm_success": False,
        "evidence": evidence,
        "executive_hypothesis": (
            f"(降级摘要, 无 LLM 定性假设) 异动月 {evidence['anomaly']['month']} 环比 "
            f"{evidence['anomaly']['mom_pct']}%, ΔGMV {evidence['anomaly']['delta_gmv']}。"
            f"PVM 分解显示主导因子为【{dominant}效应】({effects[dominant]})。"
            f"跌幅最大品类: {', '.join(evidence['top_loser_categories'][:3])}。"
            f"该结论为因素分解, 非因果结论。"
        ),
        "hypotheses": [],
        "recommendations": [],
        "evidence_boundary": "本分析为因素分解+贡献分析, 非因果结论; LLM 定性假设未生成。",
    }


# --- LLM 调用与解析 (保留原有实现) ---

def call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """调用 Anthropic-compatible API, 兼容 ThinkingBlock + TextBlock 混合返回。"""
    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        if not api_key:
            print("  ⚠ 未检测到 ANTHROPIC_AUTH_TOKEN 环境变量。")
            return None

        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        response = client.messages.create(
            model=model,
            max_tokens=16384,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        block_types = [type(b).__name__ for b in response.content]
        print(f"  🔍 content blocks: {len(response.content)} 个 ({block_types})")
        if hasattr(response, "usage"):
            u = response.usage
            print(f"  📊 token 用量: input={u.input_tokens}, output={u.output_tokens}")

        text_blocks = [
            block.text
            for block in response.content
            if hasattr(block, "text") and getattr(block, "text", "").strip()
        ]
        if not text_blocks:
            return str(response.content)
        full_text = "\n".join(text_blocks)
        print(f"  📝 输出文本长度: {len(full_text)} chars")
        return full_text

    except Exception as exc:
        print(f"  ❌ LLM 调用异常: {exc}")
        traceback.print_exc()
        return None


def parse_llm_json(raw: str) -> Optional[Dict]:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def extract_json_from_response(raw: str) -> Optional[Dict]:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
#  Step 9: ⑧ 自动商业报告落盘
# ============================================================================

def step9_generate_report(quant: Dict[str, Any], hyp: Dict[str, Any]) -> str:
    print_stage("Step 9  ⑧ 自动商业报告落盘")

    an = quant["anomaly"]
    pvm = quant["pvm"]
    cat = quant["category"]
    reg = quant["region"]
    sel = quant["seller"]
    rev = quant["review"]

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 月度表
    mom_rows = ""
    for r in an["monthly"]:
        pct = r.get("mom_pct")
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            pct_str = "— (首月)"
        else:
            flag = "🔴" if pct < MOM_CHANGE_THRESHOLD_PCT else ("🟢" if pct > 0 else "🟡")
            pct_str = f"{pct:+.2f}% {flag}"
        mom_rows += f"| {r['year_month']} | R$ {r['gmv']:,.2f} | {pct_str} |\n"

    def contrib_table(rows, dim_name, top):
        if not rows:
            return "> 暂无数据\n"
        header = f"| {dim_name} | ΔGMV | 基期 GMV | 异动月 GMV | 环比 |\n|---|---|---|---|---|\n"
        body = ""
        for r in rows[:top]:
            pct = r.get("d_pct")
            pct_s = "—" if pct is None else f"{pct:+.2f}%"
            body += (f"| {r[dim_name]} | R$ {r['d_gmv']:,.2f} | "
                     f"R$ {r['gmv_prev']:,.2f} | R$ {r['gmv_worst']:,.2f} | {pct_s} |\n")
        return header + body

    # PVM per-category table (top losers + gainers)
    pvm_cat_rows = ""
    show_cats = (cat["losers"][:5] + list(reversed(cat["gainers"]))[:5])
    for c in show_cats:
        pvm_cat_rows += (
            f"| {c['category']} | R$ {c['d_gmv']:,.2f} | "
            f"R$ {c['volume_effect']:,.2f} | R$ {c['price_effect']:,.2f} | "
            f"R$ {c['mix_effect']:,.2f} |\n"
        )

    # review terms
    term_rows = ""
    for t in rev["top_terms_by_freq"][:15]:
        term_rows += f"| {t['term']} | {t['freq']} | {t['salience']:+.4f} |\n"

    # hypotheses
    hyp_rows = ""
    for h in hyp.get("hypotheses", []):
        hyp_rows += (f"| {h.get('id','?')} | {h.get('hypothesis','')} | "
                     f"{h.get('evidence','')} | {h.get('confidence','?')} |\n")

    rec_rows = ""
    for r in hyp.get("recommendations", []):
        rec_rows += (f"| {r.get('priority','?')} | {r.get('action','')} | "
                     f"{r.get('rationale','')} |\n")

    llm_ok = hyp.get("llm_success", False)
    exec_hyp = hyp.get("executive_hypothesis", "")
    boundary = hyp.get("evidence_boundary", "")

    report = f"""# 📊 Olist 大盘 GMV 环比异动归因诊断报告

> **生成时间**: {now_str}
> **分析引擎**: auto_analyzer.py (量化为主, LLM 为辅)
> **口径**: GMV = Σ price (不含运费) | 基期 = 异动月上月 (环比)
> **边界声明**: 本报告含因素分解与贡献分析(可对账) + LLM 定性假设(标注为假设); **非因果结论**。

---

## 一、① 异动定位

| 月份 | GMV (BRL) | 环比 |
|------|-----------|------|
{mom_rows}

### 🔴 异动聚焦

| 指标 | 数值 |
|------|------|
| **异动月** | **{an['worst_month']}** |
| 基期(上月) | {an['prev_month']} |
| 异动月 GMV | R$ {an['worst_gmv']:,.2f} |
| 基期 GMV | R$ {an['prev_gmv']:,.2f} |
| ΔGMV | R$ {an['worst_abs']:,.2f} |
| **环比** | **{an['worst_pct']:+.2f}%** |

---

## 二、② PVM 因素分解 (可对账)

将 ΔGMV 拆为销量 / 价格 / 结构三因子 (品类级加法桥):

| 因子 | 贡献 (BRL) | 占 ΔGMV |
|------|-----------|---------|
| 销量效应 | R$ {pvm['volume_effect']:,.2f} | {pvm['volume_effect']/pvm['delta_gmv']*100 if pvm['delta_gmv'] else 0:+.1f}% |
| 价格效应 | R$ {pvm['price_effect']:,.2f} | {pvm['price_effect']/pvm['delta_gmv']*100 if pvm['delta_gmv'] else 0:+.1f}% |
| 结构效应 | R$ {pvm['mix_effect']:,.2f} | {pvm['mix_effect']/pvm['delta_gmv']*100 if pvm['delta_gmv'] else 0:+.1f}% |
| **ΔGMV** | **R$ {pvm['delta_gmv']:,.2f}** | 100% |

> 对账: Σ三因子 − ΔGMV = R$ {pvm['reconcile_diff']:+.4f} (应≈0)

**主导因子**: 销量效应 R$ {pvm['volume_effect']:,.2f} / 价格效应 R$ {pvm['price_effect']:,.2f} / 结构效应 R$ {pvm['mix_effect']:,.2f}

### 品类级 PVM (跌幅 Top5 + 涨幅 Top5)

| 品类 | ΔGMV | 销量效应 | 价格效应 | 结构效应 |
|------|------|----------|----------|----------|
{pvm_cat_rows if pvm_cat_rows else '| — | — | — | — | — |'}

---

## 三、③ 品类贡献分析

### 跌幅 Top {TOP_N_CONTRIBUTION} 品类
{contrib_table(cat['losers'], 'category', TOP_N_CONTRIBUTION)}
### 涨幅 Top 品类
{contrib_table(cat['gainers'], 'category', TOP_N_CONTRIBUTION)}

---

## 四、④ 地区贡献分析 (买家所在州)

### 跌幅 Top {TOP_N_CONTRIBUTION} 州
{contrib_table(reg['losers'], 'customer_state', TOP_N_CONTRIBUTION)}
### 涨幅 Top 州
{contrib_table(reg['gainers'], 'customer_state', TOP_N_CONTRIBUTION)}

---

## 五、⑤ Seller 贡献分析

### 跌幅 Top {TOP_N_SELLERS} Seller
{contrib_table(sel['losers'], 'seller_id', TOP_N_SELLERS)}
> 其余 {sel['long_tail_count']} 个 seller 合计 ΔGMV R$ {sel['long_tail_d_gmv']:,.2f}

### 涨幅 Top Seller
{contrib_table(sel['gainers'], 'seller_id', TOP_N_CONTRIBUTION)}

---

## 六、⑥ 评论主题抽取 (纯 Python, 可复算)

| 维度 | 异动月 {an['worst_month']} | 基期 {an['prev_month']} |
|------|------|------|
| 低分评论(≤{NEGATIVE_SCORE_THRESHOLD}★)总数 | {rev['worst_stat']['neg_reviews']} | {rev['prev_stat']['neg_reviews']} |
| 1★ | {rev['worst_stat']['score_1']} | {rev['prev_stat']['score_1']} |
| 2★ | {rev['worst_stat']['score_2']} | {rev['prev_stat']['score_2']} |

### 异动月高频主题词 (葡语, 已去停用词)

| 主题词 | 词频 | 相对基期 salience |
|--------|------|------------------|
{term_rows if term_rows else '| — | — | — |'}

---

## 七、⑦ LLM 定性假设 ({'✅ 已生成' if llm_ok else '⚠️ 降级'})

> **执行假设**: {exec_hyp if exec_hyp else '(未生成)'}

### 假设清单 (标注为假设, 非因果)

| # | 假设 | 量化证据 | 置信度 |
|---|------|----------|--------|
{hyp_rows if hyp_rows else '| — | 暂无 | — | — |'}

### 运营建议

| 优先级 | 行动项 | 依据 |
|--------|--------|------|
{rec_rows if rec_rows else '| — | 暂无 | — |'}

---

## 八、边界与方法论声明

- **量化层 (可对账)**: ①异动定位、②PVM 因素分解(Σ三因子=ΔGMV)、③④⑤各维度贡献分析、⑥评论主题词频。均由 Python 确定性计算, 可复算。
- **定性层 (假设)**: ⑦由 LLM 基于量化证据生成, 明确标注为假设, 不构成因果结论。
- **不做因果推断**: Olist 公开数据缺乏干预变量与反事实信息, 本项目不做因果主张 (见 ADR-0001)。
- {boundary}

---

> *本报告由 auto_analyzer.py 生成。量化在前、定性在后; 数字可对账, 假设有标注。*
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  ✅ 报告已落盘: {REPORT_PATH} ({REPORT_PATH.stat().st_size:,} bytes)")
    return report


# ============================================================================
#  主入口
# ============================================================================

def main() -> int:
    print("\n" + "█" * 78)
    print("█  Olist 大盘 GMV 环比异动归因分析 —— 量化为主, LLM 为辅")
    print("█  启动时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 78)

    try:
        df_items, df_reviews = step1_load_and_clean()
    except Exception as e:
        print(f"\n❌ Step 1 失败: {e}"); traceback.print_exc(); return 1

    try:
        anomaly = step2_anomaly_detection(df_items)
    except Exception as e:
        print(f"\n❌ Step 2 失败: {e}"); traceback.print_exc(); return 2

    worst = anomaly["worst_month"]
    prev = anomaly["prev_month"]

    try:
        pvm = step3_pvm_decomposition(df_items, worst, prev)
        category = step4_category_contribution(pvm)
        region = step5_region_contribution(df_items, worst, prev)
        seller = step6_seller_contribution(df_items, worst, prev)
        review = step7_review_themes(df_reviews, worst, prev)
    except Exception as e:
        print(f"\n❌ 量化分析失败: {e}"); traceback.print_exc(); return 3

    quant = {
        "anomaly": anomaly, "pvm": pvm, "category": category,
        "region": region, "seller": seller, "review": review,
    }

    try:
        hyp = step8_llm_hypothesis(quant)
    except Exception as e:
        print(f"\n⚠ Step 8 (LLM) 异常, 降级: {e}")
        hyp = fallback_hypothesis({
            "anomaly": anomaly, "pvm": pvm,
            "top_loser_categories": [c["category"] for c in category["losers"][:5]],
        })

    try:
        step9_generate_report(quant, hyp)
    except Exception as e:
        print(f"\n❌ Step 9 失败: {e}"); traceback.print_exc(); return 9

    # 中间数据
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        intermediate = {
            "generated_at": datetime.now().isoformat(),
            "quant": quant,
            "hypothesis": {k: v for k, v in hyp.items() if k != "evidence"},
        }
        INTERMEDIATE_JSON.write_text(
            json.dumps(sanitize_for_json(intermediate), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  📄 中间数据: {INTERMEDIATE_JSON}")
    except Exception as e:
        print(f"  ⚠ 中间数据保存失败 (非致命): {e}")

    print("\n" + "█" * 78)
    print(f"█  异动月: {worst}  环比: {anomaly['worst_pct']:+.2f}%")
    print(f"█  报告: {REPORT_PATH}")
    print("█" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
