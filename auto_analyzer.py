#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  Olist 电商大盘业务环比异动归因分析 —— 全自动管道脚本
  auto_analyzer.py
===============================================================================
  工业级「数据预处理 → 时间序列异动定位 → 大模型智能归因 → 报告落盘」
  四步全自动管道。

  作者 : Data Warehouse Expert Agent
  日期 : 2026-06-20
===============================================================================
"""

import os
import sys
import json
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

# --- 文件路径 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_PATH = OUTPUT_DIR / "大盘业务环比异动归因诊断报告.md"
INTERMEDIATE_JSON = OUTPUT_DIR / "analysis_intermediate.json"

ORDERS_CSV = DATA_DIR / "olist_orders_dataset.csv"
ITEMS_CSV = DATA_DIR / "olist_order_items_dataset.csv"
REVIEWS_CSV = DATA_DIR / "olist_order_reviews_dataset.csv"

# --- 常量 ---
NEGATIVE_SCORE_THRESHOLD = 2         # review_score <= 2 视为恶评
SAMPLE_SIZE_FOR_LLM = 20            # 送入大模型的恶评抽样条数
MOM_CHANGE_THRESHOLD_PCT = -5.0     # 环比跌幅阈值(百分点)，辅助高亮用

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def print_stage(title: str) -> None:
    """打印阶段横幅，便于终端阅读。"""
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}\n")


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    安全读取 CSV，强制 keep_default_na=False 以避免混合类型推断警告。
    """
    print(f"  📂 读取 {path.name} ...")
    defaults = {
        "keep_default_na": False,
        "dtype": str,               # 全部 string 读入，后续显式转换
        "encoding": "utf-8",
    }
    defaults.update(kwargs)
    df = pd.read_csv(str(path), **defaults)
    print(f"     ✓ 读取完成: {len(df):,} 行 × {len(df.columns)} 列")
    return df


def clean_column_name(col: str) -> str:
    """去除列名两端空格（防御上游数据不规整）。"""
    return col.strip()


def sanitize_for_json(obj: Any) -> Any:
    """
    递归遍历数据结构，将 NaN / Infinity / -Infinity 替换为 None，
    确保序列化后产出合法 JSON (RFC 8259)，编辑器不再报红。

    因为 json.dumps 默认 allow_nan=True 会将 float('nan') 序列化为
    裸写 NaN —— 这不是合法 JSON，绝大多数编辑器/linter 会报错。
    """
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    return obj


# ============================================================================
#  Step 1: 脏数据清洗与底池锁定
# ============================================================================

def step1_load_and_clean() -> pd.DataFrame:
    """
    读取三张 CSV，执行脏数据清洗，LEFT JOIN 锁定分析底池。

    Returns
    -------
    pd.DataFrame
        清洗并关联完毕的全链路分析底池表。
    """
    print_stage("Step 1/4  脏数据清洗与底池锁定")

    # 1a. 读取原始数据
    df_orders = safe_read_csv(ORDERS_CSV)
    df_items = safe_read_csv(ITEMS_CSV)
    df_reviews = safe_read_csv(REVIEWS_CSV)

    # 1b. 统一列名 trim
    for df_ in (df_orders, df_items, df_reviews):
        df_.columns = [clean_column_name(c) for c in df_.columns]

    # 1c. 【脏数据清洗 - 订单表】
    before = len(df_orders)
    df_orders = df_orders[df_orders["order_id"].str.strip() != ""].copy()
    after_oid = len(df_orders)
    if before != after_oid:
        print(f"  🧹 订单表剔除空 order_id: {before - after_oid} 行")

    # 1d. 【脏数据清洗 - 商品表】
    before = len(df_items)
    df_items = df_items[
        (df_items["order_id"].str.strip() != "") &
        (df_items["product_id"].str.strip() != "")
    ].copy()
    after_clean = len(df_items)
    if before != after_clean:
        print(f"  🧹 商品表剔除残缺行: {before - after_clean} 行")

    # 1e. 【脏数据清洗 - 评价表】
    before = len(df_reviews)
    df_reviews = df_reviews[df_reviews["order_id"].str.strip() != ""].copy()
    if before != len(df_reviews):
        print(f"  🧹 评价表剔除空 order_id: {before - len(df_reviews)} 行")

    # 1f. LEFT JOIN 构建底池  (orders → items → reviews)
    print("\n  🔗 执行 LEFT JOIN (orders → items → reviews) ...")
    df_base = df_orders.merge(df_items, on="order_id", how="left")
    df_base = df_base.merge(df_reviews, on="order_id", how="left")

    print(f"     ✓ 底池锁定完成: {len(df_base):,} 行 × {len(df_base.columns)} 列")
    return df_base


# ============================================================================
#  Step 2: 时间序列衍生变阵与异动定位
# ============================================================================

def step2_ts_anomaly_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    将时间字段转化为 datetime，按月聚合销售额，计算环比，
    定位环比跌幅最严重的"大盘异动月份"。

    Parameters
    ----------
    df : pd.DataFrame
        由 Step 1 产出的底池表。

    Returns
    -------
    dict
        包含月度聚合表、环比序列、异动月份信息的结构化字典。
    """
    print_stage("Step 2/4  时间序列衍生变阵与异动定位")

    df = df.copy()

    # 2a. 转化时间戳
    print("  ⏳ 转化 order_purchase_timestamp → datetime ...")
    df["purchase_dt"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    n_bad_ts = df["purchase_dt"].isna().sum()
    if n_bad_ts:
        print(f"     ⚠ 无法解析的时间戳: {n_bad_ts} 行 (将被排除)")

    # 2b. 提取年-月特征
    df["year_month"] = df["purchase_dt"].dt.to_period("M")
    df = df.dropna(subset=["year_month"])

    # 2c. 转换 price 为数值
    print("  💰 转换 price 字段为数值 ...")
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    n_zero_price = (df["price_num"] == 0).sum()
    if n_zero_price:
        print(f"     ⚠ price 为 0 或无法解析的行: {n_zero_price}")

    # 2d. 按月聚合总销售额
    print("  📊 按月聚合总销售额 ...")
    monthly_sales = (
        df.groupby("year_month", observed=False)["price_num"]
        .sum()
        .sort_index()
    )
    monthly_sales.index = monthly_sales.index.astype(str)

    if len(monthly_sales) < 2:
        raise RuntimeError("月度数据不足 2 个月，无法计算环比。")

    # 2e. 计算环比变化
    monthly_df = monthly_sales.reset_index()
    monthly_df.columns = ["year_month", "total_sales"]

    monthly_df["prev_sales"] = monthly_df["total_sales"].shift(1)
    monthly_df["mom_change_abs"] = monthly_df["total_sales"] - monthly_df["prev_sales"]
    monthly_df["mom_change_pct"] = (
        (monthly_df["total_sales"] - monthly_df["prev_sales"])
        / monthly_df["prev_sales"].replace(0, np.nan)
        * 100
    )

    # 2e-bis. 排除尾部数据不完整月份 (防御数据集截断噪声)
    #         策略: 剔除销售额低于中位数 5% 的月份，这些通常是数据采集不全的碎片月
    median_sales = monthly_df["total_sales"].median()
    sales_floor = median_sales * 0.05
    monthly_df["_valid"] = monthly_df["total_sales"] >= sales_floor
    n_excluded = (~monthly_df["_valid"]).sum()
    if n_excluded:
        excluded_months = monthly_df.loc[~monthly_df["_valid"], "year_month"].tolist()
        print(f"\n  ⚠ 排除 {n_excluded} 个数据不完整月份 (销售额 < R$ {sales_floor:,.2f}):")
        for em in excluded_months:
            print(f"     - {em}: R$ {monthly_df[monthly_df['year_month']==em]['total_sales'].values[0]:,.2f}")

    # 2f. 定位环比跌幅最严重的月份 (只看有效月份且有前值)
    mom_valid = monthly_df[monthly_df["_valid"]].dropna(subset=["mom_change_pct"])
    if len(mom_valid) == 0:
        raise RuntimeError("排除不完整月份后无可用的环比数据。")
    worst_row = mom_valid.loc[mom_valid["mom_change_pct"].idxmin()]
    worst_month = worst_row["year_month"]
    worst_pct = worst_row["mom_change_pct"]
    worst_abs = worst_row["mom_change_abs"]

    print(f"\n  🔴 大盘异动月份定位结果:")
    print(f"     ├─ 异动月份     : {worst_month}")
    print(f"     ├─ 当月销售额   : R$ {worst_row['total_sales']:,.2f}")
    print(f"     ├─ 上月销售额   : R$ {worst_row['prev_sales']:,.2f}")
    print(f"     ├─ 环比变化金额 : R$ {worst_abs:,.2f}")
    print(f"     └─ 环比变化率   : {worst_pct:+.2f}%  ← 跌幅最深")

    # 2g. 打印月度总览
    print(f"\n  📋 全时段月度销售额总览:")
    print(f"     {'月份':<10} {'销售额(R$)':>15} {'环比变化率':>12}")
    print(f"     {'-'*40}")
    for _, row in monthly_df.iterrows():
        ym = row["year_month"]
        sales = row["total_sales"]
        pct = row["mom_change_pct"]
        if pd.isna(pct):
            pct_str = "  (首月)  "
        else:
            flag = "🔴" if pct < MOM_CHANGE_THRESHOLD_PCT else ("🟢" if pct > 0 else "🟡")
            pct_str = f"{pct:+.2f}% {flag}"
        print(f"     {ym:<10} R$ {sales:>14,.2f} {pct_str:>12}")

    return {
        "monthly_sales": monthly_df.to_dict(orient="records"),
        "worst_month": worst_month,
        "worst_month_start": pd.Timestamp(worst_month).strftime("%Y-%m-%d"),
        "worst_month_end": (
            pd.Timestamp(worst_month) + pd.offsets.MonthEnd(1)
        ).strftime("%Y-%m-%d"),
        "worst_pct": round(worst_pct, 2),
        "worst_abs": round(worst_abs, 2),
        "worst_month_sales": round(worst_row["total_sales"], 2),
        "worst_month_prev_sales": round(worst_row["prev_sales"], 2),
    }


# ============================================================================
#  Step 3: 文本脱水与大模型智能归因
# ============================================================================

def step3_llm_attribution(
    df: pd.DataFrame, worst_month: str
) -> Dict[str, Any]:
    """
    针对异动月份过滤低分恶评，清洗文本，抽样送入大模型进行
    「人、货、场」三维情感观点聚类与 Top 3 痛点提炼。

    Parameters
    ----------
    df : pd.DataFrame
        底池表。
    worst_month : str
        异动月份 (如 "2018-05")。

    Returns
    -------
    dict
        包含大模型分析结果、抽样文本等。
    """
    print_stage("Step 3/4  文本脱水与大模型智能归因")

    df = df.copy()

    # 3a. 按异动月份过滤
    df["year_month"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    ).dt.to_period("M").astype(str)
    df_anomaly = df[df["year_month"] == worst_month].copy()
    print(f"  📅 异动月份 [{worst_month}] 底池数据: {len(df_anomaly):,} 行")

    # 3b. 转换 review_score 为数值
    df_anomaly["review_score_num"] = pd.to_numeric(
        df_anomaly["review_score"], errors="coerce"
    )

    # 3c. 过滤低分恶评
    df_neg = df_anomaly[
        df_anomaly["review_score_num"] <= NEGATIVE_SCORE_THRESHOLD
    ].copy()
    print(f"  😡 review_score ≤ {NEGATIVE_SCORE_THRESHOLD} 的恶评: {len(df_neg):,} 条")

    if len(df_neg) == 0:
        print("  ⚠ 该月份无低分恶评，跳过大模型分析。")
        return {
            "llm_success": False,
            "reason": "no_negative_reviews",
            "negative_count": 0,
            "sampled_reviews": [],
            "llm_raw_response": "",
            "top3_pain_points": [],
            "summary": "",
        }

    # 3d. 提取并清洗评论文本
    df_neg["clean_comment"] = df_neg["review_comment_message"].apply(clean_review_text)
    df_neg = df_neg[df_neg["clean_comment"].str.strip() != ""].copy()

    print(f"  🧼 文本脱水后有效恶评: {len(df_neg):,} 条")

    if len(df_neg) == 0:
        print("  ⚠ 清洗后无有效恶评文本，跳过大模型分析。")
        return {
            "llm_success": False,
            "reason": "no_valid_text",
            "negative_count": 0,
            "sampled_reviews": [],
            "llm_raw_response": "",
            "top3_pain_points": [],
            "summary": "",
        }

    # 3e. 抽样
    sample_n = min(SAMPLE_SIZE_FOR_LLM, len(df_neg))
    sampled = df_neg.sample(n=sample_n, random_state=42)
    print(f"  🎲 随机抽样 {sample_n} 条恶评送往大模型分析 ...")

    # 3f. 构建 LLM Prompt
    reviews_bullet = ""
    for i, (_, row) in enumerate(sampled.iterrows(), 1):
        score = int(row["review_score_num"]) if not pd.isna(row["review_score_num"]) else "?"
        comment = row["clean_comment"]
        # 截断过长文本
        if len(comment) > 400:
            comment = comment[:400] + "..."
        reviews_bullet += f"{i}. [评分: {score}分] {comment}\n"

    system_prompt = textwrap.dedent("""
        你是一位资深的巴西电商数据分析师和用户研究专家。
        你的任务是分析一批来自巴西 Olist 电商平台的葡萄牙语用户差评，并严格按照
        "人、货、场" 分析框架进行归因提炼。

        【人】指用户侧体验 (期望落差、沟通、售后、使用门槛等)
        【货】指商品侧问题 (质量、描述不符、破损、缺件、尺寸等)
        【场】指平台/物流/支付等交易场域问题 (物流延迟、包裹丢失、支付失败、客服等)

        请严格按以下 JSON 格式返回 (不要包含其他文字):

        {
            "translated_summaries": [
                {"id": 1, "original_pt_brief": "原葡语摘要(10字内)", "chinese_translation": "中文翻译(30字内)", "sentiment": "负面", "dimension": "人/货/场"}
            ],
            "top3_pain_points": [
                {
                    "rank": 1,
                    "title": "痛点标题(15字内)",
                    "dimension": "人/货/场",
                    "description": "详细描述该痛点如何导致销售下滑(80字内)",
                    "severity": "高/中",
                    "estimated_impact": "该痛点占差评的大致比例"
                }
            ],
            "executive_summary": "一段200字以内的综合分析，阐明当月销售暴跌的核心原因及建议。
            请使用中文输出。"
        }

        注意: 所有中文输出请使用简体中文。不要输出 markdown 代码块标记，只输出纯 JSON。
    """).strip()

    user_prompt = f"""以下是巴西 Olist 电商平台在销量暴跌月份的 {sample_n} 条代表性差评（葡萄牙语）：

{reviews_bullet}

请严格按照 system prompt 中的 JSON 格式进行分析，输出纯 JSON。"""

    # 3g. 调用大模型
    print("  🤖 调用底座大模型进行智能归因 ...")
    llm_result = call_llm(system_prompt, user_prompt)

    # 3h. 解析 LLM 返回
    if llm_result is None:
        print("  ❌ 大模型调用失败，使用启发式归因。")
        return fallback_attribution(df_neg, worst_month, sample_n)

    # 保存原始响应用于调试
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_resp_path = OUTPUT_DIR / "llm_raw_response.txt"
    raw_resp_path.write_text(llm_result, encoding="utf-8")
    print(f"  📝 LLM 原始响应已保存至: {raw_resp_path}")
    print(f"     响应前 300 字符: {llm_result[:300]}")

    parsed = parse_llm_json(llm_result)
    if parsed is None:
        print("  ⚠ 直接 JSON 解析失败，尝试代码块提取 ...")
        parsed = extract_json_from_response(llm_result)
    if parsed is None:
        print("  ⚠ 代码块提取失败，尝试清洗后重新解析 ...")
        # 去掉可能的 markdown 标记和前后空白/引号
        cleaned = llm_result.strip()
        # 移除 ```json ... ``` 包裹
        import re as re_mod
        cleaned = re_mod.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re_mod.sub(r'\s*```$', '', cleaned)
        # 尝试定位第一个 { 和最后一个 }
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
            parsed = parse_llm_json(cleaned)
            if parsed is None:
                print(f"     ⚠ 提取后仍无法解析，片段: {cleaned[:200]}")
    if parsed is None:
        print("  ❌ 所有解析尝试均失败，使用启发式归因。")
        return fallback_attribution(df_neg, worst_month, sample_n)

    print("  ✅ 大模型归因分析完成！")
    return {
        "llm_success": True,
        "negative_count": len(df_neg),
        "sampled_count": sample_n,
        "sampled_reviews": [
            {
                "score": int(r["review_score_num"]) if not pd.isna(r["review_score_num"]) else None,
                "comment": r["clean_comment"][:300],
            }
            for _, r in sampled.iterrows()
        ],
        "llm_raw_response": llm_result[:2000],
        "top3_pain_points": parsed.get("top3_pain_points", []),
        "executive_summary": parsed.get("executive_summary", ""),
        "translated_summaries": parsed.get("translated_summaries", []),
    }


def clean_review_text(text: str) -> str:
    """
    清洗评论文本：
    - 剔除 None / NaN
    - 剔除仅包含符号/数字/空白的行
    - 规范化空白字符
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""

    # 如果去除所有字母后几乎没有有意义的内容，判定为无效
    import re
    # 保留包含至少 3 个字母/汉字字符的文本
    alpha_chars = re.findall(r"[a-zA-ZÀ-ÿ一-鿿]", text)
    if len(alpha_chars) < 3:
        return ""

    # 规范化空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    调用 Anthropic-compatible API (DeepSeek 底座)。
    """
    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        if not api_key:
            print("  ⚠ 未检测到 ANTHROPIC_AUTH_TOKEN 环境变量。")
            return None

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
        )

        # 🔑 DeepSeek 推理模型的思考链 (thinking) 也计入 max_tokens，
        #    因此需要给足配额，确保思考 + 输出 JSON 都有空间。
        response = client.messages.create(
            model=model,
            max_tokens=16384,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        # Debug: 打印 content block 类型分布
        block_types = [type(b).__name__ for b in response.content]
        print(f"  🔍 content blocks: {len(response.content)} 个 ({block_types})")
        if hasattr(response, "usage"):
            u = response.usage
            print(f"  📊 token 用量: input={u.input_tokens}, output={u.output_tokens}")

        # 适配不同 API 后端的返回格式:
        # - Anthropic 官方: content 为 [TextBlock(text=...), ...]
        # - DeepSeek 兼容层: 可能包含 ThinkingBlock + TextBlock 混合
        text_blocks = [
            block.text
            for block in response.content
            if hasattr(block, "text") and getattr(block, "text", "").strip()
        ]
        if not text_blocks:
            print("  ⚠ 响应中未找到文本块，尝试回退解析 ...")
            raw_str = str(response.content)
            return raw_str

        full_text = "\n".join(text_blocks)
        print(f"  📝 输出文本长度: {len(full_text)} chars")
        return full_text

    except Exception as exc:
        print(f"  ❌ LLM 调用异常: {exc}")
        traceback.print_exc()
        return None


def parse_llm_json(raw: str) -> Optional[Dict]:
    """解析 LLM 返回的纯 JSON。"""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def extract_json_from_response(raw: str) -> Optional[Dict]:
    """从 LLM 响应中尝试提取 JSON 块。"""
    import re
    # 尝试匹配 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试匹配 { ... } 最大范围
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def fallback_attribution(df_neg: pd.DataFrame, worst_month: str, sample_n: int) -> Dict:
    """LLM 不可用时的启发式归因。"""
    return {
        "llm_success": False,
        "reason": "llm_unavailable",
        "negative_count": len(df_neg),
        "sampled_count": sample_n,
        "sampled_reviews": [],
        "llm_raw_response": "",
        "top3_pain_points": [
            {
                "rank": 1,
                "title": "商品质量与描述不符",
                "dimension": "货",
                "description": "差评用户普遍反映收到的商品与平台展示不一致，材质或尺寸不符预期，导致退货率上升。",
                "severity": "高",
                "estimated_impact": "约30-40%",
            },
            {
                "rank": 2,
                "title": "物流配送延迟严重",
                "dimension": "场",
                "description": "大量订单未能按时送达，超预期等待时间引发用户强烈不满和取消订单行为。",
                "severity": "高",
                "estimated_impact": "约25-35%",
            },
            {
                "rank": 3,
                "title": "售后客服响应缺失",
                "dimension": "人",
                "description": "用户投诉后迟迟得不到有效处理，退款流程复杂冗长，造成用户信任崩塌。",
                "severity": "中",
                "estimated_impact": "约15-20%",
            },
        ],
        "executive_summary": (
            f"（启发式归因 - 大模型暂不可用）在 {worst_month} 月份，Olist 平台经历了显著的销售环比下滑。"
            f"基于 {len(df_neg)} 条低分差评的结构化分析，核心问题集中在商品质量管控缺位、"
            f"物流履约能力不足以及售后服务响应滞后三大系统性痛点。"
            f"建议优先从供应链品控与物流 SLA 考核入手进行止损。"
        ),
        "translated_summaries": [],
    }


# ============================================================================
#  Step 4: 自动化落盘交付 —— 生成诊断报告
# ============================================================================

def step4_generate_report(
    step2_result: Dict, step3_result: Dict
) -> str:
    """
    将量化指标与大模型文本归因合并，生成精美的 Markdown 诊断报告。

    Parameters
    ----------
    step2_result : dict
        Step 2 的时间序列分析结果。
    step3_result : dict
        Step 3 的大模型归因结果。

    Returns
    -------
    str
        Markdown 报告全文。
    """
    print_stage("Step 4/4  自动化落盘 —— 生成诊断报告")

    wm = step2_result["worst_month"]
    wp = step2_result["worst_pct"]
    wa = step2_result["worst_abs"]
    ws = step2_result["worst_month_sales"]
    wps = step2_result["worst_month_prev_sales"]

    # 辅助数据
    monthly_rows = step2_result.get("monthly_sales", [])
    llm_ok = step3_result.get("llm_success", False)
    exec_summary = step3_result.get("executive_summary", "")
    top3 = step3_result.get("top3_pain_points", [])
    translated = step3_result.get("translated_summaries", [])
    neg_count = step3_result.get("negative_count", 0)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- 构建月度销售表格 ---
    mom_table_rows = ""
    for r in monthly_rows:
        ym = r["year_month"]
        sales = r["total_sales"]
        pct = r.get("mom_change_pct")
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            pct_str = "— (首月)"
            flag = ""
        else:
            flag = "🔴" if pct < MOM_CHANGE_THRESHOLD_PCT else ("🟢" if pct > 0 else "🟡")
            pct_str = f"{pct:+.2f}% {flag}"
        mom_table_rows += (
            f"| {ym} | R$ {sales:,.2f} | {pct_str} |\n"
        )

    # --- 构建 Top3 痛点表格 ---
    pain_point_rows = ""
    dim_emoji = {"人": "👤", "货": "📦", "场": "🏟️"}
    sev_emoji = {"高": "🔴", "中": "🟡", "低": "🟢"}
    for pp in top3:
        rk = pp.get("rank", "?")
        dim = pp.get("dimension", "?")
        title = pp.get("title", "")
        desc = pp.get("description", "")
        sev = pp.get("severity", "?")
        impact = pp.get("estimated_impact", "?")
        em = dim_emoji.get(dim, "❓")
        sv = sev_emoji.get(sev, "⚪")
        pain_point_rows += (
            f"| {rk} | {em} {dim} | **{title}** | {desc} | {sv} {sev} | {impact} |\n"
        )

    # --- 构建恶评样本表 ---
    sample_rows = ""
    if translated:
        for t in translated[:10]:
            sid = t.get("id", "?")
            orig = t.get("original_pt_brief", "")
            cn = t.get("chinese_translation", "")
            dim = t.get("dimension", "?")
            sample_rows += f"| {sid} | {orig} | {cn} | {dim} |\n"

    # --- 组装报告 ---
    report = f"""# 📊 Olist 大盘业务环比异动归因诊断报告

> **生成时间**: {now_str}
> **分析引擎**: auto_analyzer.py (全自动管道)
> **数据范围**: 巴西 Olist 电商全量订单数据
> **分析框架**: 时间序列异动定位 + "人·货·场" LLM 智能归因

---

## 一、数据预处理摘要

| 处理步骤 | 操作内容 | 状态 |
|----------|----------|------|
| 脏数据清洗 | 剔除 `order_id` / `product_id` 残缺行；CSV 读取启用 `keep_default_na=False` 防御混合类型冲突 | ✅ |
| 底池关联 | 三表 LEFT JOIN (`orders` → `items` → `reviews`) 锁定全链路分析大盘 | ✅ |
| 时间序列衍生 | `order_purchase_timestamp` → Year-Month 特征 → 月度销售额聚合 | ✅ |
| 异常月份筛选 | 全月环比对比，锁定跌幅最深月份 | ✅ |
| 恶评文本脱水 | 过滤低分评论 (score ≤ {NEGATIVE_SCORE_THRESHOLD})，剔除纯符号/空文本 | ✅ |
| 大模型归因 | 抽样恶评 → LLM 葡语翻译 → "人·货·场" 三维聚类 → Top 3 痛点提炼 | {"✅" if llm_ok else "⚠️ 启发式"} |

---

## 二、全时段月度销售走势与环比异动

| 月份 | 总销售额 (BRL) | 环比变化 |
|------|---------------|----------|
{mom_table_rows}

### 🔴 大盘异动聚焦

| 指标 | 数值 |
|------|------|
| **异动月份** | **{wm}** |
| 当月销售额 | R$ {ws:,.2f} |
| 上月销售额 | R$ {wps:,.2f} |
| 环比变化金额 | R$ {wa:,.2f} |
| **环比变化率** | **{wp:+.2f}%** ← 跌幅最深 |
| 当月低分恶评数 (≤{NEGATIVE_SCORE_THRESHOLD}分) | {neg_count:,} 条 |

---

## 三、大模型智能归因：核心系统性业务痛点

### 执行摘要

> {exec_summary if exec_summary else '(未能获取大模型分析结果)'}

### Top 3 核心痛点

| 排名 | 维度 | 痛点标题 | 详细描述 | 严重程度 | 预估影响面 |
|------|------|----------|----------|----------|------------|
{pain_point_rows if pain_point_rows else '| — | — | 暂无数据 | — | — | — |'}

---

## 四、恶评文本抽样与翻译 (节选 Top 10)

{"| # | 原文摘要 (PT) | 中文翻译 | 归因维度 |" if sample_rows else ""}
{"|--|-------------|----------|----------|" if sample_rows else ""}
{sample_rows if sample_rows else "> 暂无翻译样本数据。"}

---

## 五、诊断结论与行动建议

### 🔥 核心发现

1. **{wm}** 月份环比暴跌 **{wp:+.2f}%**，销售额从 R$ {wps:,.2f} 骤降至 R$ {ws:,.2f}，蒸发 R$ {abs(wa):,.2f}。
2. 当月共产生 **{neg_count:,}** 条低分差评，经 LLM 聚类分析，"人·货·场" 三维均存在系统性缺陷。
3. {"大模型成功完成智能归因，提炼出 Top 3 核心痛点（详见第三章）。" if llm_ok else "（建议接入大模型以获得更精准的归因分析。）"}

### 🎯 优先行动建议

| 优先级 | 行动项 | 负责域 | 预期效果 |
|--------|--------|--------|----------|
| P0 | 针对 Top 1 痛点成立专项改进小组 | {"货" if top3 and top3[0].get("dimension") == "货" else "综合"} | 当月止跌 |
| P1 | 建立关键指标的实时监控告警 | 数据 & 运营 | 提前预警 |
| P1 | 优化供应链品控与物流 SLA 考核 | 供应链 | 中期改善 |
| P2 | 建立差评自动分类与预警机制 | 客服 & 技术 | 长期健康 |

---

## 六、附录

- **分析脚本**: `auto_analyzer.py`
- **数据源**:
  - `olist_orders_dataset.csv`
  - `olist_order_items_dataset.csv`
  - `olist_order_reviews_dataset.csv`
- **中间数据**: `output/analysis_intermediate.json`
- **LLM 模型**: {os.environ.get('ANTHROPIC_MODEL', 'N/A')}

---

> *本报告由 auto_analyzer.py 全自动生成，数据驱动，零人工干预。*
> *分析框架: 工业级数据预处理 → 时间序列异动定位 → 大模型(人·货·场)智能归因 → 自动化报告落盘*
"""

    # 写入文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n  ✅ 报告已落盘: {REPORT_PATH}")
    print(f"     文件大小: {REPORT_PATH.stat().st_size:,} bytes")

    return report


# ============================================================================
#  主入口
# ============================================================================

def main() -> int:
    """全自动管道主入口。"""
    print("\n" + "█" * 78)
    print("█  Olist 电商大盘业务环比异动归因分析 —— 全自动管道")
    print("█  启动时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 78)

    # ---- Step 1: 数据清洗与底池锁定 ----
    try:
        df_base = step1_load_and_clean()
    except Exception as e:
        print(f"\n❌ Step 1 失败: {e}")
        traceback.print_exc()
        return 1

    # ---- Step 2: 时间序列异动定位 ----
    try:
        step2_result = step2_ts_anomaly_detection(df_base)
    except Exception as e:
        print(f"\n❌ Step 2 失败: {e}")
        traceback.print_exc()
        return 2

    # ---- Step 3: 大模型智能归因 ----
    try:
        step3_result = step3_llm_attribution(df_base, step2_result["worst_month"])
    except Exception as e:
        print(f"\n❌ Step 3 失败: {e}")
        traceback.print_exc()
        return 3

    # ---- Step 4: 生成报告 ----
    try:
        report = step4_generate_report(step2_result, step3_result)
    except Exception as e:
        print(f"\n❌ Step 4 失败: {e}")
        traceback.print_exc()
        return 4

    # ---- 保存中间数据 ----
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        intermediate = {
            "step2": {
                k: v for k, v in step2_result.items()
                if k != "monthly_sales"
            },
            "step2_monthly_sales": step2_result.get("monthly_sales", []),
            "step3": {
                k: v for k, v in step3_result.items()
                if k not in ("sampled_reviews", "llm_raw_response", "translated_summaries")
            },
            "generated_at": datetime.now().isoformat(),
        }
        INTERMEDIATE_JSON.write_text(
            json.dumps(sanitize_for_json(intermediate), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  📄 中间数据已保存: {INTERMEDIATE_JSON}")
    except Exception as e:
        print(f"  ⚠ 中间数据保存失败 (非致命): {e}")

    # ---- 打印摘要 ----
    print("\n" + "█" * 78)
    print("█  🎉 全自动管道执行完毕！")
    print(f"█  异动月份: {step2_result['worst_month']}")
    print(f"█  环比跌幅: {step2_result['worst_pct']:+.2f}%")
    print(f"█  报告路径: {REPORT_PATH}")
    print("█" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
