"""発注数と「お金のインパクト」の計算（TypeScript版の lib/ordering.ts と同じことをする）"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from inventory_agent.dataset import Product
from inventory_agent.forecast import HORIZON, ForecastResult, forecast_product, js_round

REVIEW_WEEKS = 2  # 発注は月2回＝次の発注まで約2週間
EXCESS_WEEKS = 26  # これより先まで売れ残りそうな在庫を「眠っている在庫」とみなす


@dataclass
class OrderLine:
    product: Product
    forecast: ForecastResult
    lead_weeks: int  # 届くまでの週数
    cover_weeks: int  # 今回の発注でまかなう週数
    demand_cover: int  # その期間に売れる見込み
    safety: int  # 念のための余裕（安全在庫）
    available: int  # 在庫＋発注済み
    need: int  # 足りない数
    order_qty: int  # 推奨の発注数
    order_cost: int  # 発注金額（仕入れ値ベース）
    stockout_week: int | None  # 発注しなかった場合、何週目に在庫が切れるか
    lost4_units: int  # 次の4週の売り逃し見込み
    lost4_value: int
    savable_value: int  # そのうち、今日発注すれば防げる金額
    late_units: int  # 今日発注しても届く前に売り切れる数
    late_value: int
    excess_units: int  # 26週先まで売れ残りそうな在庫
    excess_value: int  # その仕入れ金額（眠っている資金）
    reason: str  # 計算の説明


def plan_order(p: Product, sale_weeks: list[int], demand_multiplier: float = 1.0, extra_lead_days: int = 0) -> OrderLine:
    """1商品の発注案。demand_multiplier と extra_lead_days は「もしも」の試算用"""
    base = forecast_product(p, sale_weeks)
    k = demand_multiplier
    f = base if k == 1 else replace(
        base,
        forecast=[v * k for v in base.forecast],
        lower=[v * k for v in base.lower],
        upper=[v * k for v in base.upper],
    )
    lead_weeks = math.ceil((p.lead_days + extra_lead_days) / 7)
    cover_weeks = min(HORIZON, lead_weeks + REVIEW_WEEKS)
    demand_cover = sum(f.forecast[:cover_weeks])

    # 安全在庫：予測の外れやすさ × 期間の長さ。大型商品は少なめに持つ
    weekly_sigma = (f.upper[0] - f.forecast[0]) / 1.28
    z = 1.0 if p.large else 1.65
    safety = js_round(z * weekly_sigma * math.sqrt(cover_weeks))

    available = p.stock + p.on_order
    need = max(0, js_round(demand_cover + safety - available))
    lot = max(1, p.min_lot)
    order_qty = 0 if p.discontinued else (math.ceil(need / lot) * lot if need > 0 else 0)

    left = float(available)
    stockout_week = None
    for i in range(HORIZON):
        left -= f.forecast[i]
        if left < 0 and stockout_week is None:
            stockout_week = i + 1

    lost4_units = max(0, js_round(sum(f.forecast[:4]) - available))
    lost_before_arrival = max(0, js_round(sum(f.forecast[: min(4, lead_weeks)]) - available))
    savable_units = max(0, lost4_units - lost_before_arrival)
    late_units = 0 if p.discontinued else max(0, js_round(sum(f.forecast[:lead_weeks]) - available))

    tail_avg = sum(f.forecast[-4:]) / 4
    demand26 = sum(f.forecast) + tail_avg * (EXCESS_WEEKS - HORIZON)
    excess_units = max(0, js_round(available - (sum(f.forecast) if p.discontinued else demand26)))

    if p.discontinued:
        reason = "販売終了予定のため発注しません"
    else:
        reason = (
            f"{cover_weeks}週分（届くまで{lead_weeks}週＋次の発注まで{REVIEW_WEEKS}週）の予測{js_round(demand_cover)}個"
            f"＋余裕{safety}個−在庫と発注済み{available}個＝{need}個"
        )
        if order_qty > 0 and order_qty != need:
            reason += f" → {p.min_lot}個単位で{order_qty}個"

    return OrderLine(
        product=p,
        forecast=f,
        lead_weeks=lead_weeks,
        cover_weeks=cover_weeks,
        demand_cover=js_round(demand_cover),
        safety=safety,
        available=available,
        need=need,
        order_qty=order_qty,
        order_cost=order_qty * p.cost,
        stockout_week=stockout_week,
        lost4_units=lost4_units,
        lost4_value=lost4_units * p.price,
        savable_value=savable_units * p.price,
        late_units=late_units,
        late_value=late_units * p.price,
        excess_units=excess_units,
        excess_value=excess_units * p.cost,
        reason=reason,
    )


@dataclass
class Impact:
    lost4_value: int
    savable_value: int
    late_value: int
    excess_value: int
    order_total: int
    top_lost: list[OrderLine]
    late_items: list[OrderLine]
    top_excess: list[OrderLine]


def summarize(lines: list[OrderLine]) -> Impact:
    """お金のまとめ（最初の画面のフックに出す数字）"""
    return Impact(
        lost4_value=sum(l.lost4_value for l in lines),
        savable_value=sum(l.savable_value for l in lines),
        late_value=sum(l.late_value for l in lines),
        excess_value=sum(l.excess_value for l in lines),
        order_total=sum(l.order_cost for l in lines),
        top_lost=sorted([l for l in lines if l.lost4_value > 0], key=lambda l: -l.lost4_value)[:5],
        late_items=sorted([l for l in lines if l.late_units > 0], key=lambda l: -l.late_value),
        top_excess=sorted([l for l in lines if l.excess_value > 0], key=lambda l: -l.excess_value)[:5],
    )


def yen(v: float) -> str:
    return f"{js_round(v):,}円"
