"""売れ行きの予測（TypeScript版の lib/forecast-models.ts と同じことをする）

3つの方法で予測し、「過去の答え合わせ」で一番当たった方法を商品ごとに選ぶ。
すべてコードで計算する。AIは使わない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from inventory_agent.dataset import Product

HORIZON = 12  # 何週先まで予測するか
HOLDOUT = 12  # 答え合わせに使う、直近の週数
SEASON = 52  # 1年＝52週

METHOD_LABEL = {
    "seasonal": "前年同期×伸び率",
    "moving": "直近8週の平均",
    "holt": "季節＋傾向の平滑化",
}


@dataclass
class ForecastResult:
    method: str
    forecast: list[float]  # 次の12週の予測
    lower: list[float]  # 予測の幅（下側）
    upper: list[float]  # 予測の幅（上側）
    wape: float | None  # 答え合わせの誤差率（0.2なら20%）
    comparisons: list[dict] = field(default_factory=list)
    confidence: str = "低"  # 高・中・低
    note: str = ""


def mean(a: list[float]) -> float:
    return sum(a) / len(a) if a else 0.0


def js_round(x: float) -> int:
    """TypeScript の Math.round と同じ四捨五入（Pythonの round は 0.5 の扱いが違うため）"""
    return math.floor(x + 0.5)


def clean_sales(weekly: list[int], sale_weeks: list[int]) -> list[float]:
    """セールの週は、前後の普段の週の平均に置き換える"""
    sale = set(sale_weeks)
    out: list[float] = []
    for i, v in enumerate(weekly):
        if i not in sale:
            out.append(v)
            continue
        p = i - 1
        while p >= 0 and p in sale:
            p -= 1
        n = i + 1
        while n < len(weekly) and n in sale:
            n += 1
        around = [weekly[j] for j in (p, n) if 0 <= j < len(weekly)]
        out.append(mean(around) if around else v)
    return out


# ===== 3つの予測方法 =====


def seasonal_naive(y: list[float], h: int) -> list[float] | None:
    """方法1：前年同期×伸び率"""
    n = len(y)
    if n < SEASON + 8:
        return None
    recent = sum(y[n - 8 :])
    last_year = sum(y[n - 8 - SEASON : n - SEASON])
    growth = min(2, max(0.5, recent / last_year)) if last_year > 0 else 1
    return [max(0, y[n - SEASON + i] * growth) for i in range(h)]


def moving_average(y: list[float], h: int) -> list[float] | None:
    """方法2：直近8週の平均"""
    if len(y) < 4:
        return None
    return [mean(y[-8:])] * h


def holt_winters(y: list[float], h: int) -> list[float] | None:
    """方法3：ホルト・ウィンタース法（加法型）。3つの重みは、当てはまりが一番よい組み合わせを選ぶ"""
    if len(y) < SEASON + 12:
        return None
    best: tuple[float, list[float]] | None = None
    for a in (0.2, 0.4, 0.6):
        for b in (0, 0.05):
            for g in (0.2, 0.4):
                level = mean(y[:SEASON])
                trend = 0.0
                s = [v - level for v in y[:SEASON]]
                sse = 0.0
                for t in range(SEASON, len(y)):
                    si = s[t % SEASON]
                    sse += (y[t] - (level + trend + si)) ** 2
                    prev = level
                    level = a * (y[t] - si) + (1 - a) * (level + trend)
                    trend = b * (level - prev) + (1 - b) * trend
                    s[t % SEASON] = g * (y[t] - level) + (1 - g) * si
                n = len(y)
                fc = [max(0, level + (i + 1) * trend + s[(n + i) % SEASON]) for i in range(h)]
                if best is None or sse < best[0]:
                    best = (sse, fc)
    return best[1] if best else None


METHODS: dict[str, Callable[[list[float], int], list[float] | None]] = {
    "seasonal": seasonal_naive,
    "moving": moving_average,
    "holt": holt_winters,
}


# ===== 季節品の見分け方 =====


def seasonality(y: list[float]) -> float:
    """直近1年を4週ずつ13のかたまりに分け、かたまりごとの平均のばらつき（大きいほど季節の波が強い）"""
    if len(y) < SEASON:
        return 0
    blocks = _blocks13(y[-SEASON:])
    m = mean(blocks)
    if m <= 0:
        return 0
    return math.sqrt(mean([(b - m) ** 2 for b in blocks])) / m


def _blocks13(y: list[float]) -> list[float]:
    return [mean(y[i * 4 : i * 4 + 4]) for i in range(13)]


def _correlation(a: list[float], b: list[float]) -> float:
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb) if va > 0 and vb > 0 else 0


def is_seasonal_item(y: list[float]) -> bool:
    """「1年の中で山と谷が大きい」かつ「去年と今年で山と谷の形が似ている」なら季節品"""
    if len(y) < SEASON * 2:
        return False
    this_year = _blocks13(y[-SEASON:])
    last_year = _blocks13(y[-SEASON * 2 : -SEASON])
    return seasonality(y) >= 0.35 and _correlation(this_year, last_year) >= 0.6


def wape(forecast: list[float], actual: list[float]) -> float | None:
    """誤差率：外れた個数の合計 ÷ 実際に売れた個数の合計"""
    total = sum(actual)
    if total <= 0:
        return None
    return sum(abs(f - a) for f, a in zip(forecast, actual)) / total


# ===== 1商品の予測 =====


def forecast_product(p: Product, sale_weeks: list[int]) -> ForecastResult:
    """① セールの週をならす → ② 直近12週を隠して3つの方法で予測 → ③ 一番当たった方法を選ぶ → ④ 次の12週を予測"""
    y = clean_sales(p.weekly, sale_weeks)
    n = len(y)
    weeks_of_data = 0 if p.first_sale_week < 0 else n - p.first_sale_week

    # データが少ない新商品は、答え合わせができないので「直近の平均」で、確かさ「低」
    if weeks_of_data < 13:
        fc = moving_average(y[max(0, p.first_sale_week) :], HORIZON) or [0.0] * HORIZON
        return ForecastResult(
            method="moving",
            forecast=fc,
            lower=[v * 0.5 for v in fc],
            upper=[v * 1.5 for v in fc],
            wape=None,
            confidence="低",
            note=f"販売データが{weeks_of_data}週分しかないため、答え合わせができません。幅を広めに取っています。",
        )

    train, actual = y[: n - HOLDOUT], y[n - HOLDOUT :]
    comparisons = []
    for m, fn in METHODS.items():
        fc = fn(train, HOLDOUT)
        comparisons.append({"method": m, "wape": wape(fc, actual) if fc else None, "fc": fc})
    public = [{"method": c["method"], "wape": c["wape"]} for c in comparisons]

    # 販売終了予定は、季節や伸びを読まず「直近の平均」で控えめに見る
    if p.discontinued:
        fc = moving_average(y, HORIZON) or [0.0] * HORIZON
        mv = next(c for c in comparisons if c["method"] == "moving")
        return ForecastResult(
            method="moving",
            forecast=fc,
            lower=[v * 0.5 for v in fc],
            upper=[v * 1.5 for v in fc],
            wape=mv["wape"],
            comparisons=public,
            confidence="低",
            note="販売終了予定の商品です。季節や伸びは読まず、直近の平均で控えめに見ています。発注は控えてください。",
        )

    # 季節の波が強い商品は「直近の平均」を候補から外す（閑散期の答え合わせで勝ってしまうため）
    seasonal_item = is_seasonal_item(y)
    others_ok = any(c["method"] != "moving" and c["wape"] is not None for c in comparisons)
    usable = [
        c for c in comparisons if c["wape"] is not None and not (seasonal_item and c["method"] == "moving" and others_ok)
    ]
    chosen = min(usable, key=lambda c: c["wape"]) if usable else None
    method = chosen["method"] if chosen else "moving"
    forecast = METHODS[method](y, HORIZON) or [0.0] * HORIZON

    # 予測の幅：答え合わせで外れた大きさから、おおよそ8割が入る幅を出す
    if chosen:
        sigma = math.sqrt(mean([(f - a) ** 2 for f, a in zip(chosen["fc"], actual)]))
    else:
        sigma = mean(forecast) * 0.3
    w = chosen["wape"] if chosen else None
    confidence = "低" if w is None else "高" if w < 0.2 else "中" if w < 0.35 else "低"
    return ForecastResult(
        method=method,
        forecast=forecast,
        lower=[max(0, v - 1.28 * sigma) for v in forecast],
        upper=[v + 1.28 * sigma for v in forecast],
        wape=w,
        comparisons=public,
        confidence=confidence,
        note="季節の波が強い商品のため、季節を考える方法の中から選んでいます。" if seasonal_item else "",
    )


def overall_wape(products: list[Product], sale_weeks: list[int]) -> float | None:
    """全商品をまとめた誤差率（売れる数が多い商品ほど重く数える）"""
    err = total = 0.0
    for p in products:
        r = forecast_product(p, sale_weeks)
        if r.wape is None:
            continue
        actual = sum(clean_sales(p.weekly, sale_weeks)[-HOLDOUT:])
        err += r.wape * actual
        total += actual
    return err / total if total > 0 else None


def future_weeks(last_week: str, h: int = HORIZON) -> list[str]:
    """予測する週の月曜日の日付"""
    d = date.fromisoformat(last_week)
    return [(d + timedelta(days=7 * (i + 1))).isoformat() for i in range(h)]
