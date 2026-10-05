"""エージェントが使う「道具」（TypeScript版の lib/agent/tools.ts と同じ6つ）

中身はすべて第2回で作った計算（コード）。AIは数字を作らない。
"""

from __future__ import annotations

import json
import math
from typing import Callable, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from inventory_agent.dataset import Dataset, Product
from inventory_agent.forecast import HORIZON, METHOD_LABEL, forecast_product, future_weeks, js_round
from inventory_agent.ordering import plan_order, summarize, yen

TOOL_LABEL = {
    "get_overview": "全体のお金のインパクトを確認",
    "list_products": "商品の一覧を確認",
    "get_product_detail": "商品の予測と在庫を確認",
    "get_upcoming_sales": "これからのセール予定を確認",
    "simulate_plan": "「もしも」の条件で試算",
    "propose_order": "発注案を作成（承認待ち）",
}

SaveProposal = Callable[[list[dict], str], str]


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ===== 道具ごとの入力の形（AIに渡す説明にもなる） =====


class ListArgs(BaseModel):
    filter: Literal["stockout", "order", "excess", "all"] = Field(description="絞り込みの種類")


class DetailArgs(BaseModel):
    query: str = Field(description="商品コード（例：K003）または商品名（例：土鍋、麦茶ポット）")


class SimulateArgs(BaseModel):
    demand_multiplier: float | None = Field(default=None, ge=0.3, le=3, description="売れ行きの倍率。1なら変化なし")
    extra_lead_days: int | None = Field(default=None, ge=0, le=90, description="届くまでの遅れ（日）")
    supplier: str | None = Field(default=None, description="仕入先名（例：海外C）")
    codes: list[str] | None = Field(default=None, description="商品コードの一覧")


class ProposalItem(BaseModel):
    code: str
    qty: int = Field(ge=0)
    reason: str = Field(description="その数にした理由")


class ProposeArgs(BaseModel):
    items: list[ProposalItem] = Field(min_length=1)
    note: str = Field(description="担当者への説明")


def create_tools(data: Dataset, save_proposal: SaveProposal) -> list[BaseTool]:
    """道具を6つ作って返す。発注案の保存方法は、呼び出す側から渡してもらう"""

    def find(code: str) -> Product | None:
        return next((p for p in data.products if p.code == code.strip().upper()), None)

    def norm(s: str) -> str:
        return s.replace(" ", "").replace("　", "").lower()

    def search(query: str) -> list[Product]:
        """商品コードでも、商品名（一部でも）でも探せるようにする"""
        hit = find(query)
        if hit:
            return [hit]
        q = norm(query)
        return [p for p in data.products if q in norm(p.name) or (len(q) >= 2 and norm(p.name) in q)]

    def lines(**opts):
        return [plan_order(p, data.sale_weeks, **opts) for p in data.products]

    @tool("get_overview")
    def get_overview() -> str:
        """店全体の売り逃し・間に合わない欠品・眠っている在庫の金額と、上位の商品を返す。最初に状況をつかむときに使う。"""
        s = summarize(lines())
        top3 = sum(l.lost4_value for l in s.top_lost[:3])
        return _json({
            "次の4週の売り逃し": yen(s.lost4_value),
            "今日発注すれば防げる額": yen(s.savable_value),
            "届く前に在庫が切れる額": yen(s.late_value),
            "眠っている在庫": yen(s.excess_value),
            "推奨どおり発注した場合の発注総額": yen(s.order_total),
            "売り逃し上位3商品の合計": yen(top3),
            "売り逃し上位3商品が全体に占める割合": f"{js_round(top3 / s.lost4_value * 100)}%" if s.lost4_value else "0%",
            "売り逃しと間に合わない欠品の合計": yen(s.lost4_value + s.late_value),
            "売り逃しが大きい商品": [f"{l.product.code} {l.product.name}（{yen(l.lost4_value)}、売り逃し全体の{js_round(l.lost4_value / s.lost4_value * 100)}%、{l.stockout_week}週目に在庫切れ）" for l in s.top_lost],            "間に合わない商品": [f"{l.product.code} {l.product.name}（{l.stockout_week}週目に切れる／届くのは{l.lead_weeks}週後）" for l in s.late_items],
            "眠っている在庫が多い商品": [f"{l.product.code} {l.product.name}（{l.excess_units}個、{yen(l.excess_value)}）" for l in s.top_excess],
        })

    @tool("list_products", args_schema=ListArgs)
    def list_products(filter: str) -> str:
        """商品の一覧と、在庫・予測・推奨発注数・販売終了予定かどうかを返す。1つの商品を詳しく調べるときは、これではなく get_product_detail を使う。filter：stockout＝6週以内に在庫が切れる商品、order＝発注が必要な商品、excess＝在庫が余りそうな商品、all＝すべて。"""
        all_lines = lines()
        if filter == "stockout":
            picked = [l for l in all_lines if l.stockout_week is not None and l.stockout_week <= 6]
        elif filter == "order":
            picked = [l for l in all_lines if l.order_qty > 0]
        elif filter == "excess":
            picked = [l for l in all_lines if l.excess_units > 0]
        else:
            picked = all_lines
        return _json([
            {
                "コード": l.product.code,
                "商品名": l.product.name,
                "仕入先": f"{l.product.supplier}（{l.product.lead_days}日・{l.product.min_lot}個単位）",
                "販売終了予定": "はい（発注しない）" if l.product.discontinued else "いいえ",
                "在庫と発注済み": l.available,
                "次の4週の予測": js_round(sum(l.forecast.forecast[:4])),
                "在庫切れ": f"{l.stockout_week}週目" if l.stockout_week else "なし",
                "推奨発注数": l.order_qty,
                "確かさ": l.forecast.confidence,
            }
            for l in picked
        ])

    @tool("get_product_detail", args_schema=DetailArgs)
    def get_product_detail(query: str) -> str:
        """1つの商品について、予測（4週ごと・幅つき）、予測方法と誤差、在庫、在庫切れの時期、販売終了予定かどうか、推奨発注数と計算の中身を返す。商品コードでも商品名でも探せる。予測は次の12週まで。"""
        hits = search(query)
        if not hits:
            return f"「{query}」に当てはまる商品は見つかりません。この店では扱っていない可能性があります。"
        if len(hits) > 1:
            return _json({
                "説明": f"「{query}」に当てはまる商品が{len(hits)}つあります。どれのことか、商品名で絞って調べ直してください。",
                "候補": [f"{h.code} {h.name}" for h in hits],
            })
        p = hits[0]
        f = forecast_product(p, data.sale_weeks)
        l = plan_order(p, data.sale_weeks)
        weeks = future_weeks(data.weeks[-1])
        return _json({
            "コード": p.code,
            "商品名": p.name,
            "仕入先": p.supplier,
            "届くまでの日数": p.lead_days,
            "最低発注数": p.min_lot,
            "大型": p.large,
            "販売終了予定": p.discontinued,
            "販売価格": p.price,
            "仕入れ値": p.cost,
            "在庫": p.stock,
            "発注済み": p.on_order,
            "直近4週の販売数": sum(p.weekly[-4:]),
            "予測方法": METHOD_LABEL[f.method],
            "答え合わせの誤差": "データ不足" if f.wape is None else f"{js_round(f.wape * 100)}%",
            "確かさ": f.confidence,
            "予測_4週ごと": [
                {
                    "期間": f"{weeks[i]}の週から4週",
                    "予測": js_round(sum(f.forecast[i : i + 4])),
                    "幅": f"{js_round(sum(f.lower[i : i + 4]))}〜{js_round(sum(f.upper[i : i + 4]))}",
                }
                for i in (0, 4, 8)
            ],
            "在庫切れ": f"{l.stockout_week}週目" if l.stockout_week else f"{HORIZON}週以内はなし",
            "推奨発注数": l.order_qty,
            "計算の中身": l.reason,
            "注意": f.note or "なし",
        })

    @tool("get_upcoming_sales")
    def get_upcoming_sales() -> str:
        """去年のセールの時期から推定した、次の12週のセール予定を返す。セールに向けた発注を考えるときに使う。"""
        n = len(data.weeks)
        weeks = future_weeks(data.weeks[-1])
        upcoming = [w for i, w in enumerate(weeks) if (n + i - 52) in data.sale_weeks]
        return _json({
            "説明": "去年と同じ時期にセールがあると仮定した推定です。正確な予定は担当者に確認してください。",
            "セールがありそうな週": upcoming or "次の12週はなし",
        })

    @tool("simulate_plan", args_schema=SimulateArgs)
    def simulate_plan(demand_multiplier=None, extra_lead_days=None, supplier=None, codes=None) -> str:
        """「もしも」の試算。売れ行きの倍率（例：セールで1.3）や、届くまでの遅れ（日数）を入れて、売り逃しと推奨発注数がどう変わるかを返す。仕入先（国内A／国内B／海外C）や商品コードで絞れる。"""
        k, delay = demand_multiplier or 1.0, extra_lead_days or 0
        target = [p for p in data.products if (not supplier or p.supplier == supplier) and (not codes or p.code in codes)]
        if not target:
            return "条件に合う商品がありません。仕入先名や商品コードを確認してください。"
        before_lines = [plan_order(p, data.sale_weeks) for p in target]
        after_lines = [plan_order(p, data.sale_weeks, demand_multiplier=k, extra_lead_days=delay) for p in target]
        b, a = summarize(before_lines), summarize(after_lines)
        changed = [
            f"{al.product.code} {al.product.name}：推奨{bl.order_qty}→{al.order_qty}個" + (f"、届く前に{al.late_units}個不足" if al.late_units else "")
            for bl, al in zip(before_lines, after_lines)
            if al.order_qty != bl.order_qty or al.late_units != bl.late_units
        ]
        return _json({
            "条件": f"売れ行き{k}倍、届くのが{delay}日遅れる、対象：{supplier or 'すべての仕入先'}",
            "今のまま": {"売り逃し4週": yen(b.lost4_value), "間に合わない欠品": yen(b.late_value), "推奨発注総額": yen(b.order_total)},
            "条件を入れた場合": {"売り逃し4週": yen(a.lost4_value), "間に合わない欠品": yen(a.late_value), "推奨発注総額": yen(a.order_total)},
            "増える額": {
                "売り逃し4週": yen(a.lost4_value - b.lost4_value),
                "間に合わない欠品": yen(a.late_value - b.late_value),
                "推奨発注総額": yen(a.order_total - b.order_total),
            },
            "推奨発注数が変わる商品": changed,
        })

    @tool("propose_order", args_schema=ProposeArgs)
    def propose_order(items: list, note: str) -> str:
        """発注案を「承認待ち」として保存する。発注を確定する力はなく、担当者の承認が必要。数は最低発注数の単位に自動で直される。必ず get_product_detail などで確かめた数字をもとに使う。"""
        checked, problems = [], []
        for it in items:
            it = it if isinstance(it, ProposalItem) else ProposalItem(**it)
            p = find(it.code)
            if p is None:
                problems.append(f"{it.code}：商品が見つからないため除外")
                continue
            if p.discontinued:
                problems.append(f"{p.code}：販売終了予定のため除外")
                continue
            lot = max(1, p.min_lot)
            qty = math.ceil(max(0, it.qty) / lot) * lot
            if qty == 0:
                continue
            if qty != it.qty:
                problems.append(f"{p.code}：{p.min_lot}個単位に合わせて{it.qty}→{qty}個")
            checked.append({"code": p.code, "name": p.name, "supplier": p.supplier, "qty": qty, "cost": p.cost, "reason": it.reason[:200]})
        if not checked:
            return "発注案を作れませんでした。" + "／".join(problems)
        pid = save_proposal(checked, note[:500])
        return _json({
            "結果": "発注案を「承認待ち」で保存しました。発注はまだ確定していません。担当者が承認すると確定します。",
            "提案ID": pid,
            "商品数": len(checked),
            "合計金額": yen(sum(c["qty"] * c["cost"] for c in checked)),
            "自動で直した点": problems or "なし",
        })

    return [get_overview, list_products, get_product_detail, get_upcoming_sales, simulate_plan, propose_order]
