"""データの読み込みと集計（TypeScript版の lib/forecast.ts と同じことをする）

3店舗の売上CSVを、共通の商品コードにそろえて、週ごとに合計する。
AIは使わない。計算だけの部品。
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# 店舗ごとに列の名前が違っても読めるように、候補を並べておく
DATE_COLS = ["注文日", "日付", "purchase-date", "date", "購入日"]
CODE_COLS = ["商品コード", "商品管理番号", "sku", "seller-sku", "品番"]
QTY_COLS = ["数量", "個数", "quantity", "qty"]


@dataclass
class Product:
    """1つの商品の情報と、週ごとの販売数"""

    code: str
    name: str
    supplier: str = ""
    lead_days: int = 14  # 発注してから届くまでの日数
    min_lot: int = 1  # 最低発注数（この単位で発注する）
    large: bool = False  # 大型商品（保管料が高い）
    discontinued: bool = False  # 販売終了予定
    price: int = 0  # 販売価格（円）
    cost: int = 0  # 仕入れ値（円）
    stock: int = 0
    on_order: int = 0
    weekly: list[int] = field(default_factory=list)  # 週ごとの販売数（3店舗の合計）
    first_sale_week: int = -1  # 初めて売れた週の番号


@dataclass
class Dataset:
    weeks: list[str]  # 各週の月曜日（YYYY-MM-DD）
    sale_weeks: list[int]  # セールがあった週の番号
    products: list[Product]
    unknown_codes: list[dict]  # 商品一覧にないコード（要確認）
    stores: list[str]


# ===== CSVを読む =====


def read_csv(text: str) -> list[dict[str, str]]:
    """CSVの文字列を、1行ずつの辞書のリストにする（先頭の見えない印 BOM も取り除く）"""
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))


def find_col(header: list[str], candidates: list[str]) -> str | None:
    """見出しの中から、候補のどれかに当たる列名を返す"""
    lower = {h.strip().lower(): h for h in header}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def normalize_date(s: str) -> date | None:
    """2026/9/3、2026-09-03、2026-09-03T10:00 などを日付にする"""
    s = s.strip().replace("/", "-")[:10]
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return date(y, m, d)
    except ValueError:
        return None


def monday_of(d: date) -> date:
    """その日を含む週の月曜日"""
    return d - timedelta(days=d.weekday())


# ===== 3店舗の売上を、共通の商品コードで週ごとに合計する =====


def build_dataset(
    sales: list[tuple[str, str]],  # [(店舗名, CSVの文字列), ...]
    products_text: str,
    inventory_text: str,
    calendar_text: str = "",
) -> Dataset:
    # 商品一覧
    products: dict[str, Product] = {}
    alias: dict[str, str] = {}  # 店舗のコード → 共通の商品コード
    for r in read_csv(products_text):
        code = r.get("商品コード", "").strip()
        if not code:
            continue
        products[code] = Product(
            code=code,
            name=r.get("商品名", "").strip(),
            supplier=r.get("仕入先", "").strip(),
            lead_days=int(r.get("届くまでの日数") or 14),
            min_lot=int(r.get("最低発注数") or 1),
            large=bool(r.get("大型", "").strip()),
            discontinued=bool(r.get("販売終了予定", "").strip()),
            price=int(r.get("販売価格") or 0),
            cost=int(r.get("仕入れ値") or 0),
        )
        alias[code] = code
        amazon = r.get("AmazonのSKU", "").strip()
        if amazon:
            alias[amazon] = code

    # 売上を共通コードに直して集める。対応表にないコードは、勝手に足さずに「要確認」へ
    daily: list[tuple[date, str, int]] = []
    unknown: dict[tuple[str, str], int] = {}
    for store, text in sales:
        rows = read_csv(text)
        if not rows:
            continue
        header = list(rows[0].keys())
        dc, cc, qc = find_col(header, DATE_COLS), find_col(header, CODE_COLS), find_col(header, QTY_COLS)
        if not (dc and cc and qc):
            raise ValueError(f"{store}の売上CSVに、日付・商品コード・数量の列が見つかりません：{header}")
        for r in rows:
            d = normalize_date(r[dc])
            raw = r[cc].strip()
            if d is None or not raw:
                continue
            qty = int(r[qc])
            code = alias.get(raw)
            if code is None:
                unknown[(store, raw)] = unknown.get((store, raw), 0) + qty
                continue
            daily.append((d, code, qty))

    if not daily:
        raise ValueError("売上のデータが1件も読めませんでした")

    # 週の並びを作る（売上のない週も0として並べる）
    first = monday_of(min(d for d, _, _ in daily))
    last = monday_of(max(d for d, _, _ in daily))
    weeks: list[date] = []
    w = first
    while w <= last:
        weeks.append(w)
        w += timedelta(days=7)
    index = {w: i for i, w in enumerate(weeks)}

    for p in products.values():
        p.weekly = [0] * len(weeks)
    for d, code, qty in daily:
        products[code].weekly[index[monday_of(d)]] += qty
    for p in products.values():
        p.first_sale_week = next((i for i, q in enumerate(p.weekly) if q > 0), -1)

    # 在庫
    for r in read_csv(inventory_text):
        p = products.get(r.get("商品コード", "").strip())
        if p:
            p.stock = int(r.get("在庫数") or 0)
            p.on_order = int(r.get("発注済み数") or 0)

    # セールのあった週
    sale_weeks: set[int] = set()
    for r in read_csv(calendar_text) if calendar_text else []:
        s, e = normalize_date(r.get("開始日", "")), normalize_date(r.get("終了日", ""))
        if not (s and e):
            continue
        w = monday_of(s)
        while w <= e:
            if w in index:
                sale_weeks.add(index[w])
            w += timedelta(days=7)

    return Dataset(
        weeks=[w.isoformat() for w in weeks],
        sale_weeks=sorted(sale_weeks),
        products=list(products.values()),
        unknown_codes=[{"store": s, "code": c, "count": n} for (s, c), n in unknown.items()],
        stores=[s for s, _ in sales],
    )


def load_demo(data_dir: Path) -> Dataset:
    """data フォルダのデモデータを読み込む"""
    read = lambda name: (data_dir / name).read_text(encoding="utf-8")  # noqa: E731
    return build_dataset(
        sales=[
            ("楽天", read("rakuten_sales.csv")),
            ("Amazon", read("amazon_sales.csv")),
            ("自社サイト", read("shop_sales.csv")),
        ],
        products_text=read("products.csv"),
        inventory_text=read("inventory.csv"),
        calendar_text=read("sale_calendar.csv"),
    )
