"""データの読み込みと集計のテスト（実行方法：uv run pytest）"""

from datetime import date
from pathlib import Path

from inventory_agent.dataset import build_dataset, load_demo, monday_of, normalize_date

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_日付の書き方の違いを吸収できる():
    assert normalize_date("2026/9/3") == date(2026, 9, 3)
    assert normalize_date("2026-09-03T10:00") == date(2026, 9, 3)
    assert normalize_date("きのう") is None


def test_週の月曜日を求められる():
    assert monday_of(date(2026, 10, 4)) == date(2026, 9, 28)  # 日曜日 → その週の月曜日


def test_デモデータを読み込める():
    ds = load_demo(DATA_DIR)
    assert len(ds.weeks) == 105
    assert len(ds.products) == 30
    assert len(ds.sale_weeks) == 18


def test_対応表にないコードは勝手に数えずに要確認にする():
    ds = load_demo(DATA_DIR)
    assert ds.unknown_codes == [{"store": "Amazon", "code": "AMZ-NEW-WD", "count": 65}]


def test_Amazonのコードを共通の商品コードにそろえる():
    products = "商品コード,商品名,AmazonのSKU\nK005,スパチュラ,AMZ-SPAT-01\n"
    amazon = "purchase-date,sku,quantity\n2026-09-01,AMZ-SPAT-01,3\n"
    shop = "日付,商品コード,数量\n2026-09-02,K005,2\n"
    ds = build_dataset([("Amazon", amazon), ("自社サイト", shop)], products, "商品コード,在庫数\n")
    assert ds.products[0].weekly == [5]  # 2つの店舗の売上が、同じ商品の同じ週に合計される
