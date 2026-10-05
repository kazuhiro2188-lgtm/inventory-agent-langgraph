"""予測と発注のテスト（実行方法：uv run pytest）"""

from pathlib import Path

import pytest

from inventory_agent.dataset import Product, load_demo
from inventory_agent.forecast import clean_sales, forecast_product, is_seasonal_item, js_round
from inventory_agent.ordering import plan_order, summarize

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return load_demo(DATA_DIR)


def find(ds, code):
    return next(p for p in ds.products if p.code == code)


def test_四捨五入はTypeScriptと同じ():
    assert js_round(2.5) == 3  # Pythonの round(2.5) は 2 になるので、専用の関数を使う
    assert js_round(-0.4) == 0


def test_セールの週は前後の平均に置き換える():
    assert clean_sales([10, 50, 20], [1]) == [10, 15, 20]


def test_冬物は季節品と判定され_直近の平均は選ばれない(ds):
    donabe = find(ds, "K003")
    r = forecast_product(donabe, ds.sale_weeks)
    assert r.method != "moving"
    assert sum(r.forecast[8:12]) > sum(r.forecast[0:4])  # 冬に向けて予測が増えていく


def test_急に売れ始めた商品は季節品と取り違えない(ds):
    from inventory_agent.forecast import clean_sales as cs

    assert not is_seasonal_item(cs(find(ds, "K026").weekly, ds.sale_weeks))


def test_販売終了予定は発注しない(ds):
    line = plan_order(find(ds, "K025"), ds.sale_weeks)
    assert line.order_qty == 0
    assert line.excess_value > 0  # 在庫は眠っているお金として数える


def test_発注数は最低発注数の単位に切り上げる(ds):
    for p in ds.products:
        line = plan_order(p, ds.sale_weeks)
        assert line.order_qty % max(1, p.min_lot) == 0


def test_お金のインパクトがTypeScript版と同じ(ds):
    s = summarize([plan_order(p, ds.sale_weeks) for p in ds.products])
    assert s.lost4_value == 721_540
    assert s.late_value == 815_650
    assert s.excess_value == 268_140
    assert s.order_total == 2_298_300


def test_届くのが遅れると_間に合わない欠品が増える(ds):
    overseas = [p for p in ds.products if p.supplier == "海外C"]
    before = summarize([plan_order(p, ds.sale_weeks) for p in overseas]).late_value
    after = summarize([plan_order(p, ds.sale_weeks, extra_lead_days=14) for p in overseas]).late_value
    assert after > before
