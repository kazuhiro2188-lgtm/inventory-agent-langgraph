"""動作確認用：デモデータを読み込んで、予測とお金のインパクトを表示する

実行方法：uv run python -m inventory_agent.main
"""

from collections import Counter
from pathlib import Path

from inventory_agent.dataset import load_demo
from inventory_agent.forecast import METHOD_LABEL, forecast_product, overall_wape
from inventory_agent.ordering import plan_order, summarize, yen

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    ds = load_demo(DATA_DIR)
    print(f"期間：{ds.weeks[0]} 〜 {ds.weeks[-1]}（{len(ds.weeks)}週）／商品数：{len(ds.products)}種類")

    # 予測の答え合わせ
    w = overall_wape(ds.products, ds.sale_weeks)
    counts = Counter(forecast_product(p, ds.sale_weeks).method for p in ds.products)
    print(f"\n予測の答え合わせ（直近12週）：誤差 {w * 100:.1f}%")
    for m, label in METHOD_LABEL.items():
        print(f"  {label}：{counts[m]}商品")

    # お金のインパクト
    s = summarize([plan_order(p, ds.sale_weeks) for p in ds.products])
    print("\nお金のインパクト")
    print(f"  次の4週の売り逃し：{yen(s.lost4_value)}（今日発注すれば {yen(s.savable_value)} 防げる）")
    print(f"  届く前に在庫が切れる：{yen(s.late_value)}")
    print(f"  眠っている在庫：{yen(s.excess_value)}")
    print(f"  推奨どおり発注した場合の発注総額：{yen(s.order_total)}")
    print("\n届く前に在庫が切れる商品：")
    for l in s.late_items:
        print(f"  {l.product.name}：{l.stockout_week}週目に切れる／届くのは{l.lead_weeks}週後")


if __name__ == "__main__":
    main()
