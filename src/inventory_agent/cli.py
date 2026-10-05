"""ターミナルからエージェントを使う

  uv run python -m inventory_agent.cli ask "今の状況をひと言で教えて"
  uv run python -m inventory_agent.cli proposals
  uv run python -m inventory_agent.cli approve <提案ID>
  uv run python -m inventory_agent.cli reject <提案ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from inventory_agent.dataset import load_demo
from inventory_agent.proposals import ProposalStore
from inventory_agent.tools import TOOL_LABEL, create_tools

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="在庫発注のAIエージェント")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="エージェントに依頼する")
    ask.add_argument("question")
    sub.add_parser("proposals", help="発注案の一覧を見る")
    for name in ("approve", "reject"):
        p = sub.add_parser(name, help="発注案を承認／却下する（人だけが使える）")
        p.add_argument("id")
    args = parser.parse_args()

    store = ProposalStore(ROOT / "proposals.json")

    if args.command == "ask":
        from inventory_agent.agent import run_agent

        tools = create_tools(load_demo(ROOT / "data"), store.add_pending)
        print("エージェントが道具を使って調べています…\n")
        answer, steps = run_agent(tools, args.question)
        print("【考えた過程】")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {TOOL_LABEL.get(s.tool, s.tool)}  {json.dumps(s.input, ensure_ascii=False)}")
        print("\n【答え】")
        print(answer)

    elif args.command == "proposals":
        items = store.list()
        if not items:
            print("発注案はまだありません")
        for p in items:
            total = sum(i["qty"] * i["cost"] for i in p["items"])
            print(f"[{p['id']}] {p['status']}  {p['created_at']}  {len(p['items'])}商品  合計{total:,}円")
            for i in p["items"]:
                print(f"    {i['code']} {i['name']}（{i['supplier']}）{i['qty']}個：{i['reason']}")

    else:
        status = "approved" if args.command == "approve" else "rejected"
        ok = store.decide(args.id, status)
        print("更新しました" if ok else "承認待ちの発注案が見つかりません")


if __name__ == "__main__":
    main()
