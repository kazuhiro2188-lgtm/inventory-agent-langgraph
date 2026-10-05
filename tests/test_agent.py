"""道具とエージェントのつなぎのテスト（AIは呼ばない。台本どおりに動く「練習用のAI」を使う）"""

import json
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from inventory_agent.agent import run_agent
from inventory_agent.dataset import load_demo
from inventory_agent.proposals import ProposalStore
from inventory_agent.tools import create_tools

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return load_demo(DATA_DIR)


def tools_for(ds, store=None):
    saved = []
    save = store.add_pending if store else (lambda items, note: saved.append(items) or "test")
    return {t.name: t for t in create_tools(ds, save)}, saved


def test_商品名でも探せる(ds):
    t, _ = tools_for(ds)
    out = json.loads(t["get_product_detail"].invoke({"query": "麦茶ポット"}))
    assert out["コード"] == "K001"


def test_あいまいな名前は候補を返す(ds):
    t, _ = tools_for(ds)
    out = json.loads(t["get_product_detail"].invoke({"query": "ポット"}))
    assert len(out["候補"]) == 2


def test_試算は増える額を先に計算して返す(ds):
    t, _ = tools_for(ds)
    out = json.loads(t["simulate_plan"].invoke({"extra_lead_days": 14, "supplier": "海外C"}))
    assert out["増える額"]["間に合わない欠品"] == "1,183,050円"


def test_発注案は販売終了品を除き_最低発注数に直して_承認待ちで保存する(ds, tmp_path):
    store = ProposalStore(tmp_path / "proposals.json")
    t, _ = tools_for(ds, store)
    out = t["propose_order"].invoke({
        "items": [{"code": "K001", "qty": 105, "reason": "テスト"}, {"code": "K025", "qty": 10, "reason": "終了予定"}],
        "note": "テスト",
    })
    saved = store.list()
    assert len(saved) == 1 and saved[0]["status"] == "pending"
    assert saved[0]["items"] == [{"code": "K001", "name": "麦茶ポット 1.5L", "supplier": "国内A", "qty": 110, "cost": 1090, "reason": "テスト"}]
    assert "K025：販売終了予定のため除外" in out


class ScriptedModel(BaseChatModel):
    """台本どおりに返事をする「練習用のAI」。1回目は道具を使う指示、2回目は答え"""

    script: list[AIMessage]
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_考える_道具を使う_答える_の流れがつながっている(ds):
    model = ScriptedModel(script=[
        AIMessage(content="", tool_calls=[{"name": "get_overview", "args": {}, "id": "call_1"}]),
        AIMessage(content="一番まずいのは真空断熱タンブラーです。根拠：get_overview"),
    ])
    t, _ = tools_for(ds)
    answer, steps = run_agent(list(t.values()), "今の状況は？", model=model)
    assert [s.tool for s in steps] == ["get_overview"]
    assert "721,540円" in steps[0].output  # 道具の結果がエージェントに渡っている
    assert answer.startswith("一番まずいのは")
