"""LangGraphで組んだエージェント（TypeScript版の lib/agent/graph.ts と同じ形）

「考える（AI）」→「道具を使う（コード）」を、答えが出るまで繰り返す。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

MAX_STEPS = 16  # 考える・道具を使う、の繰り返しの上限

SYSTEM_PROMPT = "\n".join([
    "あなたはキッチン雑貨のネットショップの在庫・発注を担当するアシスタントです。",
    "担当者からの依頼に、用意された道具を使って調べ、分かりやすく答えてください。",
    "",
    "【守ること】",
    "- 数字（個数・金額・週・割合）は、必ず道具が返した値だけを使う。自分で予測したり、推測で数字を作ったりしない。",
    "- 合計・差・割合も自分で計算しない。道具の結果に書かれている合計や差をそのまま使う。書かれていなければ、数字を出さずに言葉で説明する。",
    "- 1つの商品について聞かれたら、get_product_detail でその商品を調べてから答える。商品コードが分からなくても、商品名でそのまま調べられる。担当者に商品コードを聞き返さない。",
    "- 販売終了予定の商品は、発注しない・発注を勧めない。発注を頼まれたら、販売終了予定であることを伝えて断る。",
    "- 調べていない商品について、断定しない。必要なら道具で確認してから答える。",
    "- 発注を確定する権限はない。発注案は propose_order で「承認待ち」として保存し、最後は担当者が承認すると伝える。",
    "- 担当者が発注案を求めていないときは、propose_order を使わない。",
    "- 担当者が発注案を求めたときは、確認を取らずに propose_order で「承認待ち」として保存する。承認待ちなので確定にはならない。",
    "- 「確かさ：低」の商品や、販売終了予定の商品は、そのことを必ず添える。",
    "- 依頼があいまいなときは、何を前提にしたかを書く。",
    "",
    "【答え方】",
    "- 日本語で、結論を先に、短く。",
    "- 最後に「根拠」として、どの道具で確かめたかを1行で添える。",
])


@dataclass
class Step:
    tool: str
    input: dict
    output: str


def default_model() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001"), temperature=0, max_tokens=1500)


def build_agent(tools: list[BaseTool], model: BaseChatModel | None = None):
    llm = (model or default_model()).bind_tools(tools)

    def think(state: MessagesState):
        """考える：今までのやりとりを見て、次に使う道具を決める（または答える）"""
        reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def next_step(state: MessagesState):
        """次の行き先：道具を使う指示があれば道具へ、なければ終わり"""
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("think", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", next_step, ["tools", END])
    graph.add_edge("tools", "think")
    return graph.compile()


def run_agent(tools: list[BaseTool], question: str, model: BaseChatModel | None = None) -> tuple[str, list[Step]]:
    """エージェントに依頼して、答えと「考えた過程（使った道具）」を返す"""
    agent = build_agent(tools, model)
    result = agent.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": MAX_STEPS})
    messages = result["messages"]

    outputs = {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}
    steps = [
        Step(tool=c["name"], input=c["args"], output=outputs.get(c["id"], "")[:2000])
        for m in messages
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]
    last = messages[-1].content
    answer = last if isinstance(last, str) else "".join(p.get("text", "") for p in last if isinstance(p, dict)).strip()
    return answer, steps
