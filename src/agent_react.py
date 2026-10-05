"""
agent_react.py — Mẫu 1: ReAct (SE373 · BTVN#3)

Ý TƯỞNG
    ReAct = Suy luận → Hành động → Quan sát → suy luận tiếp. Mỗi vòng model nhìn
    toàn bộ lịch sử rồi chọn bước kế tiếp; không có kế hoạch định sẵn.
    Dùng create_agent của LangChain 1.x. Harness gắn vào qua MỘT middleware:

        before_model   trước mỗi lần gọi model: harness đã dừng / hết ngân sách → kết thúc
        wrap_tool_call bao quanh mỗi tool call: validate → kiểm quyền → chạy → kiểm 4 điều kiện dừng
        after_model    sau mỗi lần gọi model: tính ngân sách; nếu model thôi gọi tool (tự cho là
                       xong) thì KIỂM CHỨNG bằng is_done() trước khi tin

    Model chỉ đề xuất tool call. Mọi thứ còn lại là code (middleware + harness.py).

CHẠY
    python src/agent_react.py --scenario happy
    python src/agent_react.py --scenario timeout
    python src/agent_react.py --scenario approval
    python src/agent_react.py --scenario env_change
    python src/agent_react.py --scenario happy --style hallucinate     # S5: bịa
    python src/agent_react.py --scenario happy --model real            # cần SE373_MODEL + API key
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from common import HARNESS_SOURCE, STYLES, RunResult, make_model, print_result, render_trace
from harness import Budget, Constraints, Harness, StopReason
from tools_flight import SCENARIOS, FlightWorld, make_langchain_tools

SYSTEM_PROMPT = (
    "Bạn là trợ lý đặt vé máy bay. Chỉ dùng tool để lấy thông tin, kể cả mã chuyến, giá, ghế "
    "và mã đặt chỗ; không tự bịa. Mỗi lượt gọi MỘT tool rồi đọc kết quả trước khi làm tiếp. "
    "Nếu tool báo lỗi hoặc từ chối, đọc hint rồi đổi hướng. Nếu cần người duyệt thì dừng lại."
)


def _text(message: Any) -> str:
    content = message.content
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _parse_observation(message: Any) -> dict[str, Any]:
    """ToolMessage.content là chuỗi JSON do tool trả về. Tool ném lỗi thì không phải JSON."""
    try:
        obs = json.loads(_text(message))
        return obs if isinstance(obs, dict) else {"status": "error", "error": "unexpected_output"}
    except json.JSONDecodeError:
        return {"status": "error", "error": "tool_exception", "hint": _text(message)[:200]}


class HarnessMiddleware(AgentMiddleware):
    """Cầu nối giữa vòng lặp ReAct của LangChain và Harness (harness.py)."""

    def __init__(self, harness: Harness) -> None:
        super().__init__()
        self.h = harness
        self._lock = threading.Lock()  # ToolNode có thể chạy nhiều tool call song song; harness giữ trạng thái

    def _closing(self) -> dict[str, Any]:
        """Kết thúc vòng lặp bằng một tin nhắn do harness viết: tóm tắt đã kiểm chứng, hoặc bản bàn giao."""
        h = self.h
        if h.stop_reason is StopReason.GOAL:
            text = h.summary()  # dựng từ dữ liệu thật nên không thể bịa
        else:
            text = h.handoff.render() if h.handoff else ""
        return {"jump_to": "end", "messages": [AIMessage(content=text, additional_kwargs=HARNESS_SOURCE)]}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self.h.stop_reason is None:
            exceeded = self.h.budget_exceeded()
            if exceeded:
                self.h.finish(StopReason.BUDGET, exceeded)
        return self._closing() if self.h.stop_reason is not None else None

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1]
        usage = getattr(last, "usage_metadata", None) or {}
        self.h.charge_model_call(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        if getattr(last, "tool_calls", None):
            return None  # còn hành động: để wrap_tool_call xử lý

        # Model thôi gọi tool = nó tự cho rằng đã xong. Kiểu dừng này trông giống thành công
        # nên nguy hiểm nhất khi sai: chỉ tin khi is_done() đúng.
        verdict = self.h.verify_final(_text(last))
        if verdict.accept:
            return None
        if verdict.stop is not None:
            return self._closing()
        feedback = HumanMessage(content=verdict.feedback, additional_kwargs=HARNESS_SOURCE)
        return {"jump_to": "model", "messages": [feedback]}

    def wrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        call = request.tool_call
        name, args, call_id = call["name"], call["args"], call["id"]

        def reply(obs: dict[str, Any]) -> ToolMessage:
            return ToolMessage(content=json.dumps(obs, ensure_ascii=False), tool_call_id=call_id, name=name)

        with self._lock:
            if self.h.stop_reason is not None:  # đã dừng (vd lời gọi song song sau khi chờ duyệt)
                return reply({"status": "skipped", "hint": "Phiên đã dừng, không thực thi thêm."})
            out = self.h.run_tool_call(name, args, lambda: self._execute(handler, request))
            return reply(out.observation)

    @staticmethod
    def _execute(handler: Any, request: Any) -> dict[str, Any]:
        try:
            return _parse_observation(handler(request))
        except Exception as exc:  # tool ném lỗi: biến thành observation có cấu trúc, không làm sập vòng lặp
            return {"status": "error", "error": "tool_exception",
                    "hint": f"{type(exc).__name__}: {str(exc)[:160]}"}


def run_react(scenario: str = "happy", *, model: Any = "fake", style: str = "competent",
              budget: Budget | None = None, constraints: Constraints | None = None,
              recursion_limit: int = 100) -> RunResult:
    """Chạy MỘT lần agent ReAct trong MỘT kịch bản và trả về RunResult.

    model: "fake" (model giả), "real" (đọc SE373_MODEL), hoặc một đối tượng model.
    recursion_limit: lưới an toàn cứng của LangGraph, đặt cao hơn ngân sách của harness để
    harness luôn dừng trước và còn chẩn đoán được."""
    constraints = constraints or Constraints()
    world = FlightWorld(scenario)
    harness = Harness(world, constraints, budget=budget)
    agent = create_agent(
        model=make_model(model, style, constraints),
        tools=make_langchain_tools(world),
        system_prompt=f"{SYSTEM_PROMPT} {constraints.to_prompt()}",
        middleware=[HarnessMiddleware(harness)],
    )

    inputs = {"messages": [{"role": "user", "content": constraints.request_text()}]}
    state: dict[str, Any] = {"messages": []}
    try:
        # stream để giữ lại trạng thái cuối ngay cả khi framework tự cắt (GraphRecursionError)
        for state in agent.stream(inputs, config={"recursion_limit": recursion_limit}, stream_mode="values"):
            pass
    except GraphRecursionError:
        harness.finish(StopReason.BUDGET, "GraphRecursionError (lưới an toàn của LangGraph)")

    messages = state["messages"]
    answers = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
    return RunResult(
        pattern="react", scenario=scenario, report=harness.report(),
        final_answer=_text(answers[-1]) if answers else "",
        handoff=harness.handoff.render() if harness.handoff and harness.stop_reason is not StopReason.GOAL else "",
        trace=render_trace(messages), messages=messages, harness=harness, world=world,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # in tiếng Việt đúng trên console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chạy agent ReAct đặt vé")
    parser.add_argument("--scenario", choices=SCENARIOS, default="happy")
    parser.add_argument("--style", choices=STYLES, default="competent", help="kiểu ứng xử của model giả")
    parser.add_argument("--model", choices=("fake", "real"), default="fake")
    args = parser.parse_args()
    try:
        result = run_react(args.scenario, model=args.model, style=args.style)
    except RuntimeError as exc:  # ví dụ: --model real mà chưa đặt SE373_MODEL
        parser.error(str(exc))
    print_result(result)
