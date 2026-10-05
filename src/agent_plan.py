"""
agent_plan.py — Mẫu 2: Plan-then-Execute bằng LangGraph (SE373 · BTVN#3)

Ý TƯỞNG (đúng định nghĩa trong slide)
    Gọi model MỘT lần để sinh trọn kế hoạch, rồi thực thi từng bước theo kế hoạch đó.
    Ưu thế quyết định: kế hoạch NHÌN THẤY ĐƯỢC trước khi chạy, nên kiểm được bằng code,
    duyệt được và ước lượng chi phí được. Cái giá: kế hoạch cố định, tình huống đổi thì lỗi thời.

ĐỒ THỊ LANGGRAPH
    START → planner → approve → executor ⟲ executor → verify → finish → END
                 ↑                   │
                 └──── (replan) ─────┘     chỉ khi max_replans > 0 (mẫu Lai)

    planner   1 lần gọi model → JSON kế hoạch → kiểm tra bằng code (lint_plan)
    approve   NGƯỜI duyệt kế hoạch trước khi chạy: dùng interrupt() của LangGraph
    executor  chạy MỘT bước mỗi lần, mọi lời gọi tool đều đi qua Harness.run_tool_call
    verify    chạy hết kế hoạch thì kiểm tiêu chí hoàn thành bằng is_done()
    finish    tin nhắn cuối do harness viết (tóm tắt đã kiểm chứng hoặc bản bàn giao)

    Không cần dùng quyền hạn của model ở bước thực thi, nên executor là CODE (xác định):
    kế hoạch dùng hai biến thay cho giá trị chưa biết lúc lập kế hoạch
        "$best"          chuyến rẻ nhất thoả ràng buộc, chốt một lần ở lần dùng đầu tiên
        "$booking_code"  mã đặt chỗ do book_seat trả về
    Một executor là model nhỏ (như slide gợi ý) là hướng mở rộng; nó làm tăng số lần gọi model.

LỆCH KỲ VỌNG VÀ THAY KẾ HOẠCH
    Mỗi bước ghi "expect" (trạng thái observation kỳ vọng). Nhận khác kỳ vọng:
        status == "error"  → thử lại y nguyên (tối đa max_retries); harness vẫn bắt lặp nếu giống hệt
        khác               → LỆCH KỲ VỌNG, "kế hoạch lỗi thời":
                               max_replans = 0 (Plan thuần): dừng ở kết D, kế hoạch không có nhánh dự phòng
                               max_replans > 0 (mẫu Lai)   : loại chuyến hỏng rồi lập kế hoạch MỚI

CHẠY
    python src/agent_plan.py --scenario happy
    python src/agent_plan.py --scenario env_change               # Plan thuần: kế hoạch lỗi thời (kết D)
    python src/agent_plan.py --scenario env_change --replans 1   # có replan: thích nghi (mẫu Lai)
    python src/agent_plan.py --scenario happy --reject "quá đắt" # người từ chối kế hoạch
    python src/agent_plan.py --scenario happy --planner-style wrong_date   # kế hoạch sai bị chặn
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

from common import HARNESS_SOURCE, RunResult, closing_text, message_text, print_result, render_trace
from harness import Budget, Constraints, Harness, StopReason, estimate_tokens, fmt_call
from tools_flight import SCENARIOS, SIDE_EFFECT_TOOLS, FlightWorld, make_langchain_tools

PLACEHOLDERS = ("$best", "$booking_code")
PLANNER_STYLES = ("canonical", "wrong_date", "invalid_tool", "no_pay")

PLANNER_PROMPT = (
    "Bạn là bộ lập kế hoạch đặt vé máy bay. Chỉ trả về MỘT đối tượng JSON, không giải thích thêm:\n"
    '{"steps": [{"id": 1, "tool": "<tên tool>", "args": {...}, "why": "<lý do ngắn>", "expect": "ok"}]}\n'
    "Tool được dùng: search_flights(origin, dest, date), check_seat(flight), book_seat(flight, seat), "
    "pay(code, method), get_booking(code).\n"
    "Lúc lập kế hoạch chưa biết kết quả tool, nên dùng biến thay cho giá trị chưa biết:\n"
    '  "$best"         = chuyến rẻ nhất thoả ràng buộc trong kết quả search_flights\n'
    '  "$booking_code" = mã đặt chỗ do book_seat trả về\n'
    'Kế hoạch phải kết thúc bằng thanh toán. Dùng seat "12A" và method "corp_card".'
)


# --------------------------------------------------------------------------- #
# Kế hoạch là DỮ LIỆU
# --------------------------------------------------------------------------- #
class PlanStep(BaseModel):
    id: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = ""
    expect: str = "ok"  # trạng thái observation kỳ vọng; khác thì là LỆCH KỲ VỌNG


class Plan(BaseModel):
    steps: list[PlanStep]

    def render(self) -> str:
        return "\n".join(f"{s.id}. {fmt_call(s.tool, s.args)} · kỳ vọng {s.expect} · {s.why}" for s in self.steps)

    def estimate(self) -> dict[str, Any]:
        """Ước lượng chi phí TRƯỚC khi chạy: ưu thế của việc kế hoạch nhìn thấy được."""
        return {"steps": len(self.steps), "tool_calls": len(self.steps),
                "irreversible_steps": [s.id for s in self.steps if s.tool in SIDE_EFFECT_TOOLS]}


class PlanError(ValueError):
    """Model trả về kế hoạch không đọc được."""


def parse_plan(text: str) -> Plan:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise PlanError("không tìm thấy JSON kế hoạch trong câu trả lời")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise PlanError(f"JSON không hợp lệ: {exc.msg}") from exc
    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        raise PlanError(f"kế hoạch sai schema ({exc.error_count()} lỗi)") from exc


def lint_plan(plan: Plan, harness: Harness, max_steps: int = 8) -> list[str]:
    """Kiểm kế hoạch bằng CODE trước khi chạy (sensor computational): tên tool, tham số, ràng buộc,
    thứ tự phụ thuộc, và có đủ bước để đạt tiêu chí hoàn thành. Trả về danh sách vấn đề (rỗng = hợp lệ)."""
    steps = plan.steps
    if not steps:
        return ["kế hoạch rỗng"]
    problems: list[str] = []
    if len(steps) > max_steps:
        problems.append(f"kế hoạch dài {len(steps)} bước, tối đa {max_steps}")

    c = harness.constraints
    seen: set[str] = set()  # tool đã có kết quả (từ lần chạy trước) hoặc đã nằm trong kế hoạch
    if harness.flights:
        seen.add("search_flights")
    if harness.bookings:
        seen.add("book_seat")
    for s in steps:
        error = harness.validate_tool_call(s.tool, s.args)
        if error:
            problems.append(f"bước {s.id}: {error}")
            continue
        for value in s.args.values():
            if not (isinstance(value, str) and value.startswith("$")):
                continue
            if value not in PLACEHOLDERS:
                problems.append(f"bước {s.id}: biến {value} không hợp lệ (chỉ có {', '.join(PLACEHOLDERS)})")
            elif value == "$best" and "search_flights" not in seen:
                problems.append(f"bước {s.id}: dùng $best khi chưa có kết quả search_flights")
            elif value == "$booking_code" and "book_seat" not in seen:
                problems.append(f"bước {s.id}: dùng $booking_code khi chưa có bước book_seat")
        if s.tool == "search_flights":  # ràng buộc là dữ liệu: kế hoạch không được tìm sai tuyến/ngày
            for key, want in (("origin", c.origin), ("dest", c.dest), ("date", c.date)):
                if str(s.args.get(key, "")).upper() != want.upper():
                    problems.append(f"bước {s.id}: {key}={s.args.get(key)!r} khác ràng buộc {want}")
        seen.add(s.tool)

    tools = {s.tool for s in steps}
    if "pay" not in tools:
        problems.append("kế hoạch thiếu bước pay nên không thể đạt tiêu chí hoàn thành")
    if "book_seat" not in tools and not harness.bookings:
        problems.append("kế hoạch thiếu bước book_seat")
    return problems


# --------------------------------------------------------------------------- #
# Planner giả (tất định) và cách chọn planner
# --------------------------------------------------------------------------- #
class ScriptedPlanner(BaseChatModel):
    """Planner giả: trả về kế hoạch JSON. Có kiểu cố ý sai để kiểm lớp kiểm tra kế hoạch:
        canonical     kế hoạch đúng
        wrong_date    tìm sai ngày (vi phạm ràng buộc)
        invalid_tool  dùng tool không tồn tại (cancel_flight)
        no_pay        thiếu bước thanh toán
    Khi nhận yêu cầu thay kế hoạch (có "[REPLAN]") thì bỏ bước đã làm: không tìm lại nếu đã có kết quả,
    chỉ thanh toán nếu đã giữ chỗ."""

    style: str = "canonical"
    constraints: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted-planner"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        c: Constraints = self.constraints or Constraints()
        text = message_text(messages[-1])
        date = "2026-10-08" if self.style == "wrong_date" else c.date
        search = ("search_flights", {"origin": c.origin, "dest": c.dest, "date": date},
                  f"Tìm các chuyến {c.origin}→{c.dest} ngày {date}")
        check = ("check_seat", {"flight": "$best"}, "Kiểm tra ghế còn của chuyến rẻ nhất thoả ràng buộc")
        if self.style == "invalid_tool":
            check = ("cancel_flight", {"code": "X"}, "Huỷ chuyến cũ")
        book = ("book_seat", {"flight": "$best", "seat": "12A"}, "Giữ chỗ ghế 12A")
        pay = ("pay", {"code": "$booking_code", "method": "corp_card"}, "Thanh toán đặt chỗ vừa giữ")

        if "[REPLAN]" in text:
            seq = [pay] if "Đã giữ chỗ" in text else [check, book, pay]
        else:
            seq = [search, check, book, pay]
        if self.style == "no_pay":
            seq = [s for s in seq if s[0] != "pay"]
        steps = [{"id": i, "tool": t, "args": a, "why": w, "expect": "ok"} for i, (t, a, w) in enumerate(seq, 1)]

        content = json.dumps({"steps": steps}, ensure_ascii=False)
        ai = AIMessage(content=content)
        n_in = estimate_tokens("\n".join(message_text(m) for m in messages))
        n_out = estimate_tokens(content)
        ai.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        return ChatResult(generations=[ChatGeneration(message=ai)])


def make_planner_model(kind: Any = "fake", style: str = "canonical", constraints: Constraints | None = None) -> Any:
    """'fake' → ScriptedPlanner. 'real' → model thật từ biến môi trường SE373_MODEL.
    Truyền thẳng một đối tượng model thì trả lại nguyên (dùng khi kiểm thử)."""
    if not isinstance(kind, str):
        return kind
    if kind == "fake":
        if style not in PLANNER_STYLES:
            raise ValueError(f"planner_style phải thuộc {PLANNER_STYLES}, nhận được {style!r}")
        return ScriptedPlanner(style=style, constraints=constraints or Constraints())
    if kind == "real":
        name = os.environ.get("SE373_MODEL")
        if not name:
            raise RuntimeError("Chưa đặt biến môi trường SE373_MODEL (ví dụ 'anthropic:claude-sonnet-4-5').")
        from langchain.chat_models import init_chat_model
        return init_chat_model(name)
    raise ValueError(f"model phải là 'fake' hoặc 'real', nhận được {kind!r}")


# --------------------------------------------------------------------------- #
# Người duyệt kế hoạch
# --------------------------------------------------------------------------- #
Approver = Callable[[dict[str, Any]], "bool | str"]  # True: đồng ý · False hoặc chuỗi lý do: từ chối


def approve_all(payload: dict[str, Any]) -> bool:
    return True


# --------------------------------------------------------------------------- #
# Đồ thị
# --------------------------------------------------------------------------- #
class PlanState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: dict[str, Any]      # Plan.model_dump(): giữ dạng dict cho dễ lưu checkpoint
    cursor: int               # chỉ số bước kế tiếp
    retries: int              # số lần đã thử lại bước hiện tại
    replans: int
    bindings: dict[str, str]  # giá trị đã chốt của "$best"
    excluded: list[str]       # chuyến đã biết không dùng được
    deviation: str            # lệch kỳ vọng đang chờ planner xử lý


class Unresolvable(Exception):
    """Không điền được biến trong kế hoạch (ví dụ không còn chuyến nào thoả ràng buộc)."""


class PlanExecuteAgent:
    def __init__(self, harness: Harness, tools: list[Any], planner: Any, *, request: str,
                 max_replans: int = 0, max_retries: int = 2, approver: Approver = approve_all) -> None:
        self.h = harness
        self.tools = {t.name: t for t in tools}
        self.planner = planner
        self.request = request
        self.max_replans, self.max_retries, self.approver = max_replans, max_retries, approver
        self.prompt = f"{PLANNER_PROMPT} {harness.constraints.to_prompt()}"

    def build(self) -> Any:
        g = StateGraph(PlanState)
        for name, node in (("planner", self.planner_node), ("approve", self.approve_node),
                           ("executor", self.executor_node), ("verify", self.verify_node),
                           ("finish", self.finish_node)):
            g.add_node(name, node)
        g.add_edge(START, "planner")
        g.add_conditional_edges("planner", self.route_after_planner, ["approve", "executor", "finish"])
        g.add_conditional_edges("approve", self.route_after_approve, ["executor", "finish"])
        g.add_conditional_edges("executor", self.route_after_executor, ["executor", "planner", "verify", "finish"])
        g.add_edge("verify", "finish")
        g.add_edge("finish", END)
        return g.compile(checkpointer=InMemorySaver())  # interrupt() cần checkpointer

    # ---- định tuyến --------------------------------------------------------- #
    def _stopped(self) -> bool:
        return self.h.stop_reason is not None

    def route_after_planner(self, state: PlanState) -> str:
        if self._stopped():
            return "finish"
        return "approve" if state.get("replans", 0) == 0 else "executor"  # chỉ duyệt kế hoạch gốc

    def route_after_approve(self, state: PlanState) -> str:
        return "finish" if self._stopped() else "executor"

    def route_after_executor(self, state: PlanState) -> str:
        if self._stopped():
            return "finish"
        if state.get("deviation"):
            return "planner"
        return "verify" if state["cursor"] >= len(state["plan"]["steps"]) else "executor"

    # ---- node: planner ------------------------------------------------------ #
    def planner_node(self, state: PlanState) -> dict[str, Any]:
        replanning = bool(state.get("deviation"))
        # Ngân sách đã được run_tool_call kiểm sau mỗi tool call; tới được đây nghĩa là còn ngân sách để gọi model.
        messages = [SystemMessage(content=self.prompt),
                    HumanMessage(content=self._planner_request(state, replanning))]
        ai = self.planner.invoke(messages)
        usage = getattr(ai, "usage_metadata", None) or {}
        self.h.charge_model_call(
            usage.get("input_tokens") or estimate_tokens("\n".join(message_text(m) for m in messages)),
            usage.get("output_tokens") or estimate_tokens(message_text(ai)))

        source = {"source": "planner", "replan": replanning}
        try:
            plan = parse_plan(message_text(ai))
        except PlanError as exc:
            self.h.finish(StopReason.STALL, f"planner không sinh được kế hoạch hợp lệ: {exc}",
                          question="Planner trả về kế hoạch không đọc được. Nên chạy lại hay chỉnh prompt/model?")
            return {"messages": [AIMessage(content=message_text(ai)[:400], additional_kwargs=source)]}

        update: dict[str, Any] = {
            "messages": [AIMessage(content=plan.render(), additional_kwargs=source)],
            "plan": plan.model_dump(), "cursor": 0, "retries": 0, "deviation": "", "bindings": {},
            "replans": state.get("replans", 0) + (1 if replanning else 0),
        }
        problems = lint_plan(plan, self.h)
        if problems:
            self.h.finish(StopReason.STALL, "kế hoạch không qua kiểm tra: " + "; ".join(problems),
                          question="Planner sinh kế hoạch sai. Nên chạy lại hay chỉnh prompt/model?")
        return update

    def _planner_request(self, state: PlanState, replanning: bool) -> str:
        if not replanning:
            return self.request
        lines = [f"[REPLAN] {state.get('deviation', '')}",
                 "Đã loại các chuyến: " + (", ".join(state.get("excluded", [])) or "không có") + "."]
        if self.h.flights:
            lines.append(f"Đã có kết quả search_flights ({len(self.h.flights)} chuyến), không cần tìm lại.")
        held = [b for b in self.h.bookings.values() if not b.paid]
        if held:
            lines.append(f"Đã giữ chỗ: {held[0].code}, chỉ còn thanh toán.")
        lines.append("Hãy lập kế hoạch MỚI cho phần còn lại.")
        return self.request + "\n" + "\n".join(lines)

    # ---- node: approve ------------------------------------------------------ #
    def approve_node(self, state: PlanState) -> dict[str, Any]:
        # LƯU Ý: khi chạy tiếp, LangGraph thực thi lại node này từ đầu. Không đặt tác dụng phụ trước interrupt().
        plan = Plan.model_validate(state["plan"])
        decision = interrupt({"plan": plan.render(), "estimate": plan.estimate()})
        if decision is True:
            return {"messages": [HumanMessage(content="Đồng ý kế hoạch.", additional_kwargs={"source": "approver"})]}
        reason = decision if isinstance(decision, str) and decision else "không nêu lý do"
        self.h.finish(StopReason.HUMAN, "người từ chối kế hoạch",
                      question=f"Người duyệt từ chối kế hoạch ({reason}). Cần chỉnh kế hoạch thế nào?")
        return {"messages": [HumanMessage(content=f"Từ chối kế hoạch: {reason}",
                                          additional_kwargs={"source": "approver"})]}

    # ---- node: executor ----------------------------------------------------- #
    def executor_node(self, state: PlanState) -> dict[str, Any]:
        plan = Plan.model_validate(state["plan"])
        i = state["cursor"]
        step = plan.steps[i]
        bindings = dict(state.get("bindings", {}))
        excluded = list(state.get("excluded", []))

        try:
            args = self._resolve(step, bindings, excluded)
        except Unresolvable as exc:
            self.h.finish(StopReason.STALL, f"không thực thi được bước {step.id}: {exc}")
            return {}

        call_id = f"call_{self.h.usage.tool_calls + 1}"
        ai = AIMessage(content=step.why, tool_calls=[{"name": step.tool, "args": args, "id": call_id, "type": "tool_call"}])
        out = self.h.run_tool_call(step.tool, args, lambda: self._invoke_tool(step.tool, args))
        tool_message = ToolMessage(content=json.dumps(out.observation, ensure_ascii=False),
                                   tool_call_id=call_id, name=step.tool)
        update: dict[str, Any] = {"messages": [ai, tool_message], "bindings": bindings, "excluded": excluded}
        if out.stop is not None:  # harness đã dừng: GOAL, HUMAN, LOOP, STALL hoặc BUDGET
            return update

        status = out.observation.get("status")
        if status == step.expect:
            return {**update, "cursor": i + 1, "retries": 0}
        retries = state.get("retries", 0)
        if status == "error" and retries < self.max_retries:
            return {**update, "retries": retries + 1}  # lỗi tạm thời: thử lại y nguyên
        return {**update, **self._deviate(state, step, args, out.observation, excluded)}

    def _resolve(self, step: PlanStep, bindings: dict[str, str], excluded: list[str]) -> dict[str, Any]:
        """Điền biến "$best" và "$booking_code" bằng dữ liệu harness đã thấy."""
        resolved: dict[str, Any] = {}
        for key, value in step.args.items():
            if value == "$best":
                if "best" not in bindings:  # chốt MỘT lần: các bước sau dùng lại, không tự đổi chuyến
                    candidates = sorted(
                        (f for f in self.h.flights.values()
                         if f.code not in excluded and not f.unavailable
                         and not self.h.constraints.flight_violations(f)),
                        key=lambda f: (f.effective_price, f.depart))
                    if not candidates:
                        raise Unresolvable("không còn chuyến nào thoả ràng buộc")
                    bindings["best"] = candidates[0].code
                resolved[key] = bindings["best"]
            elif value == "$booking_code":
                held = [b for b in self.h.bookings.values() if not b.paid]
                if not held:
                    raise Unresolvable("chưa có đặt chỗ nào để thanh toán")
                resolved[key] = held[0].code
            else:
                resolved[key] = value
        return resolved

    def _invoke_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            out = self.tools[name].invoke(args)
            return out if isinstance(out, dict) else {"status": "error", "error": "unexpected_output"}
        except Exception as exc:  # tool ném lỗi: biến thành observation có cấu trúc
            return {"status": "error", "error": "tool_exception", "hint": f"{type(exc).__name__}: {str(exc)[:160]}"}

    def _deviate(self, state: PlanState, step: PlanStep, args: dict[str, Any], obs: dict[str, Any],
                 excluded: list[str]) -> dict[str, Any]:
        """Observation khác kỳ vọng. Có replan thì chuyển sang planner, không thì dừng ở kết D."""
        status = obs.get("status")
        flight = str(args.get("flight", "")).upper()
        if flight and flight not in excluded and (
                status in ("sold_out", "not_found")
                or (status == "denied" and obs.get("reason") in ("violates_constraints", "unknown_flight"))):
            excluded.append(flight)  # nhớ chuyến không dùng được để lần chọn sau bỏ qua
        detail = f"bước {step.id} ({step.tool}) kỳ vọng {step.expect}, nhận {status}"
        replans = state.get("replans", 0)
        if replans < self.max_replans:
            return {"deviation": detail, "excluded": excluded}

        if self.max_replans == 0:
            why = f"kế hoạch lỗi thời: {detail}; kế hoạch cố định, không có nhánh dự phòng"
            question = (f"Kế hoạch đã lỗi thời ({detail}). Cho phép lập lại kế hoạch (replan), "
                        "hay người chọn chuyến khác thủ công?")
        else:
            why = f"kế hoạch lỗi thời: {detail}; đã lập lại kế hoạch {replans} lần vẫn lệch"
            question = f"Lập lại kế hoạch {replans} lần vẫn lệch ({detail}). Nên nới ràng buộc hay người xử lý thủ công?"
        self.h.finish(StopReason.STALL, why, question=question)
        return {"excluded": excluded}

    # ---- node: verify, finish ------------------------------------------------ #
    def verify_node(self, state: PlanState) -> dict[str, Any]:
        done = self.h.is_done()
        if done.done:
            self.h.finish(StopReason.GOAL, "chạy hết kế hoạch và tiêu chí hoàn thành đạt")
        else:
            self.h.finish(StopReason.STALL, "đã chạy hết kế hoạch nhưng chưa đạt: " + "; ".join(done.failures))
        return {}

    def finish_node(self, state: PlanState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=closing_text(self.h), additional_kwargs=HARNESS_SOURCE)]}


# --------------------------------------------------------------------------- #
# Chạy một lần
# --------------------------------------------------------------------------- #
def run_plan(scenario: str = "happy", *, model: Any = "fake", planner_style: str = "canonical",
             budget: Budget | None = None, constraints: Constraints | None = None,
             max_replans: int = 0, max_retries: int = 2, approver: Approver | None = None,
             recursion_limit: int = 100, pattern: str | None = None) -> RunResult:
    """Chạy MỘT lần Plan-then-Execute (max_replans=0) hoặc mẫu Lai (max_replans>0) trong MỘT kịch bản.

    model: "fake" (planner giả), "real" (đọc SE373_MODEL), hoặc một đối tượng chat model.
    approver: hàm nhận {"plan", "estimate"}, trả True để đồng ý, False hoặc chuỗi lý do để từ chối."""
    constraints = constraints or Constraints()
    world = FlightWorld(scenario)
    harness = Harness(world, constraints, budget=budget)
    agent = PlanExecuteAgent(harness, make_langchain_tools(world), make_planner_model(model, planner_style, constraints),
                             request=constraints.request_text(), max_replans=max_replans,
                             max_retries=max_retries, approver=approver or approve_all)
    graph = agent.build()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": recursion_limit}

    try:
        result = graph.invoke({"messages": [HumanMessage(content=constraints.request_text())]}, config)
        while result.get("__interrupt__"):  # LangGraph dừng ở approve và chờ người duyệt
            decision = agent.approver(result["__interrupt__"][0].value)
            result = graph.invoke(Command(resume=decision), config)
    except GraphRecursionError:
        harness.finish(StopReason.BUDGET, "GraphRecursionError (lưới an toàn của LangGraph)")

    state = graph.get_state(config).values  # checkpoint giữ lại trạng thái ngay cả khi bị cắt giữa chừng
    messages = state.get("messages", [])
    answers = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
    report = harness.report()
    report["plan_steps"] = len(state.get("plan", {}).get("steps", []))
    report["replans"] = state.get("replans", 0)
    return RunResult(
        pattern=pattern or ("hybrid" if max_replans > 0 else "plan"), scenario=scenario, report=report,
        final_answer=message_text(answers[-1]) if answers else "",
        handoff=harness.handoff.render() if harness.handoff and harness.stop_reason is not StopReason.GOAL else "",
        trace=render_trace(messages), messages=messages, harness=harness, world=world,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # in tiếng Việt đúng trên console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chạy agent Plan-then-Execute (hoặc Lai nếu --replans > 0)")
    parser.add_argument("--scenario", choices=SCENARIOS, default="happy")
    parser.add_argument("--model", choices=("fake", "real"), default="fake")
    parser.add_argument("--planner-style", choices=PLANNER_STYLES, default="canonical")
    parser.add_argument("--replans", type=int, default=0, help="số lần được lập lại kế hoạch (0 = Plan thuần)")
    parser.add_argument("--retries", type=int, default=2, help="số lần thử lại một bước khi tool báo lỗi")
    parser.add_argument("--reject", metavar="LÝ DO", help="mô phỏng người duyệt TỪ CHỐI kế hoạch")
    args = parser.parse_args()
    try:
        outcome = run_plan(args.scenario, model=args.model, planner_style=args.planner_style,
                           max_replans=args.replans, max_retries=args.retries,
                           approver=(lambda payload: args.reject) if args.reject else None)
    except RuntimeError as exc:  # ví dụ: --model real mà chưa đặt SE373_MODEL
        parser.error(str(exc))
    print_result(outcome)
