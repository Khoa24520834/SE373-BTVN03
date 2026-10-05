"""
common.py — Phần dùng chung giữa ba mẫu thiết kế (SE373 · BTVN#3)

GỒM
    ScriptedModel   Model GIẢ, tất định, ra quyết định theo observation đã thấy.
                    Chạy lại được và không tốn tiền, nên so sánh 3 mẫu công bằng.
                    Dùng khi cần lặp lại lỗi (S2 lặp, S5 bịa) mà model thật khó tái hiện.
    make_model      "fake" → ScriptedModel; "real" → model thật đọc từ biến môi trường SE373_MODEL.
    RunResult       Kết quả MỘT lần chạy của MỘT mẫu (cùng khuôn cho ReAct, Plan, Lai).
    render_trace    In trace theo vòng: Suy luận → Hành động → Quan sát (đọc để tìm vòng sai đầu tiên).

KIỂU ỨNG XỬ CỦA MODEL GIẢ (tham số style)
    competent    Làm đúng: tìm → kiểm ghế → đặt → trả tiền, chọn chuyến rẻ nhất thoả ràng buộc.
                 Gặp lỗi thì thử lại y nguyên (đúng thói quen hay gây lặp của model thật).
    greedy       Bỏ qua ràng buộc giờ bay và giá, chọn chuyến rẻ nhất (kiểm lỗi "quên yêu cầu").
    hallucinate  Khẳng định đã đặt vé mà không gọi tool nào (kiểm lỗi bịa, kịch bản S5).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from harness import Constraints, Harness, estimate_tokens, fmt_call
from tools_flight import FlightWorld

STYLES = ("competent", "greedy", "hallucinate")
HARNESS_SOURCE = {"source": "harness"}  # đánh dấu tin nhắn do harness chèn vào, không phải của model


def message_text(m: BaseMessage) -> str:
    """Chữ của một tin nhắn, gồm cả tham số tool call (để ước lượng token)."""
    text = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
    if isinstance(m, AIMessage) and m.tool_calls:
        text += json.dumps(m.tool_calls, ensure_ascii=False)
    return text


def _history(messages: list[BaseMessage]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Ghép mỗi tool call với observation của nó: [(tool, args, observation)]."""
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                calls[c["id"]] = (c["name"], c["args"])
        elif isinstance(m, ToolMessage):
            name, args = calls.get(m.tool_call_id, (m.name or "?", {}))
            try:
                obs = json.loads(m.content) if isinstance(m.content, str) else {}
            except json.JSONDecodeError:
                obs = {"status": "error", "error": "non_json"}
            out.append((name, args, obs if isinstance(obs, dict) else {}))
    return out


# --------------------------------------------------------------------------- #
# Model giả
# --------------------------------------------------------------------------- #
class ScriptedModel(BaseChatModel):
    """Model giả: đọc lịch sử tool call + observation rồi chọn hành động kế tiếp.

    Giống model thật ở chỗ nó PHẢN ỨNG với observation (hết ghế thì đổi chuyến,
    bị từ chối thì loại chuyến đó), nhưng tất định nên chạy lại ra đúng kết quả cũ.
    """

    style: str = "competent"
    constraints: Any = None
    seat: str = "12A"
    method: str = "corp_card"

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":  # create_agent gọi hàm này
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        ai = self._decide(messages)
        # Mỗi vòng gửi lại TOÀN BỘ lịch sử, nên input_tokens tăng dần theo số vòng (chi phí bậc hai).
        n_in = estimate_tokens("\n".join(message_text(m) for m in messages))
        n_out = estimate_tokens(message_text(ai))
        ai.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ---- chính sách ra quyết định ----------------------------------------- #
    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        if self.style == "hallucinate":
            return AIMessage(content="Done! Booked VN999, seat 5C, for 1,200,000 VND.")

        c: Constraints = self.constraints or Constraints()
        hist = _history(messages)
        flights: list[dict[str, Any]] | None = None
        excluded: set[str] = set()
        booking: dict[str, Any] | None = None
        paid = False
        for tool, args, obs in hist:
            status = obs.get("status")
            if tool == "search_flights" and status == "ok":
                flights = obs["flights"]
            elif tool in ("check_seat", "book_seat") and status in ("sold_out", "not_found", "denied"):
                excluded.add(str(args.get("flight", "")).upper())
            elif tool == "book_seat" and status == "ok":
                booking = obs["booking"]
            elif tool == "pay" and status == "ok":
                paid = True

        n = sum(len(m.tool_calls) for m in messages if isinstance(m, AIMessage))

        def act(thought: str, tool: str, **args: Any) -> AIMessage:
            call = {"name": tool, "args": args, "id": f"call_{n + 1}", "type": "tool_call"}
            return AIMessage(content=thought, tool_calls=[call])

        if paid:
            return AIMessage(content=f"Đã đặt xong vé {booking['flight']}, mã đặt chỗ {booking['code']}." if booking
                             else "Đã thanh toán xong.")
        if booking is not None:
            return act(f"Đã giữ chỗ, giờ thanh toán đặt chỗ {booking['code']}.",
                       "pay", code=booking["code"], method=self.method)
        if flights is None:
            return act(f"Cần tìm các chuyến {c.origin}→{c.dest} ngày {c.date} trước.",
                       "search_flights", origin=c.origin, dest=c.dest, date=c.date)

        def acceptable(f: dict[str, Any]) -> bool:
            if self.style == "greedy":
                return True  # bỏ qua ràng buộc
            return f["depart"] < c.depart_before and f["price"] <= c.max_price

        candidates = sorted((f for f in flights if f["flight"] not in excluded and acceptable(f)),
                            key=lambda f: f["price"])
        if not candidates:
            return AIMessage(content="Không tìm thấy chuyến nào thoả yêu cầu.")
        pick = candidates[0]["flight"]
        checked = any(t == "check_seat" and str(a.get("flight", "")).upper() == pick and o.get("status") == "ok"
                      for t, a, o in hist)
        if not checked:  # chưa kiểm ghế được (kể cả vì lỗi) thì kiểm tiếp: thử lại y nguyên
            why = "rẻ nhất" if self.style == "greedy" else "rẻ nhất thoả yêu cầu"
            return act(f"Chuyến {why} là {pick}, kiểm tra ghế còn không.", "check_seat", flight=pick)
        return act(f"{pick} còn ghế, giữ chỗ ghế {self.seat}.", "book_seat", flight=pick, seat=self.seat)


def make_model(kind: Any = "fake", style: str = "competent", constraints: Constraints | None = None) -> Any:
    """'fake' → ScriptedModel. 'real' → tên model trong biến môi trường SE373_MODEL
    (ví dụ 'anthropic:claude-sonnet-4-5'); create_agent tự khởi tạo từ chuỗi này.
    Truyền thẳng một đối tượng model thì trả lại nguyên (dùng khi kiểm thử)."""
    if not isinstance(kind, str):
        return kind
    if kind == "fake":
        if style not in STYLES:
            raise ValueError(f"style phải thuộc {STYLES}, nhận được {style!r}")
        return ScriptedModel(style=style, constraints=constraints or Constraints())
    if kind == "real":
        name = os.environ.get("SE373_MODEL")
        if not name:
            raise RuntimeError("Chưa đặt biến môi trường SE373_MODEL (ví dụ 'anthropic:claude-sonnet-4-5').")
        return name
    raise ValueError(f"model phải là 'fake' hoặc 'real', nhận được {kind!r}")


# --------------------------------------------------------------------------- #
# Kết quả một lần chạy (cùng khuôn cho cả ba mẫu → evaluate.py gom được)
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    pattern: str                   # "react" | "plan" | "hybrid"
    scenario: str
    report: dict[str, Any]         # Harness.report(): kết thúc ra sao, tốn bao nhiêu
    final_answer: str
    handoff: str                   # bản bàn giao (rỗng nếu đạt mục tiêu)
    trace: str                     # trace đã định dạng
    messages: list[BaseMessage] = field(default_factory=list, repr=False)
    harness: Harness | None = field(default=None, repr=False)
    world: FlightWorld | None = field(default=None, repr=False)


def render_trace(messages: list[BaseMessage], width: int = 140) -> str:
    """Trace theo vòng, đúng khuôn ReAct: Suy luận → Hành động → Quan sát."""

    def cut(text: str) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= width else text[: width - 1] + "…"

    lines: list[str] = []
    rnd = 0
    for m in messages:
        harness_made = (m.additional_kwargs or {}).get("source") == "harness"
        if isinstance(m, HumanMessage):
            lines.append(f"[Harness → model] {cut(m.content)}" if harness_made else f"[Yêu cầu] {cut(m.content)}")
        elif isinstance(m, AIMessage) and harness_made:
            lines.append("── Harness dừng vòng lặp ──")
            lines += [f"   {ln}" for ln in str(m.content).splitlines()]
        elif isinstance(m, AIMessage):
            rnd += 1
            if m.tool_calls:
                if m.content:
                    lines.append(f"[V{rnd}] Suy luận : {cut(m.content)}")
                for c in m.tool_calls:
                    lines.append(f"[V{rnd}] Hành động: {fmt_call(c['name'], c['args'])}")
            else:
                lines.append(f"[V{rnd}] Trả lời   : {cut(m.content)}")
        elif isinstance(m, ToolMessage):
            lines.append(f"[V{rnd}] Quan sát : {cut(m.content)}")
    return "\n".join(lines)
