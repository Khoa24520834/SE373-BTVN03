"""
harness.py — Lớp harness cho agent đặt vé máy bay (SE373 · BTVN#3)

HARNESS LÀ GÌ
    Model chỉ làm MỘT việc: đề xuất lời gọi tool (bước 02 của vòng lặp). Mọi thứ
    còn lại là CODE CỦA BẠN, nằm ở file này. File độc lập với framework nên dùng
    chung cho cả ba mẫu (ReAct, Plan-then-Execute, Lai): ba mẫu chỉ khác nhau ở
    cách điều phối model, còn harness giống hệt nhau → so sánh công bằng.

CÁC LỚP
    Bốn lớp đề bài yêu cầu:
      1. Constraints          Ràng buộc là DỮ LIỆU, không nằm trong prompt dài ra theo log.
      2. Harness.is_done      Tiêu chí hoàn thành kiểm bằng CODE, không tin model tự nói "xong".
      3. Harness.check_permission  Kiểm quyền TRƯỚC khi thực thi tool có tác dụng phụ.
      4. Handoff              Bàn giao: đã làm tới đâu, đã thử gì, cần hỏi người điều gì.
    Lớp phụ trợ: LoopDetector (lặp/bế tắc), Budget (ngân sách),
                 validate_tool_call (tên tool, tham số), grounding_check (chống bịa).

THỨ TỰ CHẠY MỖI VÒNG: Harness.run_tool_call (checklist trong slide)
      0  trước khi thực thi : validate tham số → kiểm quyền   → HUMAN (E) hoặc từ chối
      1  sau observation    : tiêu chí hoàn thành             → GOAL   (A)
      2  sau observation    : (tool, args) trùng              → LOOP   (C)
      3  sau observation    : đại lượng tiến triển đứng yên   → STALL  (D)
      4  sau observation    : vòng · token · giây · tiền      → BUDGET (B)
    Ngân sách kiểm CUỐI CÙNG: đặt lên đầu thì mọi lỗi đều thành "hết ngân sách"
    và mất chẩn đoán.

CÁCH MỘT MẪU DÙNG HARNESS
    world = FlightWorld("happy")
    h = Harness(world)
    h.charge_model_call(prompt_tokens, completion_tokens)       # mỗi lần gọi model
    out = h.run_tool_call(name, args, lambda: getattr(world, name)(**args))
    # out.observation → đưa lại cho model; out.stop khác None → dừng, out.handoff.render()
    v = h.verify_final(answer)       # khi model thôi gọi tool và đưa câu trả lời cuối
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from enum import Enum
from typing import Any, Callable, ClassVar

from pydantic import ValidationError

from tools_flight import SIDE_EFFECT_TOOLS, TOOL_NAMES, Booking, FlightWorld

# Giá GIẢ ĐỊNH (USD / 1.000 token) để quy đổi chi phí. Chỉnh theo model thật khi đo.
PRICE_IN_PER_1K = 0.003
PRICE_OUT_PER_1K = 0.015


def vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + "đ"


def estimate_tokens(text: str) -> int:
    """Ước lượng thô ~4 ký tự/token. Với model thật, ưu tiên usage_metadata."""
    return max(1, math.ceil(len(text) / 4))


def fmt_call(tool: str, args: dict[str, Any]) -> str:
    return f"{tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def _valid_date(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Năm kiểu dừng (slide: "5 điều kiện dừng")
# --------------------------------------------------------------------------- #
class StopReason(str, Enum):
    GOAL = "goal_reached"        # A · đạt mục tiêu
    BUDGET = "budget_exhausted"  # B · hết ngân sách
    LOOP = "loop"                # C · phát hiện lặp
    STALL = "stall"              # D · bế tắc
    HUMAN = "needs_human"        # E · cần con người

    @property
    def letter(self) -> str:
        return _STOP_LETTER[self]

    @property
    def label(self) -> str:
        return _STOP_LABEL[self]

    @property
    def normal(self) -> bool:
        """Dừng bình thường: đạt mục tiêu, hoặc chủ động chờ người duyệt."""
        return self in (StopReason.GOAL, StopReason.HUMAN)


_STOP_LETTER = {StopReason.GOAL: "A", StopReason.BUDGET: "B", StopReason.LOOP: "C",
                StopReason.STALL: "D", StopReason.HUMAN: "E"}
_STOP_LABEL = {StopReason.GOAL: "Đạt mục tiêu", StopReason.BUDGET: "Hết ngân sách",
               StopReason.LOOP: "Phát hiện lặp", StopReason.STALL: "Bế tắc",
               StopReason.HUMAN: "Cần con người"}


# --------------------------------------------------------------------------- #
# Lớp 1: ràng buộc là DỮ LIỆU
# --------------------------------------------------------------------------- #
@dataclass
class FlightFact:
    """Một chuyến bay mà agent ĐÃ THẤY trong observation (nguồn sự thật của harness)."""

    code: str
    origin: str
    dest: str
    date: str
    depart: str
    price: int
    refundable: bool
    checked_price: int | None = None  # giá thấy ở check_seat, dùng để kiểm chứng chéo

    @property
    def effective_price(self) -> int:
        return self.checked_price if self.checked_price is not None else self.price


@dataclass(frozen=True)
class Constraints:
    """Yêu cầu của người dùng, ghi thành DỮ LIỆU ở một chỗ cố định.

    Yêu cầu không "lùi xa dần" khi nhật ký dài ra (lỗi 'Quên yêu cầu'), vì harness
    kiểm lại bằng code ngay trước book_seat, pay và khi xét tiêu chí hoàn thành.
    """

    origin: str = "SGN"
    dest: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"
    max_price: int = 2_000_000

    TOTAL: ClassVar[int] = 4  # số ràng buộc: tuyến, ngày, giờ, giá

    def violations(self, *, origin: str, dest: str, date: str, depart: str, price: int) -> list[str]:
        out: list[str] = []
        if (origin, dest) != (self.origin, self.dest):
            out.append(f"tuyến {origin}→{dest}, cần {self.origin}→{self.dest}")
        if date != self.date:
            out.append(f"ngày {date}, cần {self.date}")
        if not depart < self.depart_before:
            out.append(f"giờ bay {depart}, cần trước {self.depart_before}")
        if price > self.max_price:
            out.append(f"giá {vnd(price)}, tối đa {vnd(self.max_price)}")
        return out

    def flight_violations(self, f: FlightFact) -> list[str]:
        return self.violations(origin=f.origin, dest=f.dest, date=f.date,
                               depart=f.depart, price=f.effective_price)

    def booking_violations(self, b: Booking) -> list[str]:
        return self.violations(origin=b.origin, dest=b.dest, date=b.depart_date,
                               depart=b.depart_time, price=b.price)

    def satisfied_by(self, f: FlightFact) -> int:
        return self.TOTAL - len(self.flight_violations(f))

    def request_text(self) -> str:
        """Câu yêu cầu gửi cho agent (dùng chung cho cả 3 mẫu)."""
        return (f"Đặt giúp tôi một vé bay {self.origin} → {self.dest} ngày {self.date}, "
                f"khởi hành trước {self.depart_before}, giá không quá {vnd(self.max_price)}. "
                "Chọn chuyến rẻ nhất thoả các yêu cầu.")

    def to_prompt(self) -> str:
        """Đoạn nhắc ràng buộc để chèn vào system prompt hoặc nhắc lại trước khi trả lời."""
        return (f"RÀNG BUỘC (không được vi phạm): tuyến {self.origin}→{self.dest}; ngày {self.date}; "
                f"khởi hành trước {self.depart_before}; giá tối đa {vnd(self.max_price)}. "
                "Chỉ dùng tool để lấy thông tin (mã chuyến, giá, ghế, mã đặt chỗ); không tự bịa dữ liệu.")


# --------------------------------------------------------------------------- #
# Lớp 3: kiểm quyền
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Policy:
    allowed_tools: tuple[str, ...] = TOOL_NAMES
    approval_limit: int = 1_500_000      # vượt hạn mức tự quyết thì phải hỏi người
    approve_non_refundable: bool = True  # vé không hoàn cũng phải hỏi người
    max_bookings: int = 1                # yêu cầu chỉ MỘT vé

    def approval_reasons(self, price: int, refundable: bool) -> list[str]:
        reasons: list[str] = []
        if price > self.approval_limit:
            reasons.append(f"vượt hạn mức {vnd(self.approval_limit)}")
        if self.approve_non_refundable and not refundable:
            reasons.append("vé không hoàn")
        return reasons


class Permission(str, Enum):
    ALLOW = "allow"
    DENY = "deny"              # không thực thi, trả lý do cho model để nó đổi hướng
    NEED_HUMAN = "need_human"  # không thực thi, DỪNG và bàn giao cho người


@dataclass
class PermissionResult:
    decision: Permission
    reason: str = ""
    hint: str = ""
    pending: dict[str, Any] | None = None  # hành động đang chờ duyệt (khi NEED_HUMAN)


# --------------------------------------------------------------------------- #
# Ngân sách vòng lặp (slide: bước · token · thời gian · chi phí)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Budget:
    max_rounds: int = 12        # số lần gọi model (một vòng = một lần gọi model)
    max_tool_calls: int = 20
    max_tokens: int = 40_000
    max_seconds: float = 120.0
    max_cost_usd: float = 0.50


@dataclass
class Usage:
    rounds: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens / 1000 * PRICE_IN_PER_1K
                + self.completion_tokens / 1000 * PRICE_OUT_PER_1K)


# --------------------------------------------------------------------------- #
# Phát hiện lặp và bế tắc (code bạn tự viết, framework không làm hộ)
# --------------------------------------------------------------------------- #
@dataclass
class LoopSignal:
    kind: str  # "LOOP" | "STALL"
    detail: str


class LoopDetector:
    """Dựa trên bộ phát hiện lặp trong slide, có ba chỉnh sửa:

    - repeat_k=3 (slide mặc định 2): báo động ở lần gọi THỨ BA giống hệt, đúng ví dụ
      "V2, V3, V4 check_seat timeout → V4 báo động". Hai lần là thử lại hợp lệ.
    - Tool "polling" (get_booking): dấu vân tay gồm cả observation, nên chờ
      pending → confirmed không bị báo nhầm; chỉ báo khi observation cũng không đổi.
    - Thêm tín hiệu "trùng observation LỖI dù tham số khác" (đổi cách viết rồi gọi lại).
    """

    def __init__(self, window: int = 6, repeat_k: int = 3, stall_n: int = 5,
                 poll_tools: frozenset[str] = frozenset({"get_booking"})) -> None:
        self.k, self.n, self.poll_tools = repeat_k, stall_n, poll_tools
        self.recent_actions: deque[str] = deque(maxlen=window)  # chỉ so cửa sổ gần
        self.recent_errors: deque[str] = deque(maxlen=window)
        self.last_progress: int | None = None
        self.stall = 0

    @staticmethod
    def _dump(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)

    def check(self, tool: str, args: dict[str, Any], observation: dict[str, Any],
              progress: int) -> LoopSignal | None:
        # Tín hiệu 1: trùng (tool, args)
        fp = f"{tool}|{self._dump(args)}"
        if tool in self.poll_tools:
            fp += f"|{self._dump(observation)}"
        times = self.recent_actions.count(fp) + 1
        if times >= self.k:
            return LoopSignal("LOOP", f"{fmt_call(tool, args)} lặp {times} lần")
        self.recent_actions.append(fp)

        # Tín hiệu 2: cùng một observation LỖI dù tham số khác
        if observation.get("status") != "ok":
            efp = self._dump(observation)
            times = self.recent_errors.count(efp) + 1
            if times >= self.k:
                return LoopSignal("LOOP", f"cùng một lỗi lặp {times} lần: {efp[:90]}")
            self.recent_errors.append(efp)

        # Tín hiệu 3: đại lượng tiến triển của bài toán đứng yên
        self.stall = self.stall + 1 if progress == self.last_progress else 0
        self.last_progress = progress
        if self.stall >= self.n:
            return LoopSignal("STALL", f"tiến triển đứng yên {self.stall} vòng liền (progress={progress})")
        return None


# --------------------------------------------------------------------------- #
# Lớp 4: bàn giao
# --------------------------------------------------------------------------- #
@dataclass
class Handoff:
    """Bản bàn giao: người nhận phải trả lời được trong 30 giây."""

    reason: StopReason
    status: str              # đã làm tới đâu
    side_effects: list[str]  # hành động nào ĐÃ có tác dụng phụ
    tried: list[str]         # hướng nào đã hỏng và vì sao
    question: str            # câu hỏi cụ thể cho người
    pending: str = ""        # hành động đang chờ duyệt (kết E)
    detail: str = ""

    def render(self) -> str:
        kind = "bình thường" if self.reason.normal else "BẤT THƯỜNG"
        lines = [f"[Kết {self.reason.letter} · {self.reason.label}] dừng {kind}",
                 f"Trạng thái     : {self.status}",
                 "Tác dụng phụ   : " + ("; ".join(self.side_effects)
                                        if self.side_effects else "chưa có (chưa giữ chỗ, chưa trả tiền)"),
                 "Đã thử         : " + ("; ".join(self.tried) if self.tried else "không có lỗi nào")]
        if self.pending:
            lines.append(f"Đang chờ duyệt : {self.pending}")
        if self.question:
            lines.append(f"Câu hỏi        : {self.question}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reason"] = self.reason.value
        return d


# --------------------------------------------------------------------------- #
# Kết quả các bước
# --------------------------------------------------------------------------- #
@dataclass
class Completion:
    done: bool
    booking: Booking | None
    failures: list[str]


@dataclass
class StepRecord:
    n: int
    tool: str
    args: dict[str, Any]
    observation: dict[str, Any]
    executed: bool  # False: bị harness chặn, tool KHÔNG chạy


@dataclass
class StepOutcome:
    observation: dict[str, Any]          # đưa lại cho model làm ToolMessage
    stop: StopReason | None = None       # khác None: DỪNG vòng lặp
    detail: str = ""
    handoff: Handoff | None = None
    executed: bool = True


@dataclass
class FinalVerdict:
    accept: bool                          # is_done() đúng → chấp nhận câu trả lời cuối
    stop: StopReason | None = None
    problems: list[str] = field(default_factory=list)  # thông tin không có nguồn
    feedback: str = ""                    # đưa lại cho model nếu bị từ chối


_REQUIRED_ARGS = {
    "search_flights": ("origin", "dest", "date"),
    "check_seat": ("flight",),
    "book_seat": ("flight", "seat"),
    "pay": ("code", "method"),
    "get_booking": ("code",),
}

_FLIGHT_RE = re.compile(r"\b[A-Z]{2}\d{3,4}\b")
_SEAT_RE = re.compile(r"\b\d{1,2}[A-F]\b")
_CODE_RE = re.compile(r"\b(?=[0-9A-Z]*\d)(?=[0-9A-Z]*[A-Z])[0-9A-Z]{4}\b")
_AMOUNT_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+|\d{6,9})(?!\d)")
_BOOKED_RE = re.compile(r"\bbooked\b|đã đặt|đặt thành công|đã giữ chỗ", re.IGNORECASE)
_PAID_RE = re.compile(r"(?<!not )\bpaid\b|đã thanh toán|thanh toán thành công", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class Harness:
    def __init__(self, world: FlightWorld, constraints: Constraints | None = None,
                 policy: Policy | None = None, budget: Budget | None = None, *,
                 loop: LoopDetector | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 max_false_claims: int = 2) -> None:
        self.world = world
        self.constraints = constraints or Constraints()
        self.policy = policy or Policy()
        self.budget = budget or Budget()
        self.loop = loop or LoopDetector()
        self.clock = clock
        self.max_false_claims = max_false_claims
        self.usage = Usage(started=clock())

        # Những gì agent ĐÃ THẤY (rút ra từ observation, không phải từ trí nhớ model)
        self.flights: dict[str, FlightFact] = {}
        self.bookings: dict[str, Booking] = {}
        self.approved: set[str] = set()      # mã chuyến người đã duyệt
        self.pending: dict[str, Any] | None = None

        self.log: list[StepRecord] = []
        self.interventions: Counter[str] = Counter()  # denied, invalid_call, need_human, false_claim
        self.schema_errors: list[str] = []
        self.ungrounded: list[str] = []
        self.false_claims = 0

        self.stop_reason: StopReason | None = None
        self.stop_detail = ""
        self.handoff: Handoff | None = None

    # ---- ngân sách -------------------------------------------------------- #
    def charge_model_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Gọi mỗi lần model được gọi. Mỗi vòng gửi lại TOÀN BỘ lịch sử nên prompt_tokens tăng dần."""
        self.usage.rounds += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens

    def elapsed(self) -> float:
        return self.clock() - self.usage.started

    def budget_exceeded(self) -> str | None:
        """Trả về mô tả giới hạn đã chạm trần (vd 'rounds 12/12'), hoặc None."""
        u, b = self.usage, self.budget
        for name, used, limit in (("rounds", u.rounds, b.max_rounds),
                                  ("tool_calls", u.tool_calls, b.max_tool_calls),
                                  ("tokens", u.total_tokens, b.max_tokens),
                                  ("seconds", round(self.elapsed(), 1), b.max_seconds),
                                  ("cost_usd", round(u.cost_usd, 4), b.max_cost_usd)):
            if used >= limit:
                return f"{name} {used}/{limit}"
        return None

    # ---- tiêu chí hoàn thành bằng code (lớp 2) ----------------------------- #
    def is_done(self) -> Completion:
        """Kiểm bằng CODE, độc lập với phán đoán của model. Dùng cả bốn dạng kiểm chứng được:
        vị từ chạy bằng code · schema hợp lệ · kiểm chứng chéo · người duyệt (ở check_permission).

        Đọc thẳng kho đặt chỗ của world (system of record) thay vì gọi tool, để không
        làm bẩn trace và không tính vào số lần gọi tool của agent.
        """
        paid = [b for b in self.world.bookings.values() if b.paid]
        if not paid:
            return Completion(False, None, ["chưa có đặt chỗ nào đã thanh toán"])
        failures: list[str] = []
        if len(paid) > 1:
            failures.append(f"có {len(paid)} vé đã thanh toán, yêu cầu chỉ một vé")
        b = paid[0]
        if b.status != "confirmed":
            failures.append(f"trạng thái đặt chỗ là {b.status}, cần confirmed")
        failures += [f"vi phạm ràng buộc: {v}" for v in self.constraints.booking_violations(b)]
        seen = self.flights.get(b.flight)
        if seen is not None and seen.checked_price is not None and seen.checked_price != b.price:
            failures.append(f"giá đặt chỗ {vnd(b.price)} khác giá đã thấy ở check_seat {vnd(seen.checked_price)}")
        failures += self.schema_errors
        return Completion(not failures, b, failures)

    def progress(self) -> int:
        """Đại lượng tiến triển của BÀI TOÁN (không có bộ đo tổng quát):
        10 × số ràng buộc thoả tốt nhất của một chuyến đã thấy + giai đoạn đặt chỗ (0/1 giữ chỗ/2 đã trả)."""
        best = max((self.constraints.satisfied_by(f) for f in self.flights.values()), default=0)
        stage = max((2 if b.paid else 1 for b in self.bookings.values()), default=0)
        return best * 10 + stage

    # ---- kiểm tra lời gọi và quyền (chạy TRƯỚC khi thực thi) ---------------- #
    def validate_tool_call(self, tool: str, args: dict[str, Any]) -> str | None:
        """Ba kiểu dùng sai công cụ mà code bắt được: tool không tồn tại, sai định dạng tham số."""
        if tool not in self.policy.allowed_tools:
            return f"Tool '{tool}' không tồn tại. Chỉ được gọi: {', '.join(self.policy.allowed_tools)}"
        if not isinstance(args, dict):
            return "args phải là một object"
        need = _REQUIRED_ARGS[tool]
        missing = [k for k in need if k not in args]
        extra = [k for k in args if k not in need]
        if missing:
            return f"Thiếu tham số: {', '.join(missing)}"
        if extra:
            return f"Tham số thừa: {', '.join(extra)}; {tool} chỉ nhận {', '.join(need)}"
        if tool == "search_flights" and not _valid_date(args["date"]):
            return "date phải có dạng YYYY-MM-DD, ví dụ 2026-10-07"
        if tool == "book_seat" and not re.fullmatch(r"\d{1,2}[A-F]", str(args["seat"]).upper()):
            return 'seat phải là số hàng + chữ A-F, ví dụ "12A"'
        return None

    def check_permission(self, tool: str, args: dict[str, Any]) -> PermissionResult:
        """Chỉ book_seat và pay (tool có tác dụng phụ) cần kiểm. Thứ tự: từ chối vì sai
        dữ liệu/ràng buộc trước, rồi mới hỏi người (hỏi người vì một chuyến sai ràng buộc là vô nghĩa)."""
        if tool not in SIDE_EFFECT_TOOLS:
            return PermissionResult(Permission.ALLOW)

        if tool == "book_seat":
            code = str(args["flight"]).upper()
            fact = self.flights.get(code)
            if fact is None:  # chặn mã chuyến bịa (VN999) ngay tại hành động
                return PermissionResult(Permission.DENY, "unknown_flight",
                                        f"Chuyến {code} chưa có trong kết quả search_flights. "
                                        "Hãy tìm chuyến trước và chỉ đặt mã có trong kết quả.")
            bad = self.constraints.flight_violations(fact)
            if bad:  # kiểm ràng buộc ngay TRƯỚC book_seat
                return PermissionResult(Permission.DENY, "violates_constraints",
                                        f"Chuyến {code} vi phạm ràng buộc: {'; '.join(bad)}. Hãy chọn chuyến khác.")
            if len(self.bookings) >= self.policy.max_bookings:
                held = next(iter(self.bookings.values()))
                return PermissionResult(Permission.DENY, "already_booked",
                                        f"Đã có đặt chỗ {held.code}. Yêu cầu chỉ một vé: "
                                        "hãy thanh toán đặt chỗ này thay vì đặt thêm.")
            price, refundable, subject = fact.effective_price, fact.refundable, code
            pending = {"tool": tool, "args": dict(args), "flight": code, "price": price, "refundable": refundable}
        else:  # pay
            b = self.bookings.get(str(args["code"]))
            if b is None:
                return PermissionResult(Permission.DENY, "unknown_booking",
                                        f"Mã đặt chỗ {args['code']} chưa được tạo bởi book_seat. "
                                        "Chỉ thanh toán mã lấy từ kết quả book_seat.")
            bad = self.constraints.booking_violations(b)
            if bad:
                return PermissionResult(Permission.DENY, "violates_constraints",
                                        f"Đặt chỗ {b.code} vi phạm ràng buộc: {'; '.join(bad)}. Không thanh toán.")
            price, refundable, subject = b.price, b.refundable, b.flight
            pending = {"tool": tool, "args": dict(args), "flight": b.flight, "price": price, "refundable": refundable}

        reasons = self.policy.approval_reasons(price, refundable)
        if reasons and subject not in self.approved:
            pending["reasons"] = reasons
            return PermissionResult(Permission.NEED_HUMAN, "; ".join(reasons), pending=pending)
        return PermissionResult(Permission.ALLOW)

    def approve(self, flight: str) -> None:
        """Gọi khi NGƯỜI đã duyệt chuyến này. Xoá trạng thái dừng để chạy tiếp."""
        self.approved.add(flight.upper())
        if self.stop_reason is StopReason.HUMAN:
            self.stop_reason, self.stop_detail, self.handoff, self.pending = None, "", None, None

    # ---- một vòng: checklist trong slide ----------------------------------- #
    def run_tool_call(self, tool: str, args: dict[str, Any],
                      execute: Callable[[], dict[str, Any]]) -> StepOutcome:
        self.usage.tool_calls += 1

        # Bước 0: validate rồi kiểm quyền, TRƯỚC khi thực thi
        problem = self.validate_tool_call(tool, args)
        if problem:
            self.interventions["invalid_call"] += 1
            obs = {"status": "denied", "reason": "invalid_call", "hint": problem}
            return self._conclude(tool, args, obs, executed=False)

        perm = self.check_permission(tool, args)
        if perm.decision is Permission.DENY:
            self.interventions["denied"] += 1
            obs = {"status": "denied", "reason": perm.reason, "hint": perm.hint}
            return self._conclude(tool, args, obs, executed=False)
        if perm.decision is Permission.NEED_HUMAN:
            self.interventions["need_human"] += 1
            self.pending = perm.pending
            obs = {"status": "needs_approval", "reason": perm.reason,
                   "hint": "Dừng lại: hành động này cần người duyệt. Không thử lại."}
            self._record(tool, args, obs, executed=False)
            return self._stop(StopReason.HUMAN, perm.reason, obs, executed=False)

        obs = execute()
        self._learn(tool, args, obs)
        return self._conclude(tool, args, obs, executed=True)

    def _conclude(self, tool: str, args: dict[str, Any], obs: dict[str, Any], *, executed: bool) -> StepOutcome:
        self._record(tool, args, obs, executed)
        if self.is_done().done:                                   # 1. đạt mục tiêu
            return self._stop(StopReason.GOAL, "tiêu chí hoàn thành đạt", obs, executed)
        sig = self.loop.check(tool, args, obs, self.progress())   # 2-3. lặp, bế tắc
        if sig:
            reason = StopReason.LOOP if sig.kind == "LOOP" else StopReason.STALL
            return self._stop(reason, sig.detail, obs, executed)
        exceeded = self.budget_exceeded()                         # 4. ngân sách, kiểm CUỐI CÙNG
        if exceeded:
            return self._stop(StopReason.BUDGET, exceeded, obs, executed)
        return StepOutcome(observation=obs, executed=executed)

    def _record(self, tool: str, args: dict[str, Any], obs: dict[str, Any], executed: bool) -> None:
        self.log.append(StepRecord(len(self.log) + 1, tool, dict(args), obs, executed))

    def _stop(self, reason: StopReason, detail: str, obs: dict[str, Any], executed: bool) -> StepOutcome:
        return StepOutcome(observation=obs, stop=reason, detail=detail,
                           handoff=self.finish(reason, detail), executed=executed)

    def finish(self, reason: StopReason, detail: str = "") -> Handoff:
        """Ghi nhận lý do dừng và lập bản bàn giao. Dừng bất thường mà im lặng là lỗi ẩn."""
        self.stop_reason, self.stop_detail = reason, detail
        self.handoff = self.make_handoff(reason, detail)
        return self.handoff

    # ---- học từ observation ------------------------------------------------ #
    def _learn(self, tool: str, args: dict[str, Any], obs: dict[str, Any]) -> None:
        if tool == "search_flights" and obs.get("status") == "ok":
            for f in obs.get("flights", []):
                self.flights[f["flight"]] = FlightFact(
                    code=f["flight"], origin=str(args["origin"]).upper(), dest=str(args["dest"]).upper(),
                    date=args["date"], depart=f["depart"], price=f["price"], refundable=f["refundable"])
        elif tool == "check_seat" and obs.get("status") == "ok":
            fact = self.flights.get(obs["flight"])
            if fact is not None:
                fact.checked_price, fact.refundable = obs["price"], obs["refundable"]
        if "booking" in obs:  # schema hợp lệ: parse theo model Booking
            try:
                booking = Booking.model_validate(obs["booking"])
                self.bookings[booking.code] = booking
            except ValidationError as exc:
                self.schema_errors.append(f"observation của {tool} không đúng schema Booking: {exc.error_count()} lỗi")

    # ---- chống bịa --------------------------------------------------------- #
    def grounding_check(self, answer: str) -> list[str]:
        """Đối chiếu số, ngày, ID, tên riêng trong câu trả lời với kết quả tool đã nhận.
        Bắt được lỗi bịa mà code không thấy qua tham số: chỉ phân tích câu trả lời mới biết."""
        problems: list[str] = []
        for code in sorted(set(_FLIGHT_RE.findall(answer))):
            if code not in self.flights:
                problems.append(f"chuyến {code}: không có trong kết quả tìm kiếm")
        known_seats = {b.seat for b in self.bookings.values()}
        for seat in sorted(set(_SEAT_RE.findall(answer))):
            if seat not in known_seats:
                problems.append(f"ghế {seat}: chưa từng được đặt")
        for code in sorted(set(_CODE_RE.findall(answer))):
            if code not in self.bookings:
                problems.append(f"mã đặt chỗ {code}: không có trong kết quả book_seat")
        known_amounts = {self.constraints.max_price, self.policy.approval_limit}
        for f in self.flights.values():
            known_amounts |= {f.price} | ({f.checked_price} if f.checked_price is not None else set())
        known_amounts |= {b.price for b in self.bookings.values()}
        for raw in sorted(set(_AMOUNT_RE.findall(answer))):
            amount = int(re.sub(r"[.,]", "", raw))
            if amount not in known_amounts:
                problems.append(f"số tiền {vnd(amount)}: không có trong kết quả tool")
        done_tools = {t["tool"] for t in self.world.side_effects}
        if _BOOKED_RE.search(answer) and "book_seat" not in done_tools:
            problems.append("khẳng định đã đặt chỗ nhưng không có lời gọi book_seat thành công")
        if _PAID_RE.search(answer) and "pay" not in done_tools:
            problems.append("khẳng định đã thanh toán nhưng không có lời gọi pay thành công")
        return problems

    def verify_final(self, answer: str) -> FinalVerdict:
        """Gọi khi model thôi gọi tool và đưa câu trả lời cuối. Đây là kiểu dừng 'trông giống
        thành công' nên nguy hiểm nhất khi sai: chỉ chấp nhận khi is_done() đúng."""
        comp = self.is_done()
        problems = self.grounding_check(answer)
        self.ungrounded += problems
        if comp.done:
            self.finish(StopReason.GOAL, "model dừng và tiêu chí hoàn thành đạt")
            return FinalVerdict(True, StopReason.GOAL, problems)

        self.false_claims += 1
        self.interventions["false_claim"] += 1
        feedback = "Chưa đạt tiêu chí hoàn thành: " + "; ".join(comp.failures) + "."
        if problems:
            feedback += " Thông tin không có nguồn trong kết quả tool: " + "; ".join(problems) + "."
        feedback += " Hãy tiếp tục dùng tool, hoặc báo rõ nếu chưa làm được."
        if self.false_claims >= self.max_false_claims:
            self.finish(StopReason.STALL, f"model khẳng định xong {self.false_claims} lần nhưng is_done() sai")
            return FinalVerdict(False, StopReason.STALL, problems, feedback)
        return FinalVerdict(False, None, problems, feedback)

    # ---- bàn giao ---------------------------------------------------------- #
    def make_handoff(self, reason: StopReason, detail: str = "") -> Handoff:
        return Handoff(reason=reason, status=self._status_line(),
                       side_effects=self._side_effect_lines(), tried=self._tried_lines(),
                       question=self._question(reason, detail),
                       pending=self._pending_line() if reason is StopReason.HUMAN else "",
                       detail=detail)

    def _status_line(self) -> str:
        if not self.flights:
            parts = ["chưa tìm được chuyến nào"]
        else:
            feasible = sorted((f for f in self.flights.values() if not self.constraints.flight_violations(f)),
                              key=lambda f: f.effective_price)
            s = f"đã thấy {len(self.flights)} chuyến, {len(feasible)} chuyến thoả mọi ràng buộc"
            if feasible:
                s += f" (rẻ nhất: {feasible[0].code} {vnd(feasible[0].effective_price)})"
            parts = [s]
        for b in self.bookings.values():
            state = "đã thanh toán" if b.paid else "đã giữ chỗ, CHƯA thanh toán"
            parts.append(f"{state} {b.code}: {b.flight} ghế {b.seat}, {vnd(b.price)}")
        if not self.bookings:
            parts.append("chưa giữ chỗ")
        return "; ".join(parts)

    def _side_effect_lines(self) -> list[str]:
        lines: list[str] = []
        for t in self.world.side_effects:  # world là nguồn sự thật về điều ĐÃ xảy ra
            b = t["result"]["booking"]
            if t["tool"] == "book_seat":
                lines.append(f"giữ chỗ {b['flight']} ghế {b['seat']} (mã {b['code']}, {vnd(b['price'])})")
            else:
                lines.append(f"đã trả tiền mã {b['code']} ({vnd(b['price'])})")
        return lines

    def _tried_lines(self, limit: int = 6) -> list[str]:
        counts: Counter[tuple[str, str]] = Counter()
        for rec in self.log:
            status = rec.observation.get("status")
            if status in ("ok", "needs_approval"):
                continue
            why = rec.observation.get("error") or rec.observation.get("reason") or ""
            counts[(fmt_call(rec.tool, rec.args), f"{status}" + (f"/{why}" if why else ""))] += 1
        lines = [f"{call} → {res}" + (f" ×{n}" if n > 1 else "") for (call, res), n in counts.items()]
        return lines[-limit:]

    def _pending_line(self) -> str:
        p = self.pending
        if not p:
            return ""
        return (f"{fmt_call(p['tool'], p['args'])} · {vnd(p['price'])} · "
                f"{'vé hoàn được' if p['refundable'] else 'vé không hoàn'} · {'; '.join(p.get('reasons', []))}")

    def _question(self, reason: StopReason, detail: str) -> str:
        if reason is StopReason.HUMAN and self.pending:
            p = self.pending
            what = (f"đặt {p['flight']} ghế {p['args']['seat']}" if p["tool"] == "book_seat"
                    else f"thanh toán đặt chỗ {p['args']['code']} ({p['flight']})")
            return f"Duyệt {what} ({vnd(p['price'])}, {'vé hoàn được' if p['refundable'] else 'vé không hoàn'}) không?"
        if reason is StopReason.LOOP:
            return f"Agent bị kẹt: {detail}. Nên thử lại sau hay đổi cách làm (ví dụ bỏ qua bước lỗi)?"
        if reason is StopReason.STALL:
            return f"Agent không tiến triển: {detail}. Nên nới ràng buộc (giờ bay, giá) hay đổi ngày/tuyến?"
        if reason is StopReason.BUDGET:
            return f"Đã chạm trần ngân sách ({detail}). Tăng ngân sách hay thu hẹp yêu cầu?"
        return ""

    # ---- đầu ra ------------------------------------------------------------ #
    def summary(self) -> str:
        """Câu trả lời cuối dựng từ dữ liệu ĐÃ KIỂM CHỨNG, nên không thể bịa."""
        comp = self.is_done()
        if comp.done and comp.booking:
            b = comp.booking
            return (f"Đã đặt vé {b.flight} {b.origin}→{b.dest} ngày {b.depart_date} lúc {b.depart_time}, "
                    f"ghế {b.seat}, giá {vnd(b.price)} ({'hoàn được' if b.refundable else 'không hoàn'}). "
                    f"Mã đặt chỗ {b.code}, đã thanh toán.")
        return "Chưa hoàn thành: " + "; ".join(comp.failures) + "."

    def report(self) -> dict[str, Any]:
        """Số liệu một lần chạy, dùng cho evaluate.py."""
        stop = self.stop_reason
        return {
            "stop": stop.value if stop else None,
            "stop_letter": stop.letter if stop else None,
            "stop_detail": self.stop_detail,
            "success": self.is_done().done,
            "rounds": self.usage.rounds,
            "tool_calls": self.usage.tool_calls,
            "tokens": self.usage.total_tokens,
            "cost_usd": round(self.usage.cost_usd, 5),
            "seconds": round(self.elapsed(), 3),
            "interventions": dict(self.interventions),
            "ungrounded": list(self.ungrounded),
            "side_effects": [t["tool"] for t in self.world.side_effects],
        }
