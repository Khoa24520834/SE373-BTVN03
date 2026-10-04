"""
tools_flight.py — Tool mockup cho agent đặt vé máy bay (SE373 · BTVN#3)

MỤC ĐÍCH
    Dựng một "thế giới giả" XÁC ĐỊNH (deterministic): cùng kịch bản, cùng chuỗi
    lời gọi thì lần nào cũng ra đúng cùng kết quả. Nhờ vậy lỗi tái hiện được
    và so sánh 3 mẫu (ReAct, Plan-then-Execute, Lai) là công bằng.

NGUYÊN TẮC THIẾT KẾ (theo phần Debugging của slide)
    1. Mọi tool trả JSON có trường "status" rõ ràng:
           ok · invalid_param · not_found · sold_out · seat_taken · already_paid · error
       "Không có dữ liệu" ({"status": "ok", "flights": []}) KHÁC "công cụ lỗi"
       ({"status": "error", ...}). Nếu gộp chung, agent sẽ tin dữ liệu sai.
    2. Lỗi luôn kèm "hint" chỉ đường đi khác, để agent không lặp vô ích.
    3. Trạng thái nằm trong FlightWorld (không dùng biến toàn cục): mỗi lần chạy
       tạo một world mới, không rò trạng thái giữa các lần chạy.
    4. Tool KHÔNG tự kiểm ràng buộc hay quyền hạn. Đó là việc của harness.

KỊCH BẢN (khớp kế hoạch đánh giá)
    happy       S1  Không chèn lỗi. Chuyến rẻ nhất thoả ràng buộc là VJ604
                    (1.290.000đ, hoàn được) nên không cần người duyệt.
    timeout     S2  check_seat luôn lỗi timeout. Agent dễ gọi lặp: kiểm tra
                    bộ phát hiện lặp.
    approval    S4  VJ604 và VJ606 không còn. Chuyến duy nhất thoả ràng buộc là
                    VN122 (1.850.000đ, KHÔNG hoàn): harness phải dừng chờ người duyệt.
    env_change  S6  Sau lần tìm kiếm đầu tiên, VJ604 hết ghế. Kế hoạch lập sẵn
                    bị lỗi thời, agent phải chuyển sang VJ606.

CÁCH DÙNG
    world = FlightWorld("happy")
    world.search_flights("SGN", "DAD", "2026-10-07")
    tools = make_langchain_tools(world)          # truyền vào create_agent(...)

    python src/tools_flight.py --scenario approval   # chạy tay để quan sát
"""
from __future__ import annotations

import argparse
import functools
import inspect
import json
import re
import sys
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Literal

from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Hằng số dùng chung (harness import lại các hằng số này)
# --------------------------------------------------------------------------- #
SCENARIOS = ("happy", "timeout", "approval", "env_change")

# Danh sách tên tool được phép gọi (harness dùng làm whitelist, chống gọi tool không tồn tại)
TOOL_NAMES = ("search_flights", "check_seat", "book_seat", "pay", "get_booking")

# Tool có tác dụng phụ không hoàn tác: chỉ các tool này cần kiểm quyền trước khi chạy
SIDE_EFFECT_TOOLS = frozenset({"book_seat", "pay"})

AIRPORTS = frozenset({"SGN", "HAN", "DAD", "HUI", "CXR"})
PAYMENT_METHODS = frozenset({"corp_card", "personal_card"})

_SEAT_RE = re.compile(r"\d{1,2}[A-F]")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_BOOKING_CODES = ("4XJ2", "7KQ9", "2MZ5", "8TR3", "6PD1")  # mã cố định, lần lượt theo thứ tự đặt


# --------------------------------------------------------------------------- #
# Mô hình dữ liệu
# --------------------------------------------------------------------------- #
class Booking(BaseModel):
    """Schema của một đặt chỗ.

    Harness dùng Booking.model_validate(...) để kiểm chứng "schema hợp lệ", và
    đọc các trường này để kiểm tiêu chí hoàn thành bằng code.
    """

    code: str
    flight: str
    origin: str
    dest: str
    seat: str
    status: Literal["held", "confirmed"]  # held: đã giữ chỗ · confirmed: đã thanh toán
    paid: bool
    price: int
    refundable: bool
    depart_date: str  # "2026-10-07"
    depart_time: str  # "08:10"


@dataclass
class Flight:
    code: str
    origin: str
    dest: str
    date: str
    depart: str
    price: int
    refundable: bool
    seats: int


def _base_flights() -> list[Flight]:
    """Dữ liệu chuyến bay cố định.

    Có chủ ý đặt "mồi nhử" cho từng ràng buộc (ngày 07/10, trước 12:00, ≤ 2 triệu,
    SGN→DAD) để kiểm tra agent có tuân thủ ràng buộc hay không.
    Tập chuyến THOẢ cả bốn ràng buộc: VJ604, VJ606, VN122.
    """
    d = "2026-10-07"
    return [
        #      mã        đi     đến    ngày          giờ      giá        hoàn?  ghế
        Flight("VJ604",  "SGN", "DAD", d,            "06:30", 1_290_000, True,  4),
        Flight("VJ606",  "SGN", "DAD", d,            "07:20", 1_420_000, True,  2),
        Flight("VN122",  "SGN", "DAD", d,            "08:10", 1_850_000, False, 3),
        Flight("VJ612",  "SGN", "DAD", d,            "10:05", 2_080_000, True,  6),  # sáng, nhưng > 2 triệu
        Flight("QH120",  "SGN", "DAD", d,            "11:20", 2_450_000, True,  3),  # sáng, nhưng > 2 triệu
        Flight("BL6180", "SGN", "DAD", d,            "12:30", 1_320_000, True,  7),  # rẻ, nhưng sau 12:00
        Flight("VN124",  "SGN", "DAD", d,            "13:30", 2_310_000, True,  4),
        Flight("QH118",  "SGN", "DAD", d,            "15:40", 1_640_000, True,  5),  # rẻ, nhưng buổi chiều
        Flight("VN134",  "SGN", "DAD", d,            "18:20", 2_190_000, True,  2),
        Flight("VJ608",  "SGN", "DAD", d,            "20:15", 1_190_000, False, 8),
        Flight("VJ610",  "SGN", "DAD", "2026-10-08", "06:45", 1_150_000, True,  5),  # sai ngày
        Flight("VN126",  "SGN", "HUI", d,            "07:30", 1_700_000, True,  4),  # sai điểm đến
    ]


# --------------------------------------------------------------------------- #
# Hàm tiện ích tạo kết quả lỗi có cấu trúc
# --------------------------------------------------------------------------- #
def _invalid(param: str, hint: str) -> dict[str, Any]:
    return {"status": "invalid_param", "param": param, "hint": hint}


def _not_found(what: str, hint: str) -> dict[str, Any]:
    return {"status": "not_found", "what": what, "hint": hint}


def _valid_date(value: Any) -> bool:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _traced(method):
    """Ghi (tool, args, result) vào world.trace sau mỗi lần gọi.

    Nhật ký này phục vụ ba việc: đối chiếu câu trả lời với kết quả thật (chống bịa),
    lập bản bàn giao (tác dụng phụ nào đã xảy ra), và thống kê khi đánh giá.
    """
    names = list(inspect.signature(method).parameters)[1:]  # bỏ 'self'

    @functools.wraps(method)
    def wrapper(self: "FlightWorld", *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = method(self, *args, **kwargs)
        call_args = {**dict(zip(names, args)), **kwargs}
        self.trace.append({"tool": method.__name__, "args": call_args, "result": result})
        return result

    return wrapper


# --------------------------------------------------------------------------- #
# Thế giới đặt vé
# --------------------------------------------------------------------------- #
class FlightWorld:
    """Dữ liệu chuyến bay, đặt chỗ và nhật ký gọi tool của MỘT lần chạy."""

    def __init__(self, scenario: str = "happy") -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario phải thuộc {SCENARIOS}, nhận được {scenario!r}")
        self.scenario = scenario
        self.flights: dict[str, Flight] = {f.code: f for f in _base_flights()}
        self.taken: dict[str, set[str]] = {}
        self.bookings: dict[str, Booking] = {}
        self.trace: list[dict[str, Any]] = []
        self._world_changed = False

        if scenario == "approval":
            # Chỉ còn VN122 thoả ràng buộc (1.850.000đ, không hoàn, vượt hạn mức tự quyết)
            for code in ("VJ604", "VJ606"):
                del self.flights[code]

    @property
    def side_effects(self) -> list[dict[str, Any]]:
        """Các lời gọi book_seat/pay đã thành công (tác dụng phụ). Dùng cho bàn giao."""
        return [
            t for t in self.trace
            if t["tool"] in SIDE_EFFECT_TOOLS and t["result"].get("status") == "ok"
        ]

    # ---- 1. tìm chuyến ---------------------------------------------------- #
    @_traced
    def search_flights(self, origin: str, dest: str, date: str) -> dict[str, Any]:
        origin, dest = str(origin).upper(), str(dest).upper()
        for param, value in (("origin", origin), ("dest", dest)):
            if value not in AIRPORTS:
                return _invalid(param, f"Dùng mã sân bay IATA hợp lệ: {', '.join(sorted(AIRPORTS))}")
        if not _valid_date(date):
            return _invalid("date", "Dùng YYYY-MM-DD, ví dụ 2026-10-07")

        found = sorted(
            (f for f in self.flights.values() if (f.origin, f.dest, f.date) == (origin, dest, date)),
            key=lambda f: f.depart,
        )
        result = {
            "status": "ok",  # flights == [] nghĩa là THẬT SỰ không có chuyến, không phải lỗi
            "flights": [
                {"flight": f.code, "depart": f.depart, "price": f.price, "refundable": f.refundable}
                for f in found
            ],
        }

        # Kịch bản S6: thế giới đổi SAU khi agent đã nhìn thấy kết quả tìm kiếm
        if self.scenario == "env_change" and not self._world_changed:
            self._world_changed = True
            self.flights["VJ604"].seats = 0
        return result

    # ---- 2. kiểm tra ghế -------------------------------------------------- #
    @_traced
    def check_seat(self, flight: str) -> dict[str, Any]:
        if self.scenario == "timeout":  # kịch bản S2: luôn lỗi
            return {
                "status": "error",
                "error": "timeout",
                "hint": "Dịch vụ kiểm tra ghế đang lỗi, thử lại sau",
            }
        f = self.flights.get(str(flight).upper())
        if f is None:
            return _not_found("flight", "Dùng mã chuyến lấy từ kết quả search_flights")
        if f.seats <= 0:
            return {
                "status": "sold_out",
                "flight": f.code,
                "seats_left": 0,
                "hint": "Chuyến đã hết ghế, chọn chuyến khác trong kết quả tìm kiếm",
            }
        return {
            "status": "ok",
            "flight": f.code,
            "seats_left": f.seats,
            "price": f.price,
            "refundable": f.refundable,
        }

    # ---- 3. giữ chỗ (có tác dụng phụ) ------------------------------------- #
    @_traced
    def book_seat(self, flight: str, seat: str) -> dict[str, Any]:
        seat = str(seat).upper()
        if not _SEAT_RE.fullmatch(seat):
            return _invalid("seat", 'Dạng số hàng + chữ cái A-F, ví dụ "12A"')
        f = self.flights.get(str(flight).upper())
        if f is None:
            return _not_found("flight", "Dùng mã chuyến lấy từ kết quả search_flights")
        if f.seats <= 0:
            return {
                "status": "sold_out",
                "flight": f.code,
                "hint": "Chuyến đã hết ghế, chọn chuyến khác trong kết quả tìm kiếm",
            }
        if seat in self.taken.setdefault(f.code, set()):
            return {"status": "seat_taken", "flight": f.code, "seat": seat, "hint": "Chọn ghế khác"}

        f.seats -= 1
        self.taken[f.code].add(seat)
        n = len(self.bookings)
        code = _BOOKING_CODES[n] if n < len(_BOOKING_CODES) else f"BK{n + 1:03d}"
        booking = Booking(
            code=code, flight=f.code, origin=f.origin, dest=f.dest, seat=seat,
            status="held", paid=False, price=f.price, refundable=f.refundable,
            depart_date=f.date, depart_time=f.depart,
        )
        self.bookings[code] = booking
        return {"status": "ok", "booking": booking.model_dump()}

    # ---- 4. thanh toán (có tác dụng phụ, không hoàn tác) ------------------ #
    @_traced
    def pay(self, code: str, method: str) -> dict[str, Any]:
        if method not in PAYMENT_METHODS:
            return _invalid("method", f"Dùng một trong: {', '.join(sorted(PAYMENT_METHODS))}")
        booking = self.bookings.get(str(code))
        if booking is None:
            return _not_found("booking", "Dùng booking code lấy từ kết quả book_seat")
        if booking.paid:  # chống trừ tiền hai lần
            return {"status": "already_paid", "booking": booking.model_dump(),
                    "hint": "Đặt chỗ này đã thanh toán, không cần trả lại"}
        booking.paid = True
        booking.status = "confirmed"
        return {"status": "ok", "booking": booking.model_dump()}

    # ---- 5. tra cứu đặt chỗ (dùng để kiểm chứng chéo) ---------------------- #
    @_traced
    def get_booking(self, code: str) -> dict[str, Any]:
        booking = self.bookings.get(str(code))
        if booking is None:
            return _not_found("booking", "Dùng booking code lấy từ kết quả book_seat")
        return {"status": "ok", "booking": booking.model_dump()}


# --------------------------------------------------------------------------- #
# Bọc thành LangChain tool (gắn với MỘT world cụ thể)
# --------------------------------------------------------------------------- #
def make_langchain_tools(world: FlightWorld) -> list:
    """Trả về 5 LangChain tool, mỗi tool gọi vào `world`.

    Docstring của từng hàm chính là mô tả tool mà model đọc được.
    Import langchain_core ở đây để file vẫn dùng/kiểm thử được khi chưa cài LangChain.
    """
    from langchain_core.tools import tool

    @tool
    def search_flights(origin: str, dest: str, date: str) -> dict:
        """Tìm chuyến bay theo tuyến và ngày. origin, dest là mã sân bay IATA (ví dụ SGN, DAD). date có dạng YYYY-MM-DD (ví dụ 2026-10-07)."""
        return world.search_flights(origin, dest, date)

    @tool
    def check_seat(flight: str) -> dict:
        """Kiểm tra số ghế còn, giá và chính sách hoàn vé của một chuyến bay (mã chuyến lấy từ search_flights)."""
        return world.check_seat(flight)

    @tool
    def book_seat(flight: str, seat: str) -> dict:
        """Giữ chỗ một ghế trên chuyến bay (ví dụ seat=12A). Trả về booking code; chưa thanh toán."""
        return world.book_seat(flight, seat)

    @tool
    def pay(code: str, method: str) -> dict:
        """Thanh toán một đặt chỗ đã giữ chỗ. method là corp_card hoặc personal_card. Không hoàn tác được."""
        return world.pay(code, method)

    @tool
    def get_booking(code: str) -> dict:
        """Tra cứu trạng thái một đặt chỗ theo booking code."""
        return world.get_booking(code)

    return [search_flights, check_seat, book_seat, pay, get_booking]


# --------------------------------------------------------------------------- #
# Chạy tay để quan sát đầu ra: python src/tools_flight.py --scenario happy
# --------------------------------------------------------------------------- #
def _demo(scenario: str) -> None:
    world = FlightWorld(scenario)
    print(f"=== kịch bản: {scenario} ===")

    def show(name: str, **kwargs: Any) -> dict[str, Any]:
        result = getattr(world, name)(**kwargs)
        shown = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        print(f"\n{name}({shown})\n  -> {json.dumps(result, ensure_ascii=False)}")
        return result

    show("search_flights", origin="SGN", dest="DAD", date="07/10")  # cố ý sai định dạng
    show("search_flights", origin="SGN", dest="DAD", date="2026-10-07")
    show("check_seat", flight="VJ604")
    booked = show("book_seat", flight="VJ604", seat="12A")
    code = booked.get("booking", {}).get("code", "XXXX")
    show("pay", code=code, method="corp_card")
    show("get_booking", code=code)
    print(f"\nTác dụng phụ đã xảy ra: {[t['tool'] for t in world.side_effects]}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # in tiếng Việt đúng trên console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chạy tay các tool đặt vé để quan sát đầu ra")
    parser.add_argument("--scenario", choices=SCENARIOS, default="happy")
    _demo(parser.parse_args().scenario)
