"""Kiểm thử tool mockup (Ngày 1). Chạy: pytest -v"""
import json

import pytest

from tools_flight import (
    SCENARIOS,
    SIDE_EFFECT_TOOLS,
    TOOL_NAMES,
    Booking,
    FlightWorld,
    make_langchain_tools,
)

DATE = "2026-10-07"


def feasible(world: FlightWorld) -> list[dict]:
    """Chuyến thoả ràng buộc của bài: SGN→DAD, 07/10, trước 12:00, ≤ 2 triệu."""
    flights = world.search_flights("SGN", "DAD", DATE)["flights"]
    return [f for f in flights if f["depart"] < "12:00" and f["price"] <= 2_000_000]


# ---------------------------------------------------------------- tìm chuyến
def test_search_tra_ve_danh_sach_sap_xep_theo_gio():
    w = FlightWorld("happy")
    r = w.search_flights("SGN", "DAD", DATE)
    assert r["status"] == "ok"
    departs = [f["depart"] for f in r["flights"]]
    assert departs == sorted(departs)
    assert len(r["flights"]) == 10  # loại chuyến sai ngày và sai điểm đến
    assert set(r["flights"][0]) == {"flight", "depart", "price", "refundable"}


def test_search_sai_dinh_dang_ngay_tra_loi_co_huong_dan():
    r = FlightWorld().search_flights("SGN", "DAD", "07/10")
    assert r["status"] == "invalid_param"
    assert r["param"] == "date"
    assert "YYYY-MM-DD" in r["hint"]


def test_search_ngay_khong_ton_tai_bi_tu_choi():
    assert FlightWorld().search_flights("SGN", "DAD", "2026-13-45")["status"] == "invalid_param"


def test_search_san_bay_la_bi_tu_choi():
    r = FlightWorld().search_flights("SGN", "XYZ", DATE)
    assert r["status"] == "invalid_param" and r["param"] == "dest"


def test_khong_co_chuyen_khac_voi_cong_cu_loi():
    """Rỗng thật sự phải là ok + [], KHÔNG phải error (chống lỗi 'tin dữ liệu sai')."""
    r = FlightWorld().search_flights("SGN", "HAN", DATE)
    assert r == {"status": "ok", "flights": []}


# ---------------------------------------------------------------- luồng đặt vé
def test_luong_dat_ve_day_du():
    w = FlightWorld("happy")
    booked = w.book_seat("VJ604", "12A")
    assert booked["status"] == "ok"
    b = Booking.model_validate(booked["booking"])
    assert (b.status, b.paid) == ("held", False)  # giữ chỗ chưa phải đã đặt xong

    paid = w.pay(b.code, "corp_card")
    assert paid["status"] == "ok"

    final = Booking.model_validate(w.get_booking(b.code)["booking"])
    assert final.status == "confirmed" and final.paid is True
    assert final.price == 1_290_000
    assert (final.depart_date, final.depart_time) == (DATE, "06:30")


def test_ma_dat_cho_co_dinh_theo_thu_tu():
    w = FlightWorld("happy")
    assert w.book_seat("VJ604", "12A")["booking"]["code"] == "4XJ2"
    assert w.book_seat("VJ604", "12B")["booking"]["code"] == "7KQ9"


def test_thanh_toan_hai_lan_khong_tru_tien_lan_hai():
    w = FlightWorld()
    code = w.book_seat("VJ604", "12A")["booking"]["code"]
    assert w.pay(code, "corp_card")["status"] == "ok"
    assert w.pay(code, "corp_card")["status"] == "already_paid"


def test_ghe_da_co_nguoi_dat_va_ghe_sai_dinh_dang():
    w = FlightWorld()
    assert w.book_seat("VJ604", "12A")["status"] == "ok"
    assert w.book_seat("VJ604", "12A")["status"] == "seat_taken"
    assert w.book_seat("VJ604", "99Z")["status"] == "invalid_param"


def test_ma_khong_ton_tai_tra_not_found_co_huong_dan():
    w = FlightWorld()
    for r in (w.get_booking("NOPE"), w.pay("NOPE", "corp_card"), w.check_seat("XX000")):
        assert r["status"] == "not_found" and r["hint"]


def test_phuong_thuc_thanh_toan_khong_hop_le():
    w = FlightWorld()
    code = w.book_seat("VJ604", "12A")["booking"]["code"]
    assert w.pay(code, "bitcoin")["status"] == "invalid_param"


# ---------------------------------------------------------------- kịch bản
def test_happy_chuyen_re_nhat_thoa_rang_buoc_la_vj604_va_hoan_duoc():
    f = min(feasible(FlightWorld("happy")), key=lambda x: x["price"])
    assert (f["flight"], f["price"], f["refundable"]) == ("VJ604", 1_290_000, True)


def test_timeout_check_seat_luon_loi_va_giong_het_nhau():
    w = FlightWorld("timeout")
    results = [w.check_seat("VN122") for _ in range(3)]
    assert all(r["status"] == "error" and r["error"] == "timeout" for r in results)
    assert results[0] == results[1] == results[2]  # tín hiệu để bộ phát hiện lặp bắt
    assert results[0]["hint"]


def test_approval_chi_con_vn122_khong_hoan_va_dat_hon_1_5_trieu():
    flights = feasible(FlightWorld("approval"))
    assert [f["flight"] for f in flights] == ["VN122"]
    assert flights[0]["refundable"] is False and flights[0]["price"] > 1_500_000


def test_env_change_vj604_het_ghe_sau_lan_tim_dau_tien():
    w = FlightWorld("env_change")
    flights = w.search_flights("SGN", "DAD", DATE)["flights"]
    assert "VJ604" in [f["flight"] for f in flights]  # agent vẫn thấy VJ604 trong kết quả
    assert w.check_seat("VJ604")["status"] == "sold_out"  # nhưng thực tế đã hết
    assert w.book_seat("VJ604", "12A")["status"] == "sold_out"
    assert w.check_seat("VJ606")["status"] == "ok"  # còn đường lui


def test_scenario_la_bi_tu_choi():
    with pytest.raises(ValueError):
        FlightWorld("khong-ton-tai")


# ---------------------------------------------------------------- nhật ký & tính xác định
def run_chuoi_goi(scenario: str) -> list[dict]:
    w = FlightWorld(scenario)
    w.search_flights("SGN", "DAD", DATE)
    w.check_seat("VJ604")
    code = w.book_seat("VJ604", "12A").get("booking", {}).get("code", "XXXX")
    w.pay(code, "corp_card")
    w.get_booking(code)
    return w.trace


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_xac_dinh_hai_lan_chay_cho_ket_qua_giong_het_nhau(scenario):
    assert run_chuoi_goi(scenario) == run_chuoi_goi(scenario)


def test_world_moi_khong_ro_trang_thai_tu_world_cu():
    a = FlightWorld()
    a.book_seat("VJ604", "12A")
    assert FlightWorld().bookings == {}


def test_trace_va_tac_dung_phu():
    w = FlightWorld()
    w.search_flights("SGN", "DAD", DATE)
    w.check_seat("VJ604")
    code = w.book_seat("VJ604", "12A")["booking"]["code"]
    w.book_seat("VJ604", "99Z")  # lỗi: không tính là tác dụng phụ
    w.pay(code, "corp_card")
    assert [t["tool"] for t in w.trace] == ["search_flights", "check_seat", "book_seat", "book_seat", "pay"]
    assert w.trace[0]["args"] == {"origin": "SGN", "dest": "DAD", "date": DATE}
    assert [t["tool"] for t in w.side_effects] == ["book_seat", "pay"]
    assert SIDE_EFFECT_TOOLS <= set(TOOL_NAMES)


# ---------------------------------------------------------------- LangChain
def test_langchain_tools_bao_dung_5_tool_va_dung_chung_world():
    pytest.importorskip("langchain_core")
    w = FlightWorld("happy")
    tools = make_langchain_tools(w)
    assert tuple(t.name for t in tools) == TOOL_NAMES
    assert all(t.description for t in tools)

    by_name = {t.name: t for t in tools}
    out = by_name["search_flights"].invoke({"origin": "SGN", "dest": "DAD", "date": DATE})
    assert out["status"] == "ok"
    assert w.trace[-1]["tool"] == "search_flights"  # lời gọi đi qua đúng world


def test_langchain_observation_la_json_hop_le():
    """Agent sẽ thấy ToolMessage; nội dung phải là chuỗi JSON parse được."""
    pytest.importorskip("langchain_core")
    tools = {t.name: t for t in make_langchain_tools(FlightWorld("happy"))}
    msg = tools["check_seat"].invoke(
        {"type": "tool_call", "name": "check_seat", "args": {"flight": "VJ604"}, "id": "call_1"}
    )
    assert msg.tool_call_id == "call_1"
    data = json.loads(msg.content)
    assert data["status"] == "ok" and data["flight"] == "VJ604"
