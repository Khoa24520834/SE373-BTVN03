"""Kiểm thử harness (Ngày 1). Chạy: pytest -v"""
import pytest

from harness import (
    Budget,
    Constraints,
    Harness,
    LoopDetector,
    Permission,
    StopReason,
    estimate_tokens,
    vnd,
)
from tools_flight import FlightWorld

DATE = "2026-10-07"


class FakeClock:
    """Đồng hồ giả để kiểm thử giới hạn thời gian mà không phải chờ."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make(scenario: str = "happy", **kwargs):
    world = FlightWorld(scenario)
    return world, Harness(world, **kwargs)


def call(h: Harness, w: FlightWorld, tool: str, **args):
    """Một vòng: harness bao quanh lời gọi tool thật."""
    return h.run_tool_call(tool, args, lambda: getattr(w, tool)(**args))


def search(h, w, date=DATE):
    return call(h, w, "search_flights", origin="SGN", dest="DAD", date=date)


# =========================================================== lớp 1: ràng buộc
def test_constraints_thoa_va_vi_pham_tung_rang_buoc():
    c = Constraints()
    assert c.violations(origin="SGN", dest="DAD", date=DATE, depart="06:30", price=1_290_000) == []
    assert len(c.violations(origin="SGN", dest="HUI", date="2026-10-08", depart="12:30", price=2_080_000)) == 4
    assert c.violations(origin="SGN", dest="DAD", date=DATE, depart="12:00", price=1)  # đúng 12:00 là vi phạm
    assert c.violations(origin="SGN", dest="DAD", date=DATE, depart="06:30", price=2_000_000) == []  # đúng trần là hợp lệ


def test_constraints_la_du_lieu_bat_bien_va_sinh_prompt():
    c = Constraints()
    with pytest.raises(Exception):
        c.max_price = 1  # type: ignore[misc]
    assert "2.000.000đ" in c.to_prompt() and "12:00" in c.to_prompt()
    assert "SGN" in c.request_text() and "rẻ nhất" in c.request_text()


# ============================================== lớp 2: tiêu chí hoàn thành bằng code
def test_is_done_chi_dung_khi_ve_da_thanh_toan_va_thoa_rang_buoc():
    w, h = make()
    assert not h.is_done().done
    code = w.book_seat("VJ604", "12A")["booking"]["code"]
    held = h.is_done()
    assert not held.done and "chưa có đặt chỗ nào đã thanh toán" in held.failures[0]  # giữ chỗ chưa phải xong
    w.pay(code, "corp_card")
    assert h.is_done().done and h.is_done().booking.code == code


def test_is_done_bat_ve_sai_rang_buoc_va_ve_thua():
    w, h = make()
    w.pay(w.book_seat("VJ612", "12A")["booking"]["code"], "corp_card")  # 2.080.000 > 2 triệu (đi vòng qua harness)
    c = h.is_done()
    assert not c.done and any("giá" in f for f in c.failures)

    w2, h2 = make()
    for flight in ("VJ604", "VJ606"):
        w2.pay(w2.book_seat(flight, "12A")["booking"]["code"], "corp_card")
    assert any("2 vé" in f for f in h2.is_done().failures)


def test_is_done_kiem_chung_cheo_gia_voi_check_seat():
    w, h = make()
    search(h, w)
    call(h, w, "check_seat", flight="VJ604")  # harness ghi nhớ giá thấy ở check_seat
    code = w.book_seat("VJ604", "12A")["booking"]["code"]
    w.pay(code, "corp_card")
    assert h.is_done().done
    w.bookings[code].price = 1_000_000  # giá đặt chỗ lệch giá đã thấy
    assert any("khác giá đã thấy" in f for f in h.is_done().failures)


# ======================================================== lớp 3: kiểm quyền
def test_ve_khong_hoan_vuot_han_muc_dung_cho_nguoi_va_chua_co_tac_dung_phu():
    w, h = make("approval")
    search(h, w)
    call(h, w, "check_seat", flight="VN122")
    out = call(h, w, "book_seat", flight="VN122", seat="12A")
    assert out.stop is StopReason.HUMAN and out.executed is False
    assert out.observation["status"] == "needs_approval"
    assert w.bookings == {} and w.side_effects == []  # kiểm quyền chạy TRƯỚC khi thực thi
    text = out.handoff.render()
    for needle in ("Kết E", "VN122", "1.850.000đ", "không hoàn", "1.500.000đ", "Duyệt đặt VN122 ghế 12A"):
        assert needle in text


def test_nguoi_duyet_roi_thi_chay_tiep_den_dich():
    w, h = make("approval")
    search(h, w)
    call(h, w, "book_seat", flight="VN122", seat="12A")  # dừng chờ duyệt
    h.approve("vn122")
    assert h.stop_reason is None
    booked = call(h, w, "book_seat", flight="VN122", seat="12A")
    assert booked.executed and booked.stop is None
    paid = call(h, w, "pay", code=booked.observation["booking"]["code"], method="corp_card")
    assert paid.stop is StopReason.GOAL


def test_chuyen_re_hoan_duoc_khong_can_duyet():
    w, h = make()
    search(h, w)
    assert h.check_permission("book_seat", {"flight": "VJ604", "seat": "12A"}).decision is Permission.ALLOW


def test_tu_choi_tool_khong_ton_tai_va_tham_so_sai_dinh_dang():
    w, h = make()
    out = h.run_tool_call("cancel_flight", {"code": "X"}, lambda: pytest.fail("không được thực thi"))
    assert out.observation["status"] == "denied" and "không tồn tại" in out.observation["hint"]
    out = call(h, w, "search_flights", origin="SGN", dest="DAD", date="07/10")
    assert out.observation["reason"] == "invalid_call" and "YYYY-MM-DD" in out.observation["hint"]
    assert w.trace == []  # cả hai lời gọi bị chặn trước khi chạm vào tool
    assert h.interventions["invalid_call"] == 2


def test_chan_ma_chuyen_bia_chuyen_sai_rang_buoc_va_dat_hai_ve():
    w, h = make()
    search(h, w)
    unseen = call(h, w, "book_seat", flight="VN999", seat="5C")
    assert unseen.observation["reason"] == "unknown_flight"

    expensive = call(h, w, "book_seat", flight="VJ612", seat="12A")  # sáng nhưng 2.080.000đ
    assert expensive.observation["reason"] == "violates_constraints" and "giá" in expensive.observation["hint"]

    afternoon = call(h, w, "book_seat", flight="QH118", seat="12A")  # rẻ nhưng buổi chiều
    assert "giờ bay" in afternoon.observation["hint"]
    assert w.bookings == {}

    assert call(h, w, "book_seat", flight="VJ604", seat="12A").executed
    second = call(h, w, "book_seat", flight="VJ606", seat="12A")
    assert second.observation["reason"] == "already_booked"


def test_khong_tra_tien_cho_ma_dat_cho_la():
    w, h = make()
    out = call(h, w, "pay", code="ZZZZ", method="corp_card")
    assert out.observation["reason"] == "unknown_booking" and w.side_effects == []


# ============================================== phát hiện lặp, bế tắc (kết C, D)
def test_lap_ba_lan_check_seat_timeout_thi_bao_dong_o_lan_thu_ba():
    w, h = make("timeout")
    search(h, w)
    assert call(h, w, "check_seat", flight="VN122").stop is None
    assert call(h, w, "check_seat", flight="VN122").stop is None  # hai lần là thử lại hợp lệ
    out = call(h, w, "check_seat", flight="VN122")
    assert out.stop is StopReason.LOOP and "lặp 3 lần" in out.detail
    text = out.handoff.render()
    assert "Kết C" in text and "BẤT THƯỜNG" in text and "timeout" in text and "×3" in text


def test_doi_cach_viet_nhung_cung_mot_loi_van_la_lap():
    w, h = make()
    outs = [call(h, w, "check_seat", flight=f"AA00{i}") for i in range(3)]
    assert [o.stop for o in outs] == [None, None, StopReason.LOOP]
    assert "cùng một lỗi" in outs[2].detail


def test_polling_get_booking_thay_doi_khong_bi_bao_nham():
    d = LoopDetector()
    held = {"status": "ok", "booking": {"status": "held"}}
    confirmed = {"status": "ok", "booking": {"status": "confirmed"}}
    assert d.check("get_booking", {"code": "X"}, held, 1) is None
    assert d.check("get_booking", {"code": "X"}, held, 2) is None
    assert d.check("get_booking", {"code": "X"}, confirmed, 3) is None  # observation đổi: polling hợp lệ
    d2 = LoopDetector()
    signals = [d2.check("get_booking", {"code": "X"}, held, 1) for _ in range(3)]
    assert signals[-1] is not None and signals[-1].kind == "LOOP"  # observation không đổi: lặp thật


def test_doi_tool_moi_vong_nhung_dung_yen_thi_la_be_tac():
    w, h = make()
    outs = [search(h, w, date=f"2026-10-{d:02d}") for d in range(8, 14)]  # 6 ngày khác nhau, đều sai
    assert [o.stop for o in outs[:5]] == [None] * 5
    assert outs[5].stop is StopReason.STALL
    assert "Kết D" in outs[5].handoff.render()


def test_progress_tang_theo_tien_do_dat_ve():
    w, h = make()
    assert h.progress() == 0
    search(h, w)
    assert h.progress() == 40  # có chuyến thoả cả 4 ràng buộc, chưa giữ chỗ
    call(h, w, "book_seat", flight="VJ604", seat="12A")
    assert h.progress() == 41
    call(h, w, "pay", code="4XJ2", method="corp_card")
    assert h.progress() == 42


# ================================================================ ngân sách (kết B)
def test_ngan_sach_theo_vong_token_chi_phi_va_thoi_gian():
    w, h = make(budget=Budget(max_rounds=3))
    for i in range(3):
        assert h.budget_exceeded() is None
        h.charge_model_call(100, 50)
    assert h.budget_exceeded() == "rounds 3/3"

    w, h = make(budget=Budget(max_tokens=1000))
    h.charge_model_call(600, 500)
    assert h.budget_exceeded() == "tokens 1100/1000"

    w, h = make(budget=Budget(max_cost_usd=0.01))
    h.charge_model_call(2000, 400)  # 0,006 + 0,006
    assert h.budget_exceeded().startswith("cost_usd")

    clock = FakeClock()
    w, h = make(budget=Budget(max_seconds=30), clock=clock)
    clock.t = 29.0
    assert h.budget_exceeded() is None
    clock.t = 31.0
    assert h.budget_exceeded().startswith("seconds")


def test_het_ngan_sach_khi_khong_co_tin_hieu_nao_khac():
    w, h = make(budget=Budget(max_tool_calls=2))
    search(h, w)
    out = call(h, w, "check_seat", flight="VJ604")
    assert out.stop is StopReason.BUDGET and out.detail == "tool_calls 2/2"
    assert "Kết B" in out.handoff.render()


def test_ngan_sach_kiem_cuoi_cung_de_khong_mat_chan_doan():
    """Cùng chạm trần VÀ lặp: phải báo LẶP (nguyên nhân), không báo HẾT NGÂN SÁCH."""
    w, h = make("timeout", budget=Budget(max_tool_calls=3))
    outs = [call(h, w, "check_seat", flight="VN122") for _ in range(3)]
    assert outs[2].stop is StopReason.LOOP


def test_dat_xong_dung_ngay_ca_khi_cung_luc_cham_tran():
    w, h = make(budget=Budget(max_tool_calls=4))
    search(h, w)
    code = call(h, w, "book_seat", flight="VJ604", seat="12A").observation["booking"]["code"]
    call(h, w, "check_seat", flight="VJ604")
    out = call(h, w, "pay", code=code, method="corp_card")  # lời gọi thứ 4: chạm trần và đạt mục tiêu
    assert out.stop is StopReason.GOAL


def test_uoc_luong_token_va_dinh_dang_tien():
    assert estimate_tokens("abcd") == 1 and estimate_tokens("a" * 9) == 3
    assert vnd(1_290_000) == "1.290.000đ"


# ============================================================ chống bịa (S5)
def test_bia_dat_ve_khong_goi_tool_bi_tu_choi_va_liet_ke_cho_bia():
    w, h = make()
    verdict = h.verify_final("Done! Booked VN999, seat 5C, for 1,200,000 VND.")
    assert verdict.accept is False and verdict.stop is None
    joined = " ".join(verdict.problems)
    for needle in ("VN999", "5C", "1.200.000đ", "đã đặt chỗ", "Chưa đạt tiêu chí"):
        assert needle in joined or needle in verdict.feedback
    again = h.verify_final("Done! Booked VN999, seat 5C, for 1,200,000 VND.")
    assert again.stop is StopReason.STALL  # khẳng định xong hai lần mà is_done() vẫn sai
    assert "Kết D" in h.handoff.render()
    assert h.interventions["false_claim"] == 2


def test_cau_tra_loi_dung_su_that_duoc_chap_nhan_khong_loi():
    w, h = make()
    search(h, w)
    code = call(h, w, "book_seat", flight="VJ604", seat="12A").observation["booking"]["code"]
    call(h, w, "pay", code=code, method="corp_card")
    verdict = h.verify_final("Đã đặt vé VJ604 ghế 12A lúc 06:30 ngày 2026-10-07, giá 1.290.000đ, mã 4XJ2, đã thanh toán.")
    assert verdict.accept and verdict.stop is StopReason.GOAL and verdict.problems == []


def test_ve_da_dat_dung_nhung_cau_tra_loi_co_so_lieu_la_van_bi_danh_dau():
    w, h = make()
    search(h, w)
    code = call(h, w, "book_seat", flight="VJ604", seat="12A").observation["booking"]["code"]
    call(h, w, "pay", code=code, method="corp_card")
    verdict = h.verify_final("Đã đặt VJ604 ghế 5C, giá 900.000đ.")
    assert verdict.accept  # mục tiêu thật sự đạt
    assert any("5C" in p for p in verdict.problems) and any("900.000đ" in p for p in verdict.problems)


# ============================================================ bàn giao và báo cáo
def test_ban_giao_ghi_ro_tac_dung_phu_da_xay_ra():
    w, h = make("timeout")
    search(h, w)
    call(h, w, "book_seat", flight="VJ604", seat="12A")  # đã giữ chỗ nhưng chưa trả tiền
    for _ in range(3):
        out = call(h, w, "check_seat", flight="VJ604")
    text = out.handoff.render()
    assert "giữ chỗ VJ604 ghế 12A" in text and "CHƯA thanh toán" in text
    assert "chưa có (chưa giữ chỗ" not in text
    assert out.handoff.to_dict()["reason"] == "loop"


def test_ban_giao_khi_dat_xong_khong_co_cau_hoi():
    w, h = make()
    search(h, w)
    code = call(h, w, "book_seat", flight="VJ604", seat="12A").observation["booking"]["code"]
    out = call(h, w, "pay", code=code, method="corp_card")
    assert out.stop is StopReason.GOAL and out.handoff.question == ""
    assert "Kết A" in out.handoff.render() and "bình thường" in out.handoff.render()


def test_summary_dung_tu_du_lieu_da_kiem_chung():
    w, h = make()
    assert h.summary().startswith("Chưa hoàn thành")
    search(h, w)
    code = call(h, w, "book_seat", flight="VJ604", seat="12A").observation["booking"]["code"]
    call(h, w, "pay", code=code, method="corp_card")
    s = h.summary()
    assert "VJ604" in s and "4XJ2" in s and "1.290.000đ" in s and "đã thanh toán" in s
    assert h.verify_final(s).problems == []  # câu trả lời dựng từ dữ liệu thật thì không bao giờ bị đánh dấu bịa


# ================================================== chạy trọn vẹn theo 4 kịch bản
def test_happy_dat_xong_o_vong_pay_va_khong_can_hoi_nguoi():
    w, h = make("happy")
    outs = [search(h, w), call(h, w, "check_seat", flight="VJ604"),
            call(h, w, "book_seat", flight="VJ604", seat="12A")]
    assert all(o.stop is None for o in outs)
    out = call(h, w, "pay", code="4XJ2", method="corp_card")
    assert out.stop is StopReason.GOAL
    rep = h.report()
    assert (rep["stop_letter"], rep["success"], rep["tool_calls"]) == ("A", True, 4)
    assert rep["side_effects"] == ["book_seat", "pay"]


def test_env_change_doi_sang_vj606_thi_van_dat_duoc():
    w, h = make("env_change")
    search(h, w)
    assert call(h, w, "check_seat", flight="VJ604").observation["status"] == "sold_out"
    assert call(h, w, "book_seat", flight="VJ604", seat="12A").observation["status"] == "sold_out"
    booked = call(h, w, "book_seat", flight="VJ606", seat="12A")
    out = call(h, w, "pay", code=booked.observation["booking"]["code"], method="corp_card")
    assert out.stop is StopReason.GOAL and "VJ606" in h.summary()


def test_report_co_du_truong_cho_evaluate():
    w, h = make("timeout")
    h.charge_model_call(1000, 100)
    search(h, w)
    for _ in range(3):
        call(h, w, "check_seat", flight="VN122")
    rep = h.report()
    assert set(rep) == {"stop", "stop_letter", "stop_detail", "success", "rounds", "tool_calls",
                        "tokens", "cost_usd", "seconds", "interventions", "ungrounded", "side_effects"}
    assert rep["stop"] == "loop" and rep["success"] is False and rep["tokens"] == 1100
