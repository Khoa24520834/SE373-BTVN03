"""Kiểm thử Plan-then-Execute và replan (Ngày 2, buổi sáng). Chạy: pytest -v"""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import tools_flight
from agent_plan import (
    PLANNER_STYLES,
    Plan,
    PlanError,
    PlanExecuteAgent,
    ScriptedPlanner,
    lint_plan,
    make_planner_model,
    parse_plan,
    run_plan,
)
from agent_react import run_react
from harness import Budget, Constraints, Harness, StopReason
from tools_flight import FlightWorld, make_langchain_tools

DATE = "2026-10-07"


class TextPlanner(ScriptedPlanner):
    """Planner giả trả về đúng một đoạn chữ cho trước (không có usage_metadata)."""

    text: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])


def observations(result):
    return [json.loads(m.content) for m in result.messages if isinstance(m, ToolMessage)]


def canonical_plan(**overrides) -> Plan:
    steps = [
        {"id": 1, "tool": "search_flights", "args": {"origin": "SGN", "dest": "DAD", "date": DATE}},
        {"id": 2, "tool": "check_seat", "args": {"flight": "$best"}},
        {"id": 3, "tool": "book_seat", "args": {"flight": "$best", "seat": "12A"}},
        {"id": 4, "tool": "pay", "args": {"code": "$booking_code", "method": "corp_card"}},
    ]
    return Plan.model_validate({"steps": overrides.get("steps", steps)})


# ------------------------------------------- bốn kịch bản đánh giá, Plan thuần
def test_s1_happy_mot_lan_goi_model_roi_chay_het_ke_hoach():
    r = run_plan("happy")
    rep = r.report
    assert (rep["stop_letter"], rep["success"], rep["rounds"], rep["tool_calls"]) == ("A", True, 1, 4)
    assert (rep["plan_steps"], rep["replans"]) == (4, 0) and r.pattern == "plan"
    assert rep["side_effects"] == ["book_seat", "pay"] and rep["interventions"] == {}
    assert "4XJ2" in r.final_answer and r.handoff == ""
    assert r.trace.count("[Kế hoạch]") == 1  # node duyệt chạy lại khi resume nhưng planner KHÔNG chạy lại
    assert "[Người duyệt] Đồng ý kế hoạch." in r.trace


def test_plan_ton_it_model_call_va_token_hon_react_o_s1():
    plan, react = run_plan("happy").report, run_react("happy").report
    assert plan["rounds"] < react["rounds"] and plan["tokens"] < react["tokens"]
    assert plan["tool_calls"] == react["tool_calls"] == 4


def test_s2_timeout_thu_lai_roi_harness_bat_lap_o_lan_thu_ba():
    r = run_plan("timeout")
    assert r.report["stop_letter"] == "C" and r.report["side_effects"] == []
    assert (r.report["rounds"], r.report["tool_calls"]) == (1, 4)  # search + ba lần check_seat y hệt
    assert "×3" in r.handoff


def test_s4_approval_dung_cho_nguoi_truoc_khi_co_tac_dung_phu():
    r = run_plan("approval")
    assert r.report["stop_letter"] == "E" and r.report["side_effects"] == [] and r.world.bookings == {}
    assert (r.report["rounds"], r.report["tool_calls"]) == (1, 3)
    assert "Duyệt đặt VN122 ghế 12A" in r.handoff


def test_s6_plan_thuan_ke_hoach_loi_thoi_dung_o_ket_d():
    r = run_plan("env_change")
    rep = r.report
    assert (rep["stop_letter"], rep["success"], rep["side_effects"]) == ("D", False, [])
    assert rep["stop_detail"].startswith("kế hoạch lỗi thời: bước 2 (check_seat) kỳ vọng ok, nhận sold_out")
    assert (rep["rounds"], rep["tool_calls"], rep["replans"]) == (1, 2, 0)
    assert "replan" in r.handoff and "check_seat(flight='VJ604') → sold_out" in r.handoff
    assert "rẻ nhất: VJ606" in r.handoff and "đã hết ghế: VJ604" in r.handoff  # bàn giao không gợi ý chuyến đã hỏng


# ----------------------------------------------- mẫu Lai: Plan + replan
def test_s6_co_replan_thich_nghi_sang_vj606_chi_hai_lan_goi_model():
    r = run_plan("env_change", max_replans=1)
    rep = r.report
    assert (rep["stop_letter"], rep["success"], rep["rounds"], rep["tool_calls"], rep["replans"]) == ("A", True, 2, 5, 1)
    assert r.pattern == "hybrid" and "VJ606" in r.final_answer
    assert r.trace.count("[Kế hoạch mới]") == 1 and rep["plan_steps"] == 3  # kế hoạch mới bỏ bước search
    assert [o["status"] for o in observations(r)][:3] == ["ok", "sold_out", "ok"]


def test_lai_ton_it_model_call_hon_react_o_s6():
    hybrid, react = run_plan("env_change", max_replans=1).report, run_react("env_change").report
    assert hybrid["success"] and react["success"] and hybrid["rounds"] < react["rounds"]


def test_replan_khong_cuu_duoc_loi_lap_va_khong_duyet_lai_ke_hoach():
    r = run_plan("timeout", max_replans=2)
    assert r.report["stop_letter"] == "C" and r.report["replans"] == 0 and r.report["rounds"] == 1
    calls = []
    run_plan("env_change", max_replans=1, approver=lambda p: calls.append(p) or True)
    assert len(calls) == 1  # chỉ duyệt kế hoạch gốc; kế hoạch thay thế chỉ cần qua lint


def test_replan_het_luot_van_lech_thi_dung_o_ket_d(monkeypatch):
    original = tools_flight._base_flights

    def both_sold_out():
        flights = original()
        for f in flights:
            if f.code == "VJ606":
                f.seats = 0  # VJ606 cũng hết ghế: kế hoạch thay thế cũng lỗi thời
        return flights

    monkeypatch.setattr(tools_flight, "_base_flights", both_sold_out)
    r = run_plan("env_change", max_replans=1)
    assert r.report["stop_letter"] == "D" and "đã lập lại kế hoạch 1 lần vẫn lệch" in r.report["stop_detail"]
    assert r.report["side_effects"] == [] and "đã hết ghế: VJ604, VJ606" in r.handoff


def test_khong_con_chuyen_nao_thoa_rang_buoc_thi_dung_va_hoi_nho_noi_rang_buoc():
    r = run_plan("happy", constraints=Constraints(max_price=1_000_000))
    assert r.report["stop_letter"] == "D" and "không còn chuyến nào thoả ràng buộc" in r.report["stop_detail"]
    assert r.report["tool_calls"] == 1 and "nới ràng buộc" in r.handoff


def test_khong_thu_lai_thi_mot_loi_tool_lam_hong_ke_hoach():
    r = run_plan("timeout", max_retries=0)  # đúng rủi ro "lỗi ở bước đầu làm hỏng toàn bộ sau"
    assert r.report["stop_letter"] == "D" and "nhận error" in r.report["stop_detail"]
    assert r.report["tool_calls"] == 2


# ---------------------------------------------------------- người duyệt (interrupt)
def test_nguoi_duyet_nhan_dung_ke_hoach_va_uoc_luong_chi_phi_truoc_khi_chay():
    payloads = []
    r = run_plan("happy", approver=lambda p: payloads.append(p) or True)
    assert len(payloads) == 1
    assert payloads[0]["estimate"] == {"steps": 4, "tool_calls": 4, "irreversible_steps": [3, 4]}
    assert "check_seat(flight='$best')" in payloads[0]["plan"]
    assert r.report["success"]


@pytest.mark.parametrize("decision, shown", [("quá đắt", "quá đắt"), (False, "không nêu lý do")])
def test_tu_choi_ke_hoach_thi_khong_tool_nao_duoc_chay(decision, shown):
    r = run_plan("happy", approver=lambda p: decision)
    assert r.report["stop_letter"] == "E" and r.report["stop"] == "needs_human"
    assert r.world.trace == [] and r.report["tool_calls"] == 0 and r.report["rounds"] == 1
    assert f"Từ chối kế hoạch: {shown}" in r.trace and "Người duyệt từ chối kế hoạch" in r.handoff


# ------------------------------------------------- kiểm tra kế hoạch bằng code
@pytest.mark.parametrize("style, reason", [
    ("wrong_date", "khác ràng buộc"),
    ("invalid_tool", "không tồn tại"),
    ("no_pay", "thiếu bước pay"),
])
def test_ke_hoach_sai_bi_chan_truoc_ca_khi_nguoi_duyet(style, reason):
    calls = []
    r = run_plan("happy", planner_style=style, approver=lambda p: calls.append(p) or True)
    assert r.report["stop_letter"] == "D" and reason in r.report["stop_detail"]
    assert r.world.trace == [] and calls == []  # không tool nào chạy, người duyệt cũng không bị làm phiền
    assert "[Kế hoạch]" in r.trace and "chạy lại hay chỉnh prompt/model" in r.handoff


def test_planner_tra_ve_rac_thi_dung_o_ket_d_va_van_ghi_lai_cau_tra_loi():
    r = run_plan("happy", model=TextPlanner(text="Xin chào, tôi sẽ đặt vé giúp bạn!"))
    assert r.report["stop_letter"] == "D" and "không sinh được kế hoạch hợp lệ" in r.report["stop_detail"]
    assert r.report["tokens"] > 0  # không có usage_metadata thì harness tự ước lượng
    assert "Xin chào" in r.trace and r.world.trace == []


def test_parse_plan():
    text = 'Đây là kế hoạch:\n```json\n{"steps": [{"id": 1, "tool": "pay", "args": {"code": "X", "method": "corp_card"}}]}\n```'
    plan = parse_plan(text)
    assert plan.steps[0].tool == "pay" and plan.steps[0].expect == "ok" and plan.steps[0].why == ""
    for bad in ("không có json", "{steps: [}", '{"steps": "không phải danh sách"}', '{"khac": 1}'):
        with pytest.raises(PlanError):
            parse_plan(bad)


def make_harness(scenario="happy", **kw):
    world = FlightWorld(scenario)
    return world, Harness(world, **kw)


def test_lint_ke_hoach_dung_va_cac_loi_thuong_gap():
    world, h = make_harness()
    assert lint_plan(canonical_plan(), h) == []

    def problems(steps):
        return " | ".join(lint_plan(canonical_plan(steps=steps), h))

    s = lambda i, tool, **args: {"id": i, "tool": tool, "args": args}  # noqa: E731
    search = s(1, "search_flights", origin="SGN", dest="DAD", date=DATE)
    check = s(2, "check_seat", flight="$best")
    book = s(3, "book_seat", flight="$best", seat="12A")
    pay = s(4, "pay", code="$booking_code", method="corp_card")

    assert "kế hoạch rỗng" in problems([])
    assert "dùng $best khi chưa có kết quả search_flights" in problems([check, book, pay])
    assert "dùng $booking_code khi chưa có bước book_seat" in problems([search, check, pay, book])
    assert "biến $abc không hợp lệ" in problems([search, s(2, "check_seat", flight="$abc"), book, pay])
    assert "origin='HAN' khác ràng buộc SGN" in problems([s(1, "search_flights", origin="HAN", dest="DAD", date=DATE), check, book, pay])
    assert "YYYY-MM-DD" in problems([s(1, "search_flights", origin="SGN", dest="DAD", date="07/10"), check, book, pay])
    assert "thiếu bước book_seat" in problems([search, check, pay])
    assert "tối đa 8" in problems([search] + [check] * 8 + [book, pay])


def test_lint_ke_hoach_thay_the_duoc_bo_qua_buoc_da_lam():
    world, h = make_harness()
    steps_after_search = canonical_plan().steps[1:]
    assert "dùng $best khi chưa có kết quả" in " ".join(lint_plan(Plan(steps=steps_after_search), h))
    h.run_tool_call("search_flights", {"origin": "SGN", "dest": "DAD", "date": DATE},
                    lambda: world.search_flights("SGN", "DAD", DATE))
    assert lint_plan(Plan(steps=steps_after_search), h) == []  # đã có kết quả tìm kiếm
    h.run_tool_call("book_seat", {"flight": "VJ604", "seat": "12A"}, lambda: world.book_seat("VJ604", "12A"))
    assert lint_plan(Plan(steps=[canonical_plan().steps[3]]), h) == []  # đã giữ chỗ: chỉ còn pay


def test_planner_gia_biet_bo_buoc_da_lam_khi_thay_ke_hoach():
    planner = ScriptedPlanner()
    ask = lambda text: json.loads(planner.invoke([("human", text)]).content)["steps"]  # noqa: E731
    assert [s["tool"] for s in ask("đặt vé")] == ["search_flights", "check_seat", "book_seat", "pay"]
    assert [s["tool"] for s in ask("[REPLAN] lệch")] == ["check_seat", "book_seat", "pay"]
    assert [s["tool"] for s in ask("[REPLAN] lệch\nĐã giữ chỗ: 4XJ2")] == ["pay"]


# ---------------------------------------------------------- an toàn và tính chất chung
def test_ngan_sach_cat_ke_hoach_giua_chung():
    r = run_plan("env_change", max_replans=1, budget=Budget(max_rounds=1))
    assert r.report["stop_letter"] == "B" and r.report["stop_detail"] == "rounds 1/1" and r.report["tool_calls"] == 1
    r = run_plan("env_change", max_replans=1, budget=Budget(max_rounds=2))
    assert r.report["stop_detail"] == "rounds 2/2" and r.report["tool_calls"] == 3
    assert r.report["side_effects"] == [] and "Kết B" in r.handoff


def test_langgraph_recursion_limit_la_luoi_an_toan_van_giu_duoc_trace():
    r = run_plan("happy", recursion_limit=3)
    assert r.report["stop_letter"] == "B" and "GraphRecursionError" in r.report["stop_detail"]
    assert "[Kế hoạch]" in r.trace


def test_tool_nem_loi_thi_thu_lai_roi_harness_bat_lap(monkeypatch):
    def boom(self, flight):
        raise RuntimeError("máy chủ ghế sập")

    monkeypatch.setattr(FlightWorld, "check_seat", boom)
    r = run_plan("happy")
    errors = [o for o in observations(r) if o["status"] == "error"]
    assert errors and errors[0]["error"] == "tool_exception" and "RuntimeError" in errors[0]["hint"]
    assert r.report["stop_letter"] == "C"


def test_verify_khi_chay_het_ke_hoach_ma_chua_dat_muc_tieu():
    world, h = make_harness()
    agent = PlanExecuteAgent(h, make_langchain_tools(world), ScriptedPlanner(), request="x")
    agent.verify_node({})
    assert h.stop_reason is StopReason.STALL and "đã chạy hết kế hoạch nhưng chưa đạt" in h.stop_detail


def test_chay_hai_lan_cho_trace_va_bao_cao_giong_het_nhau():
    for kwargs in ({}, {"max_replans": 1}):
        a, b = run_plan("env_change", **kwargs), run_plan("env_change", **kwargs)
        assert a.trace == b.trace
        strip = lambda rep: {k: v for k, v in rep.items() if k != "seconds"}  # noqa: E731
        assert strip(a.report) == strip(b.report)


@pytest.mark.parametrize("scenario", tools_flight.SCENARIOS)
@pytest.mark.parametrize("replans", [0, 1, 2])
def test_moi_to_hop_deu_ket_thuc_va_khong_co_tac_dung_phu_ngoai_y_muon(scenario, replans):
    r = run_plan(scenario, max_replans=replans)
    assert r.report["stop"] is not None and r.final_answer
    if r.report["stop_letter"] != "A":  # dừng mà chưa xong thì không được để lại vé thừa đã trả tiền
        assert "pay" not in r.report["side_effects"]
    assert len(r.world.bookings) <= 1  # chính sách: tối đa một vé


def test_make_planner_model(monkeypatch):
    assert isinstance(make_planner_model("fake", "no_pay"), ScriptedPlanner)
    sentinel = object()
    assert make_planner_model(sentinel) is sentinel
    assert set(PLANNER_STYLES) == {"canonical", "wrong_date", "invalid_tool", "no_pay"}
    with pytest.raises(ValueError):
        make_planner_model("fake", "khong-co")
    with pytest.raises(ValueError):
        make_planner_model("abc")
    monkeypatch.delenv("SE373_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="SE373_MODEL"):
        make_planner_model("real")
