"""Kiểm thử agent ReAct + middleware harness (Ngày 1, buổi tối). Chạy: pytest -v"""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_react import run_react
from common import STYLES, ScriptedModel, make_model
from harness import Budget
from tools_flight import FlightWorld


# ----------------------------------------------------------------- tiện ích
def call(call_id: str, name: str, **args):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai(text: str = "", *calls) -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


class ListModel(ScriptedModel):
    """Phát lại một kịch bản AIMessage cho trước; hết kịch bản thì lặp lại tin cuối."""

    script: list = []

    def _decide(self, messages):
        n = sum(isinstance(m, AIMessage) for m in messages)
        return self.script[min(n, len(self.script) - 1)].model_copy(deep=True)


def tool_messages(result):
    return [m for m in result.messages if isinstance(m, ToolMessage)]


def observations(result):
    return [json.loads(m.content) for m in tool_messages(result)]


# ------------------------------------------- bốn kịch bản đánh giá (khớp kế hoạch)
def test_s1_happy_dat_xong_ket_a_sau_bon_vong():
    r = run_react("happy")
    rep = r.report
    assert (rep["stop_letter"], rep["success"]) == ("A", True)
    assert (rep["rounds"], rep["tool_calls"]) == (4, 4)
    assert rep["side_effects"] == ["book_seat", "pay"] and rep["interventions"] == {}
    assert "4XJ2" in r.final_answer and "VJ604" in r.final_answer
    assert r.handoff == ""


def test_s2_timeout_ket_c_o_vong_v4_va_chua_co_tac_dung_phu():
    r = run_react("timeout")
    assert r.report["stop_letter"] == "C" and r.report["success"] is False
    assert r.report["rounds"] == 4  # V1 search, V2 V3 V4 check_seat giống hệt → báo động ở V4
    assert r.report["side_effects"] == []
    assert "Kết C" in r.handoff and "×3" in r.handoff


def test_s4_approval_ket_e_va_khong_dat_gi_ca():
    r = run_react("approval")
    assert r.report["stop_letter"] == "E" and r.report["stop"] == "needs_human"
    assert r.report["side_effects"] == [] and r.world.bookings == {}
    assert "Duyệt đặt VN122 ghế 12A" in r.handoff
    assert observations(r)[-1]["status"] == "needs_approval"


def test_s6_env_change_react_thich_nghi_sang_vj606():
    r = run_react("env_change")
    assert r.report["stop_letter"] == "A" and r.report["rounds"] == 5
    assert "VJ606" in r.final_answer
    assert [o["status"] for o in observations(r)][:2] == ["ok", "sold_out"]


# --------------------------------------------------- các kiểu lỗi của model
def test_s5_bia_khong_goi_tool_nao_bi_chan_o_lan_khang_dinh_thu_hai():
    r = run_react("happy", style="hallucinate")
    rep = r.report
    assert (rep["stop_letter"], rep["success"]) == ("D", False)
    assert (rep["rounds"], rep["tool_calls"]) == (2, 0)
    assert rep["side_effects"] == [] and rep["interventions"] == {"false_claim": 2}
    assert any("VN999" in p for p in rep["ungrounded"])
    assert len(rep["ungrounded"]) == len(set(rep["ungrounded"]))
    assert "khẳng định đã xong" in r.handoff


def test_greedy_bo_qua_rang_buoc_bi_tu_choi_roi_doi_chuyen():
    r = run_react("happy", style="greedy")
    assert r.report["stop_letter"] == "A" and r.report["interventions"] == {"denied": 1}
    assert "VJ604" in r.final_answer and "VJ608" not in r.final_answer
    denied = [o for o in observations(r) if o["status"] == "denied"]
    assert denied and "giờ bay 20:15" in denied[0]["hint"]


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("scenario", ["happy", "timeout", "approval", "env_change"])
def test_moi_to_hop_deu_ket_thuc_va_ghi_ly_do_dung(scenario, style):
    r = run_react(scenario, style=style)
    assert r.report["stop"] is not None  # không có lần chạy nào kết thúc mà không rõ lý do
    assert r.final_answer


# ------------------------------------------------------------ ngân sách, lưới an toàn
def test_ngan_sach_vong_cat_dung_luc():
    r = run_react("happy", budget=Budget(max_rounds=2))
    assert r.report["stop_letter"] == "B" and r.report["stop_detail"] == "rounds 2/2"
    assert r.report["rounds"] == 2 and r.report["side_effects"] == []
    assert "Kết B" in r.handoff


def test_ngan_sach_tool_call():
    r = run_react("happy", budget=Budget(max_tool_calls=3))
    assert r.report["stop_letter"] == "B" and r.report["stop_detail"] == "tool_calls 3/3"
    assert r.report["side_effects"] == ["book_seat"]  # đã giữ chỗ nhưng chưa trả tiền: bàn giao phải nói rõ
    assert "CHƯA thanh toán" in r.handoff


def test_langgraph_recursion_limit_la_luoi_an_toan_van_giu_duoc_trace():
    r = run_react("happy", recursion_limit=3)
    assert r.report["stop_letter"] == "B" and "GraphRecursionError" in r.report["stop_detail"]
    assert r.messages and "[V1]" in r.trace


# ---------------------------------------------- model làm điều ngoài dự kiến
def test_tool_khong_ton_tai_bi_chan_truoc_khi_framework_xu_ly():
    model = ListModel(script=[ai("Huỷ chuyến.", call("c1", "cancel_flight", code="X")),
                              AIMessage(content="Không thể huỷ chuyến.")])
    r = run_react("happy", model=model)
    first = observations(r)[0]
    assert first["status"] == "denied" and "không tồn tại" in first["hint"]
    assert r.world.trace == []  # tool thật không bị chạm tới
    assert r.report["interventions"]["invalid_call"] == 1


def test_goi_tool_song_song_van_tuan_thu_harness():
    model = ListModel(script=[
        ai("", call("c1", "search_flights", origin="SGN", dest="DAD", date="2026-10-07")),
        ai("", call("c2", "check_seat", flight="VJ604"), call("c3", "check_seat", flight="VJ606")),
        ai("", call("c4", "book_seat", flight="VJ604", seat="12A")),
        ai("", call("c5", "pay", code="4XJ2", method="corp_card")),
    ])
    r = run_react("happy", model=model)
    assert r.report["stop_letter"] == "A" and r.report["tool_calls"] == 5
    assert {m.tool_call_id for m in tool_messages(r)} == {"c1", "c2", "c3", "c4", "c5"}


def test_goi_song_song_sau_khi_cho_duyet_thi_bo_qua_va_van_du_tool_message():
    model = ListModel(script=[
        ai("", call("c1", "search_flights", origin="SGN", dest="DAD", date="2026-10-07")),
        ai("", call("c2", "book_seat", flight="VN122", seat="12A"), call("c3", "check_seat", flight="VJ612")),
    ])
    r = run_react("approval", model=model)
    assert r.report["stop_letter"] == "E" and r.report["side_effects"] == []
    # API model yêu cầu MỌI tool_call đều có ToolMessage tương ứng
    assert {m.tool_call_id for m in tool_messages(r)} == {"c1", "c2", "c3"}


def test_tool_nem_loi_thanh_observation_co_cau_truc(monkeypatch):
    def boom(self, flight):
        raise RuntimeError("máy chủ ghế sập")

    monkeypatch.setattr(FlightWorld, "check_seat", boom)
    r = run_react("happy")  # model giả thử lại y nguyên → ba lần giống hệt → LOOP
    obs = [o for o in observations(r) if o["status"] == "error"]
    assert obs and obs[0]["error"] == "tool_exception" and "RuntimeError" in obs[0]["hint"]
    assert r.report["stop_letter"] == "C"


# ---------------------------------------------------- tính chất chung
def test_chay_hai_lan_cho_trace_va_bao_cao_giong_het_nhau():
    a, b = run_react("env_change"), run_react("env_change")
    assert a.trace == b.trace
    strip = lambda rep: {k: v for k, v in rep.items() if k != "seconds"}  # noqa: E731
    assert strip(a.report) == strip(b.report)


def test_chi_phi_lich_su_input_token_tang_dan_theo_vong():
    r = run_react("happy")
    inputs = [m.usage_metadata["input_tokens"] for m in r.messages
              if isinstance(m, AIMessage) and m.usage_metadata]
    assert len(inputs) == 4 and inputs == sorted(set(inputs))  # mỗi vòng gửi lại toàn bộ lịch sử


def test_trace_theo_khuon_suy_luan_hanh_dong_quan_sat():
    t = run_react("approval").trace
    for needle in ("[Yêu cầu]", "[V1] Suy luận :", "[V1] Hành động: search_flights(", "[V1] Quan sát :",
                   "── Harness dừng vòng lặp ──", "Kết E"):
        assert needle in t


def test_make_model(monkeypatch):
    assert isinstance(make_model("fake", "greedy"), ScriptedModel)
    sentinel = object()
    assert make_model(sentinel) is sentinel
    with pytest.raises(ValueError):
        make_model("fake", "khong-co")
    with pytest.raises(ValueError):
        make_model("abc")
    monkeypatch.delenv("SE373_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="SE373_MODEL"):
        make_model("real")
    monkeypatch.setenv("SE373_MODEL", "provider:model-x")
    assert make_model("real") == "provider:model-x"
