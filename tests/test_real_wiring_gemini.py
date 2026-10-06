"""Đường chạy GEMINI qua máy chủ Gemini giả (không tốn tiền). Cần: pip install langchain-google-genai."""
import csv

import pytest

pytest.importorskip("langchain_google_genai")

import check_api  # noqa: E402
import evaluate  # noqa: E402
from agent_hybrid import run_hybrid  # noqa: E402
from agent_plan import run_plan  # noqa: E402
from agent_react import run_react  # noqa: E402
from common import build_real_model  # noqa: E402
from fake_gemini_server import FakeGemini  # noqa: E402


@pytest.fixture()
def server(monkeypatch):
    with FakeGemini() as srv:
        monkeypatch.setenv("GOOGLE_API_KEY", srv.api_key)
        monkeypatch.setenv("GOOGLE_BASE_URL", srv.url)
        yield srv


def test_build_real_model_gui_dung_key_va_model(server):
    msg = build_real_model().invoke("Trả lời đúng một từ: OK")
    req = server.requests[-1]
    assert msg.content in ("OK", [{"type": "text", "text": "OK"}])
    assert req["path"] == "/v1beta/models/gemini-3.8-flash:generateContent" and req["auth"] == server.api_key


def test_ba_mau_chay_qua_giao_thuc_gemini(server):
    react = run_react("happy", model="real")
    assert (react.report["stop_letter"], react.report["rounds"]) == ("A", 4)
    assert "[V1] Suy luận : Cần tìm các chuyến SGN→DAD" in react.trace and "'type'" not in react.trace
    assert run_plan("happy", model="real").report["stop_letter"] == "A"
    hybrid = run_hybrid("env_change", model="real").report
    assert (hybrid["stop_letter"], hybrid["rounds"], hybrid["replans"]) == ("A", 2, 1)


def test_check_api_thanh_cong(server, capsys):
    assert check_api.main([]) == 0
    out = capsys.readouterr().out
    assert "Google Gemini API" in out and "gemini-3.8-flash" in out and "KẾT LUẬN: sẵn sàng" in out
    assert server.api_key not in out


def test_check_api_sai_key(server, monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-SAI-KEY-0000000000")
    assert check_api.main([]) == 1
    out = capsys.readouterr().out
    assert "API key bị từ chối" in out and "aistudio.google.com" in out and "SAI-KEY-0000000000" not in out


def test_evaluate_gemini(server, tmp_path, capsys):
    code = evaluate.main(["--model", "real", "--scenarios", "happy", "--patterns", "react", "plan",
                          "--repeats", "1", "--yes", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "Nhà cung cấp: Google (Gemini)" in out and "0.75 USD vào / 3.75 USD ra" in out
    with (tmp_path / "ket_qua_real-gemini-3.8-flash.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["pattern"], r["stop_letter"]) for r in rows] == [("react", "A"), ("plan", "A")]
