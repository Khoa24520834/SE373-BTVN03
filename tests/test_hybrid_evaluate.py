"""Kiểm thử mẫu Lai (agent_hybrid) và công cụ đánh giá (evaluate). Chạy: pytest -v"""
import csv

import pytest

import evaluate
import tools_flight
from agent_hybrid import run_hybrid
from evaluate import COLUMNS, SCENARIO_ID, evaluate as run_eval, summarize, to_markdown
from harness import Budget

# Kết quả mong đợi của model giả: (kịch bản, mẫu) → kiểu dừng
EXPECTED_STOPS = {
    ("happy", "react"): "A", ("happy", "plan"): "A", ("happy", "hybrid"): "A",
    ("timeout", "react"): "C", ("timeout", "plan"): "C", ("timeout", "hybrid"): "C",
    ("approval", "react"): "E", ("approval", "plan"): "E", ("approval", "hybrid"): "E",
    ("env_change", "react"): "A", ("env_change", "plan"): "D", ("env_change", "hybrid"): "A",
}


# ------------------------------------------------------------------ mẫu Lai
def test_lai_thich_nghi_o_s6_va_gan_nhan_hybrid():
    r = run_hybrid("env_change")
    assert r.pattern == "hybrid" and r.report["stop_letter"] == "A" and r.report["replans"] == 1
    assert (r.report["rounds"], r.report["tool_calls"]) == (2, 5)


def test_lai_bat_buoc_phai_co_it_nhat_mot_luot_replan():
    with pytest.raises(ValueError, match="max_replans phải >= 1"):
        run_hybrid("happy", max_replans=0)


def test_lai_het_ca_hai_chuyen_re_thi_den_cong_duyet_cua_nguoi(monkeypatch):
    original = tools_flight._base_flights

    def both_sold_out():
        flights = original()
        for f in flights:
            if f.code == "VJ606":
                f.seats = 0
        return flights

    monkeypatch.setattr(tools_flight, "_base_flights", both_sold_out)
    r = run_hybrid("env_change", max_replans=2)
    # VJ604 và VJ606 hỏng → replan hai lần → còn VN122 (không hoàn, vượt hạn mức) → harness chờ người duyệt
    assert (r.report["stop_letter"], r.report["replans"], r.report["side_effects"]) == ("E", 2, [])
    assert "Duyệt đặt VN122" in r.handoff


# ------------------------------------------------------------- evaluate: lưới đầy đủ
def test_luoi_day_du_voi_model_gia(tmp_path):
    rows = run_eval(out_dir=tmp_path, log=lambda m: None)
    assert len(rows) == 12
    assert {(r["scenario"], r["pattern"]): r["stop_letter"] for r in rows} == EXPECTED_STOPS
    by_key = {(r["scenario"], r["pattern"]): r for r in rows}
    assert not by_key[("env_change", "plan")]["as_expected"]  # Plan thuần gãy ở S6
    assert all(r["as_expected"] for k, r in by_key.items() if k != ("env_change", "plan"))
    assert all(r["scenario_id"] == SCENARIO_ID[r["scenario"]] and not r["error"] for r in rows)
    assert by_key[("approval", "react")]["side_effects"] == ""  # dừng chờ duyệt: chưa tác dụng phụ nào


def test_ghi_csv_markdown_va_trace(tmp_path):
    rows = run_eval(("react", "hybrid"), ("happy", "env_change"), out_dir=tmp_path, log=lambda m: None)
    csv_path, md_path = evaluate.write_outputs(rows, tmp_path, "fake", "giả định")
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM để Excel đọc đúng tiếng Việt
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        read = list(csv.DictReader(f))
    assert list(read[0]) == COLUMNS and len(read) == 4
    assert {(r["scenario"], r["pattern"]) for r in read} == {("happy", "react"), ("happy", "hybrid"),
                                                              ("env_change", "react"), ("env_change", "hybrid")}
    md = md_path.read_text(encoding="utf-8")
    for needle in ("# Kết quả đánh giá (fake)", "## S1 · happy", "## S6 · env_change", "## Tổng hợp theo mẫu",
                   "Model GIẢ", "KHÔNG đặt được vé là kết quả đúng"):
        assert needle in md
    traces = sorted(p.name for p in (tmp_path / "traces_fake").iterdir())
    assert traces == ["S1_happy_hybrid_1.txt", "S1_happy_react_1.txt",
                      "S6_env_change_hybrid_1.txt", "S6_env_change_react_1.txt"]
    text = (tmp_path / "traces_fake" / "S6_env_change_hybrid_1.txt").read_text(encoding="utf-8")
    assert "[Kế hoạch mới]" in text and "--- Số liệu ---" in text


def test_chay_hai_lan_cho_ket_qua_giong_het_nhau():
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]  # noqa: E731
    assert strip(run_eval(log=lambda m: None)) == strip(run_eval(log=lambda m: None))


def test_lap_nhieu_lan_va_chon_tap_con():
    rows = run_eval(("plan",), ("happy", "timeout"), repeats=3, log=lambda m: None)
    assert [(r["scenario"], r["run"]) for r in rows] == [("happy", 1), ("happy", 2), ("happy", 3),
                                                         ("timeout", 1), ("timeout", 2), ("timeout", 3)]


# --------------------------------------------------- evaluate: lỗi, trần chi phí
def test_loi_giong_nhau_hai_lan_lien_thi_dung_som_va_ghi_thanh_dong_loi(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("máy chủ sập")

    monkeypatch.setattr(evaluate, "run_pattern", boom)
    logs = []
    rows = run_eval(log=logs.append)
    assert len(rows) == 2 and all(r["stop"] == "error" and "RuntimeError: máy chủ sập" in r["error"] for r in rows)
    assert any("DỪNG SỚM: cùng một lỗi" in m and "check_api.py" in m for m in logs)


def test_loi_khac_nhau_khong_dung_som_va_van_chay_tiep(monkeypatch):
    original, calls = evaluate.run_pattern, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise ValueError(f"lỗi số {len(calls)}")  # hai lỗi KHÁC nhau
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate, "run_pattern", flaky)
    rows = run_eval(("plan",), ("happy",), repeats=4, log=lambda m: None)
    assert [bool(r["error"]) for r in rows] == [True, True, False, False]


def test_tran_chi_phi_ca_dot(monkeypatch):
    logs = []
    rows = run_eval(("react",), ("happy", "timeout", "approval"), max_total_usd=0.005, log=logs.append)
    assert len(rows) == 1  # lần đầu đã tốn ~0,007 USD theo giá giả định, vượt trần 0,005
    assert any("tổng chi phí" in m and "chạm trần" in m for m in logs)


def test_ngan_sach_moi_lan_chay_duoc_truyen_xuong():
    rows = run_eval(("react",), ("happy",), budget=Budget(max_rounds=2), log=lambda m: None)
    assert rows[0]["stop_letter"] == "B" and not rows[0]["as_expected"]


# --------------------------------------------------------------- tổng hợp
def row(scenario, pattern, letter, success, expected, tokens=100, error=""):
    r = {c: None for c in COLUMNS}
    r.update(scenario=scenario, pattern=pattern, stop_letter=letter, success=success, as_expected=expected,
             rounds=2, tool_calls=3, tokens=tokens, cost_usd=0.01, seconds=1.0, interventions=1, error=error)
    return r


def test_summarize_bo_dong_loi_khoi_trung_binh_nhung_van_dem():
    rows = [row("happy", "react", "A", True, True, tokens=100), row("happy", "react", "A", True, True, tokens=300),
            row("happy", "react", "B", False, False, tokens=200), row("happy", "react", "", False, False, error="boom")]
    by_cell, by_pattern = summarize(rows)
    cell = by_cell[0]
    assert (cell["n"], cell["errors"], cell["success"], cell["as_expected"]) == (4, 1, 2, 2)
    assert cell["stops"] == "A×2 B×1" and cell["tokens"] == pytest.approx(200)  # (100+300+200)/3, không tính dòng lỗi
    assert by_pattern[0]["pattern"] == "react"


def test_markdown_chi_ghi_chu_model_gia_khi_la_model_gia():
    rows = [row("happy", "plan", "A", True, True)]
    assert "Model GIẢ" in to_markdown(rows, "fake", "x")
    assert "Model GIẢ" not in to_markdown(rows, "real-gpt-5.6-luna", "0.2 / 1.2")
    assert "0.2 / 1.2" in to_markdown(rows, "real-gpt-5.6-luna", "0.2 / 1.2")


# ------------------------------------------------------ giao diện dòng lệnh
def test_main_model_gia_ghi_file_va_tra_ve_0(tmp_path, capsys):
    code = evaluate.main(["--out", str(tmp_path), "--scenarios", "happy", "--patterns", "react", "plan"])
    out = capsys.readouterr().out
    assert code == 0 and "Đã ghi:" in out and "# Kết quả đánh giá (fake)" in out
    assert (tmp_path / "ket_qua_fake.csv").exists() and (tmp_path / "tong_hop_fake.md").exists()


def test_main_model_that_thieu_cau_hinh_thi_bao_loi_va_khong_chay(capsys):
    with pytest.raises(SystemExit) as exc:
        evaluate.main(["--model", "real"])
    assert exc.value.code == 2 and "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_model_that_huy_khi_khong_xac_nhan_thi_khong_goi_api(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(evaluate, "run_pattern", lambda *a, **k: pytest.fail("không được gọi API khi chưa xác nhận"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert evaluate.main(["--model", "real", "--scenarios", "happy"]) == 1
    out = capsys.readouterr().out
    assert "Sẽ chạy   : 9 lượt (1 kịch bản × 3 mẫu × 3 lần)" in out
    assert "0.2 USD vào / 1.2 USD ra" in out and "Đã huỷ, chưa tốn xu nào" in out
    assert "0123456789" not in out  # khoá chỉ hiện dạng che
