"""Kiểm thử nạp .env, che khoá, bảng giá và cấu hình model thật (không cần mạng, không cần langchain-openai)."""
import os
import sys

import pytest

import common
import harness
from langchain_core.messages import AIMessage

from common import (
    apply_prices,
    build_real_model,
    content_text,
    message_text,
    render_trace,
    find_env_files as real_find_env_files,  # import lúc thu thập test, TRƯỚC khi fixture hermetic_env thay thế
    load_env,
    mask_secret,
    real_model_settings,
    redact_secrets,
    resolve_prices,
)

# Đúng mẫu .env mà người dùng đang dùng ở bài trước
USER_TEMPLATE = """# Copy this file to .env in this directory or at the repository root.
OPENAI_API_KEY=your_api_key_here
# Required. Include the provider's OpenAI-compatible API version path when needed.
OPENAI_BASE_URL=https://api.openai.com/v1
# Required. Use a model ID offered by the configured provider.
OPENAI_MODEL=gpt-5.6-luna
"""


def write(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding))
    return path


# --------------------------------------------------------------- an toàn khi chạy pytest
def test_pytest_khong_bao_gio_doc_env_that():
    assert common.find_env_files() == []  # fixture hermetic_env trong conftest.py
    assert load_env() == []
    assert "OPENAI_API_KEY" not in os.environ


# ----------------------------------------------------------------- load_env
def test_doc_dung_mau_env_cua_nguoi_dung(tmp_path):
    names = load_env([write(tmp_path / ".env", USER_TEMPLATE)])
    assert names == ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
    assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert os.environ["OPENAI_MODEL"] == "gpt-5.6-luna"


def test_cu_phap_dau_nhay_export_khoang_trang_va_xuong_dong_windows(tmp_path):
    text = ('export OPENAI_API_KEY = "sk-abc-123"\r\n'
            "OPENAI_MODEL='gpt-5.6-luna'\r\n"
            "\r\n   # chú thích có thụt lề\r\n"
            "OPENAI_BASE_URL=https://example.com/v1/#khong-phai-chu-thich\r\n"
            "dong-khong-co-dau-bang\r\n")
    load_env([write(tmp_path / ".env", text)])
    assert os.environ["OPENAI_API_KEY"] == "sk-abc-123"
    assert os.environ["OPENAI_MODEL"] == "gpt-5.6-luna"
    assert os.environ["OPENAI_BASE_URL"] == "https://example.com/v1/#khong-phai-chu-thich"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_doc_duoc_file_env_tao_tu_notepad_va_powershell(tmp_path, encoding):
    """Windows PowerShell 5 ghi file bằng UTF-16, Notepad hay thêm BOM: cả hai đều phải đọc được."""
    load_env([write(tmp_path / ".env", "OPENAI_MODEL=gpt-5.6-luna\n# tiếng Việt: được\n", encoding)])
    assert os.environ["OPENAI_MODEL"] == "gpt-5.6-luna"


def test_giu_nguyen_bien_da_co_va_file_uu_tien_cao_thang(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "tu-moi-truong")
    first = write(tmp_path / "a.env", "OPENAI_API_KEY=KEY-A\nOPENAI_MODEL=tu-file-a\n")
    second = write(tmp_path / "b.env", "OPENAI_API_KEY=KEY-B\nOPENAI_BASE_URL=https://b/v1\n")
    names = load_env([first, second])
    assert os.environ["OPENAI_MODEL"] == "tu-moi-truong"  # biến đã đặt trong PowerShell không bị file ghi đè
    assert os.environ["OPENAI_API_KEY"] == "KEY-A"  # file đứng trước thắng
    assert os.environ["OPENAI_BASE_URL"] == "https://b/v1"
    assert names == ["OPENAI_API_KEY", "OPENAI_BASE_URL"]


def test_load_env_chi_tra_ve_ten_khong_bao_gio_tra_ve_gia_tri(tmp_path):
    names = load_env([write(tmp_path / ".env", "OPENAI_API_KEY=sk-rat-bi-mat-123456\n")])
    assert names == ["OPENAI_API_KEY"] and "sk-rat-bi-mat" not in repr(names)


def test_find_env_files_thu_muc_hien_tai_duoc_uu_tien(tmp_path, monkeypatch):
    env = write(tmp_path / ".env", "OPENAI_MODEL=x\n")
    monkeypatch.chdir(tmp_path)
    assert real_find_env_files()[0] == env.resolve()


# ---------------------------------------------------------- che khoá bí mật
def test_mask_secret():
    assert mask_secret(None) == "(chưa đặt)" and mask_secret("") == "(chưa đặt)"
    assert mask_secret("abc") == "*** (dài 3 ký tự)"
    masked = mask_secret("sk-proj-0123456789abcdef")
    assert masked == "sk-…cdef (dài 24 ký tự)" and "0123456789" not in masked


def test_redact_secrets_che_khoa_lo_trong_thong_bao_loi(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-0123456789abcdef")
    out = redact_secrets("Error 401: key sk-proj-0123456789abcdef is invalid")
    assert "0123456789" not in out and "sk-…cdef" in out
    monkeypatch.delenv("OPENAI_API_KEY")
    assert redact_secrets("không có khoá nào") == "không có khoá nào"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-ABCDEFGHIJKLMNOP")  # khoá Claude cũng được che
    assert "ABCDEFGHIJ" not in redact_secrets("401: sk-ant-api03-ABCDEFGHIJKLMNOP bị từ chối")


# ------------------------------------------------------ cấu hình model thật
def test_real_model_settings_chua_co_key_nao_thi_chi_ca_hai_cach():
    with pytest.raises(RuntimeError) as err:
        real_model_settings()
    msg = str(err.value)
    assert "ANTHROPIC_API_KEY (Claude)" in msg and "OPENAI_API_KEY" in msg and ".env.example" in msg


def test_real_model_settings_thieu_model_cua_openai_thi_bao_ro(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(RuntimeError) as err:
        real_model_settings()
    assert "OPENAI_API_KEY, OPENAI_MODEL" in str(err.value)


def test_real_model_settings_tu_choi_gia_tri_mau(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", "your_api_key_here")
    with pytest.raises(RuntimeError, match="giá trị mẫu"):
        real_model_settings()


def test_real_model_settings_mac_dinh_va_cat_khoang_trang(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-test-123456789  ")
    monkeypatch.setenv("OPENAI_MODEL", " gpt-5.6-luna ")
    s = real_model_settings()
    assert (s["provider"], s["api_key"], s["model"]) == ("openai", "sk-test-123456789", "gpt-5.6-luna")
    assert (s["base_url"], s["reasoning_effort"]) == ("https://api.openai.com/v1", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    assert real_model_settings()["base_url"] == "https://proxy.example.com/v1"
    assert real_model_settings()["reasoning_effort"] == "low"


def test_thieu_langchain_openai_thi_huong_dan_cai_dat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123456789")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setitem(sys.modules, "langchain_openai", None)  # mô phỏng chưa cài
    with pytest.raises(RuntimeError, match="pip install langchain-openai"):
        build_real_model()


# ---------------------------------------------------------------------- giá
def test_resolve_prices():
    assert resolve_prices("gpt-5.6-luna", None, None) == (0.20, 1.20, "bảng giá OpenAI đã kiểm tra (10/2026)")
    for name in ("claude-haiku-4-5-20251001", "claude-haiku-4-5"):  # mã đầy đủ và bí danh cùng giá
        assert resolve_prices(name, None, None) == (1.00, 5.00, "bảng giá Anthropic đã kiểm tra (10/2026)")
    assert resolve_prices("gpt-5.6-luna", 0.5, None)[:2] == (0.5, 1.20)  # chỉ ghi đè một giá
    assert resolve_prices("model-la", 1.0, 2.0) == (1.0, 2.0, "do bạn truyền vào")
    p_in, p_out, source = resolve_prices("model-la", None, None)
    assert (p_in, p_out) == (3.0, 15.0) and "GIẢ ĐỊNH" in source and "--price-in" in source


def test_apply_prices_doi_cach_tinh_chi_phi(monkeypatch):
    monkeypatch.setattr(harness, "PRICE_IN_PER_1K", harness.PRICE_IN_PER_1K)  # để monkeypatch khôi phục sau test
    monkeypatch.setattr(harness, "PRICE_OUT_PER_1K", harness.PRICE_OUT_PER_1K)
    usage = harness.Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert usage.cost_usd == pytest.approx(18.0)  # giá giả định 3 + 15
    apply_prices(0.20, 1.20)
    assert usage.cost_usd == pytest.approx(1.40)


# ===================================================== chọn nhà cung cấp: Claude hoặc OpenAI
def test_chi_co_key_claude_thi_dung_claude_va_model_mac_dinh_la_haiku(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")
    s = real_model_settings()
    assert (s["provider"], s["label"]) == ("anthropic", "Anthropic (Claude)")
    assert s["model"] == "claude-haiku-4-5-20251001"  # không đặt ANTHROPIC_MODEL thì dùng Haiku 4.5 (rẻ nhất)
    assert (s["base_url"], s["reasoning_effort"]) == ("https://api.anthropic.com", "")
    assert (s["key_env"], s["model_env"], s["base_env"]) == ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL")


def test_chi_co_key_openai_thi_dung_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    assert real_model_settings()["provider"] == "openai"


def test_co_ca_hai_key_thi_uu_tien_claude_va_llm_provider_doi_duoc(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    assert real_model_settings()["provider"] == "anthropic"  # vừa chuyển sang Claude mà .env còn dòng OpenAI cũ
    monkeypatch.setenv("LLM_PROVIDER", " OpenAI ")  # không phân biệt hoa thường, bỏ khoảng trắng
    assert real_model_settings()["provider"] == "openai"


def test_llm_provider_sai_thi_bao_loi_ro(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER phải là google, anthropic, openai"):
        real_model_settings()


@pytest.mark.parametrize("raw, expected", [
    ("https://api.anthropic.com/v1", "https://api.anthropic.com"),   # thói quen gõ /v1 như bên OpenAI
    ("https://api.anthropic.com/v1/", "https://api.anthropic.com"),
    ("https://api.anthropic.com/", "https://api.anthropic.com"),
    ("https://proxy.example.com/anthropic/v1", "https://proxy.example.com/anthropic"),
    ("https://api.anthropic.com", "https://api.anthropic.com"),
])
def test_dia_chi_claude_tu_bo_v1_o_cuoi(monkeypatch, raw, expected):
    """Thư viện Anthropic tự thêm /v1/messages; dư /v1 thì thành /v1/v1/messages và báo 404 khó hiểu."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", raw)
    assert real_model_settings()["base_url"] == expected


def test_openai_van_giu_nguyen_v1(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    assert real_model_settings()["base_url"] == "https://api.openai.com/v1"


def test_reasoning_effort_chi_ap_dung_cho_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")
    assert real_model_settings()["reasoning_effort"] == ""  # Claude không dùng tham số này


def test_thieu_langchain_anthropic_thi_huong_dan_cai_dat(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    with pytest.raises(RuntimeError, match="pip install langchain-anthropic"):
        build_real_model()


def test_temperature_cua_claude_phai_la_so(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "mot")
    pytest.importorskip("langchain_anthropic")
    with pytest.raises(RuntimeError, match="ANTHROPIC_TEMPERATURE phải là một số"):
        build_real_model()


# ===================================================== content dạng danh sách khối (Claude)
def test_content_text_doc_duoc_chuoi_danh_sach_khoi_va_none():
    assert content_text("xin chào") == "xin chào"
    assert content_text(None) == ""
    blocks = [{"type": "text", "text": "Cần tìm chuyến. "}, {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
              {"type": "text", "text": "Xong."}, "rời"]
    assert content_text(blocks) == "Cần tìm chuyến. Xong.rời"  # bỏ khối tool_use, giữ chữ


def test_message_text_va_trace_dung_chu_that_khi_content_la_danh_sach_khoi():
    ai = AIMessage(content=[{"type": "text", "text": "Tìm chuyến bay"}], tool_calls=[
        {"name": "search_flights", "args": {"origin": "SGN"}, "id": "t1", "type": "tool_call"}])
    assert message_text(ai).startswith("Tìm chuyến bay") and "search_flights" in message_text(ai)  # không còn là JSON thô
    trace = render_trace([ai, AIMessage(content=[{"type": "text", "text": "Đã đặt xong."}])])
    assert "[V1] Suy luận : Tìm chuyến bay" in trace and "[V2] Trả lời   : Đã đặt xong." in trace
    assert "'type'" not in trace and '"type"' not in trace


# ===================================================== Google (Gemini)
def test_key_google_duoc_uu_tien_va_model_mac_dinh_gemini_3_8_flash(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-0123456789")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")  # .env còn key Claude cũ
    s = real_model_settings()
    assert (s["provider"], s["model"], s["base_url"]) == ("google", "gemini-3.8-flash", "")
    assert resolve_prices("gemini-3.8-flash", None, None) == (0.75, 3.75, "bảng giá Google đã kiểm tra (10/2026)")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert real_model_settings()["provider"] == "anthropic"


def test_thieu_langchain_google_genai_thi_huong_dan_cai_dat(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-0123456789")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    with pytest.raises(RuntimeError, match="pip install langchain-google-genai"):
        build_real_model()
