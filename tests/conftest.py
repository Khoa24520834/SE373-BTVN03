"""Cấu hình chung cho pytest: bảo đảm kiểm thử KHÔNG BAO GIỜ đọc file .env thật hay dùng API key thật."""
import pytest

import common
import harness

ENV_NAMES = ("LLM_PROVIDER", "GOOGLE_API_KEY", "GOOGLE_MODEL", "GOOGLE_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ANTHROPIC_TEMPERATURE",
             "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT")


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    # Bạn sẽ có .env thật (với API key thật) ở thư mục gốc repo: không test nào được phép đọc nó, kể cả gián tiếp
    # qua load_env() trong các hàm main(). Test nào cần biến môi trường thì tự đặt bằng monkeypatch.setenv.
    monkeypatch.setattr(common, "find_env_files", lambda: [])
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def hermetic_prices(monkeypatch):
    # evaluate.main() đặt giá quy đổi chi phí cho harness: khôi phục sau mỗi test để các test khác không bị lệch số.
    monkeypatch.setattr(harness, "PRICE_IN_PER_1K", harness.PRICE_IN_PER_1K)
    monkeypatch.setattr(harness, "PRICE_OUT_PER_1K", harness.PRICE_OUT_PER_1K)
