"""
check_api.py — Kiểm tra Google API key, model Gemini và tool calling TRƯỚC khi chạy đánh giá thật (SE373 · BTVN#3)

NÓ KIỂM TRA GÌ, THEO THỨ TỰ
    1  Tìm và đọc file .env
    2  Cấu hình đủ: GOOGLE_API_KEY, GOOGLE_MODEL (khoá chỉ hiện dạng che)
    3  Đã cài google-genai và langchain-google-genai
    4  Gọi THẲNG bằng thư viện google-genai → chứng minh key và tên model đúng
    5  Gọi qua LangChain (ChatGoogleGenerativeAI) → chứng minh phần nối LangChain đúng
    6  Gọi tool (function calling)              → điều kiện BẮT BUỘC để agent chạy được

CHẠY
    python src/check_api.py
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import common
from common import build_real_model, load_env, mask_secret, real_model_settings, resolve_prices

CONSOLE = "aistudio.google.com"
MODEL_DOCS = "ai.google.dev/gemini-api/docs/models"
PING = "Trả lời đúng một từ: OK"
TOOL_PROMPT = "Hãy gọi tool echo với text='xin chào'."


def diagnose(exc: BaseException) -> str:
    """Biến lỗi của thư viện Google thành lời giải thích và cách sửa."""
    code = getattr(exc, "code", None)
    status = code if isinstance(code, int) else getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "api key not valid" in text or "api_key_invalid" in text or status == 401:
        return f"API key bị từ chối: key sai hoặc đã bị xoá. Tạo key mới tại {CONSOLE} rồi dán lại vào GOOGLE_API_KEY."
    if status == 403 or "permission_denied" in text:
        return "Không có quyền (403): key không được dùng Gemini API, hoặc API chưa được bật cho project của key."
    if status == 404 or "not_found" in text:
        return f"Không tìm thấy (404): sai mã model trong GOOGLE_MODEL. Mã model hợp lệ xem tại {MODEL_DOCS}."
    if status == 429 or "resource_exhausted" in text:
        return "Bị giới hạn (429): gọi quá nhanh hoặc hết hạn mức của gói. Đợi một chút rồi chạy lại."
    if status == 400:
        return "Yêu cầu bị từ chối (400): xem thông điệp lỗi bên trên."
    if "connect" in name or "timeout" in name or "connect" in text:
        return "Không kết nối được: kiểm tra mạng, VPN hoặc proxy."
    return f"Lỗi không phân loại được ({type(exc).__name__}). Hãy gửi nguyên thông báo lỗi để được hỗ trợ."


def main(argv: list[str] | None = None) -> int:
    loaded = load_env()
    parser = argparse.ArgumentParser(description="Kiểm tra Google API key, model Gemini và tool calling")
    parser.add_argument("--price-in", type=float, help="giá USD / 1 triệu token vào (để ước tính chi phí)")
    parser.add_argument("--price-out", type=float, help="giá USD / 1 triệu token ra")
    args = parser.parse_args(argv)

    def safe(text: Any) -> str:  # không bao giờ để khoá lọt vào thông báo lỗi in ra màn hình
        return common.redact_secrets(str(text))[:400]

    def ok(msg: str) -> None:
        print(f"  ✓ {msg}")

    def bad(msg: str) -> None:
        print(f"  ✗ {msg}")

    print("=== Kiểm tra kết nối Google Gemini API ===\n")

    # 1. .env
    print("[1/6] File .env")
    files = common.find_env_files()
    if files:
        ok("Đã đọc: " + "; ".join(str(f) for f in files))
        ok(f"Nạp {len(loaded)} biến mới: {', '.join(loaded) or '(không có, đã đặt sẵn trong môi trường)'}")
    else:
        print("  · Không thấy file .env. Tạo .env ở thư mục gốc repo, sao chép từ .env.example.")

    # 2. cấu hình
    print("\n[2/6] Cấu hình")
    try:
        settings = real_model_settings()
    except RuntimeError as exc:
        bad(str(exc))
        return 1
    if settings["provider"] != "google":
        bad(f"Đang chọn {settings['label']}, không phải Google. Đặt GOOGLE_API_KEY trong .env "
            "và bỏ dòng LLM_PROVIDER (hoặc đặt LLM_PROVIDER=google).")
        return 1
    ok(f"{'GOOGLE_API_KEY':<16}= {mask_secret(settings['api_key'])}")
    ok(f"{'GOOGLE_MODEL':<16}= {settings['model']}")
    price_in, price_out, price_source = resolve_prices(settings["model"], args.price_in, args.price_out)

    # 3. thư viện
    print("\n[3/6] Thư viện")
    try:
        from google import genai
        import langchain_google_genai
    except ImportError as exc:
        bad(f"Thiếu thư viện: {exc.name}. Chạy: pip install langchain-google-genai")
        return 1
    ok(f"google-genai {getattr(genai, '__version__', '?')} · "
       f"langchain-google-genai {getattr(langchain_google_genai, '__version__', '?')}")

    tokens_in = tokens_out = 0
    failed = False

    # 4. gọi thẳng bằng google-genai
    print("\n[4/6] Gọi trực tiếp bằng thư viện google-genai")
    try:
        http_options = {"base_url": settings["base_url"]} if settings["base_url"] else None
        client = genai.Client(api_key=settings["api_key"], http_options=http_options)
        t0 = time.perf_counter()
        resp = client.models.generate_content(model=settings["model"], contents=PING)
        seconds = time.perf_counter() - t0
        u = resp.usage_metadata
        used_in = getattr(u, "prompt_token_count", 0) or 0
        used_out = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        tokens_in += used_in
        tokens_out += used_out
        ok(f"Trả lời {(resp.text or '').strip()!r} · {used_in} token vào, {used_out} token ra · {seconds:.1f}s")
    except Exception as exc:  # noqa: BLE001 - mọi lỗi mạng/API đều phải được giải thích, không văng traceback
        failed = True
        bad(safe(exc))
        print(f"    → {diagnose(exc)}")

    # 5. qua LangChain
    print("\n[5/6] Gọi qua LangChain (ChatGoogleGenerativeAI)")
    llm = None
    if failed:
        print("  · Bỏ qua vì bước 4 đã lỗi: sửa bước 4 trước.")
    else:
        try:
            llm = build_real_model()
            t0 = time.perf_counter()
            msg = llm.invoke(PING)
            seconds = time.perf_counter() - t0
            usage = getattr(msg, "usage_metadata", None) or {}
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
            ok(f"Trả lời {common.content_text(msg.content).strip()!r} · {usage.get('input_tokens', '?')} token vào, "
               f"{usage.get('output_tokens', '?')} token ra · {seconds:.1f}s")
        except Exception as exc:  # noqa: BLE001
            failed = True
            bad(safe(exc))
            print(f"    → {diagnose(exc)}")

    # 6. tool calling
    print("\n[6/6] Gọi tool (function calling): điều kiện bắt buộc để agent chạy được")
    if llm is None:
        print("  · Bỏ qua vì bước trước đã lỗi.")
    else:
        try:
            from langchain_core.tools import tool

            @tool
            def echo(text: str) -> str:
                """Trả lại đúng chuỗi text."""
                return text

            reply = llm.bind_tools([echo]).invoke(TOOL_PROMPT)
            usage = getattr(reply, "usage_metadata", None) or {}
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
            if reply.tool_calls and reply.tool_calls[0]["name"] == "echo":
                ok(f"Model gọi echo({', '.join(f'{k}={v!r}' for k, v in reply.tool_calls[0]['args'].items())})")
            else:
                failed = True
                bad("Model KHÔNG gọi tool. Agent sẽ không chạy được: đổi GOOGLE_MODEL sang model hỗ trợ function calling.")
        except Exception as exc:  # noqa: BLE001
            failed = True
            bad(safe(exc))
            print(f"    → {diagnose(exc)}")

    cost = tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out
    print(f"\n--- Tổng cộng: {tokens_in} token vào, {tokens_out} token ra ≈ {cost:.5f} USD "
          f"(giá {price_in}/{price_out} USD mỗi 1M token, nguồn: {price_source})")
    if failed:
        print("KẾT LUẬN: CHƯA sẵn sàng. Sửa lỗi ở trên rồi chạy lại.")
        return 1
    print("KẾT LUẬN: sẵn sàng. Lệnh tiếp theo:\n  python src/evaluate.py --model real --scenarios happy --repeats 1")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # in tiếng Việt đúng trên console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
