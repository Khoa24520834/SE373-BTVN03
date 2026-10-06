"""
common.py — Phần dùng chung giữa ba mẫu thiết kế (SE373 · BTVN#3)

GỒM
    ScriptedModel   Model GIẢ, tất định, ra quyết định theo observation đã thấy.
                    Chạy lại được và không tốn tiền, nên so sánh 3 mẫu công bằng.
                    Dùng khi cần lặp lại lỗi (S2 lặp, S5 bịa) mà model thật khó tái hiện.
    make_model      "fake" → ScriptedModel; "real" → Claude (ChatAnthropic) hoặc OpenAI (ChatOpenAI) dựng từ file .env.
    load_env        Nạp file .env (không cần thư viện ngoài). CHỈ các lệnh chạy từ dòng lệnh mới gọi;
                    pytest và các hàm thư viện không bao giờ tự đọc .env.
    RunResult       Kết quả MỘT lần chạy của MỘT mẫu (cùng khuôn cho ReAct, Plan, Lai).
    render_trace    In trace theo vòng: Suy luận → Hành động → Quan sát (đọc để tìm vòng sai đầu tiên).

KIỂU ỨNG XỬ CỦA MODEL GIẢ (tham số style)
    competent    Làm đúng: tìm → kiểm ghế → đặt → trả tiền, chọn chuyến rẻ nhất thoả ràng buộc.
                 Gặp lỗi thì thử lại y nguyên (đúng thói quen hay gây lặp của model thật).
    greedy       Bỏ qua ràng buộc giờ bay và giá, chọn chuyến rẻ nhất (kiểm lỗi "quên yêu cầu").
    hallucinate  Khẳng định đã đặt vé mà không gọi tool nào (kiểm lỗi bịa, kịch bản S5).
"""
from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from harness import Constraints, Harness, StopReason, estimate_tokens, fmt_call
from tools_flight import FlightWorld

STYLES = ("competent", "greedy", "hallucinate")
HARNESS_SOURCE = {"source": "harness"}  # đánh dấu tin nhắn do harness chèn vào, không phải của model


def content_text(content: Any) -> str:
    """Chữ trong content của một tin nhắn. Claude (và vài model khác) trả content là DANH SÁCH các khối,
    ví dụ [{"type": "text", "text": "..."}, {"type": "tool_use", ...}], khi tin nhắn có tool call."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else str(b.get("text", ""))
                       for b in content if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text"))
    return "" if content is None else str(content)


def message_text(m: BaseMessage) -> str:
    """Chữ của một tin nhắn, gồm cả tham số tool call (để ước lượng token)."""
    text = content_text(m.content)
    if isinstance(m, AIMessage) and m.tool_calls:
        text += json.dumps(m.tool_calls, ensure_ascii=False)
    return text


def _history(messages: list[BaseMessage]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Ghép mỗi tool call với observation của nó: [(tool, args, observation)]."""
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                calls[c["id"]] = (c["name"], c["args"])
        elif isinstance(m, ToolMessage):
            name, args = calls.get(m.tool_call_id, (m.name or "?", {}))
            try:
                obs = json.loads(m.content) if isinstance(m.content, str) else {}
            except json.JSONDecodeError:
                obs = {"status": "error", "error": "non_json"}
            out.append((name, args, obs if isinstance(obs, dict) else {}))
    return out


# --------------------------------------------------------------------------- #
# Model giả
# --------------------------------------------------------------------------- #
class ScriptedModel(BaseChatModel):
    """Model giả: đọc lịch sử tool call + observation rồi chọn hành động kế tiếp.

    Giống model thật ở chỗ nó PHẢN ỨNG với observation (hết ghế thì đổi chuyến,
    bị từ chối thì loại chuyến đó), nhưng tất định nên chạy lại ra đúng kết quả cũ.
    """

    style: str = "competent"
    constraints: Any = None
    seat: str = "12A"
    method: str = "corp_card"

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":  # create_agent gọi hàm này
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        ai = self._decide(messages)
        # Mỗi vòng gửi lại TOÀN BỘ lịch sử, nên input_tokens tăng dần theo số vòng (chi phí bậc hai).
        n_in = estimate_tokens("\n".join(message_text(m) for m in messages))
        n_out = estimate_tokens(message_text(ai))
        ai.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ---- chính sách ra quyết định ----------------------------------------- #
    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        if self.style == "hallucinate":
            return AIMessage(content="Done! Booked VN999, seat 5C, for 1,200,000 VND.")

        c: Constraints = self.constraints or Constraints()
        hist = _history(messages)
        flights: list[dict[str, Any]] | None = None
        excluded: set[str] = set()
        booking: dict[str, Any] | None = None
        paid = False
        for tool, args, obs in hist:
            status = obs.get("status")
            if tool == "search_flights" and status == "ok":
                flights = obs["flights"]
            elif tool in ("check_seat", "book_seat") and status in ("sold_out", "not_found", "denied"):
                excluded.add(str(args.get("flight", "")).upper())
            elif tool == "book_seat" and status == "ok":
                booking = obs["booking"]
            elif tool == "pay" and status == "ok":
                paid = True

        n = sum(len(m.tool_calls) for m in messages if isinstance(m, AIMessage))

        def act(thought: str, tool: str, **args: Any) -> AIMessage:
            call = {"name": tool, "args": args, "id": f"call_{n + 1}", "type": "tool_call"}
            return AIMessage(content=thought, tool_calls=[call])

        if paid:
            return AIMessage(content=f"Đã đặt xong vé {booking['flight']}, mã đặt chỗ {booking['code']}." if booking
                             else "Đã thanh toán xong.")
        if booking is not None:
            return act(f"Đã giữ chỗ, giờ thanh toán đặt chỗ {booking['code']}.",
                       "pay", code=booking["code"], method=self.method)
        if flights is None:
            return act(f"Cần tìm các chuyến {c.origin}→{c.dest} ngày {c.date} trước.",
                       "search_flights", origin=c.origin, dest=c.dest, date=c.date)

        def acceptable(f: dict[str, Any]) -> bool:
            if self.style == "greedy":
                return True  # bỏ qua ràng buộc
            return f["depart"] < c.depart_before and f["price"] <= c.max_price

        candidates = sorted((f for f in flights if f["flight"] not in excluded and acceptable(f)),
                            key=lambda f: f["price"])
        if not candidates:
            return AIMessage(content="Không tìm thấy chuyến nào thoả yêu cầu.")
        pick = candidates[0]["flight"]
        checked = any(t == "check_seat" and str(a.get("flight", "")).upper() == pick and o.get("status") == "ok"
                      for t, a, o in hist)
        if not checked:  # chưa kiểm ghế được (kể cả vì lỗi) thì kiểm tiếp: thử lại y nguyên
            why = "rẻ nhất" if self.style == "greedy" else "rẻ nhất thoả yêu cầu"
            return act(f"Chuyến {why} là {pick}, kiểm tra ghế còn không.", "check_seat", flight=pick)
        return act(f"{pick} còn ghế, giữ chỗ ghế {self.seat}.", "book_seat", flight=pick, seat=self.seat)


# --------------------------------------------------------------------------- #
# Model thật (Claude của Anthropic hoặc OpenAI) và file .env
# --------------------------------------------------------------------------- #
ENV_PROVIDER, ENV_EFFORT, ENV_TEMPERATURE = "LLM_PROVIDER", "OPENAI_REASONING_EFFORT", "ANTHROPIC_TEMPERATURE"
PLACEHOLDER_KEYS = ("your_api_key_here", "sk-...", "sk-ant-...", "changeme")


@dataclass(frozen=True)
class ProviderSpec:
    """Mọi thứ khác nhau giữa các nhà cung cấp: tên biến .env, địa chỉ mặc định, thư viện LangChain, nơi tạo key."""

    name: str
    label: str
    key_env: str
    model_env: str
    base_env: str
    default_base: str
    default_model: str  # rỗng: bắt buộc phải đặt model trong .env
    package: str        # tên gói pip
    module: str         # tên module để import
    console: str        # nơi tạo key và xem số dư


PROVIDERS: dict[str, ProviderSpec] = {
    "google": ProviderSpec("google", "Google (Gemini)", "GOOGLE_API_KEY", "GOOGLE_MODEL", "GOOGLE_BASE_URL",
                           "", "gemini-3.8-flash", "langchain-google-genai", "langchain_google_genai",
                           "aistudio.google.com"),
    "anthropic": ProviderSpec("anthropic", "Anthropic (Claude)", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                              "ANTHROPIC_BASE_URL", "https://api.anthropic.com", "claude-haiku-4-5-20251001",
                              "langchain-anthropic", "langchain_anthropic", "console.anthropic.com"),
    "openai": ProviderSpec("openai", "OpenAI", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
                           "https://api.openai.com/v1", "", "langchain-openai", "langchain_openai",
                           "platform.openai.com"),
}
ALL_KEY_ENVS = tuple(p.key_env for p in PROVIDERS.values())

# Giá USD / 1 triệu token (vào, ra, hãng), theo trang model chính thức khi viết (10/2026). Nhà cung cấp khác
# hoặc model khác có thể tính khác: dùng --price-in / --price-out để ghi đè. Model không có trong bảng thì phải tự truyền giá.
KNOWN_PRICES_PER_1M: dict[str, tuple[float, float, str]] = {
    "gemini-3.8-flash": (0.75, 3.75, "Google"),  # giá giới thiệu đến 31/12/2026
    "claude-haiku-4-5-20251001": (1.00, 5.00, "Anthropic"),
    "claude-haiku-4-5": (1.00, 5.00, "Anthropic"),
    "gpt-5.6-luna": (0.20, 1.20, "OpenAI"),
    "gpt-5.4-nano": (0.20, 1.25, "OpenAI"),
    "gpt-5.4-mini": (0.75, 4.50, "OpenAI"),
}


def find_env_files() -> list[Path]:
    """Các file .env sẽ được đọc, theo thứ tự ưu tiên: thư mục hiện tại, thư mục src/, thư mục gốc repo."""
    here = Path(__file__).resolve().parent
    seen: list[Path] = []
    for path in (Path.cwd() / ".env", here / ".env", here.parent / ".env"):
        path = path.resolve()
        if path.is_file() and path not in seen:
            seen.append(path)
    return seen


def _read_env_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # PowerShell 5 ghi file bằng UTF-16 theo mặc định
        return raw.decode("utf-16")
    return raw.decode("utf-8")


def load_env(paths: list[Path] | None = None) -> list[str]:
    """Nạp các file .env vào os.environ (không cần thư viện ngoài). Biến ĐÃ có trong môi trường được giữ
    nguyên, file ưu tiên cao hơn thắng file ưu tiên thấp hơn. Chỉ trả về TÊN biến đã nạp, không bao giờ trả về giá trị.
    paths: danh sách file tường minh (dùng khi kiểm thử); bỏ trống thì tìm bằng find_env_files()."""
    loaded: list[str] = []
    for path in (find_env_files() if paths is None else paths):
        for line in _read_env_text(path).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def mask_secret(value: str | None) -> str:
    """Hiển thị khoá bí mật an toàn: vài ký tự đầu/cuối và độ dài, KHÔNG bao giờ in nguyên khoá."""
    if not value:
        return "(chưa đặt)"
    if len(value) <= 8:
        return f"*** (dài {len(value)} ký tự)"
    return f"{value[:3]}…{value[-4:]} (dài {len(value)} ký tự)"


def redact_secrets(text: str) -> str:
    """Che mọi API key nếu nó lỡ xuất hiện trong một chuỗi sắp được in hoặc ghi ra file (thông báo lỗi, CSV)."""
    for name in ALL_KEY_ENVS:
        key = os.environ.get(name, "").strip()
        if len(key) > 8:
            text = text.replace(key, mask_secret(key))
    return text


def detect_provider() -> ProviderSpec:
    """LLM_PROVIDER nếu có; không thì theo key nào đang được đặt, ưu tiên Google > Claude > OpenAI."""
    explicit = os.environ.get(ENV_PROVIDER, "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise RuntimeError(f"{ENV_PROVIDER} phải là {', '.join(PROVIDERS)}, nhận được {explicit!r}.")
        return PROVIDERS[explicit]
    for name in ("google", "anthropic", "openai"):
        if os.environ.get(PROVIDERS[name].key_env, "").strip():
            return PROVIDERS[name]
    raise RuntimeError(
        f"Chưa có API key nào. Đặt {PROVIDERS['google'].key_env} (Gemini), {PROVIDERS['anthropic'].key_env} (Claude) "
        f"hoặc {PROVIDERS['openai'].key_env} (OpenAI) trong file .env ở thư mục gốc repo (sao chép từ .env.example), hoặc đặt biến trong PowerShell.")


def real_model_settings() -> dict[str, str]:
    """Cấu hình model thật từ biến môi trường. Thiếu hoặc còn là giá trị mẫu thì báo rõ phải làm gì."""
    spec, env = detect_provider(), os.environ
    key = env.get(spec.key_env, "").strip()
    model = (env.get(spec.model_env) or spec.default_model).strip()
    missing = [name for name, value in ((spec.key_env, key), (spec.model_env, model)) if not value]
    if missing:
        raise RuntimeError(f"Thiếu biến môi trường: {', '.join(missing)}. Tạo file .env ở thư mục gốc repo "
                           "(sao chép từ .env.example) hoặc đặt biến trong PowerShell.")
    if key.lower() in PLACEHOLDER_KEYS:
        raise RuntimeError(f"{spec.key_env} vẫn là giá trị mẫu. Mở file .env và dán API key thật.")
    base = (env.get(spec.base_env) or spec.default_base).strip()
    if spec.name == "anthropic":  # thư viện Anthropic tự thêm /v1/messages; dư /v1 sẽ thành /v1/v1/messages và báo 404
        base = re.sub(r"/v1$", "", base.rstrip("/"))
    return {"provider": spec.name, "label": spec.label, "api_key": key, "model": model, "base_url": base,
            "key_env": spec.key_env, "model_env": spec.model_env, "base_env": spec.base_env,
            "reasoning_effort": (env.get(ENV_EFFORT) or "").strip() if spec.name == "openai" else ""}


def build_real_model() -> Any:
    """Dựng model thật từ file .env: ChatAnthropic (Claude) hoặc ChatOpenAI (ép dùng Chat Completions)."""
    settings = real_model_settings()
    spec = PROVIDERS[settings["provider"]]
    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        raise RuntimeError(f"Chưa cài {spec.package}. Chạy: pip install {spec.package}") from exc
    if spec.name == "google":
        kwargs = {"model": settings["model"], "api_key": settings["api_key"], "max_retries": 2, "timeout": 90}
        if settings["base_url"]:
            kwargs["base_url"] = settings["base_url"]
        return module.ChatGoogleGenerativeAI(**kwargs)
    common: dict[str, Any] = {"model": settings["model"], "api_key": settings["api_key"],
                              "base_url": settings["base_url"], "max_retries": 2}
    if spec.name == "anthropic":
        kwargs = {**common, "max_tokens": 4096, "timeout": 90}  # Claude bắt buộc có max_tokens
        temperature = os.environ.get(ENV_TEMPERATURE, "").strip()
        if temperature:
            try:
                kwargs["temperature"] = float(temperature)
            except ValueError:
                raise RuntimeError(f"{ENV_TEMPERATURE} phải là một số, nhận được {temperature!r}.") from None
        return module.ChatAnthropic(**kwargs)
    kwargs = {**common, "use_responses_api": False, "timeout": 90}
    if settings["reasoning_effort"]:
        kwargs["reasoning_effort"] = settings["reasoning_effort"]
    return module.ChatOpenAI(**kwargs)


def resolve_prices(model: str, price_in: float | None, price_out: float | None) -> tuple[float, float, str]:
    """(giá vào, giá ra, nguồn) tính theo USD / 1 triệu token. Ưu tiên: tham số truyền vào > bảng đã biết > giả định."""
    if price_in is not None and price_out is not None:
        return price_in, price_out, "do bạn truyền vào"
    if model in KNOWN_PRICES_PER_1M:
        known_in, known_out, vendor = KNOWN_PRICES_PER_1M[model]
        return (known_in if price_in is None else price_in), (known_out if price_out is None else price_out), \
            f"bảng giá {vendor} đã kiểm tra (10/2026)"
    return (3.0 if price_in is None else price_in), (15.0 if price_out is None else price_out), \
        "GIẢ ĐỊNH (không biết giá của model này, hãy truyền --price-in/--price-out)"


def apply_prices(price_in_per_1m: float, price_out_per_1m: float) -> None:
    """Đặt giá quy đổi chi phí cho harness (Usage.cost_usd đọc hai hằng số này lúc tính)."""
    import harness
    harness.PRICE_IN_PER_1K, harness.PRICE_OUT_PER_1K = price_in_per_1m / 1000, price_out_per_1m / 1000


def make_model(kind: Any = "fake", style: str = "competent", constraints: Constraints | None = None) -> Any:
    """'fake' → ScriptedModel. 'real' → ChatOpenAI dựng từ file .env (xem build_real_model).
    Truyền thẳng một đối tượng model thì trả lại nguyên (dùng khi kiểm thử)."""
    if not isinstance(kind, str):
        return kind
    if kind == "fake":
        if style not in STYLES:
            raise ValueError(f"style phải thuộc {STYLES}, nhận được {style!r}")
        return ScriptedModel(style=style, constraints=constraints or Constraints())
    if kind == "real":
        return build_real_model()
    raise ValueError(f"model phải là 'fake' hoặc 'real', nhận được {kind!r}")


# --------------------------------------------------------------------------- #
# Kết quả một lần chạy (cùng khuôn cho cả ba mẫu → evaluate.py gom được)
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    pattern: str                   # "react" | "plan" | "hybrid"
    scenario: str
    report: dict[str, Any]         # Harness.report(): kết thúc ra sao, tốn bao nhiêu
    final_answer: str
    handoff: str                   # bản bàn giao (rỗng nếu đạt mục tiêu)
    trace: str                     # trace đã định dạng
    messages: list[BaseMessage] = field(default_factory=list, repr=False)
    harness: Harness | None = field(default=None, repr=False)
    world: FlightWorld | None = field(default=None, repr=False)


def closing_text(harness: Harness) -> str:
    """Tin nhắn cuối do HARNESS viết: tóm tắt đã kiểm chứng (đạt mục tiêu) hoặc bản bàn giao (dừng khác)."""
    if harness.stop_reason is StopReason.GOAL:
        return harness.summary()  # dựng từ dữ liệu thật nên không thể bịa
    return harness.handoff.render() if harness.handoff else ""


_PATTERN_LABEL = {"react": "ReAct", "plan": "Plan-then-Execute", "hybrid": "Lai (Plan + replan)"}


def print_result(r: RunResult) -> None:
    """In trace và số liệu một lần chạy (dùng chung cho giao diện dòng lệnh của cả ba mẫu)."""
    rep = r.report
    print(f"=== {_PATTERN_LABEL.get(r.pattern, r.pattern)} · kịch bản {r.scenario} ===\n")
    print(r.trace)
    print("\n--- Kết quả ---")
    print(f"Kiểu dừng : {rep['stop_letter']} ({rep['stop']}) · {rep['stop_detail']}")
    print(f"Thành công: {rep['success']} (theo is_done)")
    print(f"Tốn       : {rep['rounds']} lần gọi model · {rep['tool_calls']} lần gọi tool · "
          f"{rep['tokens']} token · {rep['cost_usd']} USD · {rep['seconds']}s")
    if "plan_steps" in rep:
        print(f"Kế hoạch  : {rep['plan_steps']} bước · lập lại {rep['replans']} lần")
    print(f"Can thiệp : {rep['interventions'] or 'không'} · tác dụng phụ: {rep['side_effects'] or 'không'}")
    if rep["ungrounded"]:
        print(f"Bịa       : {rep['ungrounded']}")


def render_trace(messages: list[BaseMessage], width: int = 140) -> str:
    """Trace theo vòng, đúng khuôn ReAct: Suy luận → Hành động → Quan sát."""

    def cut(text: str) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= width else text[: width - 1] + "…"

    lines: list[str] = []
    rnd = 0
    for m in messages:
        source = (m.additional_kwargs or {}).get("source")
        harness_made = source == "harness"
        if isinstance(m, HumanMessage):
            label = {"harness": "[Harness → model]", "approver": "[Người duyệt]"}.get(source, "[Yêu cầu]")
            lines.append(f"{label} {cut(content_text(m.content))}")
        elif isinstance(m, AIMessage) and source == "planner":  # kế hoạch do model lập (không phải một vòng)
            label = "[Kế hoạch mới]" if m.additional_kwargs.get("replan") else "[Kế hoạch]"
            attempt = m.additional_kwargs.get("attempt", 1)
            lines.append(label + (f" (lần {attempt}, sau khi sửa)" if attempt > 1 else ""))
            lines += [f"   {ln}" for ln in content_text(m.content).splitlines()]
        elif isinstance(m, AIMessage) and harness_made:
            lines.append("── Harness dừng vòng lặp ──")
            lines += [f"   {ln}" for ln in content_text(m.content).splitlines()]
        elif isinstance(m, AIMessage):
            rnd += 1
            if m.tool_calls:
                if content_text(m.content):
                    lines.append(f"[V{rnd}] Suy luận : {cut(content_text(m.content))}")
                for c in m.tool_calls:
                    lines.append(f"[V{rnd}] Hành động: {fmt_call(c['name'], c['args'])}")
            else:
                lines.append(f"[V{rnd}] Trả lời   : {cut(content_text(m.content))}")
        elif isinstance(m, ToolMessage):
            lines.append(f"[V{rnd}] Quan sát : {cut(content_text(m.content))}")
    return "\n".join(lines)
