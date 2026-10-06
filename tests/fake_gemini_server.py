"""
fake_gemini_server.py — Máy chủ GEMINI (Google) GIẢ trên localhost để kiểm thử đường chạy Gemini mà không tốn tiền.

Nói đúng giao thức generateContent: POST /v1beta/models/<model>:generateContent, header x-goog-api-key,
phần functionCall / functionResponse. Nội dung trả lời do model giả tất định quyết định (ScriptedModel cho agent,
ScriptedPlanner cho planner).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent_plan import ScriptedPlanner
from common import ScriptedModel, message_text
from harness import Constraints, estimate_tokens


def gemini_to_langchain(body: dict[str, Any]) -> list[BaseMessage]:
    """Đổi request generateContent thành danh sách tin nhắn LangChain."""
    out: list[BaseMessage] = []
    system = body.get("systemInstruction") or {}
    if system.get("parts"):
        out.append(SystemMessage(content="".join(p.get("text", "") for p in system["parts"])))
    pending: list[tuple[str, str]] = []  # (tên tool, id) của lời gọi chưa có kết quả
    n = 0
    for c in body.get("contents", []):
        parts = c.get("parts", [])
        if c.get("role") == "model":
            calls = []
            for p in parts:
                if "functionCall" in p:
                    n += 1
                    fc = p["functionCall"]
                    call_id = fc.get("id") or f"gc_{n}"
                    calls.append({"name": fc["name"], "args": fc.get("args", {}), "id": call_id, "type": "tool_call"})
                    pending.append((fc["name"], call_id))
            out.append(AIMessage(content="".join(p.get("text", "") for p in parts if "text" in p), tool_calls=calls))
            continue
        for p in parts:
            if "functionResponse" in p:
                fr = p["functionResponse"]
                call_id = fr.get("id") or next((i for name, i in pending if name == fr["name"]), "gc_?")
                pending = [x for x in pending if x[1] != call_id]
                resp = fr.get("response", {})
                if isinstance(resp, dict) and len(resp) == 1 and isinstance(next(iter(resp.values())), str):
                    content = next(iter(resp.values()))
                else:
                    content = json.dumps(resp, ensure_ascii=False)
                out.append(ToolMessage(content=content, tool_call_id=call_id))
            elif "text" in p:
                out.append(HumanMessage(content=p["text"]))
    return out


def decide(messages: list[BaseMessage], tools: list[str]) -> AIMessage:
    """Nội dung trả lời: gọi echo (bài kiểm tra tool), agent giả, planner giả, hoặc 'OK'."""
    if tools == ["echo"]:
        return AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "xin chào"},
                                                  "id": "gc_echo", "type": "tool_call"}])
    if tools:
        return ScriptedModel(style="competent", constraints=Constraints())._decide(messages)
    system = " ".join(message_text(m) for m in messages if isinstance(m, SystemMessage))
    if "bộ lập kế hoạch" in system:
        return ScriptedPlanner()._generate(messages).generations[0].message
    return AIMessage(content="OK")


class FakeGemini:
    def __init__(self, api_key: str = "AIzaSy-test-0123456789abcdef", model: str = "gemini-3.8-flash") -> None:
        self.api_key, self.model = api_key, model
        self.requests: list[dict[str, Any]] = []  # mọi request đã nhận: path, x-goog-api-key, body
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "FakeGemini":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # im lặng
                pass

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _error(self, code: int, status: str, message: str) -> None:
                self._send(code, {"error": {"code": code, "message": message, "status": status}})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.requests.append({"path": self.path, "auth": self.headers.get("x-goog-api-key"), "body": body})
                if self.headers.get("x-goog-api-key") != outer.api_key:
                    return self._error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")
                if self.path != f"/v1beta/models/{outer.model}:generateContent":
                    return self._error(404, "NOT_FOUND", f"models/{self.path} is not found for API version v1beta")
                tools = [d["name"] for t in body.get("tools", []) for d in t.get("functionDeclarations", [])]
                messages = gemini_to_langchain(body)
                ai = decide(messages, tools)
                parts: list[dict[str, Any]] = [{"text": ai.content}] if ai.content else []
                parts += [{"functionCall": {"name": c["name"], "args": c["args"]}} for c in ai.tool_calls]
                n_in = estimate_tokens("\n".join(message_text(m) for m in messages))
                n_out = estimate_tokens(message_text(ai))
                self._send(200, {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                                 "usageMetadata": {"promptTokenCount": n_in, "candidatesTokenCount": n_out,
                                                   "totalTokenCount": n_in + n_out},
                                 "modelVersion": outer.model})

        return Handler
