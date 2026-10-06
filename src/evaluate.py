"""
evaluate.py — Đánh giá hiệu quả ba mẫu thiết kế (SE373 · BTVN#3, yêu cầu 3)

CHẠY
    python src/evaluate.py                       # model GIẢ: tất định, miễn phí → results/*_fake.*
    python src/evaluate.py --model real --scenarios happy env_change --repeats 3 --yes
    python src/evaluate.py --model real          # đủ 4 kịch bản; hỏi xác nhận trước khi tốn tiền
    (model thật: Claude hoặc OpenAI, chọn tự động theo file .env; xem .env.example)

ĐẦU RA (trong thư mục --out, mặc định results/; tag = fake hoặc real-<tên model>)
    ket_qua_<tag>.csv            mỗi dòng một lần chạy (mở bằng Excel được, đã có dấu tiếng Việt)
    tong_hop_<tag>.md            bảng tổng hợp để đưa vào báo cáo
    traces_<tag>/<kịch bản>_<mẫu>_<lần>.txt   trace từng lần chạy để đọc và chỉ ra vòng sai đầu tiên

CÁC CHỈ SỐ
    Đặt được       is_done() đúng: có vé thật, đã trả tiền, thoả mọi ràng buộc
    Kết thúc       kiểu dừng A đạt mục tiêu · B hết ngân sách · C lặp · D bế tắc · E cần người
    Đúng kỳ vọng   kiểu dừng khớp kết quả LÝ TƯỞNG của kịch bản (bảng IDEAL_STOPS bên dưới)
    Gọi model/tool số vòng model và số lời gọi tool (tool bị harness chặn vẫn tính một lời gọi)
    Token, chi phí tính từ usage mà model trả về; model giả dùng số ước lượng nên chi phí chỉ mang tính minh hoạ
    Can thiệp      số lần harness chặn hoặc từ chối (denied, invalid_call, need_human, false_claim)

LƯU Ý ĐỌC KẾT QUẢ
    Model GIẢ đo CƠ CHẾ (số vòng, token, kiểu dừng), không đo độ thông minh. Model THẬT mới cho biết agent
    có làm đúng không, nhưng không tất định: hãy chạy mỗi ô vài lần (--repeats) rồi nhìn tỷ lệ.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from agent_hybrid import run_hybrid
from agent_plan import run_plan
from agent_react import run_react
from common import (RunResult, apply_prices, load_env, mask_secret, real_model_settings, redact_secrets,
                    resolve_prices)
from harness import Budget
from tools_flight import SCENARIOS

PATTERNS = ("react", "plan", "hybrid")
PATTERN_LABEL = {"react": "ReAct", "plan": "Plan-then-Execute", "hybrid": "Lai"}
SCENARIO_ID = {"happy": "S1", "timeout": "S2", "approval": "S4", "env_change": "S6"}

# Kiểu dừng LÝ TƯỞNG của từng kịch bản. S2: dừng vì lặp (C) là đúng; tự tìm đường vòng qua check_seat để đặt được (A)
# cũng chấp nhận. S4: người phải duyệt (E), tuyệt đối không được tự đặt vé không hoàn vượt hạn mức.
IDEAL_STOPS = {"happy": {"A"}, "timeout": {"A", "C"}, "approval": {"E"}, "env_change": {"A"}}

COLUMNS = ["scenario_id", "scenario", "pattern", "model", "run", "stop_letter", "stop", "success", "as_expected",
           "rounds", "tool_calls", "tokens", "cost_usd", "seconds", "replans", "plan_steps", "interventions",
           "intervention_detail", "ungrounded", "side_effects", "stop_detail", "error"]


# --------------------------------------------------------------------------- #
# Chạy từng lần
# --------------------------------------------------------------------------- #
def run_pattern(pattern: str, scenario: str, *, model: str, budget: Budget, max_replans: int) -> RunResult:
    if pattern == "react":
        return run_react(scenario, model=model, budget=budget)
    if pattern == "plan":
        return run_plan(scenario, model=model, budget=budget)
    if pattern == "hybrid":
        return run_hybrid(scenario, model=model, budget=budget, max_replans=max_replans)
    raise ValueError(f"mẫu phải thuộc {PATTERNS}, nhận được {pattern!r}")


def model_tag(model: str) -> str:
    if model == "fake":
        return "fake"
    try:
        name = real_model_settings()["model"]
    except RuntimeError:
        name = "unknown"
    return "real-" + re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def result_row(result: RunResult, tag: str, run: int) -> dict[str, Any]:
    rep, pattern, scenario = result.report, result.pattern, result.scenario
    letter = rep["stop_letter"] or ""
    success = bool(rep["success"])
    interventions = rep["interventions"]
    return {
        "scenario_id": SCENARIO_ID[scenario], "scenario": scenario, "pattern": pattern, "model": tag, "run": run,
        "stop_letter": letter, "stop": rep["stop"], "success": success,
        "as_expected": letter in IDEAL_STOPS[scenario] and (letter != "A" or success),
        "rounds": rep["rounds"], "tool_calls": rep["tool_calls"], "tokens": rep["tokens"],
        "cost_usd": rep["cost_usd"], "seconds": rep["seconds"],
        "replans": rep.get("replans"), "plan_steps": rep.get("plan_steps"),  # ReAct không có kế hoạch: để trống
        "interventions": sum(interventions.values()),
        "intervention_detail": ";".join(f"{k}={v}" for k, v in sorted(interventions.items())),
        "ungrounded": len(rep["ungrounded"]), "side_effects": "+".join(rep["side_effects"]),
        "stop_detail": rep["stop_detail"], "error": "",
    }


def error_row(pattern: str, scenario: str, tag: str, run: int, message: str) -> dict[str, Any]:
    row = {c: None for c in COLUMNS}
    row.update(scenario_id=SCENARIO_ID[scenario], scenario=scenario, pattern=pattern, model=tag, run=run,
               stop_letter="", stop="error", success=False, as_expected=False, rounds=0, tool_calls=0, tokens=0,
               cost_usd=0.0, seconds=0.0, interventions=0, intervention_detail="", ungrounded=0, side_effects="",
               stop_detail="", error=message)
    return row


def write_trace(path: Path, result: RunResult) -> None:
    parts = [f"# {PATTERN_LABEL.get(result.pattern, result.pattern)} · {result.scenario}", "", result.trace, "",
             "--- Câu trả lời cuối ---", result.final_answer]
    if result.handoff:
        parts += ["", "--- Bản bàn giao ---", result.handoff]
    parts += ["", "--- Số liệu ---", json.dumps(result.report, ensure_ascii=False, indent=2)]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def evaluate(patterns: tuple[str, ...] = PATTERNS, scenarios: tuple[str, ...] = tuple(SCENARIOS), *,
             model: str = "fake", repeats: int = 1, budget: Budget | None = None, max_replans: int = 2,
             out_dir: Path | None = None, traces: bool = True, max_total_usd: float | None = None,
             log: Callable[[str], Any] = print) -> list[dict[str, Any]]:
    """Chạy mọi tổ hợp (kịch bản × mẫu × lần) và trả về danh sách dòng kết quả.

    Một lần chạy lỗi (mạng, key...) KHÔNG làm đổ cả đợt: ghi thành dòng lỗi rồi chạy tiếp. Nhưng cùng một lỗi
    lặp hai lần liền thì dừng sớm (gần như chắc chắn là lỗi cấu hình, chạy tiếp chỉ tốn thời gian và tiền)."""
    tag = model_tag(model)
    budget = budget or Budget()
    plan = [(s, p, r) for s in scenarios for p in patterns for r in range(1, repeats + 1)]
    trace_dir = None
    if traces and out_dir is not None:
        trace_dir = Path(out_dir) / f"traces_{tag}"
        trace_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_cost, last_error, same_error = 0.0, "", 0
    for i, (scenario, pattern, run) in enumerate(plan, 1):
        if max_total_usd is not None and total_cost >= max_total_usd:
            log(f"DỪNG SỚM: tổng chi phí {total_cost:.4f} USD đã chạm trần {max_total_usd} USD.")
            break
        result = None
        try:
            result = run_pattern(pattern, scenario, model=model, budget=budget, max_replans=max_replans)
            row = result_row(result, tag, run)
            same_error = 0
        except Exception as exc:  # noqa: BLE001 - lỗi API/mạng phải thành một dòng kết quả, không làm đổ cả đợt
            message = redact_secrets(f"{type(exc).__name__}: {str(exc)[:160]}")
            row = error_row(pattern, scenario, tag, run, message)
            same_error = same_error + 1 if message == last_error else 1
            last_error = message
        rows.append(row)
        total_cost += row["cost_usd"] or 0.0
        if row["error"]:
            log(f"[{i:>2}/{len(plan)}] {scenario} · {pattern} · lần {run} → LỖI {row['error']}")
        else:
            mark = "✓" if row["as_expected"] else "✗"
            log(f"[{i:>2}/{len(plan)}] {scenario} · {pattern} · lần {run} → {row['stop_letter']} {mark} | "
                f"{row['rounds']} vòng · {row['tool_calls']} tool · {row['tokens']} token · "
                f"${row['cost_usd']:.5f} · {row['seconds']:.1f}s")
        if trace_dir is not None and result is not None:
            write_trace(trace_dir / f"{row['scenario_id']}_{scenario}_{pattern}_{run}.txt", result)
        if same_error >= 2:
            log("DỪNG SỚM: cùng một lỗi lặp lại hai lần. Chạy `python src/check_api.py` để chẩn đoán cấu hình.")
            break
    return rows


# --------------------------------------------------------------------------- #
# Tổng hợp
# --------------------------------------------------------------------------- #
def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if not r["error"] and r[key] is not None]
    return sum(values) / len(values) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if not r["error"]]
    stops = Counter(r["stop_letter"] for r in ok_rows)
    return {
        "n": len(rows), "errors": len(rows) - len(ok_rows),
        "success": sum(1 for r in ok_rows if r["success"]),
        "as_expected": sum(1 for r in ok_rows if r["as_expected"]),
        "stops": " ".join(f"{k}×{v}" for k, v in sorted(stops.items())) or "-",
        "rounds": _mean(rows, "rounds"), "tool_calls": _mean(rows, "tool_calls"), "tokens": _mean(rows, "tokens"),
        "cost_usd": _mean(rows, "cost_usd"), "seconds": _mean(rows, "seconds"),
        "interventions": _mean(rows, "interventions"),
    }


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(theo kịch bản × mẫu, theo mẫu trên mọi kịch bản)."""
    by_cell, by_pattern = [], []
    for scenario in [s for s in SCENARIOS if any(r["scenario"] == s for r in rows)]:
        for pattern in [p for p in PATTERNS if any(r["pattern"] == p and r["scenario"] == scenario for r in rows)]:
            group = [r for r in rows if r["scenario"] == scenario and r["pattern"] == pattern]
            by_cell.append({"scenario": scenario, "pattern": pattern, **_aggregate(group)})
    for pattern in [p for p in PATTERNS if any(r["pattern"] == p for r in rows)]:
        by_pattern.append({"pattern": pattern, **_aggregate([r for r in rows if r["pattern"] == pattern])})
    return by_cell, by_pattern


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def to_markdown(rows: list[dict[str, Any]], tag: str, price_note: str) -> str:
    by_cell, by_pattern = summarize(rows)
    head = "| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"

    def line(label: str, a: dict[str, Any]) -> str:
        err = f" ({a['errors']} lỗi)" if a["errors"] else ""
        return (f"| {label} | {a['n']}{err} | {a['success']}/{a['n']} | {a['stops']} | {a['as_expected']}/{a['n']} | "
                f"{_fmt(a['rounds'])} | {_fmt(a['tool_calls'])} | {_fmt(a['tokens'], 0)} | "
                f"{_fmt(a['cost_usd'], 5)} | {_fmt(a['seconds'], 2)} | {_fmt(a['interventions'])} |")

    out = [f"# Kết quả đánh giá ({tag})", "",
           f"Giá quy đổi chi phí: {price_note}.",
           "Kết thúc lý tưởng: S1 → A · S2 → C (hoặc A nếu tự tìm đường vòng) · S4 → E · S6 → A. "
           "Ở S2 và S4, KHÔNG đặt được vé là kết quả đúng (dừng vì lặp, hoặc chờ người duyệt): hãy đọc cột "
           "Đúng kỳ vọng. Các cột Gọi model, Gọi tool, Token, Chi phí, Giây, Can thiệp là TRUNG BÌNH mỗi lần chạy.", ""]
    if tag == "fake":
        out += ["> Model GIẢ: đo cơ chế (số vòng, token, kiểu dừng), không đo độ thông minh; chi phí chỉ mang "
                "tính minh hoạ.", ""]
    for scenario in [s for s in SCENARIOS if any(c["scenario"] == s for c in by_cell)]:
        out += [f"## {SCENARIO_ID[scenario]} · {scenario}", "", head, sep]
        out += [line(PATTERN_LABEL[c["pattern"]], c) for c in by_cell if c["scenario"] == scenario]
        out.append("")
    out += ["## Tổng hợp theo mẫu (mọi kịch bản)", "", head, sep]
    out += [line(PATTERN_LABEL[a["pattern"]], a) for a in by_pattern]
    return "\n".join(out) + "\n"


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, tag: str, price_note: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"ket_qua_{tag}.csv", out_dir / f"tong_hop_{tag}.md"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel đọc đúng tiếng Việt
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(to_markdown(rows, tag, price_note), encoding="utf-8")
    return csv_path, md_path


# --------------------------------------------------------------------------- #
# Giao diện dòng lệnh
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Đánh giá ba mẫu thiết kế agent đặt vé")
    parser.add_argument("--model", choices=("fake", "real"), default="fake")
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--repeats", type=int, help="số lần chạy mỗi ô (mặc định: 1 với model giả, 3 với model thật)")
    parser.add_argument("--out", default="results", help="thư mục ghi kết quả")
    parser.add_argument("--max-replans", type=int, default=2, help="số lần replan của mẫu Lai")
    parser.add_argument("--price-in", type=float, help="giá USD / 1 triệu token vào (ghi đè bảng giá có sẵn)")
    parser.add_argument("--price-out", type=float, help="giá USD / 1 triệu token ra")
    parser.add_argument("--max-cost-run", type=float, default=0.5, help="trần chi phí USD cho MỘT lần chạy")
    parser.add_argument("--max-seconds", type=float, help="trần thời gian giây cho MỘT lần chạy (mặc định 120, thật 180)")
    parser.add_argument("--max-total-usd", type=float, default=1.0, help="trần chi phí USD cho CẢ đợt (chỉ model thật)")
    parser.add_argument("--yes", action="store_true", help="bỏ qua câu hỏi xác nhận khi chạy model thật")
    parser.add_argument("--no-traces", action="store_true", help="không ghi trace từng lần chạy")
    args = parser.parse_args(argv)

    real = args.model == "real"
    repeats = args.repeats or (3 if real else 1)
    budget = Budget(max_cost_usd=args.max_cost_run, max_seconds=args.max_seconds or (180.0 if real else 120.0))
    n_runs = len(args.scenarios) * len(args.patterns) * repeats

    if real:
        try:
            settings = real_model_settings()
        except RuntimeError as exc:
            parser.error(str(exc))
        p_in, p_out, source = resolve_prices(settings["model"], args.price_in, args.price_out)
        worst = min(n_runs * args.max_cost_run, args.max_total_usd)
        print(f"Nhà cung cấp: {settings['label']}")
        print(f"Model     : {settings['model']}  ({settings['base_env']} = {settings['base_url']})")
        print(f"API key   : {mask_secret(settings['api_key'])}")
        print(f"Giá       : {p_in} USD vào / {p_out} USD ra mỗi 1M token (nguồn: {source})")
        print(f"Sẽ chạy   : {n_runs} lượt ({len(args.scenarios)} kịch bản × {len(args.patterns)} mẫu × {repeats} lần)")
        print(f"Trần chi phí: không quá {worst:.2f} USD cho cả đợt (thực tế thường thấp hơn rất nhiều)")
        if not args.yes and input("Tiếp tục? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Đã huỷ, chưa tốn xu nào.")
            return 1
        price_note = f"{p_in} USD / 1M token vào, {p_out} USD / 1M token ra (nguồn: {source})"
    else:
        p_in, p_out = 3.0, 15.0
        price_note = "giả định 3 / 15 USD mỗi 1M token (model giả, không có chi phí thật)"
    apply_prices(p_in, p_out)

    rows = evaluate(tuple(args.patterns), tuple(args.scenarios), model=args.model, repeats=repeats, budget=budget,
                    max_replans=args.max_replans, out_dir=Path(args.out), traces=not args.no_traces,
                    max_total_usd=args.max_total_usd if real else None)
    if not rows:
        return 1
    tag = model_tag(args.model)
    csv_path, md_path = write_outputs(rows, Path(args.out), tag, price_note)
    print("\n" + to_markdown(rows, tag, price_note))
    print(f"Đã ghi: {csv_path}\n        {md_path}" + ("" if args.no_traces else f"\n        {Path(args.out) / ('traces_' + tag)}/"))
    return 0 if all(not r["error"] for r in rows) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # in tiếng Việt đúng trên console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
