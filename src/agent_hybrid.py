"""
agent_hybrid.py — Mẫu 3: Lai "Plan + replan" (SE373 · BTVN#3)

Ý TƯỞNG (đúng sơ đồ "ReAct + Plan" trong slide)
    Lập kế hoạch → thực thi từng bước → Observation đổi đáng kể? → có thì LẬP LẠI kế hoạch dựa trên
    điều vừa quan sát → không thì chạy tiếp → xong.
    Nhận lợi thế của cả hai: kế hoạch nhìn thấy trước (Plan) và thích nghi theo thực tế (ReAct).

CÁCH CÀI ĐẶT: dùng chung đồ thị với Plan-then-Execute (agent_plan.py), chỉ khác MỘT tham số
    max_replans = 0   Plan thuần: kế hoạch lỗi thời thì dừng ở kết D
    max_replans > 0   Lai: kế hoạch lỗi thời thì loại chuyến hỏng rồi gọi planner lập kế hoạch MỚI

    Hai chỗ slide để mở, ở đây được chốt bằng CODE (xác định, kiểm thử được):
    - "Thực thi k bước rồi mới xét": ta xét SAU MỖI BƯỚC (k = 1) vì mỗi bước đều có thể có tác dụng phụ.
    - "Observation đổi đáng kể": observation có trường status khác kỳ vọng (expect) của bước đó.

    Những điều Lai KHÔNG làm, có chủ ý:
    - Lỗi lặp (cùng tool, cùng tham số) không được "cứu" bằng replan: vẫn dừng ở kết C.
    - Chỉ kế hoạch gốc cần người duyệt; kế hoạch thay thế chỉ cần qua kiểm tra bằng code (lint_plan).
    - Số lần replan bị chặn (max_replans). Hết lượt mà vẫn lệch thì dừng ở kết D và bàn giao.

    Hướng khác mà LangChain 1.x cung cấp sẵn là TodoListMiddleware (model tự giữ một danh sách việc cần làm).
    Ở đây tự dựng bằng LangGraph để harness kiểm soát được từng bước.

CHẠY
    python src/agent_hybrid.py --scenario env_change              # thích nghi sang VJ606
    python src/agent_hybrid.py --scenario env_change --replans 1
    python src/agent_hybrid.py --scenario env_change --model real # planner là model thật (cần file .env)
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from agent_plan import PLANNER_STYLES, Approver, run_plan
from common import RunResult, load_env, print_result
from harness import Budget, Constraints
from tools_flight import SCENARIOS


def run_hybrid(scenario: str = "happy", *, model: Any = "fake", planner_style: str = "canonical",
               budget: Budget | None = None, constraints: Constraints | None = None,
               max_replans: int = 2, max_retries: int = 2, max_plan_retries: int = 1,
               approver: Approver | None = None, recursion_limit: int = 100) -> RunResult:
    """Chạy MỘT lần mẫu Lai trong MỘT kịch bản. max_replans phải >= 1 (bằng 0 là Plan thuần: dùng run_plan)."""
    if max_replans < 1:
        raise ValueError("max_replans phải >= 1; max_replans=0 là Plan thuần (agent_plan.run_plan)")
    return run_plan(scenario, model=model, planner_style=planner_style, budget=budget, constraints=constraints,
                    max_replans=max_replans, max_retries=max_retries, max_plan_retries=max_plan_retries,
                    approver=approver, recursion_limit=recursion_limit, pattern="hybrid")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # in tiếng Việt đúng trên console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()  # nạp .env cho --model real (pytest và hàm thư viện không bao giờ tự đọc .env)
    parser = argparse.ArgumentParser(description="Chạy agent Lai (Plan + replan)")
    parser.add_argument("--scenario", choices=SCENARIOS, default="happy")
    parser.add_argument("--model", choices=("fake", "real"), default="fake")
    parser.add_argument("--planner-style", choices=PLANNER_STYLES, default="canonical")
    parser.add_argument("--replans", type=int, default=2, help="số lần được lập lại kế hoạch (>= 1)")
    parser.add_argument("--retries", type=int, default=2, help="số lần thử lại một bước khi tool báo lỗi")
    parser.add_argument("--plan-retries", type=int, default=1, help="số lần planner được sửa kế hoạch sai")
    args = parser.parse_args()
    try:
        outcome = run_hybrid(args.scenario, model=args.model, planner_style=args.planner_style,
                             max_replans=args.replans, max_retries=args.retries, max_plan_retries=args.plan_retries)
    except (RuntimeError, ValueError) as exc:  # ví dụ: --model real mà chưa có file .env
        parser.error(str(exc))
    print_result(outcome)
