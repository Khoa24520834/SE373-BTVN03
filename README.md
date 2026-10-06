# SE373 · BTVN#3 — Agent đặt vé máy bay bằng LangChain/LangGraph

Agent đặt vé máy bay chạy trên **tool mockup**, được bao bởi một **lớp harness** dùng chung, cài đặt theo **3 mẫu thiết kế** (ReAct, Plan-then-Execute, Lai) và đánh giá trên 4 kịch bản. Model thật: **Google Gemini 3.8 Flash**.

Báo cáo: [`report/BaoCao_BTVN03.pdf`](report/BaoCao_BTVN03.pdf)

## Cấu trúc

```text
src/
  tools_flight.py   Tool mockup (5 tool, 4 kịch bản lỗi), dữ liệu cố định
  harness.py        Ràng buộc là dữ liệu, tiêu chí hoàn thành bằng code, kiểm quyền, bàn giao,
                    phát hiện lặp/bế tắc, ngân sách, chống bịa
  common.py         Model giả, đọc .env, dựng model Gemini, định dạng trace
  agent_react.py    Mẫu 1: ReAct (create_agent + middleware harness)
  agent_plan.py     Mẫu 2: Plan-then-Execute (LangGraph, interrupt cho người duyệt)
  agent_hybrid.py   Mẫu 3: Lai (Plan + lập lại kế hoạch khi lệch kỳ vọng)
  evaluate.py       Chạy 3 mẫu × 4 kịch bản, ghi CSV, bảng tổng hợp, trace
  check_api.py      Kiểm tra API key, model, tool calling
tests/              Test pytest (chạy offline, không cần API key)
results/            Kết quả đánh giá (CSV, bảng tổng hợp, trace từng lần chạy)
report/             Báo cáo PDF
```

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cấu hình API key (chỉ cần khi chạy model thật)

Sao chép `.env.example` thành `.env` rồi điền key lấy tại aistudio.google.com:

```text
GOOGLE_API_KEY=key-cua-ban
GOOGLE_MODEL=gemini-3.8-flash
```

Không commit file `.env`. Kiểm tra kết nối:

```powershell
python src/check_api.py
```

## Chạy

```powershell
pytest -q                                              # toàn bộ test, không cần key
python src/agent_react.py --scenario happy             # một mẫu, một kịch bản, model giả
python src/agent_plan.py --scenario env_change
python src/agent_hybrid.py --scenario env_change
python src/agent_react.py --scenario happy --model real   # dùng Gemini
```

Kịch bản: `happy` (S1), `timeout` (S2), `approval` (S4), `env_change` (S6).

## Đánh giá

```powershell
python src/evaluate.py                                                      # model giả, tất định
python src/evaluate.py --model real --scenarios happy env_change --repeats 3   # Gemini
```

Kết quả ghi vào `results/`:

| File | Nội dung |
|---|---|
| `ket_qua_<model>.csv` | Mỗi dòng một lần chạy |
| `tong_hop_<model>.md` | Bảng tổng hợp theo kịch bản và theo mẫu |
| `traces_<model>/` | Trace từng lần chạy (Suy luận → Hành động → Quan sát) |

Kiểu dừng: **A** đạt mục tiêu · **B** hết ngân sách · **C** phát hiện lặp · **D** bế tắc · **E** cần con người.
