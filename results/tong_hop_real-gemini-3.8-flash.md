# Kết quả đánh giá (real-gemini-3.8-flash)

Giá quy đổi chi phí: 0.75 USD / 1M token vào, 3.75 USD / 1M token ra (nguồn: bảng giá Google đã kiểm tra (10/2026)).
Kết thúc lý tưởng: S1 → A · S2 → C (hoặc A nếu tự tìm đường vòng) · S4 → E · S6 → A. Ở S2 và S4, KHÔNG đặt được vé là kết quả đúng (dừng vì lặp, hoặc chờ người duyệt): hãy đọc cột Đúng kỳ vọng. Các cột Gọi model, Gọi tool, Token, Chi phí, Giây, Can thiệp là TRUNG BÌNH mỗi lần chạy.

## S1 · happy

| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | 3 (2 lỗi) | 0/3 | B×1 | 0/3 | 5.0 | 4.0 | 8702 | 0.01097 | 190.90 | 1.0 |
| Plan-then-Execute | 1 (1 lỗi) | 0/1 | - | 0/1 | - | - | - | - | - | - |

## Tổng hợp theo mẫu (mọi kịch bản)

| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | 3 (2 lỗi) | 0/3 | B×1 | 0/3 | 5.0 | 4.0 | 8702 | 0.01097 | 190.90 | 1.0 |
| Plan-then-Execute | 1 (1 lỗi) | 0/1 | - | 0/1 | - | - | - | - | - | - |
