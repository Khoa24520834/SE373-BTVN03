# Kết quả đánh giá (fake)

Giá quy đổi chi phí: giả định 3 / 15 USD mỗi 1M token (model giả, không có chi phí thật).
Kết thúc lý tưởng: S1 → A · S2 → C (hoặc A nếu tự tìm đường vòng) · S4 → E · S6 → A. Ở S2 và S4, KHÔNG đặt được vé là kết quả đúng (dừng vì lặp, hoặc chờ người duyệt): hãy đọc cột Đúng kỳ vọng. Các cột Gọi model, Gọi tool, Token, Chi phí, Giây, Can thiệp là TRUNG BÌNH mỗi lần chạy.

> Model GIẢ: đo cơ chế (số vòng, token, kiểu dừng), không đo độ thông minh; chi phí chỉ mang tính minh hoạ.

## S1 · happy

| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | 1 | 1/1 | A×1 | 1/1 | 4.0 | 4.0 | 1712 | 0.00697 | 0.02 | 0.0 |
| Plan-then-Execute | 1 | 1/1 | A×1 | 1/1 | 1.0 | 4.0 | 382 | 0.00286 | 0.02 | 0.0 |
| Lai | 1 | 1/1 | A×1 | 1/1 | 1.0 | 4.0 | 382 | 0.00286 | 0.02 | 0.0 |

## S2 · timeout

| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | 1 | 0/1 | C×1 | 1/1 | 4.0 | 4.0 | 1688 | 0.00697 | 0.02 | 0.0 |
| Plan-then-Execute | 1 | 0/1 | C×1 | 1/1 | 1.0 | 4.0 | 382 | 0.00286 | 0.02 | 0.0 |
| Lai | 1 | 0/1 | C×1 | 1/1 | 1.0 | 4.0 | 382 | 0.00286 | 0.02 | 0.0 |

## S4 · approval

| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | 1 | 0/1 | E×1 | 1/1 | 3.0 | 3.0 | 1046 | 0.00454 | 0.01 | 1.0 |
| Plan-then-Execute | 1 | 0/1 | E×1 | 1/1 | 1.0 | 3.0 | 382 | 0.00286 | 0.02 | 1.0 |
| Lai | 1 | 0/1 | E×1 | 1/1 | 1.0 | 3.0 | 382 | 0.00286 | 0.02 | 1.0 |

## S6 · env_change

| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | 1 | 1/1 | A×1 | 1/1 | 5.0 | 5.0 | 2358 | 0.00937 | 0.02 | 0.0 |
| Plan-then-Execute | 1 | 0/1 | D×1 | 0/1 | 1.0 | 2.0 | 382 | 0.00286 | 0.01 | 0.0 |
| Lai | 1 | 1/1 | A×1 | 1/1 | 2.0 | 5.0 | 768 | 0.00524 | 0.02 | 0.0 |

## Tổng hợp theo mẫu (mọi kịch bản)

| Mẫu | Số lần | Đặt được | Kết thúc | Đúng kỳ vọng | Gọi model | Gọi tool | Token | Chi phí (USD) | Giây | Can thiệp |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | 4 | 2/4 | A×2 C×1 E×1 | 4/4 | 4.0 | 4.0 | 1701 | 0.00696 | 0.02 | 0.2 |
| Plan-then-Execute | 4 | 1/4 | A×1 C×1 D×1 E×1 | 3/4 | 1.0 | 3.2 | 382 | 0.00286 | 0.02 | 0.2 |
| Lai | 4 | 2/4 | A×2 C×1 E×1 | 4/4 | 1.2 | 4.0 | 478 | 0.00346 | 0.02 | 0.2 |
