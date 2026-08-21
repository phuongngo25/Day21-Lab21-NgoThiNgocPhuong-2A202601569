# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

`attn_only` (chỉ gắn LoRA vào q,v, rank nâng lên 283 để khớp ngân sách tham số) đạt đúng
target=0.97 — bằng tuyệt đối với `correct` (gắn vào cả 12 lớp linear, rank 16). Tôi nghĩ
gắn adapter vào nhiều lớp hơn kiểu gì cũng phải nhỉnh hơn một chút, nhưng dữ liệu cho thấy
ở ngân sách tham số này, rank cao bù được hoàn toàn cho vị trí hẹp trên bài toán 225 mẫu.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở việc train — 4 run GPU chạy suôn sẻ trên Colab T4 sau khi pipeline đã được
vá các lỗi trong `SIMULATION-FINDINGS.md`. Chỗ tốn thời gian nhất là một lỗi test suite
không liên quan gì đến model: `tests/test_env_and_silent_defaults.py` dùng
`from tests.fake_tokenizer import FakeTokenizer` (import kiểu package) trong khi mọi file
test khác dùng `from fake_tokenizer import FakeTokenizer` (import trần) — venv sạch trên
máy cá nhân thì chạy được (vì `tests/` được đưa thẳng vào `sys.path`), nhưng trên Colab
(rất nhiều package cài sẵn) một gói khác chiếm mất tên `tests`, khiến 3 test fail chỉ khi
chạy chung cả suite chứ không fail khi chạy riêng lẻ — mất công tái hiện đúng lỗi mới sửa
được, dù bản sửa cuối cùng chỉ là đổi 3 dòng.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ chọn đúng lớp để gắn LoRA (attention vs toàn bộ linear layer) là quyết định
quan trọng nhất. Số liệu ở NB4 cho thấy learning rate mới là biến số quyết định — sai LR
10× (`wrong_lr`) làm target rơi thẳng từ 0.97 xuống 0.00, y hệt model chưa train, trong
khi đổi vị trí gắn adapter (`attn_only` so với `correct`) không tạo ra chênh lệch nào.

**4. Bạn dùng AI assistant vào việc gì trong lab này? Chỗ nào nó sai?**

Dùng Claude Code để: đọc toàn bộ repo và lập kế hoạch chạy pipeline, hướng dẫn từng bước
chạy trên Colab, chẩn đoán lỗi `ModuleNotFoundError: tests.fake_tokenizer` (tái hiện được
lỗi cục bộ bằng cách cài torch vào venv rồi so sánh chạy riêng lẻ vs chạy cả suite), và
tổng hợp số liệu từ `results/*.json` thành `REPORT.md`. Chỗ nó sai: lần đầu đoán nhầm
nguyên nhân lỗi 3 test fail là do phiên bản torch/bf16 (dựa theo các phát hiện F-15/F-23
trong `SIMULATION-FINDINGS.md`) — chạy thử với cùng phiên bản torch 2.13.0 trên máy local
thì tất cả pass, chứng minh giả thuyết đó sai, phải xin traceback thật từ Colab mới tìm ra
nguyên nhân thật (xung đột tên package `tests`).

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Kiểm tra phân bố nhãn trong tập train trước khi train, không phải sau. `results/qualitative.json`
cho thấy toàn bộ 7/50 ca fine-tune trả lời sai đều sai đúng một nhãn (`urgency=thap` bị
đoán thành `trung_binh`) — nhiều khả năng do 225 mẫu train có quá ít ví dụ nhãn `thap`.
Nếu kiểm tra phân bố nhãn từ đầu, có thể đã phát hiện và cân bằng lại trước khi tốn một
lượt train + eval đầy đủ.
