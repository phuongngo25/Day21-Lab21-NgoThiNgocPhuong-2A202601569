# Lab 21 — Evaluation Report

**Họ tên**: Ngô Thị Ngọc Phượng  **MSSV**: 2A202601569  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free — Tesla T4 16GB (14.6GB khả dụng)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH tiếng Việt → JSON triage, corpus mặc định của lab (250 mẫu train) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (mặc định tier T4) — p95 đo được chỉ là **98** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 (`train.planned_steps(225, T4, 2) = 30`) |

**`max_length` lệch với p95 — giải thích:** `max_length=1024` là hằng số cố định theo tier
(áp dụng chung cho mọi corpus có thể đổi vào), không phải số đo riêng cho corpus này.
p95 thực đo chỉ 98 token (max toàn bộ 250 mẫu là 101 token) — nghĩa là không mẫu nào bị
cắt, biên an toàn rất rộng (10×), chỉ đánh đổi bằng việc tốn `max_length` token đệm không
cần thiết trong cấu hình LoRA. Không sửa `max_length` xuống 256 vì đó là cấu hình chung
cho tier, không riêng cho một corpus.

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json: "reasoning preserved — safe to train on traces")*

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (`results/mask_proof.json`, `supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3214.2 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1039.5 |
| (c) LoRA fine-tune | **0.970** | 0.678 | 1.000 | 1426.4 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — target nhảy từ 0.000 → 0.765, format từ
0.000 → 1.000, và latency giảm ~3.1× (naive prompt để model rambling tới trần
160 token; optimized prompt ép JSON ngắn gọn). `make verify` xác nhận `(a)=0.000 ->
(b)=0.765` và SHA của `OPTIMIZED_PROMPT` không đổi.

**Có sửa `OPTIMIZED_PROMPT` không?** Không — dùng nguyên bản của lab.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6268 | **0.97** | 940.7 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5364 | **0.97** | 776.1 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.00** | 905.1 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 985.7 | 7.09 |

format (NB5 §4): `correct`=1.00 · `attn_only`=1.00 · `wrong_lr`=0.00 · `qlora`=1.00.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

`attn_only` **hoà tuyệt đối** với `correct` trên target (0.97 = 0.97) và format (1.0 =
1.0), dù chỉ gắn vào `q,v` thay vì toàn bộ 12 lớp linear của decoder — bù lại bằng rank
283 thay vì 16 để khớp đúng ngân sách 32.46M tham số. Xét theo train loss thì thứ tự lại
*khác*: `attn_only` (0.5364) thấp hơn `correct` (0.6268), tức nếu chấm bằng loss thay vì
target thì `attn_only` trông "thắng" — nhưng trên thang đo thật cả hai bằng nhau. Điều
này khớp với đúng cái bẫy rubric 2.5 cảnh báo: train loss và target không nhất thiết xếp
hạng giống nhau. Với bài toán 225 mẫu, 4 khoá JSON này, rank cao bù cho vị trí hẹp là đủ
— nói cách khác, ở ngân sách tham số đã cho, *vị trí gắn adapter không phải đòn bẩy*; cả
hai cấu hình đều đã vượt ngưỡng năng lực cần thiết cho lượng thông tin trong dữ liệu.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` dùng LR 1e-5 (giảm 10× so với `correct`'s 1e-4 — đúng bằng thang LR của
full-fine-tune, quá nhỏ cho LoRA). Loss cuối là 1.5704 — cao hơn hẳn `correct` (0.6268)
và không hội tụ trong 30 step. Trên target/format, `wrong_lr` = 0.00/0.00, **giống hệt
baseline (a) chưa train** — nghĩa là model gần như không học được gì. Nếu chỉ nhìn con số
loss "cao hơn hẳn 3 run kia" mà không biết LR, kết luận sai dễ mắc nhất là *"cần train
nhiều step hơn"* hoặc *"cấu hình LoRA (rank/vị trí) có vấn đề"* — trong khi nguyên nhân
thật chỉ là một siêu tham số sai lệch 10×. Đây chính là lý do rubric bắt xếp hạng bằng
target chứ không phải train loss: loss không phân biệt được "chưa hội tụ" với "học sai".

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` dùng 7.09 GB so với 12.01 GB của `correct` — tiết kiệm **4.92 GB (~41%)**. Đổi
lại, target giảm 0.97 → 0.94 (**-0.03**), dù format vẫn hoàn hảo (1.0). Số đo này **ủng
hộ** khuyến nghị của nhà cung cấp: trên tier T4 (14.6 GB khả dụng), `correct` ở 12.01 GB
đã vừa thoải mái — không cần đánh đổi 3 điểm % target để tiết kiệm VRAM mà mình không
thiếu. QLoRA chỉ đáng cân nhắc khi VRAM thực sự là ràng buộc cứng (ví dụ tier `LAPTOP`
8–12 GB), không phải trên T4.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.080` · `valid_trace_rate = 0.00`

Cổng FAIL vì `regression` tụt 0.080 (0.758 → 0.678), vượt ngưỡng cho phép 0.020 —
dù `target` thắng đậm baseline (b) đã tối ưu prompt (+0.205, 0.765 → 0.970). Đây là dấu
hiệu **catastrophic forgetting** kinh điển: 30 step train toàn bộ trên JSON triage 4-khoá,
0% dữ liệu "phổ thông" xen vào, kéo trọng số lệch đủ xa để giảm khả năng trả lời câu hỏi
kiến thức chung — đúng bài học deck §14.3. `valid_trace_rate = 0.00` không phải lỗi: theo
F-30, corpus mặc định toàn JSON trần, không có khối `<think>` nào trong 250 câu trả lời,
nên số này **cấu trúc phải bằng 0** ở mọi run, không phản ánh chất lượng fine-tune.

Nói cách khác: bản fine-tune **học tác vụ hẹp rất tốt** (thắng cả baseline đã prompt kỹ)
nhưng **trả giá bằng năng lực tổng quát** — một FAILED có nguyên nhân rõ ràng và (theo
README's bảng lỗi thường gặp) có hướng sửa cụ thể: trộn 1–5% dữ liệu phổ thông vào tập
train để "tiêm chủng" chống quên thảm hoạ, chưa được thử trong lần chạy này.

---

## 6. Định tính — bắt buộc có cả ca THUA

> **Giới hạn dữ liệu, nêu rõ để không gây hiểu lầm**: `results/qualitative.json` (được
> `notebooks/05_evaluate_and_verdict.py` ghi) chỉ lưu output của fine-tune (c) cho từng
> ticket, KHÔNG lưu output của baseline (b) cho từng item riêng lẻ (chỉ có điểm tổng hợp
> trong `baselines_frozen.json`: target=0.765 trên 50 item). Vì vậy cột "(b) prompt" dưới
> đây là số liệu tổng hợp, không phải output nguyên văn per-item của (b).

| # | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune (output thật) | Nhận xét |
|---|---|---|---|---|
| 1 | "...đặt chuột không dây... Cho tôi trả lại. Gấp..." | `doi_tra / cao / chuột không dây / tich_cuc` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_c...` | ✅ FT thắng — khớp cả 4 khoá |
| 2 | "...đặt đèn bàn LED... Vỡ khi nhận. Gấp..." | `san_pham_loi / cao / đèn bàn LED / trung_tinh` | `{"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung...` | ✅ FT thắng |
| 3 | "...đặt bình giữ nhiệt... Chưa thấy tiền. Khi nào tiện..." | `hoan_tien / **thap** / bình giữ nhiệt / tich_cuc` | `{"intent": "hoan_tien", "urgency": "**trung_binh**", "product": "bình giữ nhiệt", "sentiment":` | ❌ **FT thua** — sai `urgency`: thật là `thap`, đoán `trung_binh` |
| 4 | "...đặt nồi chiên không dầu... Hoàn tiền. Khi nào tiện. Quá tệ." | `hoan_tien / **thap** / nồi chiên không dầu / tieu_cuc` | `{"intent": "hoan_tien", "urgency": "**trung_binh**", "product": "nồi chiên không dầu", "sentim` | ❌ **FT thua** — cùng lỗi `urgency` |
| 5 | "...đặt đèn bàn LED... Sai màu. Khi nào tiện. Shop hỗ trợ tốt." | `san_pham_loi / **thap** / đèn bàn LED / tich_cuc` | `{"intent": "san_pham_loi", "urgency": "**trung_binh**", "product": "đèn bàn LED", "sentiment":` | ❌ **FT thua** — cùng lỗi `urgency` |

**Có mẫu chung nào ở các ca FT thua không?** **Có, rất rõ** — toàn bộ 7/50 item bị điểm
dưới 1.0 trong `results/qualitative.json` đều sai đúng một kiểu: nhãn thật là
`urgency=thap` ("khi nào tiện" — không hối) nhưng fine-tune luôn đoán `urgency=trung_binh`.
Không có item nào sai `intent`, `product`, hay `sentiment`. Đây không phải lỗi ngẫu
nhiên mà là một **thiên lệch hệ thống**: model có vẻ hiếm khi dự đoán nhãn `thap`, có thể
do phân bố nhãn `urgency` trong 225 mẫu train nghiêng về `trung_binh`/`cao`.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Chưa nên deploy bản fine-tune này ở dạng hiện tại: cổng hồi quy
FAILED vì quên thảm hoạ (-0.080 regression), và một thiên lệch nhãn `urgency=thap` hệ
thống sẽ khiến ticket cấp thấp bị định tuyến sai mức độ ưu tiên trong thực tế. Nhưng đây
không phải bằng chứng "fine-tune vô dụng" — trên đúng tác vụ nó được huấn luyện, nó thắng
đậm cả baseline đã prompt kỹ (+0.205 target, format hoàn hảo, latency thấp hơn). Vấn đề
nằm ở thiết kế thí nghiệm, có hướng sửa cụ thể: (1) trộn 1–5% dữ liệu phổ thông để chống
quên thảm hoạ, (2) cân bằng lại phân bố nhãn `urgency` trong tập train để giảm thiên lệch
`thap`. Đòn bẩy thật sự trong lab này, xếp theo mức ảnh hưởng đo được: **learning rate**
lớn nhất (`wrong_lr` phá huỷ hoàn toàn kết quả, target 0.97→0.00), **thành phần dữ liệu**
đứng thứ hai (quyết định cả catastrophic forgetting lẫn thiên lệch nhãn), còn **vị trí
gắn adapter** và **rank** — biến số trung tâm mà lab cũ tập trung vào — hoá ra *không* là
đòn bẩy ở ngân sách tham số này: `attn_only` (q,v, r=283) và `correct` (text-linear, r=16)
hoà tuyệt đối trên target. Mask thì đúng ngay từ đầu (NB1 xanh cả hai assert) nên không
phải nguyên nhân của bất kỳ vấn đề nào quan sát được.

**Ba điều tôi học được** (cụ thể, không generic):
1. Train loss thấp hơn không đồng nghĩa mô hình tốt hơn trên tác vụ thật —
   `attn_only` có loss thấp hơn `correct` (0.5364 vs 0.6268) nhưng target bằng nhau
   tuyệt đối; nếu lab chấm bằng loss thay vì target ở NB5, kết luận về "đòn bẩy" sẽ khác.
2. Một thiên lệch hệ thống (7/7 ca lỗi đều sai đúng một nhãn `urgency=thap`) dễ bị che
   khuất nếu chỉ nhìn điểm trung bình target=0.97 — phải đọc từng ví dụ sai mới thấy
   mẫu hình, không phải mở rộng ngẫu nhiên số liệu tổng.
3. "Đủ tham số" không đồng nghĩa "đặt đúng chỗ" theo giả định trực quan: gắn adapter vào
   toàn bộ 12 lớp linear ở rank thấp và gắn vào 2 lớp attention ở rank rất cao, cùng ngân
   sách tham số, cho kết quả target giống hệt nhau trên bài toán 225 mẫu này — rank đủ
   lớn có thể bù được cho vị trí hẹp, ít nhất ở quy mô dữ liệu này.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 5% dữ liệu phổ thông (regression-style) vào
tập train rồi chạy lại `correct` để xem cổng hồi quy có PASS không, đồng thời kiểm tra lại
phân bố nhãn `urgency` trong 225 mẫu train để xác nhận giả thuyết thiên lệch dữ liệu.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
