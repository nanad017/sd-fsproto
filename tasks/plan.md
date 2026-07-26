# Plan: SD-FSProto — từ codebase sang kết quả paper

Nguồn: `docs/spec.md` REV.2. Bốn giai đoạn, có checkpoint xác minh giữa các giai đoạn.

## Đường găng (critical path)

```
CAPE dựng xong ──► trace Dataset B (~40h) ──► few-shot chính ──► bảng chính + ablation
       │                                                              ▲
       └────────► trace Dataset A subsample (~55h) ──► drift + α ─────┘

Song song, KHÔNG phụ thuộc CAPE:
  thu thập Dataset B (tải mẫu + nhãn) ─┐
  static pipeline trên Dataset A ──────┼──► pilot: static-only đã phân biệt được family chưa?
  code CPU-light + dataset_a mode ─────┘
```

Sandbox là nút thắt cổ chai (~95h). Mọi việc không phụ thuộc nó phải chạy song song từ đầu.

## Giai đoạn 1 — Chuẩn bị (không cần sandbox)

Làm được ngay, gỡ rủi ro sớm nhất với chi phí thấp nhất.

- Config `cpu_light.yaml` + `dataset_a.yaml`; thêm chế độ `--static-only` cho extractor
- Chế độ dataset ít-family: episode sampler cho Dataset A không dùng family-disjoint
  (chỉ 5 family) mà dùng **subtype-disjoint** hoặc chuyển sang chế độ classification thường
- Static extraction toàn bộ Dataset A → **pilot quan trọng**: nếu static-only đã đạt F1 rất cao
  trên 5 family thì dynamic khó chứng minh giá trị gia tăng → cần biết sớm
- Thu thập Dataset B: tải ~30 family × ~30 mẫu, chuẩn hoá nhãn, khử trùng lặp theo hash

**Checkpoint 1:** static pipeline chạy sạch trên ≥9.500/9.970 file; có `labels.csv` cho B.

## Giai đoạn 2 — Sandbox (nút thắt, chạy nền dài ngày)

- Dựng CAPE v2, cấu hình 2 VM, kiểm thử với 20 mẫu → đối chiếu parser với report thật
- **Đối chiếu parser là bắt buộc trước khi chạy hàng loạt** — sai field = mất 95h
- Chạy Dataset B trước (đường găng của bảng chính), rồi Dataset A subsample
- Giám sát tỉ lệ trace rỗng/lỗi; mẫu né sandbox vẫn giữ (dyn_rel phản ánh) chứ không loại

**Checkpoint 2:** ≥90% mẫu có report hợp lệ; 3–5 report đối chiếu tay khớp parser.

## Giai đoạn 3 — Thực nghiệm chính (Dataset B)

- Train proposed 5w5s → xác nhận hội tụ, chọn siêu tham số **chỉ trên val**
- Chạy ma trận 16 thí nghiệm tuần tự nền (~50–80h CPU)
- Semi-supervised §11 trên checkpoint tốt nhất

**Checkpoint 3:** bảng chính + ablation + baseline đầy đủ, reproducible cùng seed.

## Giai đoạn 4 — Case study (Dataset A) + tổng hợp

- Drift theo timeline thật; phân tích α packed vs không packed (`analyze_alpha.py` — cần viết)
- Per-family F1 + nhận xét định tính
- Gom tất cả thành bảng markdown/LaTeX

**Checkpoint 4:** đủ 9 Success Criteria trong spec.

## Rủi ro & giảm thiểu

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| Parser CAPE sai field | Mất toàn bộ 95h sandbox | Đối chiếu tay 20 mẫu **trước** khi chạy hàng loạt (GĐ2) |
| Dataset A không có timestamp | Mất Success Criteria #6 (drift) | Hỏi sớm; dự phòng: subtype clustering hoặc bỏ drift khỏi A |
| Static-only đã quá mạnh trên A | Claim 1 yếu trên A | Pilot ở GĐ1 phát hiện sớm; chuyển trọng tâm Claim 1 sang B |
| Nhãn family của B nhiễu | Prototype lệch (§17.2) | Ưu tiên nguồn có nhãn nhất quán; khử trùng lặp; kiểm tra tay mẫu ngoại lai |
| CPU quá chậm cho 16 thí nghiệm | Không kịp bảng ablation | `cpu_light` + early stopping + giảm episodes/epoch; đo lại sau GĐ2 |
| α collapse về 1 modality | Claim 3 không chứng minh được | Đã thấy trên dummy (α_dyn=0,998); dự phòng: entropy regularization trên α |

## Thứ tự & song song

Giai đoạn 1 làm được **ngay bây giờ** và không phụ thuộc gì. Giai đoạn 2 khởi động ngay
khi CAPE dựng xong, chạy nền suốt trong lúc Giai đoạn 1 tiếp diễn. Giai đoạn 3 và 4 độc lập
nhau sau khi có trace tương ứng — có thể xen kẽ.
