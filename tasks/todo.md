# Todo: SD-FSProto

Từ `tasks/plan.md`. Thứ tự theo phụ thuộc. `[B]` = chặn việc khác, `[?]` = chờ tác giả trả lời.

## Giai đoạn 1 — Chuẩn bị (làm được ngay, không cần sandbox)

- [ ] **T1. Config CPU-light + Dataset A**
  - Acceptance: `configs/cpu_light.yaml` (embed 128, d_model 64, bytes 8K, episodes 100,
    epochs 20, patience 5) và `configs/dataset_a.yaml`; smoke test vẫn đạt acc > 0,75
  - Verify: `train.py --config default --config cpu_light --config dummy_smoke`
  - Files: `configs/cpu_light.yaml`, `configs/dataset_a.yaml`

- [ ] **T2. Chế độ `--static-only` cho extractor** `[B]`
  - Acceptance: trích xuất được khi không có `--report-dir`; phần dynamic điền rỗng hợp lệ
    (graph 1 node, dyn_rel = 0, cờ `has_dynamic=false` trong index)
  - Verify: chạy trên 20 PE tự dựng + kiểm tra Dataset load được, model forward không lỗi
  - Files: `scripts/extract_features.py`, `src/sdfsproto/data/dataset.py`

- [ ] **T3. Chế độ ít-family cho Dataset A**
  - Acceptance: Dataset A (5 family) train được ở chế độ classification thường + episodic
    subtype-disjoint; không giả vờ là few-shot family-disjoint
  - Verify: chạy trên dummy được cắt còn 5 family, không crash, không rò rỉ family
  - Files: `src/sdfsproto/data/dataset.py`, `src/sdfsproto/engine/trainer.py`

- [ ] **T4. Static extraction toàn bộ Dataset A** `[?]` — chờ đường dẫn dataset (đang ở máy khác)
  - Acceptance: ≥9.500/9.970 file trích xuất thành công; báo cáo phân bố entropy, tỉ lệ packed
  - Verify: `extract_features.py --static-only`; kiểm tra `index.json`, thống kê `static_rel`
  - Files: `data/A_static/`

- [ ] **T5. Pilot static-only trên Dataset A** (phụ thuộc T2, T3, T4)
  - Acceptance: có số macro-F1 6 lớp chỉ dùng static → biết dynamic cần đóng góp bao nhiêu
  - Verify: `train.py --config dataset_a --set model.modality=static`
  - Files: `results/A_static_pilot/`

- [ ] **T6. Thu thập Dataset B** `[?]` — chờ chốt nguồn (khuyến nghị MalwareBazaar)
  - Acceptance: ≥30 family × ≥30 mẫu, khử trùng lặp theo SHA-256, `labels.csv` có timestamp
  - Verify: đếm family/mẫu; kiểm tra không trùng hash với Dataset A
  - Files: `data/B_raw/`, `labels.csv`

## Giai đoạn 2 — Sandbox `[?]` chờ CAPE

- [ ] **T7. Dựng CAPE v2 + 2 VM**
  - Acceptance: chạy được 20 mẫu thử, sinh `report.json` hợp lệ
  - Verify: kiểm tra report có `behavior.processes[].calls[]`

- [ ] **T8. Đối chiếu parser với report thật** `[B]` — **bắt buộc trước T9**
  - Acceptance: 3–5 report đọc tay, xác nhận `api`/`category`/`arguments` khớp parser;
    semantic tag gán đúng trên các hành vi đã biết (run key, injection, HTTP)
  - Verify: chạy `cape_report.extract_cape()` in ra tag, so với nội dung report
  - Files: `src/sdfsproto/extract/cape_report.py`

- [ ] **T9. Chạy sandbox Dataset B** (~40h) → **T10. Dataset A subsample** (~55h)
  - Acceptance: ≥90% mẫu có report hợp lệ; log rõ mẫu lỗi/timeout
  - Verify: đếm report; thống kê phân bố độ dài trace

## Giai đoạn 3 — Few-shot chính (Dataset B)

- [ ] **T11. Train proposed 5w5s, tinh chỉnh trên val**
- [ ] **T12. Ma trận 16 thí nghiệm** (~50–80h nền) → bảng chính + ablation + baseline
- [ ] **T13. Semi-supervised §11** trên checkpoint tốt nhất

## Giai đoạn 4 — Case study (Dataset A) + tổng hợp

- [ ] **T14. Drift §12 theo timestamp thật** `[?]` phụ thuộc Dataset A có timestamp
- [ ] **T15. `scripts/analyze_alpha.py`** — phân phối α theo packed/không packed, trace ngắn/đầy đủ
  - Acceptance: biểu đồ + kiểm định thống kê (Mann-Whitney) cho khác biệt α giữa hai nhóm
  - Đây là bằng chứng trực tiếp nhất cho Claim 3
- [ ] **T16. Per-family F1 + nhận xét định tính**
- [ ] **T17. Gom bảng markdown/LaTeX cho paper**

## Câu hỏi chặn cần tác giả trả lời

1. Dataset A **có timestamp** không? (quyết định T14 / Success Criteria #6)
2. Nguồn Dataset B — dùng MalwareBazaar hay anh có nguồn riêng? (T6)
3. Copy Dataset A về máy này hay chạy extraction ở máy kia? (T4)
4. CAPE dựng ở máy nào, mấy VM? (T7 — con số 95h giả định 2 VM)
