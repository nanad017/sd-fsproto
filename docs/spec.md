# Spec: SD-FSProto — Thực nghiệm & số liệu cho paper

> Trạng thái: REV.2 (2026-07-26) — chờ tác giả duyệt.
> Rev.1 giả định 1 dataset nhiều family + máy GPU. Cả hai đều sai; rev.2 sửa theo
> thực tế: **hybrid 2 dataset**, **CPU-only**, **chưa có CAPE report nào**.
> Intent gốc: `docs/intent/sd-fsproto.md`. Codebase lõi đã xong & smoke-test.

## Objective

Sinh bộ kết quả thực nghiệm reproducible cho paper SD-FSProto, dùng **hai dataset
bổ trợ nhau** vì không dataset nào một mình đáp ứng được cả 4 claim:

| | **Dataset A** (đang có) | **Dataset B** (cần thu thập) |
|---|---|---|
| Quy mô | 5 family + benign, 9.970 mẫu | ~30 family × ~30 mẫu ≈ 900 mẫu |
| Đặc điểm | ít lớp, rất nhiều mẫu/lớp | nhiều lớp, ít mẫu/lớp |
| Vai trò | drift theo timeline, phân tích α (packed vs không), per-family deep-dive, pilot static | **few-shot chính**: 5w1s / 5w5s / 10w5s, toàn bộ ablation + baseline |
| Claim phục vụ | 1, 3, 4 (multi-proto qua subtype), §12 | 1, 2, 3, 4, §11 |

Lý do tách: few-shot yêu cầu `C_base ∩ C_novel = ∅` với ≥20 base + ≥5 novel family.
Dataset A có 5 family → không thể; nhưng lại là dataset **duy nhất** đủ mẫu/family
để mô phỏng drift theo thời gian và phân tích thống kê α. Dùng cả hai là mạnh nhất.

## Ràng buộc phần cứng (đã đo, không phải ước lượng)

Train **CPU-only** (14 threads). Đo trên dummy data, 5-way 5-shot, 200 episode/epoch:

| Cấu hình | Params | ms/episode | phút/epoch |
|---|---|---|---|
| default (embed 256, bytes 16K) | 3,48M | 2200 | 7,3 |
| bỏ nhánh bytes | 2,33M | 1595 | 5,3 |
| **light (embed 128, d_model 64, bytes 8K)** | 2,34M | 1442 | **4,8** |
| light + behavior_graph | 2,79M | 612 | 2,0 |

I/O chỉ chiếm ~100ms/episode → tối ưu phải nhắm vào kích thước model, không phải cache.
**Hệ quả:** dùng cấu hình `light` làm mặc định cho data thật; ma trận 16 thí nghiệm
× ~20 epoch ≈ 27h trên dummy, ước ~50–80h trên data thật (sequence/graph dài hơn).
Chạy nền tuần tự, không kỳ vọng lặp nhanh.

## Chi phí sandbox (chi phối toàn bộ tiến độ)

CAPE ~4–5 phút/mẫu/VM (200s timeout + restore). Với 2 VM song song:

| Phạm vi | Số mẫu | Thời gian |
|---|---|---|
| Toàn bộ Dataset A | 9.970 | ~370h (15 ngày) — **không làm** |
| Dataset A subsample (drift + α) | ~1.500 | ~55h |
| Dataset B (few-shot) | ~900–1.000 | ~40h |
| **Tổng kế hoạch** | **~2.500** | **~95h** |

Subsample A: stratified theo family, ưu tiên mẫu có timestamp trải đều (cho drift).

## Tech Stack

Python 3.12 (uv venv `.venv`), PyTorch 2.13 CPU, numpy, scikit-learn, pyyaml, `pefile`.
Behavior-graph GNN viết thuần torch — không phụ thuộc torch-geometric.
Sandbox: CAPE v2 (**chưa cài xong** — `capev2/` mới chỉ có `config.md`).

## Commands

```bash
# --- Static pipeline (không cần sandbox, chạy được ngay khi có file PE) ---
.venv/bin/python scripts/extract_features.py --static-only \
  --pe-dir <PE> --labels labels.csv --out data/A_static

# --- Trích xuất đầy đủ (sau khi có CAPE report) ---
.venv/bin/python scripts/extract_features.py \
  --pe-dir <PE> --report-dir <reports> --labels labels.csv --out data/B

# --- Few-shot chính trên Dataset B ---
.venv/bin/python scripts/run_experiments.py --config configs/default.yaml \
  --config configs/cpu_light.yaml --set data.root=data/B --out results/B_v1

# --- Dataset A: drift + phân tích α ---
.venv/bin/python scripts/train.py --config configs/default.yaml \
  --config configs/cpu_light.yaml --config configs/dataset_a.yaml
.venv/bin/python scripts/eval_extensions.py --run runs/dataset_a
.venv/bin/python scripts/analyze_alpha.py --run runs/dataset_a   # CẦN VIẾT
```

## Project Structure

```
configs/     default.yaml | cpu_light.yaml (mới) | dataset_a.yaml (mới) | dummy_smoke.yaml
src/sdfsproto/
  data/      schema, dummy generator, Dataset + EpisodeSampler (family-disjoint)
  models/    static 4 nhánh | dynamic 3 backend | fusion 4 kiểu | fewshot | losses
  engine/    Trainer episodic, metrics
  baselines/ Matching/Relation/Siamese; RF/SVM/GBoost
  extensions/ semi_supervised (§11), drift (§12)
  extract/   pe_static, cape_report
scripts/     train | run_experiments | eval_extensions | extract_features | analyze_alpha(mới)
data/        dummy/ | A_static/ | A/ | B/     (KHÔNG commit)
runs/ results/ docs/ tasks/
```

## Code Style

Khớp codebase hiện có: type hints đầy đủ, docstring Anh một dòng nêu công thức/shape,
hyperparameter luôn qua YAML + dotted override, shape ghi chú inline (`# [B, T]`),
token 0 = padding, extractor không bao giờ crash cả batch vì một file hỏng.

```python
def class_distances(query, protos, proto_class, n_way, metric="euclidean") -> torch.Tensor:
    """d(x, c) = min_j d(f(x), p_c^j). Returns [Q, n_way]."""
```

## Testing Strategy

1. **Smoke cấu hình** — 14 cấu hình (3 backend × 4 fusion + 2 modality) forward/backward hữu hạn
2. **Smoke e2e** — dummy_smoke 5 epoch: test acc > 0,75; unknown AUROC > 0,8
3. **Extractor** — tỉ lệ lỗi < 5% trên data thật; đối chiếu tay 3–5 CAPE report với parser
4. **Determinism** — eval seed cố định (777/999); hai lần chạy cùng checkpoint ra cùng số

## Boundaries

- **Always:** family-disjoint split cho mọi số few-shot; smoke test trước khi chạy job dài;
  bảng sinh từ `results/*.json`; báo CI95 theo episode; lưu config+seed theo run
- **Ask first:** đổi schema `.npz`; thêm dependency; đổi protocol đánh giá; ghi đè `runs/`, `results/` đã có số
- **Never:** commit mẫu malware / dataset vào repo; chạy PE ngoài sandbox; để cùng family
  ở cả train và test; tuning theo test set

## Xử lý lớp Benign

Benign **không phải** một family → không vào episode few-shot. Dùng cho:
- Nguồn distractor bổ sung khi hiệu chỉnh ngưỡng unknown τ
- Pilot nhị phân benign/malware để sanity-check chất lượng đặc trưng (không vào bảng chính)
- Phân tích α: benign thường không packed → nhóm đối chứng tự nhiên cho Claim 3

## Success Criteria

**Dataset B (few-shot — bảng chính):**
1. `data/B` build xong: ≥30 family, ≥95% mẫu trích xuất thành công, split family-disjoint
2. Bảng chính 5w1s / 5w5s / 10w5s: accuracy±CI95, macro-F1, unknown AUROC, ms/query
3. Ablation đủ 8 dòng (static-only, dynamic-only ×3 backend, concat, attention, late-vote, single-proto) — cùng seed, cùng episode set
4. Baseline: Matching / Relation / Siamese (cùng backbone) + RF / SVM / GBoost (cùng protocol)
5. Semi-supervised §11: Δaccuracy theo mức contamination của pool unlabeled

**Dataset A (case study):**
6. Drift §12 theo timeline thật: frozen vs adaptive, số sub-prototype sinh ra theo thời gian
7. **Phân tích α**: phân phối α_static/α_dynamic trên nhóm packed vs không packed, và trên
   nhóm trace ngắn vs đầy đủ — bằng chứng định lượng trực tiếp cho Claim 3
8. Per-family F1 cho 5 family, kèm nhận xét định tính (ví dụ Zeroaccess rootkit vs Locker ransomware)

**Chung:**
9. Reproducible: chạy lại cùng seed ra đúng số cũ

*Success = pipeline sinh đủ bảng đúng chuẩn. "Proposed thắng baseline" là kết quả khoa học,
không phải điều kiện hoàn thành — thua ở đâu vẫn báo cáo trung thực ở đó.*

## Open Questions

1. **Dataset B lấy từ nguồn nào?** Khuyến nghị **MalwareBazaar** (abuse.ch): API miễn phí,
   mẫu có nhãn family, tải được binary thật. Lưu ý: **BODMAS không dùng được** cho hướng này
   vì chỉ phát hành feature vector, không có binary → không chạy sandbox được.
   VirusShare cũng khả thi nhưng cần AVClass2 để gán nhãn.
2. **Dataset A có timestamp không?** Thí nghiệm drift (§12, Success Criteria #6) phụ thuộc
   hoàn toàn vào cái này. Nếu không có, drift phải chuyển sang mô phỏng bằng subtype clustering
   (yếu hơn) hoặc bỏ khỏi paper.
3. **CAPE**: dựng ở máy nào, mấy VM song song? Con số 95h ở trên giả định 2 VM.
4. Copy dataset A về máy này để chạy static pipeline, hay chạy extraction ở máy kia?
