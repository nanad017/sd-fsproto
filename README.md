# SD-FSProto

**Static-Dynamic Few-Shot Prototypical Learning for PE Malware Family Classification.**

Phân loại malware theo family khi mỗi family mới chỉ có K mẫu gán nhãn (1/5-shot),
kết hợp đặc trưng tĩnh của PE file và hành vi động từ CAPE v2 sandbox.
Chi tiết ý tưởng: `Ý tưởng.txt`; intent đã chốt: `docs/intent/sd-fsproto.md`.

## Kiến trúc

```
PE file ──► Static encoder (4 nhánh: MalConv bytes / image CNN / imports+strings / metadata MLP) ──► z_s ─┐
                                                                                                          ├─► Reliability-aware fusion ──► z ──► Multi-prototype few-shot head
CAPE report ──► Dynamic encoder (1 trong 3: api_seq / semantic_seq / behavior_graph GNN) ──► z_d ─────────┘        (α_s·ẑ_s + α_d·ẑ_d + α_sd·z_sd)
```

- **Loss:** `L_cls + λ1·L_con (supcon) + λ2·L_align (static↔dynamic) + λ3·L_sep (đẩy prototype)`
- **Unknown detection:** mẫu xa mọi prototype hơn ngưỡng τ (calibrate trên val) → unknown family
- **Mở rộng:** semi-supervised prototype refinement (§11), drift-aware prototype bank (§12)

## Tài liệu

| File | Nội dung |
|---|---|
| **`docs/RUNBOOK.md`** | **Chạy pipeline khi đã có dữ liệu — 8 bước, có điều kiện kiểm tra từng bước** |
| `docs/handoff.md` | Cách chuẩn bị & bàn giao dữ liệu (labels.csv, report mẫu) |
| `docs/IMPLEMENTATION.md` | Đã triển khai gì, đối chiếu từng mục `Ý tưởng.txt` |
| `docs/spec.md` | Spec thực nghiệm (REV.2) — hybrid 2 dataset, CPU-only |
| `tasks/plan.md`, `tasks/todo.md` | Kế hoạch & việc còn lại |

## Cài đặt

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu   # hoặc bản CUDA trên máy GPU
uv pip install --python .venv/bin/python -e .
# trích xuất data thật cần thêm: uv pip install --python .venv/bin/python pefile
```

## Chạy nhanh (dummy data, CPU)

```bash
python scripts/make_dummy_data.py --root data/dummy          # sinh dataset synthetic
python scripts/train.py --config configs/default.yaml --config configs/dummy_smoke.yaml
python scripts/eval_extensions.py --run runs/dummy_smoke     # semi-supervised + drift
```

## Dữ liệu thật

```bash
python scripts/extract_features.py \
  --pe-dir <thư mục PE> --report-dir <thư mục CAPE report> \
  --labels labels.csv --out data/real
# labels.csv: id,family,timestamp[,labeled]
python scripts/train.py --config configs/default.yaml --set data.root=data/real
```

## Ma trận thí nghiệm (bảng cho paper)

```bash
python scripts/run_experiments.py --config configs/default.yaml [--only 'proposed|fusion']
# → results/latest/results.md (+ results.json)
```

Gồm: 5w1s / 5w5s / 10w5s; ablation modality (static-only, dynamic-only ×3 backend);
ablation fusion (concat / attention / late-vote / reliability); single- vs multi-prototype;
baseline head (Matching / Relation / Siamese trên cùng backbone); classical ML (RF / SVM / GBoost).

## Cấu trúc code

```
src/sdfsproto/
  config.py                YAML config + deep-merge + dotted overrides
  data/schema.py           schema sample .npz + reliability signals
  data/dummy.py            generator dữ liệu synthetic (có packed/evasive/subtype/drift/unlabeled)
  data/dataset.py          Dataset, collate (pad seq + batch graph), EpisodeSampler family-disjoint
  models/static_encoders.py   4 nhánh static + aggregator
  models/dynamic_encoders.py  api_seq / semantic_seq transformer, behavior graph GNN (thuần torch)
  models/fusion.py         concat / attention / late_vote / reliability-aware
  models/fewshot.py        prototype đơn & multi (k-means), distance, unknown threshold
  models/losses.py         supcon / align / prototype separation
  models/sdfsproto.py      model đầy đủ + episode_forward (4 loss)
  engine/trainer.py        episodic train/val/test, calibrate τ, metrics
  engine/metrics.py        acc, macro-F1, family-F1, unknown AUROC, compactness
  baselines/heads.py       Matching / Relation / Siamese heads
  baselines/classical.py   RF / SVM / GBoost theo protocol episodic
  extensions/semi_supervised.py  refine prototype bằng pseudo-label (§11)
  extensions/drift.py      PrototypeBank EMA + spawn sub-prototype (§12)
  extract/pe_static.py     trích xuất static từ PE thật (pefile)
  extract/cape_report.py   parse CAPE v2 report: api/semantic tokens + behavior graph
```

## Lưu ý nghiên cứu

- **Family-disjoint split là bắt buộc** — split được sinh theo family ngay trong index.json.
- Nhãn family nên làm sạch bằng AVClass trước khi đưa vào `labels.csv`.
- Trace ngắn ≠ benign: tín hiệu `dyn_rel` cho phép fusion tự hạ trọng số dynamic.
- Trên máy GPU: cài torch bản CUDA, `device: auto` sẽ tự nhận.
