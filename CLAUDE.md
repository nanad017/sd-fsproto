# SD-FSProto

Few-shot phân loại họ malware PE, kết hợp đặc trưng tĩnh (PE/EMBER) và động (CAPE v2).
Mục tiêu: **thực nghiệm cho paper**, không phải hệ thống vận hành.

Ý tưởng gốc: `Ý tưởng.txt` · Intent đã chốt: `docs/intent/sd-fsproto.md`

## Trạng thái

Code **đã hoàn tất** (~3.000 dòng, đã smoke-test). Đang chờ dữ liệu thật.
Chi tiết: `docs/IMPLEMENTATION.md` · Cách chạy: `docs/RUNBOOK.md`

## Ràng buộc dễ làm sai — đọc kỹ

1. **Dataset A chỉ có 5 family → KHÔNG phải few-shot.**
   Split của nó là sample-disjoint (`split_mode: sample`), mọi family đều có mặt ở
   train/val/test. Số từ Dataset A **không bao giờ** được gọi là kết quả few-shot.
   Few-shot cần ≥25 family với split family-disjoint (Dataset B, hiện là optional).

2. **Không tuning trên test.** Mọi siêu tham số chọn trên val. Split train/test của
   Dataset A do chủ dataset ấn định — không chia lại. Val chỉ cắt ra từ train.

3. **Không dùng `cape_verdict`** làm feature. Đó là kết luận của một detector khác;
   dùng nó = học lại đầu ra detector, độ chính xác ảo.

4. **Lọc artifact sandbox** khỏi mọi feature dạng chuỗi (`wevtutil`, `SMaster*`,
   thư mục analyzer ngẫu nhiên, `192.168.122.1` của INetSim...). Đã cài trong
   `extract/cape_features.py::ARTIFACT_PATTERNS`.

5. **Giữ mẫu `ran_ok=False`.** Trace rỗng không có nghĩa là vô dụng — `dyn_rel[7]=0`
   chính là tín hiệu cho reliability fusion (Claim 3).

6. **Không commit mẫu malware hoặc dataset** vào repo. Không chạy file PE ngoài sandbox.

## Giới hạn đã biết (nêu trong paper, đừng "sửa" bằng cách giấu)

- **Backend 2 chạy suy giảm**: `api.sequence` của dataset không có arguments per-call.
  Tag suy từ tên API (INJECT/CRYPTO/ANTI) vẫn đúng; tag phụ thuộc tham số
  (RUN_KEY/SHELL/HTTP) chỉ thành context token đầu chuỗi. `meta.json` ghi
  `degraded_semantics: true`.
- **α có thể collapse** về một modality — đã thấy trên dummy. `analyze_alpha.py` tự
  cảnh báo. Nếu xảy ra trên data thật, Claim 3 không được chứng minh; cân nhắc
  entropy regularization trên α.

## Môi trường

- `.venv` (Python 3.12 qua uv), PyTorch CPU. **Train cũng chạy CPU** — dùng
  `configs/cpu_light.yaml` (đo được ~4,8 phút/epoch).
- Luôn gọi `.venv/bin/python`, không phải `python` hệ thống.
- Dữ liệu thật nằm ở máy khác, không có trên máy này. Trỏ đường dẫn qua
  `--set data.root=...` hoặc tham số của script build.

## Lệnh hay dùng

```bash
# smoke test (dummy data, luôn chạy được)
.venv/bin/python scripts/train.py --config configs/default.yaml --config configs/dummy_smoke.yaml

# dựng dataset thật từ cape_features + EMBER
.venv/bin/python scripts/build_dataset_a.py --cape-root <...> \
  --ember ember.npy --ember-ids ember_ids.npy --pe-dir <...> \
  --out data/A --exclude-family Benign

# train trên data thật
.venv/bin/python scripts/train.py --config configs/default.yaml \
  --config configs/cpu_light.yaml --config configs/ember.yaml \
  --config configs/dataset_a.yaml --set data.root=data/A

# ma trận thí nghiệm | extensions | bằng chứng Claim 3
.venv/bin/python scripts/run_experiments.py --config ... --out results/v1
.venv/bin/python scripts/eval_extensions.py --run <run_dir>
.venv/bin/python scripts/analyze_alpha.py --run <run_dir>
```

## Quy ước code

- Mọi hyperparameter qua YAML + `--set dotted.key=value`, không hardcode.
- Type hints đầy đủ; docstring tiếng Anh nêu công thức/shape; comment shape inline `# [B, T]`.
- Token `0` = padding ở mọi sequence.
- Extractor không được crash cả batch vì một file hỏng — degrade và phản ánh vào reliability signal.
- Ablation bật/tắt qua config, không sửa code: `model.static.branches`, `model.dynamic.backend`,
  `model.fusion.kind`, `model.modality`, `model.fewshot.multi_proto`.

## Trả lời bằng tiếng Việt.
