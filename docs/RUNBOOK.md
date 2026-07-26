# Runbook — chạy pipeline khi đã có dữ liệu

Chạy theo đúng thứ tự. Mỗi bước có **điều kiện kiểm tra** trước khi sang bước sau —
đừng bỏ qua, vì bước sau tốn hàng chục giờ.

Mọi lệnh chạy từ thư mục gốc `~/Ansolo/model`, dùng `.venv/bin/python`.

---

## Chọn đường vào theo nguồn dữ liệu

Mỗi nguồn có một script dựng dataset riêng, nhưng **tất cả đổ về cùng một schema
`.npz`** nên các bước train phía sau giống hệt nhau.

### A. Avast-CTU (reduced reports) — nhanh nhất để thử

```bash
.venv/bin/python scripts/build_avast_ctu.py \
  --reports <thư mục reduced>/ --labels <public_labels.csv> \
  --out data/avast --limit 2000          # bỏ --limit khi chạy thật

.venv/bin/python scripts/train.py \
  --config configs/default.yaml --config configs/cpu_light.yaml \
  --config configs/avast_ctu.yaml --set data.root=data/avast
```

Chia train/test **theo thời gian** (mốc 2019-08-01) do script tự áp đặt.
Mốc so sánh của bài báo gốc: **94,5%** (static+dynamic) / **~63%** (chỉ static).
Ra gần 100% là có rò rỉ. Đặc tả bộ này: `docs/avast-ctu-spec.md`.

### B. CAPE features đã trích sẵn + EMBER `.npy`

```bash
.venv/bin/python scripts/build_dataset_a.py \
  --cape-root <thư mục có raw/<split>/<family>/*.json> \
  --ember ember.npy --ember-ids ember_ids.npy \
  --pe-dir <thư mục PE> \
  --raw-report-dir /opt/CAPEv2/storage/analyses \   # nếu raw report còn
  --out data/A --exclude-family Benign

.venv/bin/python scripts/train.py \
  --config configs/default.yaml --config configs/cpu_light.yaml \
  --config configs/ember.yaml --config configs/dataset_a.yaml \
  --set data.root=data/A
```

`--raw-report-dir` là tuỳ chọn nhưng **nên có**: raw report giữ tham số từng lời gọi,
thiếu nó thì backend `semantic_seq` chạy ở chế độ suy giảm.
EMBER `.npy` dạng ma trận thuần thì **bắt buộc** kèm `--ember-ids`.

### C. File PE thô + CAPE report thô

Đây là đường đi đầy đủ nhất, mô tả chi tiết ở **Bước 0–8** bên dưới.

```bash
.venv/bin/python scripts/extract_features.py \
  --pe-dir <PE> --report-dir <reports> --labels labels.csv --out data/real
```

### D. Dữ liệu giả — kiểm tra pipeline, không cần dữ liệu thật

```bash
.venv/bin/python scripts/make_dummy_data.py --root data/dummy --families 30
.venv/bin/python scripts/train.py \
  --config configs/default.yaml --config configs/dummy_smoke.yaml
```

Luôn chạy được. Dùng để xác nhận môi trường ổn trước khi đụng dữ liệu thật.

---

## Sau khi có dataset — các bước chung cho mọi nguồn

```bash
.venv/bin/python scripts/run_experiments.py --config ... --out results/v1  # ma trận thí nghiệm
.venv/bin/python scripts/eval_extensions.py --run <run_dir>                # §11 + §12
.venv/bin/python scripts/analyze_alpha.py  --run <run_dir>                 # bằng chứng Claim 3
```

Chi tiết ở Bước 6–8.

---

## Tổng quan thời gian (đường C — đầy đủ nhất)

| Bước | Việc | Thời gian |
|---|---|---|
| 0 | Chuẩn bị `labels.csv` | 10 phút |
| 1 | Trích xuất static | 2–4 giờ (Dataset A 9.970 mẫu) |
| 2 | Pilot static-only | 30 phút |
| 3 | **Đối chiếu parser CAPE** | 15 phút — **bắt buộc** |
| 4 | Chạy sandbox | ~95 giờ (chạy nền nhiều ngày) |
| 5 | Trích xuất đầy đủ | 3–5 giờ |
| 6 | Train + ma trận thí nghiệm | 50–80 giờ (chạy nền) |
| 7 | Extensions + phân tích α | 2 giờ |
| 8 | Gom bảng cho paper | 30 phút |

Bước 1–3 làm được **ngay bây giờ** nếu có file PE, không cần chờ sandbox.

---

## Bước 0 — Chuẩn bị `labels.csv`

Bốn cột, đúng tên header:

```csv
id,family,timestamp,labeled
a3f2b1c...,winwebsec,1698796800,1
d7e4a09...,zbot,1701388800,1
9c1f5b2...,benign,1699401600,1
```

- `id` — **đúng bằng tên file PE** (pipeline tìm theo `<pe-dir>/<id>`)
- `timestamp` — unix epoch; không có thì để `0` (mất thí nghiệm drift)
- `labeled` — để `0` nếu muốn đưa mẫu vào pool unlabeled cho semi-supervised

Nếu family đang nằm ở tên thư mục con:

```bash
.venv/bin/python -c "
import csv, pathlib
root = pathlib.Path('/duong/dan/dataset')
rows = [{'id': f.name, 'family': f.parent.name.lower(), 'timestamp': 0, 'labeled': 1}
        for f in root.rglob('*') if f.is_file()]
w = csv.DictWriter(open('labels_a.csv','w',newline=''),
                   fieldnames=['id','family','timestamp','labeled'])
w.writeheader(); w.writerows(rows)
print(len(rows), 'dòng')
"
```

**Kiểm tra:** số dòng khớp số file PE; `cut -d, -f2 labels_a.csv | sort | uniq -c` ra đúng các family.

---

## Bước 1 — Trích xuất static (không cần sandbox)

> **Nếu đã có vector EMBER**: thêm `--config configs/ember.yaml` vào mọi lệnh train ở
> bước 2 và 6, và đảm bảo `.npz` có mảng `ember` (xem adapter ở bước 5). EMBER đã bao trùm
> `metadata`/`imports`/`strings` nên `configs/ember.yaml` chỉ bật `[ember, bytes, image]`.
> Nếu dùng EMBER2018 (2.381 chiều) thay vì 2024: `--set model.static.ember.in_dim=2381`.


```bash
.venv/bin/python scripts/extract_features.py --static-only --split-mode sample \
  --pe-dir /duong/dan/dataset \
  --labels labels_a.csv \
  --out data/A_static
```

`--split-mode sample` cho **Dataset A** (5 family). Với **Dataset B** (~30 family) thì bỏ
cờ này đi để dùng family-disjoint mặc định.

Script tự in tỉ lệ thành công và số mẫu mỗi family ở cuối.

**Điều kiện qua bước:** success rate ≥ 95%. Nếu thấp hơn, xem `errors` in ra — thường là file
không phải PE hợp lệ hoặc tên trong CSV lệch tên file thật.

---

## Bước 2 — Pilot static-only

Câu hỏi cần trả lời: **chỉ static thôi đã phân biệt được family tới đâu?** Nếu đã rất cao thì
dynamic khó chứng minh giá trị gia tăng — biết sớm để điều chỉnh trọng tâm paper.

```bash
.venv/bin/python scripts/train.py \
  --config configs/default.yaml \
  --config configs/cpu_light.yaml \
  --config configs/dataset_a.yaml \
  --set data.root=data/A_static \
  --set model.modality=static \
  --set run_dir=runs/A_static_pilot
```

**Đọc kết quả:** macro-F1 in ra cuối. Nếu > 0,95 → static đã gần bão hoà, cần nói rõ trong paper
rằng đóng góp của dynamic nằm ở mẫu packed/obfuscated chứ không phải toàn bộ.

---

## Bước 3 — Đối chiếu parser CAPE ⚠️ BẮT BUỘC

**Làm bước này với 1 report đầu tiên, TRƯỚC khi chạy sandbox hàng loạt.**
Sai một field name = 95 giờ sandbox ra dữ liệu rác.

```bash
.venv/bin/python -c "
import json, sys
r = json.load(open(sys.argv[1]))
print('top keys:', list(r.keys()))
b = r.get('behavior', {})
print('behavior keys:', list(b.keys()))
p = (b.get('processes') or [{}])[0]
print('process keys:', list(p.keys()))
print('n_calls:', len(p.get('calls') or []))
print(json.dumps((p.get('calls') or [])[:3], indent=2)[:1500])
" /duong/dan/report.json
```

Rồi chạy parser lên chính report đó:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from sdfsproto.extract.cape_report import extract_cape, N_TAGS
d = extract_cape(sys.argv[1])
print('n_api:', len(d['api_ids']), '| n_nodes:', len(d['node_type']), '| n_edges:', len(d['edge_src']))
print('semantic tags:', sorted(set((d['sem_ids'] % N_TAGS).tolist())))
print('dyn_rel:', d['dyn_rel'].round(3).tolist())
" /duong/dan/report.json
```

**Điều kiện qua bước:**
- `n_api` > 0 và xấp xỉ số call thật trong report
- `dyn_rel` không toàn số 0
- Semantic tag có nhiều hơn một giá trị (nếu chỉ ra `[0]` = tất cả đều `OTHER` → luật gán tag
  không khớp, phải sửa)

**Nếu lệch:** gửi tôi output của cả hai lệnh trên, tôi sửa `extract/cape_report.py`.

---

## Bước 4 — Chạy sandbox

Không có lệnh của pipeline ở bước này. Lưu ý khi cấu hình:

- Đặt report ra `<report-dir>/<id>.json` (hoặc `<report-dir>/<id>/report.json` — parser đọc được cả hai)
- **Mẫu né sandbox vẫn giữ lại**, đừng loại. Trace ngắn được `dyn_rel` ghi nhận và fusion tự hạ trọng số — đó là dữ liệu cho Claim 3
- Ưu tiên chạy **Dataset B trước** (đường găng của bảng chính), Dataset A subsample sau

---

## Bước 5 — Trích xuất đầy đủ

```bash
# Dataset A (5 family)
.venv/bin/python scripts/extract_features.py --split-mode sample \
  --pe-dir /duong/dan/dataset_A --report-dir /duong/dan/reports_A \
  --labels labels_a.csv --out data/A

# Dataset B (~30 family) — family-disjoint, KHÔNG dùng --split-mode
.venv/bin/python scripts/extract_features.py \
  --pe-dir /duong/dan/dataset_B --report-dir /duong/dan/reports_B \
  --labels labels_b.csv --out data/B
```

**Điều kiện qua bước:** với B, script phải in ra ≥ 25 family và mỗi family ≥ 15 mẫu. Nếu ít hơn,
episode sampler sẽ báo lỗi ở bước sau.

---

## Bước 6 — Train + ma trận thí nghiệm

### 6a. Smoke test trước khi chạy job dài

Bắt buộc — 5 phút này tiết kiệm hàng chục giờ:

```bash
.venv/bin/python scripts/train.py \
  --config configs/default.yaml --config configs/cpu_light.yaml \
  --set data.root=data/B --set train.epochs=1 \
  --set data.episodes_per_epoch=10 --set data.val_episodes=5 \
  --set data.test_episodes=10 --set run_dir=runs/smoke_B
```

Chạy trót lọt = đường ống thông. Đừng nhìn accuracy ở đây.

### 6b. Model chính (Dataset B)

```bash
.venv/bin/python scripts/train.py \
  --config configs/default.yaml --config configs/cpu_light.yaml \
  --set data.root=data/B --set run_dir=runs/B_proposed
```

### 6c. Toàn bộ 16 thí nghiệm (chạy nền, 50–80 giờ)

```bash
nohup .venv/bin/python scripts/run_experiments.py \
  --config configs/default.yaml --config configs/cpu_light.yaml \
  --set data.root=data/B --out results/B_v1 \
  > results_B.log 2>&1 &

tail -f results_B.log        # theo dõi
```

Chạy trước một nhóm nhỏ để ước lượng thời gian thật:

```bash
.venv/bin/python scripts/run_experiments.py \
  --config configs/default.yaml --config configs/cpu_light.yaml \
  --set data.root=data/B --only 'proposed_5w5s|fusion_concat' \
  --skip-classical --out results/B_probe
```

Kết quả ghi dần vào `results/B_v1/results.json` sau mỗi thí nghiệm — mất điện giữa chừng không mất hết.

### 6d. Dataset A

```bash
.venv/bin/python scripts/train.py \
  --config configs/default.yaml --config configs/cpu_light.yaml \
  --config configs/dataset_a.yaml --set data.root=data/A
```

---

## Bước 7 — Extensions + phân tích α

```bash
# Semi-supervised (§11) + drift (§12) — trên Dataset A vì cần timestamp
.venv/bin/python scripts/eval_extensions.py --run runs/dataset_a --episodes 100

# Bằng chứng Claim 3 — chạy trên CẢ HAI dataset
.venv/bin/python scripts/analyze_alpha.py --run runs/dataset_a
.venv/bin/python scripts/analyze_alpha.py --run runs/B_proposed
```

**Đọc kết quả α:** mở `runs/*/alpha_analysis.md`.

- Nếu có dòng **CẢNH BÁO α collapse** → Claim 3 chưa chứng minh được. Cách xử lý: thêm entropy
  regularization lên α (báo tôi, tôi cài), hoặc báo cáo trung thực rằng fusion nghiêng hẳn về một
  modality trên dataset này
- Bảng 4 giả thuyết: cần **cả** `p < 0,05` **và** `|Cliff's d| ≥ 0,15` mới tính là ủng hộ

---

## Bước 8 — Gom bảng cho paper

```bash
cat results/B_v1/results.md              # bảng chính + ablation + baseline
cat runs/dataset_a/alpha_analysis.md     # bằng chứng Claim 3
cat runs/dataset_a/extensions.json       # semi-supervised + drift
cat runs/B_proposed/metrics.json         # có family_f1 chi tiết
```

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `only N families with >= K labeled samples` | Không đủ mẫu/family cho `k_shot + n_query` | Giảm `--set data.n_query=5`, hoặc gom thêm mẫu |
| `split_mode='sample' has no held-out families` | Đặt `n_unknown > 0` ở chế độ sample | `--set data.n_unknown=0` |
| `alpha analysis needs 'reliability'` | Run dùng fusion baseline | Train lại với fusion mặc định |
| success rate thấp ở bước 1 | Tên trong CSV lệch tên file | Đối chiếu `ls <pe-dir> \| head` với `cut -d, -f1 labels.csv \| head` |
| Semantic tag chỉ ra `[0]` | Luật gán tag không khớp report | Gửi tôi output bước 3 |
| Train quá chậm | Model còn lớn so với CPU | `--set model.static.branches='[image,imports,strings,metadata]'` (bỏ nhánh bytes, nhanh ~30%) |
| Máy hết RAM | Batch/episode quá lớn | Giảm `data.n_query`, `data.k_shot` |

---

## Checklist trước khi viết paper

- [ ] Dataset B ≥ 25 family, split **family-disjoint** (không phải sample)
- [ ] Số của Dataset A **không** được báo cáo như few-shot (chế độ sample-disjoint)
- [ ] Mọi siêu tham số chọn trên **val**, không chạm test
- [ ] Bảng chính có CI95 theo episode
- [ ] Chạy lại cùng seed ra đúng số cũ
- [ ] `alpha_analysis.md` không có cảnh báo collapse — hoặc nếu có thì đã nêu thẳng trong paper
- [ ] Ablation dùng **cùng seed, cùng episode set** với model chính
