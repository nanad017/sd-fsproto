# Báo cáo triển khai SD-FSProto

Cập nhật: 2026-07-26 · Codebase: 2.802 dòng Python / 39 file
Nguồn đối chiếu: `Ý tưởng.txt` (17 mục) · Intent đã chốt: `docs/intent/sd-fsproto.md`

---

## Phần I — Tổng quan trạng thái

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Toàn bộ kiến trúc model (Module 1–5) | ✅ Xong, đã smoke-test | 14 cấu hình forward/backward hữu hạn |
| Huấn luyện episodic + 4 loss | ✅ Xong | Hội tụ trên dummy, acc 83,8% |
| Unknown detection (ngưỡng τ) | ✅ Xong | AUROC 0,86 trên dummy; τ hiệu chỉnh bằng Youden J |
| Mở rộng §11 semi-supervised | ✅ Xong, đã kiểm chứng | Δacc +0,9% trên dummy |
| Mở rộng §12 concept drift | ✅ Xong, chưa chứng minh lợi ích | Drift synthetic quá nhẹ, cần data thật |
| Baseline (§16) | ✅ Xong | 3 few-shot head + 3 classical ML |
| Ma trận thí nghiệm (§15) | ✅ Xong | 16 thí nghiệm, sinh bảng markdown |
| Trích xuất PE tĩnh | ✅ Xong, test với PE tự dựng | Chưa chạy trên mẫu thật |
| Parser CAPE report | ⚠️ Xong nhưng **chưa đối chiếu report thật** | Rủi ro cao nhất còn lại |
| `analyze_alpha.py` (bằng chứng Claim 3) | ✅ Xong, đã chạy thử | Mann-Whitney + Cliff's d, tự phát hiện α collapse |
| Dataset | ❌ Chưa có | Nút thắt chính |

**Tóm lại:** phần *code* đã hoàn tất — mọi thứ làm được mà không cần dữ liệu đều đã xong.
Còn lại chỉ hai việc, cả hai đều phụ thuộc dữ liệu: đối chiếu parser CAPE với report thật,
và chạy thực nghiệm.

---

## Phần II — Đối chiếu từng mục của `Ý tưởng.txt`

### §1–2. Bài toán & đầu vào

| Ý tưởng | Triển khai | File |
|---|---|---|
| Support set N-way K-shot, query set | `EpisodeSampler` sinh episode N-way K-shot Q-query | `data/dataset.py` |
| Family-disjoint base/novel | Split theo family trong `index.json`, cưỡng chế ở sampler | `data/dataset.py`, `scripts/extract_features.py` |
| Static view: header, section, import, string, entropy, byte | Đủ 6 nhóm, chuẩn hoá thành 32 chiều metadata + token hoá | `extract/pe_static.py` |
| Dynamic view: API, arguments, file/registry/process/network | Parse đủ, kèm chuẩn hoá tham số thành semantic tag | `extract/cape_report.py` |

**Khác biệt so với ý tưởng:** thêm hai vector *reliability signal* (8 chiều mỗi bên) không có
trong bản gốc — đây là đầu vào cho fusion ở §6. Chi tiết từng tín hiệu ghi trong `data/schema.py`.

### §4. Module 1 — Static Encoder

Ý tưởng liệt kê 5 nhánh; **đã chốt bỏ CFG/FCG** (quyết định trong phỏng vấn: cần disassembly
pipeline riêng, hay hỏng trên mẫu packed — đúng loại mẫu reliability fusion nhắm tới).
**Thêm nhánh EMBER** sau khi biết tác giả dùng vector EMBER 2024 (2.568 chiều).

| Nhánh | Mô hình triển khai | Class |
|---|---|---|
| **EMBER vector** | MLP có LayerNorm + signed log1p | `EmberBranch` |
| Byte sequence | MalConv (gated conv, embedding 257 gồm padding) | `MalConvBranch` |
| PE grayscale image | CNN 3 tầng + adaptive pool | `ImageBranch` |
| Import / string | Hashed token → `EmbeddingBag` (order-free) | `BagBranch` |
| PE metadata | MLP có LayerNorm | `MetadataBranch` |
| ~~CFG/FCG~~ | Ngoài phạm vi | — |

**Quan hệ EMBER với các nhánh khác.** EMBER đã bao gồm byte histogram, byte-entropy histogram,
string stats, header, section, imports/exports đã hash → **bao trùm** nhánh `metadata`, `imports`,
`strings`. Nhưng EMBER lưu *histogram*, nên **mất thứ tự byte và bố cục 2-D** — đó chính là phần
`bytes` (MalConv) và `image` đóng góp thêm. Hai cấu hình:

| Có vector EMBER | Không có |
|---|---|
| `branches: [ember, bytes, image]` (`configs/ember.yaml`) | `branches: [bytes, image, imports, strings, metadata]` |

Aggregator nối các nhánh bật → LayerNorm → MLP → `z_s`. Ablation qua `model.static.branches`.
File: `models/static_encoders.py`.

### §5. Module 2 — Dynamic Encoder (cả 3 cách)

| Cách trong ý tưởng | Triển khai | Class |
|---|---|---|
| Cách 1: API sequence | Transformer encoder, masked mean pooling | `SeqTransformer(input_key="api_ids")` |
| Cách 2: API + argument semantic | Cùng kiến trúc, vocab = `api_id × 16 + tag` | `SeqTransformer(input_key="sem_ids")` |
| Cách 3: Behavior graph | Message passing có kiểu + GRUCell update, readout mean+max | `BehaviorGraphEncoder` |

Ba backend chung một interface `forward(batch) -> z_d`, đổi bằng `model.dynamic.backend`.
**GNN viết thuần torch** (`index_add_`, `index_reduce_`) — không phụ thuộc torch-geometric,
giảm rủi ro cài đặt trên máy train. File: `models/dynamic_encoders.py` (111 dòng).

**16 semantic tag đã cài** (§5 cách 2 chỉ nêu ví dụ, đây là bộ đầy đủ):
`RUN_KEY`, `SERVICE_REG`, `REG_WRITE`, `USER_WRITABLE_EXEC`, `TEMP_FILE`, `SYSTEM_DIR_WRITE`,
`SHELL_EXEC`, `PROC_INJECT`, `HTTP`, `IP_DIRECT`, `MUTEX`, `CRYPTO`, `ANTI_ANALYSIS`,
`PERSIST_TASK`, `SERVICE_CREATE`, `OTHER`.

### §6. Module 3 — Static-Dynamic Fusion

Công thức ý tưởng: `z = α_s·z_s + α_d·z_d + α_sd·z_sd`. Triển khai đúng công thức, trong đó:

- `α` = softmax của 3 gate, mỗi gate nhìn **cả nội dung embedding lẫn reliability signal thô**
- `z_sd = tanh(MLP([z_s ⊙ z_d ; z_s ; z_d]))` — số hạng tương tác
- Consistency = cosine(z_s, z_d), đưa vào gate `α_sd`

Bảng tình huống ở §6 được hiện thực **gián tiếp qua học** chứ không hard-code rule: mẫu packed
có `static_rel[5]=1`, entropy cao, ít import → gate tự học hạ `α_s`. Đã quan sát đúng: khi
dynamic rỗng hoàn toàn, α_static = 0,93.

Kèm 3 fusion baseline để so sánh (§15): `concat`, `attention`, `late_vote`.
File: `models/fusion.py` (119 dòng).

### §7–8. Module 4–5 — Prototype & Multi-Prototype

| Ý tưởng | Triển khai |
|---|---|
| `p_c = mean(f(x_i))` | `build_prototypes(multi=False)` |
| `P_c = {p_c^1..p_c^M}`, `d(x,c) = min_j d(...)` | `build_prototypes(multi=True)` — k-means trong từng family |
| Phân loại theo softmax(-d) | `class_distances()` + cross-entropy trên `-d` |

Thêm ngoài ý tưởng: `return_assign=True` trả về sub-prototype nào mỗi mẫu support thuộc về —
cần cho refinement §11 ở mức sub-prototype. File: `models/fewshot.py` (89 dòng).

### §9. Huấn luyện episodic + 4 loss

`L = L_cls + λ₁·L_con + λ₂·L_align + λ₃·L_sep` — cài đủ 4 số hạng:

| Loss | Triển khai | Ghi chú |
|---|---|---|
| `L_cls` | Cross-entropy trên `-d(x,c)` | Bỏ qua query nhãn -1 (unknown) |
| `L_con` | Supervised contrastive (Khosla) | Đã sửa lỗi NaN do `-inf × 0` trên đường chéo |
| `L_align` | `1 − cosine(z_s, z_d)` cùng mẫu | Chỉ khi fusion có cả hai view |
| `L_sep` | Hinge `relu(margin − d)` giữa prototype khác family | Chỉ tính cặp khác family |

File: `models/losses.py`, `models/sdfsproto.py::episode_forward`.

### §10. Unknown family detection

Đúng công thức `min_c d(f(x), p_c) > τ`. Điểm bổ sung: **τ không đặt tay** mà hiệu chỉnh trên
tập val bằng Youden's J (tối đa `TPR_known − FPR_unknown`). Episode val/test có `n_unknown`
query lấy từ family ngoài episode. File: `engine/metrics.py::calibrate_tau`, `models/fewshot.py`.

### §11. Mở rộng — Semi-supervised

Đúng công thức ý tưởng, áp dụng **ở mức sub-prototype**:

```
p_c = ( Σ_{S_c} f(x) + β·Σ_j w_j·f(x̃_j) ) / ( |S_c| + β·Σ_j w_j )
```

- Truy xuất `retrieve_k` mẫu unlabeled gần nhất, gán pseudo-label theo softmax(-d)
- Lọc `conf < conf_threshold` (chính là "loại bỏ mẫu mơ hồ" của §11)
- Mỗi pseudo sample chỉ cập nhật sub-prototype nó gần nhất

Kiểm chứng: trên cụm tách rõ, khoảng cách prototype→tâm thật giảm 2,58 → 2,00.
File: `extensions/semi_supervised.py` (75 dòng).

### §12. Mở rộng — Concept drift

`p_c^t = (1−γ)·p_c^{t−1} + γ·z̄_c^t` cài trong `PrototypeBank`. Cơ chế spawn sub-prototype mới
khi cụm mới nằm xa: **chỉ spawn khi có cả một cụm** (≥ max(3, 20% batch)) vượt ngưỡng, không
spawn vì một mẫu ngoại lai đơn lẻ. Ngưỡng `spawn_dist: auto` = 3× độ phân tán nội bộ family.

Kèm `simulate_drift_adaptation()`: sắp mẫu theo timestamp, dựng prototype từ k mẫu đầu, stream
phần còn lại theo chunk, so sánh **frozen vs adaptive** — đúng vòng lặp "analyst gán nhãn mẫu
mới" ở §13 bước 7. File: `extensions/drift.py` (148 dòng).

### §13. Pipeline đầy đủ

Cả 7 bước đều có mã tương ứng:

| Bước §13 | Thực thi bởi |
|---|---|
| 1. Static analysis | `extract/pe_static.py` |
| 2. Dynamic analysis | `extract/cape_report.py` |
| 3. Encoding | `models/static_encoders.py`, `models/dynamic_encoders.py` |
| 4. Reliability-aware fusion | `models/fusion.py::ReliabilityFusion` |
| 5. Few-shot prototype | `models/fewshot.py` |
| 6. Unknown handling | `metrics.calibrate_tau` + `predict_with_unknown` |
| 7. Continual update | `extensions/drift.py::PrototypeBank` |

### §14. Bốn claim

| Claim | Cơ chế trong code | Bằng chứng thực nghiệm |
|---|---|---|
| 1. Static-dynamic complementarity | Hai encoder + fusion | Ablation static-only / dynamic-only ⏳ |
| 2. Few-shot adaptation | Prototype, không train lại | Bảng 5w1s/5w5s/10w5s ⏳ |
| 3. Reliability-aware fusion | Gate theo reliability signal | `analyze_alpha.py` ✅ sẵn sàng ⏳ chờ dữ liệu |
| 4. Multi-prototype | k-means sub-prototype, `d=min_j` | Ablation single vs multi ⏳ |

⏳ = code sẵn sàng, chờ dữ liệu.

### §15–16. Thí nghiệm & baseline

`scripts/run_experiments.py` định nghĩa sẵn **16 thí nghiệm**:

- 3 setting chính: 5w1s, 5w5s, 10w5s
- 2 so sánh dynamic backend: api_seq, behavior_graph
- 4 ablation modality: static-only, dynamic-only ×3 backend
- 3 ablation fusion: concat, attention, late_vote
- 1 ablation prototype: single_proto
- 3 baseline head: matching, relation, siamese *(dùng chung backbone để công bằng)*
- Cộng 3 classical ML chạy riêng: RF, SVM, GradientBoosting/XGBoost

**Metric đã cài** (§16): accuracy ± CI95, macro-F1, family-wise F1, unknown AUROC,
TPR/FPR tại τ, prototype compactness, ms/query. Xuất `results.json` + bảng markdown.

### §17. Năm điểm yếu ý tưởng tự nêu — đối chiếu

| Điểm yếu | Xử lý trong triển khai |
|---|---|
| 1. Few-shot không giải quyết zero-day hoàn toàn không nhãn | Chấp nhận; có unknown detection để không ép mẫu vào family sai |
| 2. Nhãn family nhiễu | Ngoài phạm vi code — yêu cầu làm sạch bằng AVClass ghi trong spec (Boundaries) |
| 3. Dynamic trace không đầy đủ | `dyn_rel` 8 tín hiệu gồm cờ `run_completed`, trace length, anti-analysis count |
| 4. Family-disjoint split bắt buộc | Cưỡng chế ở sampler; chế độ `sample` in cảnh báo rõ không được báo cáo như few-shot |
| 5. Fusion thô gây hại | Có LayerNorm mỗi nhánh + gating; 3 fusion baseline để chứng minh bằng số |

---

## Phần III — Những gì thêm ngoài ý tưởng gốc

1. **Reliability signals (8+8 chiều)** — §6 mô tả bằng bảng tình huống định tính; tôi hiện thực
   thành vector số cụ thể để gate học được. Danh sách đầy đủ trong `data/schema.py`.
2. **τ hiệu chỉnh tự động bằng Youden J** thay vì đặt tay.
3. **Dummy data generator** (`data/dummy.py`, 287 dòng) — sinh dữ liệu synthetic có đủ packed /
   evasive / subtype / drift / unlabeled pool, cho phép test toàn pipeline trước khi có data thật.
4. **Chế độ `sample`-disjoint** — để Dataset A (5 family) dùng được mà không giả mạo giao thức few-shot.
5. **Chế độ `--static-only`** — chạy pipeline khi chưa có sandbox; dynamic điền rỗng hợp lệ.
6. **Cluster-gated spawn** cho drift — §12 chỉ nói "nếu xuất hiện cụm mới cách xa"; tôi thêm điều
   kiện số lượng tối thiểu để một mẫu ngoại lai không sinh sub-prototype rác.

---

## Phần IV — Kết quả kiểm chứng đã có (dummy data)

Dummy 30 family / 1.297 mẫu, cấu hình nhỏ, 5 epoch CPU:

```
test accuracy      0,8382 ± 0,0096   (6 family chưa từng thấy khi train)
macro-F1           0,8384
unknown AUROC      0,8560
proto compactness  5,11
ms/query           3,7
```

Kiểm chứng thành phần:
- 14 cấu hình (3 backend × 4 fusion + 2 modality) — forward/backward hữu hạn ✅
- 3 baseline head + classical RF (acc 0,821 trên 20 episode) ✅
- Semi-supervised: prototype→tâm thật 2,58 → 2,00 ✅
- Drift: spawn đúng khi cụm nhảy xa, không spawn với outlier đơn ✅
- Parser CAPE trên report giả: 8 semantic tag gán đúng ✅
- PE extractor trên PE tự dựng + file rác: không crash, degrade đúng ✅

**Cảnh báo diễn giải:** mọi số trên là *dummy data do tôi sinh ra* — chỉ chứng minh đường ống
chạy đúng, **không** phản ánh hiệu năng thật.

---

## Phần V — Còn lại

### Chặn bởi dữ liệu
- Dataset A (5 family, 9.970 mẫu) — ở máy khác, chưa trích xuất
- Dataset B (~30 family) — chưa thu thập; khuyến nghị MalwareBazaar
  (BODMAS **không dùng được**: chỉ phát hành feature vector, không có binary → không chạy sandbox)
- CAPE chưa dựng xong (`capev2/` mới có `config.md`); ước ~95h sandbox cho 2.500 mẫu / 2 VM

### Chặn bởi công việc còn lại
- **Đối chiếu parser CAPE với report thật** — rủi ro cao nhất; sai field = mất 95h sandbox
- Xác nhận Dataset A có timestamp không (quyết định thí nghiệm drift còn hay bỏ)

### Công cụ đo Claim 3 — `analyze_alpha.py`

Bốn giả thuyết **một phía, đặt trước khi xem dữ liệu**:

| Giả thuyết | Kỳ vọng |
|---|---|
| PE bị packed | α_static thấp hơn |
| Trace không hoàn chỉnh | α_dynamic thấp hơn |
| Ít import | α_static thấp hơn |
| Trace ngắn | α_dynamic thấp hơn |

Kiểm định Mann-Whitney U (không giả định phân phối chuẩn) + Cliff's delta làm effect size.
Kết luận "ủng hộ" cần **cả hai**: `p < 0,05` **và** `|d| ≥ 0,15` — vì trên hàng nghìn mẫu,
p rất nhỏ vẫn có thể là khác biệt không đáng kể. Script tự phát hiện và cảnh báo khi α collapse.

Chạy thử trên checkpoint dummy: phát hiện đúng α collapse (α_dynamic = 0,998, σ = 0,001) và
chỉ 1/4 giả thuyết được ủng hộ — tức trên dummy, **Claim 3 chưa được chứng minh**. Đây là kết
quả trung thực cần thiết: công cụ không tự động xác nhận claim.

### Rủi ro đã quan sát được, cần theo dõi trên data thật
1. **α collapse** — trên dummy α_dynamic = 0,998 vì tín hiệu synthetic quá mạnh một phía.
   Nếu lặp lại trên data thật thì Claim 3 yếu; dự phòng: entropy regularization trên α.
2. **Drift chưa chứng minh được lợi ích** — frozen ≈ adaptive trên dummy vì drift synthetic quá
   nhẹ. Cần timestamp thật để kết luận.
3. **CPU-only** — ma trận 16 thí nghiệm ước 50–80h; đã có `cpu_light.yaml` (4,8 phút/epoch đo được).

---

## Phần VI — Bản đồ file

```
configs/    default | cpu_light (profile CPU đã đo) | dataset_a (5 family) | dummy_smoke | dummy_tiny
src/sdfsproto/
  config.py               YAML deep-merge + dotted override
  data/schema.py          schema .npz + định nghĩa 16 reliability signal
  data/dummy.py           generator synthetic (packed/evasive/subtype/drift/unlabeled)
  data/dataset.py         Dataset, collate (pad seq + batch graph), EpisodeSampler 2 chế độ split
  models/static_encoders.py   §4 — 4 nhánh + aggregator
  models/dynamic_encoders.py  §5 — 3 backend chung interface
  models/fusion.py            §6 — reliability + 3 baseline
  models/fewshot.py           §7–8 — prototype đơn/multi, distance, unknown
  models/losses.py            §9 — supcon / align / separation
  models/sdfsproto.py         model đầy đủ + episode_forward (4 loss)
  engine/trainer.py           §9 — vòng episodic, hiệu chỉnh τ, test
  engine/metrics.py           §16 — toàn bộ metric
  baselines/heads.py          §16 — Matching / Relation / Siamese
  baselines/classical.py      §16 — RF / SVM / GBoost theo protocol episodic
  extensions/semi_supervised.py  §11
  extensions/drift.py            §12
  extract/pe_static.py        §2.1 — trích xuất PE thật
  extract/cape_report.py      §2.2 — parse CAPE v2 + 16 semantic tag
scripts/    train | run_experiments (16 thí nghiệm) | eval_extensions | extract_features | make_dummy_data
docs/       intent/ | spec.md (REV.2) | handoff.md (cách gửi dữ liệu) | IMPLEMENTATION.md (file này)
tasks/      plan.md | todo.md
```
