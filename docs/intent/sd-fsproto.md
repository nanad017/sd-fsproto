# Intent: SD-FSProto — Static-Dynamic Few-Shot Malware Family Classification

Chốt ngày 2026-07-26, đã xác nhận với tác giả.

- **Outcome:** Codebase hoàn chỉnh của SD-FSProto chạy được thực nghiệm và sinh kết quả
  (bảng chính + ablation) để viết paper.
- **Người dùng:** Tác giả paper — chạy thí nghiệm, không phải hệ thống production.
- **Kiến trúc:**
  - Static encoder **4 nhánh**: byte/MalConv, grayscale image CNN, import/string, PE metadata MLP.
    **Bỏ CFG/FCG** (out of scope).
  - Dynamic encoder **cả 3 cách**, interface chung cắm-rút:
    1. API sequence (Transformer)
    2. API + argument semantic tokens
    3. Behavior graph (GNN)
  - Reliability-aware fusion (α_s, α_d, α_sd) + cross-modal interaction.
  - Multi-prototype per family, khoảng cách min over sub-prototypes.
  - Episodic meta-learning, loss tổng = L_cls + λ1·L_con + λ2·L_align + λ3·L_sep.
  - Unknown detection bằng ngưỡng τ trên khoảng cách tới prototype gần nhất.
- **Mở rộng (trong scope paper):**
  - Mục 11: semi-supervised retrieval + pseudo-label prototype refinement.
  - Mục 12: drift-aware prototype update (EMA + spawn sub-prototype mới).
- **Dữ liệu:** Giả định hoàn thiện, tác giả cung cấp sau. Kỳ vọng: PE files + nhãn family
  + CAPE v2 reports; cần timestamp (cho drift) và pool unlabeled (cho semi-supervised).
- **Môi trường:** Dev trên máy này (CPU, smoke-test bằng dữ liệu synthetic);
  train thật trên máy khác có GPU → code portable, điều khiển bằng config.
- **Thí nghiệm:** 5-way 1-shot / 5-shot, 10-way 5-shot; ablation từng module;
  baseline: ProtoNet, Siamese, Matching, Relation, fusion variants, classical ML.
  Family-disjoint split bắt buộc.
- **Out of scope:** CFG/FCG encoder; hệ thống production/realtime; thu thập dataset;
  viết nội dung paper (chỉ hỗ trợ số liệu/bảng).

Nguồn ý tưởng gốc: `Ý tưởng.txt` (cùng thư mục gốc repo).
