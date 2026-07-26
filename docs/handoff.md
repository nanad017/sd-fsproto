# Cách bàn giao dữ liệu cho pipeline

Ba thứ cần gửi, theo thứ tự ưu tiên. Không cần đủ cả ba mới bắt đầu — gửi được cái nào làm cái đó.

---

## 1. CAPE report mẫu — ƯU TIÊN CAO NHẤT

**Gửi cái này ngay khi chạy được mẫu đầu tiên, đừng chờ chạy xong hàng loạt.**

Lý do: parser `cape_report.py` viết theo CAPE v2 chuẩn, nhưng field name lệch giữa các
version. Sai một field = 95 giờ sandbox ra dữ liệu rác. Đối chiếu trước tốn 10 phút.

Chỉ cần phần đầu, không cần cả file (report thật thường vài chục MB):

```bash
python3 -c "
import json; r = json.load(open('report.json'))
print('info keys:', list(r.get('info', {}).keys()))
print('top keys:', list(r.keys()))
b = r.get('behavior', {})
print('behavior keys:', list(b.keys()))
p = (b.get('processes') or [{}])[0]
print('process keys:', list(p.keys()))
print('n_calls:', len(p.get('calls') or []))
print('first 3 calls:'); print(json.dumps((p.get('calls') or [])[:3], indent=2)[:2000])
print('summary keys:', list((b.get('summary') or {}).keys()))
"
```

Dán output vào chat. Tôi đối chiếu và sửa parser nếu lệch.

---

## 2. Cấu trúc dataset

```bash
# Chạy ở máy chứa dataset
ls <thư mục dataset> | head -20
find <thư mục dataset> -maxdepth 2 -type d | head -20
ls <thư mục dataset> | wc -l
# Có file metadata/nhãn nào không?
find <thư mục dataset> -maxdepth 2 -name "*.csv" -o -maxdepth 2 -name "*.json" | head
```

Cần biết: tên file là hash hay tên gốc, family nằm ở tên thư mục hay file nhãn riêng,
**có cột timestamp/first-seen không** (quyết định thí nghiệm drift còn hay bỏ).

---

## 3. File nhãn `labels.csv`

Format pipeline cần — bốn cột, header đúng tên:

```csv
id,family,timestamp,labeled
a3f2b1c...,winwebsec,1698796800,1
d7e4a09...,zbot,1701388800,1
9c1f5b2...,benign,1699401600,1
```

- `id` — **đúng bằng tên file PE** (pipeline tìm file theo `<pe-dir>/<id>`)
- `family` — tên họ, chữ thường; benign ghi `benign`
- `timestamp` — unix epoch (first-seen). Không có thì để `0`, nhưng mất thí nghiệm drift
- `labeled` — `1` bình thường; đặt `0` để đưa mẫu vào pool unlabeled (cho semi-supervised §11)

Nếu family đang nằm ở tên thư mục, sinh CSV bằng:

```bash
python3 -c "
import csv, pathlib
root = pathlib.Path('<thư mục dataset>')
rows = [{'id': f.name, 'family': f.parent.name.lower(), 'timestamp': 0, 'labeled': 1}
        for f in root.rglob('*') if f.is_file()]
w = csv.DictWriter(open('labels.csv','w',newline=''), fieldnames=['id','family','timestamp','labeled'])
w.writeheader(); w.writerows(rows)
print(len(rows), 'rows')
"
```

---

## Gửi bằng cách nào

| Tình huống | Cách |
|---|---|
| Output lệnh, report mẫu, labels.csv nhỏ | Dán thẳng vào chat |
| Chạy lệnh ngay trong phiên này | Gõ `!` rồi lệnh — output vào thẳng hội thoại |
| File PE thật (2,8 GB) | Copy về máy này rồi cho tôi đường dẫn; hoặc tôi viết script, anh chạy ở máy kia và gửi log |

Không cần gửi file PE để tôi bắt đầu — chỉ cần **cấu trúc + 1 report mẫu** là tôi
đối chiếu parser và chuẩn bị sẵn được.

---

## Khi đã có dữ liệu, chạy theo thứ tự

```bash
# B1. Static trước (không cần sandbox, vài giờ CPU) — phát hiện lỗi sớm
.venv/bin/python scripts/extract_features.py --static-only --split-mode sample \
  --pe-dir <PE> --labels labels.csv --out data/A_static

# B2. Pilot: static-only đã phân biệt được family tới đâu?
.venv/bin/python scripts/train.py --config configs/default.yaml \
  --config configs/cpu_light.yaml --config configs/dataset_a.yaml \
  --set data.root=data/A_static --set model.modality=static

# B3. Sau khi có CAPE report — trích xuất đầy đủ
.venv/bin/python scripts/extract_features.py --split-mode sample \
  --pe-dir <PE> --report-dir <reports> --labels labels.csv --out data/A
```

Với Dataset B (~30 family) thì bỏ `--split-mode sample` để dùng family-disjoint mặc định.
