# Avast-CTU Public CAPE Dataset — đặc tả cấu trúc dữ liệu

> Tài liệu tự chứa, dùng để thiết kế input cho model phân loại họ malware.
> Nguồn: Bošanský et al., *Avast-CTU Public CAPE Dataset*, arXiv:2209.03188 (2022).
> Repo: `https://github.com/avast/avast-ctu-cape-dataset`
>
> Phần **1–6** là đặc tả dataset. Phần **7–9** là kinh nghiệm parse thực tế và
> khuyến nghị thiết kế — được đánh dấu rõ đâu là suy luận chứ không phải tài liệu gốc.

---

## 1. Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Số mẫu | 48.976 |
| Loại mẫu | **Toàn bộ là mẫu độc hại** — không có mẫu sạch |
| Nền tảng | Windows PE |
| Sandbox | CAPEv2 (fork của Cuckoo), chạy tháng 7–8/2021 |
| Môi trường guest | Windows 7 + MS Office, VM ngụy trang giống PC thật (Chrome, Firefox, Adobe Reader, Spotify, có private key, có mật khẩu lưu trong Chrome, có internet thật) |
| Thời gian phát hiện mẫu | Chủ yếu 2017–2019 (mốc `2017-01` gộp cả mẫu cũ hơn) |
| Định dạng | JSON, mỗi mẫu một file |
| Số họ | 10 |
| Số type | 6 |

**Hệ quả về bài toán:** vì không có mẫu sạch, bộ này dùng cho **phân loại đa lớp
theo họ**, không dùng trực tiếp để train detector nhị phân. Muốn làm detector cần
bổ sung mẫu sạch từ nguồn khác (EMBER, SOREL-20M).

---

## 2. Nhãn

Bốn trường metadata cho mỗi mẫu:

| Trường | Kiểu | Mô tả |
|---|---|---|
| `sha256` | string, 64 hex | Định danh; **cũng là tên file JSON** |
| `classification` | categorical | Họ malware (10 giá trị) |
| `type` | categorical | Loại malware (6 giá trị) |
| `date` | date | Ngày phát hiện — **dùng chia train/test** |

### 2.1 Phân bố họ

| Family | Số mẫu | Tỉ lệ |
|---|---|---|
| Emotet | 14.429 | 29,5% |
| Swisyn | 12.591 | 25,7% |
| Qakbot | 4.895 | 10,0% |
| Trickbot | 4.202 | 8,6% |
| Lokibot | 4.191 | 8,6% |
| njRAT | 3.372 | 6,9% |
| Zeus | 2.594 | 5,3% |
| Ursnif | 1.343 | 2,7% |
| Adload | 704 | 1,4% |
| HarHar | 655 | 1,3% |

Mất cân bằng nặng: Emotet gấp ~22 lần HarHar. Cần class weighting hoặc balanced sampling.

### 2.2 Type

`banker`, `trojan`, `pws`, `coinminer`, `rat`, `keylogger`

### 2.3 Nhiễu nhãn

Tài liệu gốc nêu rõ: gán một PE độc hại vào đúng một họ là việc khó vì các họ có thể
**dùng chung code**. Nhãn lấy từ hệ thống nội bộ của Avast và **có thể có nhiễu**.

---

## 3. Tổ chức file

```
<dataset_root>/
├── <sha256>.json          # một file JSON mỗi mẫu
├── ...
└── <file nhãn>            # ánh xạ sha256 -> classification, type, date
```

Có **hai bộ report**:

| Bộ | Nội dung | Kích thước | Dùng để |
|---|---|---|---|
| **Full** | Toàn bộ output CAPEv2: cây tiến trình, mọi system call kèm tham số, binary dump, memory info, kết quả YARA | Rất lớn, một số file **>800 MB** | Phân tích sâu |
| **Reduced** | Chỉ giữ `behavior.summary` và `static.pe` | Nhỏ | **Train model** |

> **Không train trên full report.** Một số trường trong đó — đặc biệt kết quả
> **YARA detection** — làm **lộ nhãn thật**, model học tắt và kết quả vô nghĩa.
> Đây là lý do bộ reduced tồn tại.

---

## 4. Schema của reduced report

```json
{
  "behavior": { "summary": { ... } },
  "static":   { "pe":      { ... } }
}
```

Chỉ có hai nhánh này. **Mọi nhánh khác của CAPE đều bị cắt** — xem mục 6.

### 4.1 `behavior.summary` — đặc trưng động

Tất cả các trường là **mảng chuỗi**, độ dài thay đổi tùy mẫu.

| Key | Nội dung | Ví dụ |
|---|---|---|
| `resolved_apis` | API được resolve, dạng `<dll>.<FunctionName>` | `kernel32.dll.GetNativeSystemInfo` |
| `keys` | Registry key được truy cập | `HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Nls\CustomLocale` |
| `write_keys` | Registry key bị **ghi** (có thể rỗng `[]`) | `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run\x` |
| `files` | File được truy cập | `C:\Windows\SysWOW64\kernel32.dll` |
| `executed_commands` | Lệnh/tiến trình được thực thi, đường dẫn đầy đủ | `C:\Users\comp\AppData\Local\Temp\FFFF450D574E5E5706FB.exe` |
| `mutexes` | Mutex được tạo | `PEMB40`, `Global\I5C3A8244` |

Ngoài sáu key trên (được minh hoạ trong bài báo), CAPE summary **có thể** sinh thêm
tùy mẫu: `read_keys`, `delete_keys`, `read_files`, `write_files`, `delete_files`,
`dll_loaded`, `started_services`, `directory_created`, `command_line`.

**Không giả định key nào luôn tồn tại.** Code parse phải xử lý key thiếu và mảng rỗng.

### 4.2 `static.pe` — đặc trưng tĩnh

| Key | Kiểu | Ghi chú |
|---|---|---|
| `sections` | object[] | Xem 4.2.1 |
| `imports` | object[] | Xem 4.2.2 |
| `imported_dll_count` | int | Số DLL import |
| `exports` | array | Thường rỗng với malware |
| `resources` | object[] | Xem 4.2.3 |
| `versioninfo` | array | Thông tin version từ resource |
| `entrypoint` | string hex | `"0x00403600"` |
| `imagebase` | string hex | `"0x00400000"` |
| `imphash` | string md5 | Hash của import table — **đặc trưng họ rất mạnh** |
| `reported_checksum` | string hex | Checksum ghi trong header |
| `actual_checksum` | string hex | Checksum tính lại — **lệch nhau là tín hiệu** |
| `timestamp` | string | Compile time, `"YYYY-MM-DD HH:MM:SS"`; thường bị giả mạo |
| `osversion` | string | `"6.0"` |
| `pdbpath` | string \| null | Đường dẫn PDB nếu có |
| `peid_signatures` | array \| null | Signature packer theo PEiD — **nhãn packer sẵn có** |
| `overlay` | object \| null | Dữ liệu thừa sau phần cuối PE |
| `digital_signers` | array | Chữ ký số |
| `guest_signers` | object | Xem 4.2.4 |
| `icon`, `icon_hash`, `icon_fuzzy` | string \| null | Fuzzy hash của icon |

#### 4.2.1 `sections[]`

```json
{
  "name": ".text",
  "raw_address": "0x00001000",
  "virtual_address": "0x00001000",
  "virtual_size": "0x00002786",
  "size_of_data": "0x00003000",
  "entropy": "5.83",
  "characteristics": "IMAGE_SCN_CNT_CODE|IMAGE_SCN_MEM_EXECUTE|IMAGE_SCN_MEM_READ",
  "characteristics_raw": "0x60000021"
}
```

#### 4.2.2 `imports[]`

```json
{
  "dll": "USER32.dll",
  "imports": [ { "name": "GetFocus", "address": "0x404024" } ]
}
```

#### 4.2.3 `resources[]`

```json
{
  "name": "RT_STRING",
  "filetype": null,
  "offset": "0x000210a0",
  "size": "0x00000024",
  "entropy": "0.55",
  "language": "LANG_NORWEGIAN",
  "sublanguage": "SUBLANG_NORWEGIAN_BOKMAL"
}
```

#### 4.2.4 `guest_signers`

```json
{
  "aux_error": true,
  "aux_sha1": null,
  "aux_timestamp": null,
  "aux_valid": false,
  "aux_signers": [],
  "aux_error_desc": "No signature found. SignTool Error ..."
}
```

---

## 5. Chia train/test — bắt buộc theo thời gian

| Tập | Điều kiện | Số mẫu | Tỉ lệ |
|---|---|---|---|
| Train | `date` < **2019-08-01** | 37.512 | ~76% |
| Test | `date` >= **2019-08-01** | 11.464 | ~24% |

**Không chia ngẫu nhiên.** Bài báo chứng minh: với random split, model chỉ dùng static
đạt accuracy **>95%**; cùng model đó với time split chỉ đạt **~63%**. Random split
khiến model bám vào đặc trưng dễ nhận nhưng không tổng quát theo thời gian.

Nếu cần tập validation, cắt từ **phần mới nhất của train** (không lấy ngẫu nhiên),
để val cũng nằm sau train về thời gian — giữ đúng tinh thần đánh giá trước concept drift.

---

## 6. Những gì reduced report KHÔNG có

Đây là phần quan trọng nhất khi thiết kế model, vì nó loại bỏ nhiều kiến trúc phổ biến.

| Thiếu | Hệ quả thiết kế |
|---|---|
| **Thứ tự lời gọi API** | `resolved_apis` là **tập hợp**, không phải chuỗi. Mọi model chuỗi (LSTM/GRU/Transformer trên API sequence) **mất nền tảng**. Dùng set encoder (DeepSets, EmbeddingBag, TF-IDF) thay vì sequence model. |
| **Tham số từng lời gọi** | Không gắn được ngữ nghĩa vào từng call. `RegSetValue` không biết ghi vào key nào — chỉ biết *mẫu có ghi vào key nào đó* qua `write_keys`. |
| **Số lần gọi** | Chỉ biết "có gọi", không biết tần suất. Không xây được histogram API. |
| **Cây tiến trình** | Không có pid/parent_id. Không dựng được đồ thị quan hệ tiến trình. |
| **Toàn bộ nhánh network** | Không có domain, IP, HTTP, DNS. Với Emotet/Trickbot/Ursnif — vốn đặc trưng bởi hành vi C2 — đây là mất mát lớn. |
| **Thời lượng chạy, cờ chạy thành công** | Không phân biệt được "trace rỗng vì mẫu né sandbox" và "trace rỗng vì mẫu đơn giản". |
| **File PE gốc** | Không có binary. Không chạy được MalConv, ảnh xám, hay trích EMBER. Static chỉ có những gì `static.pe` đã tóm tắt. |

---

## 7. Kinh nghiệm parse thực tế

*(Phần này rút từ việc triển khai parser thật, không có trong tài liệu gốc.)*

### 7.1 Số được lưu dưới dạng CHUỖI

Địa chỉ và kích thước là chuỗi hex (`"0x00001000"`), entropy là chuỗi thập phân
(`"5.83"`). Phải ép kiểu thủ công và **có phòng vệ**, vì giá trị có thể là `null`
hoặc chuỗi rỗng:

```python
def hexint(v, default=0):
    try:
        s = str(v).strip()
        return int(s, 16) if s.lower().startswith("0x") else int(s)
    except (TypeError, ValueError):
        return default
```

### 7.2 Schema không cố định

Key có thể **thiếu hẳn** giữa các mẫu, không chỉ là rỗng. Nhiều trường là `null`.
Luôn dùng `.get()` với giá trị mặc định, không index trực tiếp.

### 7.3 Tên ngẫu nhiên phải chuẩn hoá trước khi hash

Đây là cái bẫy dễ mắc nhất. Ví dụ thật trong `executed_commands`:

```
C:\Users\comp\AppData\Local\Temp\FFFF450D574E5E5706FB.exe
```

Tên `FFFF450D574E5E5706FB` **đổi mỗi lần chạy**. Hash thẳng chuỗi này là học nhiễu.
Thứ mang thông tin là **mẫu hình**: "file tên hex dài, thả vào Temp, rồi được chạy".

Chuẩn hoá trước khi hash:

```python
v = v.lower()
v = re.sub(r"c:\\users\\[^\\]+", "%userdir%", v)          # tên user
v = re.sub(r"[0-9a-f]{8,}", "%hex%", v)                   # tên ngẫu nhiên hex
v = re.sub(r"[0-9a-f-]{36}", "%guid%", v)                 # GUID
v = re.sub(r"\d+", "%d", v)                               # số
```

Áp dụng cho `files`, `executed_commands`, `keys`, `mutexes`, `pdbpath`.

### 7.4 `resolved_apis` dạng `<dll>.<Function>`

Tách bằng dấu chấm **cuối cùng**, vì tên DLL đã chứa dấu chấm:

```python
dll, func = api.rsplit(".", 1)     # "kernel32.dll.CreateFileW" -> ("kernel32.dll", "CreateFileW")
```

Giữ cả hai: tên DLL cho biết *nhóm năng lực*, tên hàm cho biết *hành vi cụ thể*.

### 7.5 Escape backslash

Đường dẫn trong JSON dùng `\\`. Thư viện JSON chuẩn tự xử lý — nhưng nếu ghi regex
trên chuỗi đã parse thì nhớ path là single backslash.

---

## 8. Trường có giá trị cao — theo mức độ

*(Đánh giá của người triển khai, không phải kết luận của bài báo.)*

**Rất mạnh**

- `mutexes` — malware thường tạo mutex tên **hardcode** để không tự lây nhiễm hai
  lần. Cùng họ hay dùng cùng tên qua nhiều biến thể. Gần như chữ ký họ, và **packer
  không xoá được** vì nó là hành vi runtime.
- `imphash` — vân tay của import table. Biến thể cùng họ build từ cùng source thường
  trùng imphash.
- `write_keys` — mạnh hơn `keys` nhiều. Ghi vào `\Run\` là cài khởi động cùng máy.
- `peid_signatures` — **nhãn packer có sẵn**, không cần đoán bằng heuristic entropy.

**Trung bình**

- Entropy từng section (min/max/mean/std) + tỉ lệ section entropy > 7,2
- Lệch giữa `reported_checksum` và `actual_checksum`
- `resolved_apis` dạng tập — hashing trick hoặc TF-IDF
- Tên section (`UPX0`, `.themida` lộ packer), `imported_dll_count`
- Kích thước và sự tồn tại của `overlay`

**Yếu / nhiều nhiễu**

- `files` — phần lớn là DLL hệ thống, gần như mẫu nào cũng có. Lọc bỏ đường dẫn
  `C:\Windows\` trước khi dùng.
- `timestamp` compile — thường bị giả mạo (`"1995-11-19 14:43:13"` là giá trị hay
  gặp). Nhưng **chính sự giả mạo lại là tín hiệu**: dùng cờ nhị phân "timestamp có
  hợp lý không" thay vì giá trị thô.

---

## 9. Baseline để so sánh

Model gốc: **HMIL** (Hierarchical Multi-Instance Learning), nhận trực tiếp cây JSON
làm input, không vector hoá thủ công. Thư viện `Mill.jl` + `JsonGrinder.jl` (Julia).
Cấu hình: dense 32, aggregation `meanmax`, activation `relu`, 200 bước, minibatch 500, ADAM.

| Model | Input | Acc train | **Acc test (time split)** |
|---|---|---|---|
| Reduced | `behavior.summary` + `static.pe` | 99,5% | **94,5%** |
| Static-only | chỉ `static.pe` | 96,7% | **~63%** |

Accuracy theo họ, model Reduced trên test:

| Family | Acc | | Family | Acc |
|---|---|---|---|---|
| Adload | 100,00% | | Trickbot | 98,58% |
| Swisyn | 99,88% | | Zeus | 95,83% |
| njRAT | 99,26% | | Lokibot | 95,33% |
| Qakbot | 99,24% | | Ursnif | 94,02% |
| HarHar | 98,55% | | Emotet | 86,54% |

**Kết luận từ baseline:** đặc trưng hành vi là thiết yếu. Model chỉ dùng static sụp
đổ trên Emotet (2,35%) và Qakbot (35,62%) ở tập test — hai họ này thay đổi đặc tính
static mạnh theo thời gian.

> **Ngưỡng tự kiểm tra:** nếu model của bạn đạt gần 100% trên test với time split,
> gần như chắc chắn có rò rỉ hoặc đã vô tình chia ngẫu nhiên. Mốc hợp lý là quanh
> 94,5% cho static+dynamic.

---

## 10. Đặc thù domain cần lưu ý

Ba điểm khiến bài toán này khác ML thông thường:

1. **Concept drift liên tục** — phân phối thay đổi do phần mềm mới và do chính tác
   giả malware chủ động né detection. Đây là lý do time split bắt buộc.
2. **Nhãn nhiễu** — xác định ground truth rất tốn kém, cần chuyên gia. Với họ dùng
   chung code thì ranh giới bản thân nó đã mờ.
3. **Yêu cầu false-positive cực thấp** — trong hệ thống phòng thủ nhiều lớp, false
   negative có thể được lớp khác bắt, nhưng false positive báo thẳng cho người dùng.
   Mức yêu cầu khoảng **FPR ≈ 10⁻⁴**. Metric nên là **TPR tại FPR cố định rất thấp**,
   không chỉ accuracy.

---

## 11. Gợi ý thiết kế input

**Nếu vector hoá thủ công thay vì dùng HMIL:**

| Nhóm | Cách xử lý |
|---|---|
| Tập chuỗi (`resolved_apis`, `files`, `mutexes`, `keys`, tên hàm import) | Hashing trick hoặc TF-IDF; nhớ chuẩn hoá theo 7.3 trước |
| Số | Entropy section (min/max/mean/std), số section, `imported_dll_count`, lệch checksum, kích thước overlay |
| Categorical | `imphash`, tên section, `osversion`, `peid_signatures` |
| Nhị phân/đếm | Có `digital_signers` không, `aux_valid`, số `executed_commands`, số mutex |

**Kiến trúc phù hợp với đặc thù "tập hợp, không thứ tự":**

- DeepSets / EmbeddingBag trên tập API — đúng bản chất dữ liệu
- Gradient boosting trên vector đã hash — baseline mạnh, rẻ
- HMIL trên cây JSON thô — cách của bài báo gốc, không cần feature engineering
- **Không dùng** LSTM/Transformer trên `resolved_apis` như thể nó là chuỗi

**Hướng nghiên cứu bộ này hỗ trợ tốt:**

- Robust hoá trước concept drift (có `date` cho mọi mẫu)
- Tìm phần "rẻ nhất" của report vẫn đủ phân loại chính xác — vì trích static nhanh
  hơn nhiều so với chạy sandbox
- Xử lý JSON thô không cần feature engineering
