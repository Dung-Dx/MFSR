# MFSR

Mã nguồn của bài báo **“Multi-modal DEM Super-Resolution Using Relative
Depth: A New Benchmark and Beyond”**, đã được chấp nhận tại *International
Journal of Applied Earth Observation and Geoinformation*.

Dataset gốc của công trình có tại
[Figshare](https://doi.org/10.6084/m9.figshare.26409490.v3).

Repository này cũng cung cấp `crop_merge_data.py` để crop và gộp ảnh từ hai
dataset mô bệnh học MIDOG++ và BRACS thành các patch RGB kích thước 224 x 224.

## 1. Cài đặt môi trường bằng Conda

Đi đến thư mục repository:

```bash
cd /datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/MFSR
```

Tạo và kích hoạt môi trường Conda sử dụng Python 3.10:

```bash
conda create --name mfsr python=3.10 pip -y
conda activate mfsr
```

Cài đặt các thư viện cần thiết:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` ghim PyTorch 2.8.0 và torchvision 0.23.0 bản CUDA 12.8 để
tương thích với driver hiện tại của cụm SLURM. Không cài PyTorch trực tiếp từ
PyPI không kèm phiên bản CUDA, vì pip có thể chọn bản CUDA 13.0 và gây lỗi
`The NVIDIA driver on your system is too old`.

Nếu môi trường `mfsr` đã từng cài `torch 2.14.0+cu130`, thay thế bằng bản
CUDA 12.8 như sau:

```bash
conda activate mfsr
python -m pip uninstall -y torch torchvision
python -m pip install \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Kiểm tra PyTorch và GPU:

```bash
python -c "import torch, torchvision, numpy, PIL, cv2; print('PyTorch:', torch.__version__); print('CUDA khả dụng:', torch.cuda.is_available())"
```

Lệnh kiểm tra GPU cần chạy **bên trong SLURM job đã được cấp GPU**. Nếu vẫn có
lỗi driver, cài lại đúng wheel CUDA 12.8 như hướng dẫn bên trên hoặc tham khảo
[trang các phiên bản PyTorch chính thức](https://pytorch.org/get-started/previous-versions/).
Kết quả đúng trong job phải có hậu tố `+cu128` và `CUDA khả dụng: True`.

## 2. Dataset đầu vào

Script sử dụng mặc định hai đường dẫn sau:

```text
MIDOG++:
/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/Dataset/images

BRACS:
/datastore/uittogether/LuuTru/Thuchd/Research/BRACS
```

Đối với BRACS, script sử dụng ảnh trong `BRACS_RoI/latest_version`, không xử
lý tập WSI có dung lượng gần 1 TB.

Cách chia dữ liệu:

- BRACS giữ nguyên các split `train`, `val`, `test` đã có sẵn.
- MIDOG++ được chia theo từng ảnh nguồn/case với tỷ lệ 80/10/10.
- MIDOG++ được stratify theo loại u và sử dụng seed cố định để có thể tái lập.
- Các patch sinh ra từ cùng một ảnh nguồn không bao giờ nằm ở nhiều split.

## 3. Kiểm tra trước khi tạo DataMerge

Chạy `dry-run` để kiểm tra đường dẫn, metadata và ước tính số lượng patch mà
không ghi ảnh ra ổ đĩa:

```bash
python crop_merge_data.py --dry-run
```

Script chỉ sử dụng một phương pháp crop cố định với patch 224 x 224. Các patch
bình thường không chồng lấn. Nếu phần còn lại ở mép ảnh nhỏ hơn 224 pixel,
patch cuối được dịch ngược để bám sát mép ảnh và chỉ patch này chồng lên patch
ngay trước nó. Nếu toàn bộ chiều rộng hoặc chiều cao của ảnh nguồn nhỏ hơn 224
pixel thì chiều đó được padding màu trắng vì không tồn tại patch trước để chồng.

Dữ liệu hiện tại dự kiến sinh khoảng **809.394 patch**.

## 4. Crop và gộp hai dataset

Chạy toàn bộ quá trình bằng 8 worker:

```bash
python crop_merge_data.py --workers 8
```

Kết quả mặc định được lưu tại:

```text
/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/Dataset/DataMerge40x
```

Ảnh của MIDOG++ và BRACS được trộn trực tiếp trong cùng thư mục của mỗi split:

```text
DataMerge40x/
|-- train/
|   |-- midogpp_001_x000000_y000000.png
|   |-- bracs_BRACS_1238_PB_32_x000000_y000000.png
|   `-- ...
|-- val/
|-- test/
|-- annotations/
|   |-- train.json
|   |-- val.json
|   `-- test.json
|-- manifest.csv
`-- dataset_info.json
```

Tất cả patch được lưu dưới dạng PNG lossless, RGB và có kích thước chính xác
224 x 224. Tiền tố `midogpp_` và `bracs_` chỉ dùng để nhận biết nguồn và tránh
trùng tên; ảnh của hai dataset vẫn nằm chung trong `train`, `val`, `test`.

## 5. Annotation thống nhất

Mỗi split chỉ có **một file annotation chung** cho toàn bộ patch MIDOG++ và
BRACS:

```text
annotations/train.json
annotations/val.json
annotations/test.json
```

Mỗi file JSON có ba trường chính:

- `images`: tất cả patch MIDOG++ và BRACS của split.
- `annotations`: annotation detection của MIDOG++ và classification của BRACS.
- `categories`: không gian nhãn chung của dataset mới.

Mỗi phần tử `images` chứa `source_dataset` và `task` để xác định nguồn và loại
bài toán. Mỗi phần tử `annotations` có `annotation_type` bằng `detection` hoặc
`classification`.

ID nhãn toàn cục được tổ chức như sau:

| Category ID | Tên lớp | Task | Nguồn |
|---:|---|---|---|
| 1 | mitotic figure | Detection | MIDOG++ |
| 2 | not mitotic figure | Detection | MIDOG++ |
| 100 | N | Classification | BRACS |
| 101 | PB | Classification | BRACS |
| 102 | UDH | Classification | BRACS |
| 103 | FEA | Classification | BRACS |
| 104 | ADH | Classification | BRACS |
| 105 | DCIS | Classification | BRACS |
| 106 | IC | Classification | BRACS |

Đối với MIDOG++, bounding box được chuyển sang định dạng COCO
`[x_min, y_min, width, height]` và hiệu chỉnh theo tọa độ patch. Đối với BRACS,
mỗi patch có một annotation classification và không có bounding box.

### Manifest chung

`manifest.csv` là chỉ mục chung cho tất cả patch của cả hai dataset. Các cột
quan trọng gồm:

- `relative_path`: đường dẫn patch tính từ thư mục gốc của dataset DataMerge.
- `split`: `train`, `val` hoặc `test`.
- `dataset`: `midogpp` hoặc `bracs`.
- `task`: `detection` hoặc `classification`.
- `class_id`, `class_name`, `group_name`: nhãn BRACS.
- `tumor_type`: loại u của MIDOG++.
- `x`, `y`: vị trí patch trên ảnh nguồn.
- `num_mitotic`, `num_hard_negative`: số đối tượng MIDOG++ trong patch.
- `image_id`: ID liên kết với bản ghi trong file annotation của split.
- `global_category_id`: nhãn toàn cục 100–106 đối với patch BRACS.

Ví dụ đọc manifest:

```python
import pandas as pd

manifest = pd.read_csv(
    "/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/"
    "Dataset/DataMerge40x/manifest.csv"
)

train_data = manifest[manifest["split"] == "train"]
bracs_train = train_data[train_data["dataset"] == "bracs"]
midog_train = train_data[train_data["dataset"] == "midogpp"]
```

## 6. Các tùy chọn chạy

Loại các patch có ít hơn 10% mô ước tính:

```bash
python crop_merge_data.py \
    --workers 8 \
    --min-tissue-fraction 0.10
```

Thay đổi đường dẫn dữ liệu:

```bash
python crop_merge_data.py \
    --midog-dir /duong/dan/MIDOGpp/images \
    --bracs-dir /duong/dan/BRACS \
    --output-dir /duong/dan/DataMerge40x \
    --workers 8
```

Xem toàn bộ tham số:

```bash
python crop_merge_data.py --help
```

Nếu quá trình crop bị gián đoạn, có thể chạy lại cùng câu lệnh. Script sẽ bỏ
qua các PNG đã tồn tại. Chỉ sử dụng `--overwrite` nếu muốn tạo lại ảnh cũ.

## 7. Tạo DataMerge20x và DataMerge10x

Sau khi đã tạo xong `DataMerge40x`, dùng cùng một file `resize.py` để tạo hai
độ phân giải thấp hơn. Tạo ảnh 112 x 112 và lưu vào `DataMerge20x`:

```bash
python resize.py --size 112 --workers 8
```

Tạo ảnh 56 x 56 và lưu vào `DataMerge10x`:

```bash
python resize.py --size 56 --workers 8
```

Hai lệnh trên đều đọc trực tiếp từ `DataMerge40x`, không resize nối tiếp từ
112 xuống 56. Ảnh được downsample bằng nội suy LANCZOS và vẫn giữ nguyên định
dạng RGB cùng tên file, split và cấu trúc thư mục:

```text
DataMerge20x/                 DataMerge10x/
|-- train/                    |-- train/
|-- val/                      |-- val/
|-- test/                     |-- test/
|-- annotations/              |-- annotations/
|   |-- train.json            |   |-- train.json
|   |-- val.json              |   |-- val.json
|   `-- test.json             |   `-- test.json
|-- manifest.csv              |-- manifest.csv
`-- dataset_info.json         `-- dataset_info.json
```

`resize.py` tự động cập nhật metadata như sau:

- `width`, `height` trong annotation và `patch_width`, `patch_height` trong
  manifest được đổi thành kích thước đích.
- `bbox` và `area` của annotation detection MIDOG++ được scale theo kích thước
  ảnh mới. Annotation classification BRACS không có bbox nên giữ nguyên.
- `x`, `y`, `valid_width`, `valid_height` vẫn là tọa độ/kích thước crop trên
  ảnh nguồn 40x, nhờ đó vẫn truy ngược được vị trí patch gốc.
- `dataset_info.json` ghi rõ dataset cha, tỉ lệ resize, phép nội suy và quy ước
  tọa độ.

Nếu bị gián đoạn, chạy lại đúng lệnh cũ để tiếp tục. Mặc định script bỏ qua ảnh
đã có; dùng `--overwrite` khi thật sự muốn resize lại toàn bộ. Có thể xem tất
cả tham số bằng:

```bash
python resize.py --help
```

Khi dùng đường dẫn khác hoặc một kích thước khác, truyền rõ cả đầu vào và đầu
ra:

```bash
python resize.py \
    --input-dir /duong/dan/DataMerge40x \
    --output-dir /duong/dan/DataMergeMoi \
    --size 112 \
    --workers 8
```

## 8. Kiểm tra mô hình MFSR

Repository chính thức ban đầu thiếu `common.py` và `model_MCANet/CBAM.py` dù
hai module này được import trực tiếp. Phiên bản trong thư mục này đã tái hiện
các thành phần cần thiết dựa trên kiến trúc trong paper và implementation gốc
của từng khối:

- `common.py`: RCAB/Residual Group từ RCAN và DenseProjection từ D-DBPN.
- `model_MCANet/CBAM.py`: Convolutional Block Attention Module.
- `ResNet.py`: ba backbone ResNet101 độc lập đúng kiến trúc pseudo-siamese.
- `MFSR_DEM_Depth.py`: fusion theo kích thước động nên hỗ trợ trực tiếp cặp
  56 x 56 và 224 x 224, thay vì cố định feature map 64 x 64.

Chạy kiểm tra forward không tải pretrained weights:

```bash
python MFSR_DEM_Depth.py
```

Kết quả mong đợi:

```text
Device: cuda; parameters: 180,277,633
Output shape: (1, 1, 224, 224)
```

Nếu máy không nhận GPU, dòng đầu sẽ là `Device: cpu` và forward sẽ chậm hơn.
Khi khởi tạo model để huấn luyện, `pretrained=True` sẽ tải pretrained
ResNet101 của PyTorch trong lần chạy đầu tiên:

```python
from MFSR_DEM_Depth import MFSR

model = MFSR(pretrained=True, scale=4)
```

### Quy ước đầu vào

Với `scale=4`, `MFSR.forward(lr_dem, optical, relative_depth)` yêu cầu:

| Tensor | Số kênh | Ví dụ kích thước |
|---|---:|---|
| `lr_dem` | 1 | `B x 1 x 56 x 56` |
| `optical` | 3 | `B x 3 x 224 x 224` |
| `relative_depth` | 1 | `B x 1 x 224 x 224` |
| output SR DEM | 1 | `B x 1 x 224 x 224` |

DataMerge40x/20x/10x chứa ảnh mô bệnh học RGB, còn mô hình MFSR gốc nhận ba
modality DEM/optical/depth. Vì vậy, không dùng trực tiếp class `MFSR` ở trên
để train DataMerge. Phần tiếp theo sử dụng `HistologyMFSR`, là adaptation RGB
riêng cho mục tiêu pretrain backbone WSSS-Tissue.

Nguồn kiến trúc tham khảo:

- Paper MFSR: <https://doi.org/10.1016/j.jag.2025.104865>
- Repository chính thức: <https://github.com/hwj0711/MFSR>
- RCAN: <https://github.com/yulunzhang/RCAN>
- D-DBPN implementation: <https://github.com/sanghyun-son/EDSR-PyTorch>
- CBAM implementation: <https://github.com/Jongchan/attention-module>

## 9. Train MFSR trên DataMerge cho WSSS-Tissue Stage 2

`MFSR_Histology.py` chỉ nhận ảnh RGB độ phân giải thấp. Ảnh được nội suy
bicubic lên 224 x 224; hai view luminance và một view RGB sau đó đi qua ba
nhánh encoder MFSR. Ảnh ground truth 40x chỉ được dùng để tính loss, hoàn toàn
không được truyền vào đầu vào mô hình. Nhánh RGB `encoder.OPT_resnet` là
ResNet101 sẽ được xuất riêng để nạp vào backbone của WSSS-Tissue Stage 2.

Hai bài toán được train độc lập:

| Bài toán | Input | Target | Thư mục checkpoint mặc định |
|---|---|---|---|
| 10x → 40x (×4) | DataMerge10x, 56² | DataMerge40x, 224² | `checkpoints/mfsr_10x_to_40x` |
| 20x → 40x (×2) | DataMerge20x, 112² | DataMerge40x, 224² | `checkpoints/mfsr_20x_to_40x` |

### Kiểm tra cặp dữ liệu trước khi train

Các lệnh này chỉ đọc manifest và một cặp ảnh, không cần GPU và không khởi tạo
mô hình:

```bash
python train_mfsr_histology.py \
    --scale 4 --dry-run --max-train-samples 2 --max-val-samples 1

python train_mfsr_histology.py \
    --scale 2 --dry-run --max-train-samples 2 --max-val-samples 1
```

### Chạy thử ngắn trên GPU

Nên chạy thử trước để đo VRAM và thời gian. Batch size tối thiểu là 2 vì ASPP
có BatchNorm trên feature map 1 x 1:

```bash
python train_mfsr_histology.py \
    --scale 4 \
    --epochs 1 \
    --batch-size 2 \
    --lr 1e-4 \
    --gradient-clip 1.0 \
    --amp-dtype bfloat16 \
    --workers 8 \
    --max-train-samples 100 \
    --max-val-samples 20 \
    --checkpoint-dir checkpoints/smoke_10x_to_40x
```

### Train đầy đủ hai model

Train 10x → 40x:

```bash
CUDA_VISIBLE_DEVICES=0 python train_mfsr_histology.py \
    --scale 4 \
    --epochs 100 \
    --batch-size 64 \
    --grad-accumulation 1 \
    --lr 1e-4 \
    --gradient-clip 1.0 \
    --amp-dtype bfloat16 \
    --workers 8
```

Train 20x → 40x:

```bash
CUDA_VISIBLE_DEVICES=1 python train_mfsr_histology.py \
    --scale 2 \
    --epochs 100 \
    --batch-size 64 \
    --grad-accumulation 1 \
    --lr 1e-4 \
    --gradient-clip 1.0 \
    --amp-dtype bfloat16 \
    --workers 8
```

Hai lệnh có thể chạy ở hai job/GPU riêng. Với toàn bộ 657.380 patch train,
100 epoch sẽ rất lâu; hãy dùng kết quả smoke test để chọn lại số epoch phù
hợp. Batch 64 đã sử dụng gần 65 GiB VRAM trong lần chạy thực tế; hãy giảm
`--batch-size` nếu GPU được cấp ít bộ nhớ hơn.

Trên SLURM, `run.sh` nhận scale làm đối số. Gửi hai job độc lập bằng:

```bash
sbatch run.sh 4  # DataMerge10x -> DataMerge40x
sbatch run.sh 2  # DataMerge20x -> DataMerge40x
```

Thiết lập ổn định mặc định là BF16 trên A100, learning rate `1e-4` và gradient
clipping `1.0`. Phép attention nhạy với tràn số luôn được tính ở FP32. Nếu
loss hoặc gradient vẫn không hữu hạn ở một batch cá biệt, script bỏ batch hoặc
optimizer update đó, xóa gradient, tự giảm learning rate một nửa và tiếp tục
batch kế tiếp. Cảnh báo cùng tên ảnh được ghi vào SLURM log; giá trị `NaN`
không được cộng vào loss trung bình hoặc dùng để chọn checkpoint.

Mỗi thư mục kết quả chứa:

```text
last_training_checkpoint.pth          # toàn bộ model/optimizer để resume
last_resnet101_for_wsss_stage2.pth     # backbone RGB ở epoch cuối
best_resnet101_for_wsss_stage2.pth     # backbone RGB có val L1 tốt nhất
training_config.json
tensorboard/
```

Theo dõi quá trình train:

```bash
tensorboard --logdir checkpoints --port 6006
```

Nếu job bị ngắt, resume đúng bài toán bằng full training checkpoint:

```bash
python train_mfsr_histology.py \
    --scale 4 \
    --epochs 100 \
    --batch-size 64 \
    --grad-accumulation 1 \
    --lr 1e-4 \
    --gradient-clip 1.0 \
    --amp-dtype bfloat16 \
    --workers 8 \
    --resume checkpoints/mfsr_10x_to_40x/last_training_checkpoint.pth
```

### Nạp backbone vào WSSS-Tissue Stage 2

Chọn **một** trong hai file `best_resnet101_for_wsss_stage2.pth` cho một lần
thử nghiệm Stage 2. Ngay sau khi tạo model DeepLab của WSSS-Tissue, nạp raw
state dict vào backbone:

```python
import torch

backbone_path = (
    "/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/MFSR/"
    "checkpoints/mfsr_10x_to_40x/best_resnet101_for_wsss_stage2.pth"
)
backbone_state = torch.load(backbone_path, map_location="cpu")
model.backbone.load_state_dict(backbone_state, strict=True)
```

Đổi `mfsr_10x_to_40x` thành `mfsr_20x_to_40x` để dùng checkpoint thứ hai.
Không truyền file backbone này vào tham số `--resume` hiện có của
WSSS-Tissue, vì `--resume` của repo đó dành cho checkpoint **toàn bộ DeepLab**.
Tên và shape của toàn bộ 624 tensor ResNet101 giữa checkpoint được xuất và
backbone Stage 2 đã được đối chiếu tương thích; khác biệt output-stride của
Stage 2 chỉ là stride/dilation, không làm thay đổi shape trọng số.
