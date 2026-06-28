# Fire Detection AI

Hệ thống phát hiện lửa và khói theo thời gian thực sử dụng YOLOv8, phục vụ qua REST API với FastAPI.

## Tính năng

- Phát hiện 2 lớp: **fire** (lửa) và **smoke** (khói)
- Tự động dùng GPU (CUDA) nếu có, fallback về CPU
- Visualize bản đồ nhiệt bằng **EigenCAM**
- Gửi **cảnh báo email** (Gmail) kèm ảnh bounding box và GradCAM khi phát hiện
- Web UI + REST API: upload ảnh, nhập URL, chạy ảnh mẫu
- Lịch sử 50 kết quả gần nhất

## Cài đặt

```bash
pip install -e .
```

Để training thêm:

```bash
pip install -e ".[training]"
```

Yêu cầu Python >= 3.11.

## Chạy web server

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Truy cập `http://localhost:8000`

## Scripts

```bash
# Inference ảnh tĩnh
python scripts/run_inference.py --image path/to/image.jpg --weights runs/detect/.../best.pt

# Webcam realtime
python scripts/run_webcam.py --device 0 --conf 0.45

# Headless simulation (video/folder)
python scripts/run_headless_sim.py --source path/to/video.mp4

# Training
python scripts/train_yolov8s.py
```

## Cấu hình email

Vào `http://localhost:8000/settings` để nhập Gmail sender, App Password và địa chỉ nhận. App Password tạo tại [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).

## Cấu trúc

```
src/fire_detection/
  inference/      # YOLOv8Detector
  web/            # FastAPI app, GradCAM, email alert
  capture/        # Camera thread
  detection/      # Temporal filter
scripts/          # CLI tools
training/         # Dataset utilities
```

## Model

Weights mặc định: `runs/detect/runs/train/dfire_v1_s_baseline/weights/best.pt`

Dataset: [D-Fire](https://github.com/gaiasd/DFireDataset) — 21.000+ ảnh, 2 lớp fire/smoke.
