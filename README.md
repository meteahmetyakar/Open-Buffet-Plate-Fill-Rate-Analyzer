# 📌 Open Buffet Plate Fill Rate Analyzer

An AI-driven system for real-time monitoring and visualization of plate fill rates in open buffet settings, combining advanced image processing with an intuitive dashboard interface.

---

## 📖 Project Description

Open Buffet Plate Fill Rate Analyzer analyzes buffet dishes in real time to detect fullness levels and depletion trends.  
It captures video input from an overhead camera, leverages a YOLO-based detection and segmentation model to estimate fill rates, and streams structured JSON alerts via Server-Sent Events (SSE).  
A WPF dashboard client subscribes to the SSE feed, visualizes plate fill percentages, and notifies staff of low-fill or violation events.

---

## 🛠️ Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/meteahmetyakar/Open-Buffet-Plate-Fill-Rate-Analyzer.git
   cd Open-Buffet-Plate-Fill-Rate-Analyzer
   ```

2. **Set up the Python environment:**
   - Ensure Python 3.8+ and pip are installed.
   - (Optional) Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate   # Linux/macOS
     venv\Scripts\activate      # Windows
     ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics roboflow opencv-python torch sseclient
   ```

4. **Configure and build the WPF dashboard:**
   - Open `BuffetMonitoringDashboard.sln` in Visual Studio 2019 or later.
   - Restore NuGet packages and build the solution.

---

## ▶️ Usage Example

- **Start backend server:**
  ```bash
  python app.py --source 0          # Use default webcam
  python app.py --source buffet.mp4 # Or path to a video file
  ```
  - The server loads `best.pt`, processes frames, and emits SSE messages such as:
    ```json
    { "dish": "Salad", "fill_rate": 0.75, "alert": "OK" }
    ```

- **Launch dashboard client:**
  - Run `BuffetMonitoringDashboard.exe`.
  - Enter SSE endpoint (e.g., `http://localhost:8000/stream`).
  - View live fill percentages, ⚠️ warnings, and ❗ critical alerts.

---

## 📂 File Overview

| File/Folder                     | Purpose                                            | Key Elements                                 |
|---------------------------------|----------------------------------------------------|----------------------------------------------|
| `app.py`                        | SSE server for video capture and detection         | Ultralytics YOLO, Roboflow dataset, `sse.py` |
| `train.py`                      | YOLOv8 model training script                       | `data.yaml`, augmentation configs            |
| `best.pt`                       | Pre-trained YOLO weights                           | Trained on custom buffet dataset             |
| `camera-test.py`                | Camera input testing utility                       | Basic capture & preview loop                 |
| `data.yaml`                     | YOLO dataset config                                | Class labels, image paths                    |
| `README.dataset.txt`            | Dataset creation and annotation notes              | Stats & folder structure                     |
| `README.roboflow.txt`           | Roboflow export & preprocessing steps              | Export settings, augmentation recipes        |
| `runs/`                         | YOLO train/validate/segment outputs                | `train/`, `val/`, `segment/`                 |
| `sse.py`                        | SSE implementation                                 | Flask/FastAPI integration                    |
| `BuffetMonitoringDashboard/`    | WPF dashboard application                          | XAML views, notification logic               |

---

## 📸 Screenshots

- **Python App Fill Rate**
<img src="https://github.com/meteahmetyakar/Open-Buffet-Plate-Fill-Rate-Analyzer/blob/main/images/fill-rate-example.png"/>

- **Dashboard Login Page**
<img src="https://github.com/meteahmetyakar/Open-Buffet-Plate-Fill-Rate-Analyzer/blob/main/images/login-page.png"/>

- **Dashboard Fill Rate**
<img src="https://github.com/meteahmetyakar/Open-Buffet-Plate-Fill-Rate-Analyzer/blob/main/images/dashboard-example.png"/>

