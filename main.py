# python ./data_server/app_example.py
from flask import Flask, request, render_template
print("載入 opencv...")
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os

print("載入模型...")
model = YOLO("yolov8m.pt")
app = Flask("hackathon server")

def get_class_label(results, model):
    names = model.names
    box = results[0].boxes[0]
    index = int(box.cls[0].item())
    label = names[index]
    return label

def detect_image(source_image, save_detected=True):
    # print(os.getcwd())
    filename = f"static/img/detect/latest.jpg"
    results = model.predict(
        source_image, # 影像
        conf=0.5, # 信心門檻值
        iou=0.45, # IoU 門檻值
    )
    # 根據偵測結果畫框
    if len(results[0].boxes) == 0:
        annotated  = source_image
        label = "nothing"
    else:
        annotated = results[0].plot()
        label = get_class_label(results, model)
    # 儲存影像
    ok = cv2.imwrite(filename, annotated)
    if ok:
        print(f"偵測到 {label} 並儲存圖片")
    else:
        print("[debug] 儲存影像失敗")
    return label
    
@app.route("/esp32/capture", methods=["GET", "POST"])
def esp32_capture():
    if request.method == "GET":
        return render_template("test_upload.html")
    elif request.method == "POST":
        if "file" not in request.files:
            print("[debug] /esp32-upload: No file part")
            return 400, "No file part"
        
        file = request.files["file"]
        if file.filename == "":
            print("[debug] /esp32-upload: No selected file")
            return 400, "No selected file"
        
        img_bytes = file.read() # 讀成 byte 模式資料
        np_arr = np.frombuffer(img_bytes, np.uint8) # 轉 numpy array (後面轉格式需要)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # 轉乘 image (opencv 跟 YOLO 可以吃的)
        if img is None:
            print("[debug] /esp32-upload: cv2.imdecode failed")
            return "Decode error", 400

        # 丟給 YOLO 做偵測
        label = detect_image(img, True)
        return "ok"
    
@app.route("/yolo_display")
def yolo_display():
    return render_template("yolo_display.html")

app.run(host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False)

  
