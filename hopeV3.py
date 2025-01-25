import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from statistics import median
import requests

def overlay_mask(image, mask, color, alpha=0.5):
    """
    Maskeyi görüntünün üzerine yarı saydam olarak bindirir.
    image: Orijinal görüntü
    mask: Siyah-beyaz maske (0-255)
    color: Maskenin rengi (BGR formatında bir tuple, örn. (255, 0, 0) mavi)
    alpha: Maskenin opaklık seviyesi (0.0 ile 1.0 arasında)
    """
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask == 255] = color
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return blended

def calculate_fill_rate(plate_mask, food_mask):
    """
    Tabağın doluluk oranını hesaplar.
    plate_mask: Tabak segmentasyon maskesi (beyaz pikseller 255).
    food_mask: Yemek segmentasyon maskesi (beyaz pikseller 255).
    """
    """food_mask = cv2.resize(food_mask, (plate_mask.shape[1], plate_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection_mask = cv2.bitwise_and(plate_mask, food_mask)

    cv2.imshow("plate", plate_mask)
    cv2.imshow("food", food_mask)
    cv2.imshow("intersection_mask", intersection_mask)"""


    plate_area = np.sum(plate_mask == 255)
    food_area = np.sum(food_mask == 255)

    if plate_area == 0:
        return 0

    fill_rate = (food_area / plate_area) * 100
    return fill_rate

fill_rates = {}

def calculate_fill_rate_with_median_per_food(food_id, plate_mask, food_mask):
    global fill_rates
    fill_rate = calculate_fill_rate(plate_mask, food_mask)
    current_time = time.time()  # Şu anki zaman (saniye cinsinden)

    if food_id not in fill_rates:
        fill_rates[food_id] = []

    # Yeni kaydı ekle
    fill_rates[food_id].append((current_time, fill_rate))

    # Hızlı tükenme kontrolü için son 3 dakikadaki kayıtlar
    recent_data = [(t, r) for t, r in fill_rates[food_id] if current_time - t <= 180]

    # Tüm verilerden en fazla son 5 kaydı al
    last_5_rates = [r for _, r in fill_rates[food_id][-5:]]

    # Hızlı tükenme kontrolü
    if len(recent_data) >= 2:
        initial_time, initial_rate = recent_data[0]  # İlk kayıt
        _, latest_rate = recent_data[-1]  # Son kayıt
        rate_diff = initial_rate - latest_rate

        # Eğer düşüş %20'den fazla ise hızlı tükenme durumu
        if latest_rate > initial_rate & rate_diff > 20:
            return median(last_5_rates), True

    # Medyan hesaplaması (son 5 kaydın medyanı)
    return median(last_5_rates), False



def grabcut_segmentation(frame, bbox):
    """
    GrabCut algoritmasıyla segmentasyon yapar.
    frame: Girdi görüntüsü (BGR formatında).
    bbox: ROI'nin sınırları (x, y, w, h).
    """
    height, width = frame.shape[:2]
    x, y, w, h = bbox

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (x, y, w, h)
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask * 255

def create_plate_mask(frame, bbox):
    """
    Tabağın maskesini oluşturmak için bounding box içine tam oturan bir dairesel maske yerleştirir.
    frame: Girdi görüntüsü (BGR formatında).
    bbox: ROI'nin sınırları (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    radius = min((x2 - x1), (y2 - y1)) // 2

    plate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(plate_mask, (center_x, center_y), radius, 255, thickness=cv2.FILLED)
    return plate_mask

def main():
    yolo_model_path = "best.pt"

    try:
        yolo_model = YOLO(yolo_model_path)
        print("YOLO modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"YOLO modeli yüklenirken hata oluştu: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Model testi başladı. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare okunamadı!")
            break

        results = yolo_model(frame)
        annotated_frame = frame.copy()

        foods = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = yolo_model.names.get(cls, "Unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                food_id = class_name
                foods.append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'food_id': food_id,
                    'fill_rate': 0.0,
                    'message': ""
                })

        for food in foods:
            x1, y1, x2, y2 = food['bbox']
            food_id = food['food_id']

            plate_mask = create_plate_mask(frame, (x1, y1, x2, y2))

            roi = frame[y1:y2, x1:x2]
            food_bbox = (20, 20, roi.shape[1] - 40, roi.shape[0] - 40)
            food_mask = grabcut_segmentation(roi, food_bbox)

            roi_h, roi_w = food_mask.shape[:2]
            food_mask_global = np.zeros_like(frame[:, :, 0])
            food_mask_global[y1:y1 + roi_h, x1:x1 + roi_w] = food_mask

            restricted_food_mask = cv2.bitwise_and(food_mask_global, plate_mask)
            smoothed_fill_rate, is_fast_depleting = calculate_fill_rate_with_median_per_food(food_id, plate_mask, food_mask)
            food['fill_rate'] = smoothed_fill_rate

            if is_fast_depleting:
                food['message'] = f"{food['class_name']} is running out faster than expected."

            if smoothed_fill_rate <= 20:
                food['message'] = f"{food['class_name']} supply has reached a critical threshold; a refill is recommended."


            annotated_frame = overlay_mask(annotated_frame, plate_mask, color=(255, 0, 0), alpha=0.5)
            annotated_frame = overlay_mask(annotated_frame, restricted_food_mask, color=(0, 255, 0), alpha=0.5)

            fill_rate_text = f"Fill Rate (Median): {smoothed_fill_rate:.2f}% | {food['food_id']}"
            cv2.putText(annotated_frame, fill_rate_text, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{food['class_name']} {food['confidence']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Yemek ve Tabak Tanıma - YOLOv8 & GrabCut', annotated_frame)

        formatted_foods = [
            {
                "name": food["class_name"],
                "fill_rate": food["fill_rate"],
                "message":  food['message']
            }
            for food in foods
        ]

        sendmessagetosse(formatted_foods if formatted_foods else {})

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def sendmessagetosse(foods):
    url = "http://127.0.0.1:5000/send"
    if len(foods) > 0:
        print(foods[0]['message'])
    response = requests.post(url, json=foods, headers={"Content-Type": "application/json"})


if __name__ == "__main__":
    main()
