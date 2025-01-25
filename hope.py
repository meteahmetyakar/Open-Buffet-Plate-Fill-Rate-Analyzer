import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from segment_anything import sam_model_registry, SamPredictor
from statistics import median
import requests
import random


def overlay_mask(image, mask, color, alpha=0.5):
    """
    Maskeyi görüntünün üzerine yarı saydam olarak bindirir.
    image: Orijinal görüntü
    mask: Siyah-beyaz maske (0-255)
    color: Maskenin rengi (BGR formatında bir tuple, örn. (255, 0, 0) mavi)
    alpha: Maskenin opaklık seviyesi (0.0 ile 1.0 arasında)
    """
    if image.shape[:2] != mask.shape:
        print(f"Image shape: {image.shape[:2]}, Mask shape: {mask.shape}")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(f"Mask resized to: {mask.shape}")

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
    food_mask = cv2.resize(food_mask, (plate_mask.shape[1], plate_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Yemek maskesini sadece tabak maskesine sınırla
    intersection_mask = cv2.bitwise_and(plate_mask, food_mask)

    # Maskelerdeki beyaz alanları (1 olan pikselleri) say
    plate_area = np.sum(plate_mask == 255)  # Tabağın alanı
    food_area = np.sum(intersection_mask == 255)  # Yemeğin alanı (sadece tabakta olan kısmı)

    # Eğer tabak maskesi boşsa, doluluk oranını sıfır yap
    if plate_area == 0:
        return 0

    # Doluluk oranı hesapla
    fill_rate = (food_area / plate_area) * 100
    return fill_rate


fill_rates = {}  # Global bir liste olarak tanımlayın
def calculate_fill_rate_with_median_per_food(food_id, plate_mask, food_mask):
    global fill_rates
    fill_rate = calculate_fill_rate(plate_mask, food_mask)

    if food_id not in fill_rates:
        fill_rates[food_id] = []  # Eğer food_id için bir geçmiş yoksa, yeni bir liste başlat

    fill_rates[food_id].append(fill_rate)

    # Sadece son 10 değeri tut
    if len(fill_rates[food_id]) > 10:
        fill_rates[food_id].pop(0)

    # Medyan doluluk oranını hesapla
    smoothed_fill_rate = median(fill_rates[food_id])
    return smoothed_fill_rate

def grabcut_segmentation(frame, bbox):
    """
    GrabCut algoritmasıyla segmentasyon yapar.
    frame: Girdi görüntüsü (BGR formatında).
    bbox: ROI'nin sınırları (x, y, w, h).
    """

    height, width = frame.shape[:2]
    x, y, w, h = bbox

    # ROI koordinatlarını sınırla
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    # ROI görüntüsünü kes
    roi = frame[y:y + h, x:x + w]

    rect = (x, y, w, h)
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # GrabCut algoritmasını uygula
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)

    # Sonuç maskesini oluştur
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask * 255


def main():
    sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    if sam_checkpoint is None:
        print("SAM modeli yüklenemedi. Program sonlandırılıyor.")
        return

    yolo_model_path = "best.pt"

    # SAM modelini yükle
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to("cuda")  # GPU kullanıyorsanız
    predictor = SamPredictor(sam)

    # YOLO modelini yükle
    try:
        yolo_model = YOLO(yolo_model_path)
        print("YOLO modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"YOLO modeli yüklenirken hata oluştu: {e}")
        return

    # Kamerayı başlat
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

        # Yemeği tespit et (YOLO)
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

                food_id = class_name  # Her bir food için benzersiz bir ID oluştur
                foods.append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'food_id': food_id,
                    'fill_rate': 0.0  # İlk doluluk oranını 0 olarak başlatıyoruz
                })

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        for food in foods:
            x1, y1, x2, y2 = food['bbox']
            food_id = food['food_id']
            input_box = [x1, y1, x2, y2]
            masks, scores, logits = predictor.predict(
                box=np.array([input_box]),
                multimask_output=False
            )

            if masks is not None and len(masks) > 0:
                mask = masks[0].astype(np.uint8) * 255

                cv2.imshow("mask", mask)

                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                    center = (int(cx), int(cy))
                    radius = int(radius)

                    # Tabak maskesi oluştur (tabağın dış sınırları)
                    plate_mask = np.zeros(mask.shape, dtype=np.uint8)  # uint8 olarak oluştur
                    cv2.drawContours(plate_mask, [c], -1, 255, thickness=cv2.FILLED)

                    # ROI belirle
                    roi = frame[y1:y2, x1:x2]
                    food_bbox = (20, 20, roi.shape[1] - 40, roi.shape[0] - 40)  # Yemek için daha dar bir ROI
                    food_mask = grabcut_segmentation(roi, food_bbox)

                    # Yemek maskesini ROI'nin global koordinatlarına hizala
                    roi_h, roi_w = food_mask.shape[:2]
                    food_mask_global = np.zeros_like(frame[:, :, 0])  # Tüm çerçeve boyutunda boş bir maske
                    food_mask_global[y1:y1 + roi_h, x1:x1 + roi_w] = food_mask  # Yemek maskesini global çerçeveye yerleştir

                    # Yemek maskesini tabak maskesiyle sınırla
                    restricted_food_mask = cv2.bitwise_and(food_mask_global, plate_mask)

                    # Doluluk oranını hesapla
                    smoothed_fill_rate = calculate_fill_rate_with_median_per_food(food_id, plate_mask, food_mask)
                    food['fill_rate'] = smoothed_fill_rate

                    plate_height, plate_width = plate_mask.shape[:2]
                    annotated_frame = cv2.resize(annotated_frame, (plate_width, plate_height))


                    # Maskeleri görüntüye bindir
                    annotated_frame = overlay_mask(annotated_frame, plate_mask, color=(255, 0, 0), alpha=0.5)  # Mavi: Tabak
                    annotated_frame = overlay_mask(annotated_frame, restricted_food_mask, color=(0, 255, 0), alpha=0.5)  # Yeşil: Yemek

                    # Doluluk oranını görüntüye yaz
                    fill_rate_text = f"Fill Rate (Median): {smoothed_fill_rate:.2f}%" + " | " + food["food_id"]
                    cv2.putText(annotated_frame, fill_rate_text, (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Bounding box ve sınıf adını görüntüye çiz
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{food['class_name']} {food['confidence']:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Yemekleri çiz


        cv2.imshow('Yemek ve Tabak Tanıma - YOLOv8 & SAM', annotated_frame)

        formatted_foods = [
            {
                "name": food["class_name"],  # 'class_name' değerini 'name' olarak alıyoruz
                "fill_rate": food["fill_rate"],  # Doluluk oranını alıyoruz
                "message": f"{food['class_name']} doluluk oranı hesaplandı."  # Dinamik mesaj
            }
            for food in foods
        ]


        if not formatted_foods:
            sendmessagetosse({})
        else:
            sendmessagetosse(formatted_foods)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def sendmessagetosse(foods):
    # Sunucunun POST endpoint URL'si
    url = "http://127.0.0.1:5000/send"
    print(foods)

    # JSON verisini sunucuya gönder
    response = requests.post(url, json=foods, headers={"Content-Type": "application/json"})

    # Sunucudan gelen yanıtı yazdır
    print(response.json())



if __name__ == "__main__":
    main()