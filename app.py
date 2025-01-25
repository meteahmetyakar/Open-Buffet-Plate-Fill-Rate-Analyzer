import os
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import requests

def download_model(url, save_path):
    """
    Model dosyasını indirir ve belirlenen yola kaydeder.
    """
    print(f"Model indiriliyor: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Model başarıyla indirildi: {save_path}")
    else:
        raise Exception(f"Model indirilemedi. HTTP Hatası: {response.status_code}")


def calculate_fill_rate(plate_mask, food_mask):
    """
    Tabağın doluluk oranını hesaplar.
    plate_mask: Tabak segmentasyon maskesi (SAM modeli tarafından oluşturulur).
    food_mask: Yemek maskesi (YOLO ve SAM verileriyle).
    """
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

def overlay_mask(image, mask, color, alpha=0.5):
    """
    Maskeyi görüntünün üzerine yarı saydam olarak bindirir.
    image: Orijinal görüntü
    mask: Siyah-beyaz maske (0-255)
    color: Maskenin rengi (BGR formatında bir tuple, örn. (255, 0, 0) mavi)
    alpha: Maskenin opaklık seviyesi (0.0 ile 1.0 arasında)
    """
    # Maskeyi renkli hale getir
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask == 255] = color

    # Yarı saydamlıkla birleştir
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return blended

def main():
    # SAM modelini kontrol et ve yükle
    sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    if sam_checkpoint is None:
        print("SAM modeli yüklenemedi. Program sonlandırılıyor.")
        return

    # YOLO modelini kontrol et ve yükle
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

    # Kamerayı başlatın
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı! Farklı bir kamera indeksini deneyin.")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Hiçbir kamera açılamadı!")
            return
        else:
            print("1 numaralı kamera kullanılıyor.")
    else:
        print("0 numaralı kamera kullanılıyor.")

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

                foods.append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })

        # SAM kullanarak plate tespiti
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        for food in foods:
            x1, y1, x2, y2 = food['bbox']
            input_box = [x1, y1, x2, y2]
            masks, scores, logits = predictor.predict(
                box=np.array([input_box]),
                multimask_output=False
            )

            if masks is not None and len(masks) > 0:
                mask = masks[0]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                    center = (int(cx), int(cy))
                    radius = int(radius)

                    # Tabak maskesi oluştur (tabağın dış sınırları)
                    plate_mask = np.zeros(mask.shape, dtype=np.uint8)  # uint8 olarak oluştur
                    cv2.drawContours(plate_mask, [c], -1, 255, thickness=cv2.FILLED)

                    # Yemeğin maskesini oluştur (yemek sınırları)
                    food_mask = np.zeros(mask.shape, dtype=np.uint8)  # uint8 olarak oluştur
                    cv2.rectangle(food_mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

                    # Yemek maskesini tabak maskesiyle sınırla
                    intersection_mask = cv2.bitwise_and(plate_mask, food_mask)

                    # Maskeleri yarı saydam olarak görüntünün üzerine bindir
                    annotated_frame = overlay_mask(annotated_frame, plate_mask, color=(255, 0, 0), alpha=0.5)  # Mavi: Tabak
                    annotated_frame = overlay_mask(annotated_frame, intersection_mask, color=(0, 255, 0), alpha=0.5)  # Yeşil: Yemek

                    # Doluluk oranını hesapla
                    fill_rate = calculate_fill_rate(plate_mask, intersection_mask)

                    # Doluluk oranını yazdır
                    fill_rate_text = f"Plate - Fill Rate : {fill_rate:.2f}%"
                    cv2.putText(annotated_frame, fill_rate_text, (center[0] - radius, center[1] - radius - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    cv2.circle(annotated_frame, center, radius, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, "Plate", (center[0] - radius, center[1] - radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Yemekleri çiz
        for food in foods:
            x1, y1, x2, y2 = food['bbox']
            class_name = food['class_name']
            confidence = food['confidence']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Yemek ve Tabak Tanıma - YOLOv8 & SAM', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()