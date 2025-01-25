import cv2
from ultralytics import YOLO

# 1. Modeli yükleyin
model = YOLO("best.pt")  # Eğitilmiş YOLOv8 modelinizi buraya koyun

# 2. Kamera akışını başlatın
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera içindir; başka bir kamera için ilgili indexi kullanın

# 3. Kamera akışını işle
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamıyor.")
        break

    # YOLOv8 modeliyle tahmin yap
    results = model(frame, verbose=False)  # Görüntü üzerinde algılama yap

    # Algılama sonuçlarını çiz
    annotated_frame = results[0].plot()  # Sonuçları çizilmiş şekilde al

    # Sonuçları görüntüle
    cv2.imshow("YOLOv8 Live", annotated_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
