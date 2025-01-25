from ultralytics import YOLO

def main():
    # Modeli yükleme
    model = YOLO('yolov8m.pt')  # YOLOv8'in önceden eğitilmiş segmentation modeli

    # Modeli eğitme
    results = model.train(
        data='data.yaml',  # YAML dosyasının yolu
        epochs=300,                  # Eğitim döngüsü sayısı
        batch=8,                   # Batch boyutu
    )

    # Eğitimden sonra doğrulama
    metrics = model.val()

    # Modeli kaydetme
    model.save('best.pt')

if __name__ == '__main__':
    main()
