#Kütüphane bazlı görüntü kalite değerlendirme
"""
Tespit ettiği bozulmalar
Bulanıklık (blur)
Gürültü (noise)
JPEG sıkıştırma artefaktları
Kontrast kaybı
Renk bozulmaları
Aşırı sıkıştırma

Skor: 0-100 arası (genellikle 0-50 arası daha yaygın)
Düşük skor = yüksek kalite (örn: 0-20 = çok iyi)
Yüksek skor = düşük kalite (örn: 40+ = kötü)

"""

from brisque import BRISQUE
import cv2
import os

data_path = "data"

for file in os.listdir(data_path):
    img = cv2.imread(os.path.join(data_path, file))
    if img is None:
        print(f"Dosya okunamadı: {file}")
        continue
    score = BRISQUE().score(img)
    print(f"Kalite skoru: {score} - {file}")
    if score > 40:
        print(f"Bu fotoğraf kalitesiz olabilir, lütfen yeniden çekin.")
    else:
        print(f"Bu fotoğraf kaliteli olabilir.")


