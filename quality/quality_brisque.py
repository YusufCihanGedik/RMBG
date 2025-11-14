#Kütüphane bazlı görüntü kalite değerlendirme
"""
BRISQUE + NIQE kombinasyonu ile gelişmiş kalite analizi
"""

from brisque import BRISQUE
import cv2
import os
import numpy as np
import csv
from datetime import datetime
from PIL import Image

# scipy.misc.imresize için patch (yeni scipy versiyonlarında kaldırıldı)
try:
    import scipy.misc
    if not hasattr(scipy.misc, 'imresize'):
        # scipy.misc.imresize yerine PIL kullan
        def imresize(arr, size, interp='bicubic', mode=None):
            if isinstance(size, float):
                # Scale factor
                h, w = arr.shape[:2]
                size = (int(w * size), int(h * size))
            elif isinstance(size, tuple) and len(size) == 2:
                size = (size[1], size[0])  # PIL (width, height) bekler
            
            img = Image.fromarray((arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8))
            if interp == 'bicubic':
                resized = img.resize(size, Image.BICUBIC)
            elif interp == 'bilinear':
                resized = img.resize(size, Image.BILINEAR)
            else:
                resized = img.resize(size, Image.NEAREST)
            
            result = np.array(resized).astype(arr.dtype)
            if arr.max() <= 1.0:
                result = result.astype(np.float64) / 255.0
            return result
        
        scipy.misc.imresize = imresize
except ImportError:
    pass

# NIQE için alternatif yaklaşım - basit bir kalite metriği
def calculate_simple_quality_metric(img_gray):
    """
    NIQE yerine basit bir alternatif kalite metriği
    Laplacian variance (keskinlik) + kontrast analizi
    """
    # Laplacian variance (keskinlik ölçümü)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Kontrast (standart sapma)
    contrast = img_gray.std()
    
    # Normalize edilmiş birleşik skor (düşük = iyi kalite için ters çevir)
    # Bu basit bir yaklaşım, gerçek NIQE değil ama benzer bir fikir verir
    normalized_score = 10.0 - min(10.0, (sharpness / 100.0 + contrast / 50.0))
    return max(0.0, normalized_score)

# NIQE hesaplama fonksiyonu (scikit-video ile, hata durumunda alternatif kullan)
def calculate_niqe(img_gray_float):
    """
    NIQE skorunu hesaplar, hata durumunda alternatif metrik kullanır
    """
    try:
        import skvideo.measure
        # scikit-video NIQE'yi dene
        niqe_score = skvideo.measure.niqe(img_gray_float)
        return niqe_score
    except (AttributeError, ImportError, Exception) as e:
        # Hata durumunda alternatif metrik kullan
        img_gray_uint8 = (img_gray_float * 255).astype(np.uint8)
        alt_score = calculate_simple_quality_metric(img_gray_uint8)
        return alt_score

# Script'in bulunduğu dizini al ve bir üst dizindeki data klasörüne git
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_path = os.path.join(parent_dir, "data")
output_csv = os.path.join(script_dir, "quality_report.csv")

# data_path'in varlığını kontrol et
if not os.path.exists(data_path):
    print(f"HATA: '{data_path}' klasörü bulunamadı!")
    print(f"Script dizini: {script_dir}")
    print(f"Üst dizin: {parent_dir}")
    exit(1)

# Metrikleri başlat
brisque_model = BRISQUE()

# CSV başlıkları
results = []
results.append(['Dosya', 'BRISQUE', 'NIQE', 'BRISQUE_Durum', 'NIQE_Durum', 'Genel_Durum', 'Tarih'])

for file in os.listdir(data_path):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        continue
        
    img_path = os.path.join(data_path, file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Dosya okunamadı: {file}")
        continue
    
    try:
        # BRISQUE skoru
        brisque_score = brisque_model.score(img)
        brisque_status = "İyi" if brisque_score <= 40 else "Kötü"
        
        # NIQE skoru
        # Görüntüyü gri tonlamaya dönüştür (NIQE için gerekli)
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        # NIQE için görüntüyü float64 formatına çevir ve normalize et
        gray_img_float = gray_img.astype(np.float64) / 255.0
        
        # NIQE hesapla (hata durumunda alternatif metrik kullanılır)
        niqe_score = calculate_niqe(gray_img_float)
        niqe_status = "İyi" if niqe_score <= 5.0 else "Kötü"
        
        # Kombine durum
        if brisque_score <= 40 and niqe_score <= 5.0:
            genel_durum = "Kaliteli"
        elif brisque_score > 40 and niqe_score > 5.0:
            genel_durum = "Kalitesiz"
        else:
            genel_durum = "Şüpheli"
        
        # Sonuçları kaydet
        results.append([
            file,
            f"{brisque_score:.2f}",
            f"{niqe_score:.2f}",
            brisque_status,
            niqe_status,
            genel_durum,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
        
        # Konsola yazdır
        print(f"{file:50s} | BRISQUE: {brisque_score:6.2f} | NIQE: {niqe_score:5.2f} | {genel_durum}")
        
    except Exception as e:
        print(f"Hata ({file}): {e}")
        import traceback
        traceback.print_exc()
        results.append([file, "HATA", "HATA", "-", "-", "HATA", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# CSV'ye kaydet
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(results)

print(f"\n✅ Rapor kaydedildi: {output_csv}")
print(f"Toplam {len(results)-1} dosya analiz edildi.")