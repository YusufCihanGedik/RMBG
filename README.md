# RMBG - AI Powered Background Removal Tool

Bu proje, yapay zeka tabanlı görüntü segmentasyonu kullanarak fotoğraflardan arka planı otomatik olarak kaldıran güçlü bir Python aracıdır. BRIA AI'nin RMBG-1.4 ve RMBG-2.0 modellerini destekler.

## 🎯 Amaç

RMBG aracı şu amaçlarla geliştirilmiştir:
- **E-ticaret**: Ürün fotoğraflarından arka planı kaldırarak profesyonel katalog görselleri oluşturma
- **Sosyal Medya**: Profil fotoğrafları ve içerik görselleri için temiz arka plan kaldırma
- **Grafik Tasarım**: Tasarım projelerinde kullanılmak üzere şeffaf arka planlı görseller hazırlama
- **Toplu İşlem**: Yüzlerce görsel dosyasını otomatik olarak işleme

## ✨ Özellikler

### 🤖 Çift Model Desteği
- **RMBG-1.4**: Hızlı ve etkili, genel kullanım için optimize edilmiş
- **RMBG-2.0**: Daha yüksek kalite, detaylı segmentasyon için gelişmiş

### 🎨 Gelişmiş Post-Processing
- **Feather**: Maske kenarlarını yumuşatma (0-3 seviye)
- **Morph**: Küçük delikleri kapatma ve gürültü temizleme (0-3 seviye)
- **Threshold**: İkili maske veya grayscale alpha seçimi

### 🎭 Arka Plan Seçenekleri
- **Şeffaf PNG**: Tamamen şeffaf arka plan
- **Düz Renk**: Özel renk kodları ile düz arka plan (örn: `solid:#ffffff`)

### ⚡ Performans Optimizasyonları
- **GPU Desteği**: CUDA ile hızlandırılmış işlem
- **Half Precision**: RMBG-2.0 için yarım hassasiyet desteği
- **Toplu İşlem**: Klasör bazında otomatik işleme

## 🛠️ Kurulum

### Gereksinimler
```bash
pip install torch torchvision transformers pillow numpy opencv-python tqdm
```

### CUDA Desteği (Opsiyonel)
GPU hızlandırması için PyTorch CUDA versiyonunu yükleyin:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Kullanım

### Temel Kullanım

#### Tek Dosya İşleme
```bash
python test_remove_bg2.py -i input.jpg -o output.png
```

#### Klasör İşleme
```bash
python test_remove_bg2.py -i ./input_folder -o ./output_folder
```

### Gelişmiş Parametreler

#### Model Seçimi
```bash
# RMBG-1.4 (varsayılan)
python test_remove_bg2.py -i input.jpg -o output.png --version 1.4

# RMBG-2.0 (daha yüksek kalite)
python test_remove_bg2.py -i input.jpg -o output.png --version 2.0
```

#### Post-Processing Ayarları
```bash
# Kenar yumuşatma
python test_remove_bg2.py -i input.jpg -o output.png --feather 2

# Morfolojik temizleme
python test_remove_bg2.py -i input.jpg -o output.png --morph 1

# İkili maske (keskin kenarlar)
python test_remove_bg2.py -i input.jpg -o output.png --threshold 0.5
```

#### Arka Plan Seçenekleri
```bash
# Şeffaf arka plan (varsayılan)
python test_remove_bg2.py -i input.jpg -o output.png

# Beyaz arka plan
python test_remove_bg2.py -i input.jpg -o output.png --bg solid:#ffffff

# Siyah arka plan
python test_remove_bg2.py -i input.jpg -o output.png --bg solid:#000000
```

#### Dosya Formatı Filtreleme
```bash
# Sadece JPG ve PNG dosyalarını işle
python test_remove_bg2.py -i ./photos -o ./output --pattern "*.jpg,*.png"
```

### Komple Örnek
```bash
python test_remove_bg2.py \
    -i ./product_photos \
    -o ./processed_photos \
    --version 2.0 \
    --feather 1 \
    --morph 1 \
    --bg solid:#f8f8f8 \
    --pattern "*.jpg,*.jpeg,*.png"
```

## 📁 Proje Yapısı

```
RMBG/
├── test_remove_bg2.py      # Ana uygulama
├── test_remove_bg.py       # Basit test scripti
├── scrape_images.py        # Web'den görsel indirme aracı
├── data/                   # Test görselleri
├── README.md              # Bu dosya
└── requirements.txt       # Bağımlılıklar (oluşturulacak)
```

## 🔧 Teknik Detaylar

### Model Mimarileri
- **RMBG-1.4**: Transformers pipeline tabanlı, hızlı inference
- **RMBG-2.0**: AutoModelForImageSegmentation, yüksek kalite segmentasyon

### Desteklenen Formatlar
- **Giriş**: JPG, JPEG, PNG, WebP
- **Çıkış**: PNG (şeffaf), JPG/PNG (düz arka plan)

### Performans
- **GPU**: ~2-5 saniye/görsel (boyuta bağlı)
- **CPU**: ~10-30 saniye/görsel (boyuta bağlı)

## 🚀 Gelecek Geliştirmeler

- [ ] Batch processing için daha iyi bellek yönetimi
- [ ] Web arayüzü (Gradio/Streamlit)
- [ ] Docker container desteği
- [ ] API endpoint'leri
- [ ] Daha fazla arka plan seçeneği (gradient, pattern)

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- [BRIA AI](https://www.bria.ai/) - RMBG modelleri için
- [Hugging Face](https://huggingface.co/) - Model hosting ve transformers kütüphanesi için
- [PyTorch](https://pytorch.org/) - Deep learning framework için

## 📞 İletişim

Sorularınız veya önerileriniz için issue açabilir veya pull request gönderebilirsiniz.

---

**Not**: Bu araç ticari kullanım için uygundur. Model lisansları için BRIA AI'nin şartlarını kontrol edin.