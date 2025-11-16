"""
FashionCLIP Test Scripti
Adım 2: Model yükleme ve test
"""

from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image
import numpy as np
import os

print("=" * 60)
print("🟢 Adım 2: FashionCLIP Modelini Yükleme")
print("=" * 60)

# Modeli yükle
try:
    print("\n📥 FashionCLIP modeli yükleniyor...")
    fclip = FashionCLIP("fashion-clip")
    print("✅ Model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Hata: {e}")
    exit(1)

print("\n" + "=" * 60)
print("🟢 Adım 3: Örnek Görsel ile Test")
print("=" * 60)

# Test için kategori listesi
categories = [
    "t-shirt",
    "shirt",
    "dress",
    "pants",
    "skirt",
    "hoodie",
    "jacket",
    "coat"
]

# Data klasöründeki bir görseli test et
data_path = "data"
image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"❌ {data_path} klasöründe görsel bulunamadı!")
    exit(1)

# İlk görseli kullan
#test_image_path = os.path.join(data_path, image_files[0])
test_image_path = "data/m-38-diger_1690307629.jpg"
print(f"\n📸 Test görseli: {test_image_path}")

try:
    # Görseli yükle
    image = Image.open(test_image_path).convert("RGB")
    print(f"✅ Görsel yüklendi: {image.size[0]}x{image.size[1]} piksel")
    
    # Embedding'leri hesapla
    print("\n🔄 Embedding'ler hesaplanıyor...")
    image_embeddings = fclip.encode_images([image], batch_size=1)
    text_embeddings = fclip.encode_text(categories, batch_size=len(categories))
    
    # Kosinüs benzerliği hesapla
    img_emb = image_embeddings[0]  # (D,)
    similarities = np.dot(text_embeddings, img_emb) / (
        np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(img_emb)
    )
    
    # En yüksek skorlu kategoriyi bul
    best_idx = int(np.argmax(similarities))
    predicted_category = categories[best_idx]
    best_score = float(similarities[best_idx])
    
    # Sonuçları yazdır
    print("\n" + "=" * 60)
    print("📊 SONUÇLAR")
    print("=" * 60)
    print(f"\n🎯 Tahmin edilen kategori: {predicted_category} (skor: {best_score:.3f})")
    
    print("\n📈 Tüm kategori benzerlikleri:")
    print("-" * 60)
    # Skorlara göre sırala
    sorted_results = sorted(zip(categories, similarities), key=lambda x: x[1], reverse=True)
    for i, (cat, sim) in enumerate(sorted_results, 1):
        marker = "🏆" if i == 1 else "  "
        print(f"{marker} {i}. {cat:15s}: {sim:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Test tamamlandı!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Hata: {e}")
    import traceback
    traceback.print_exc()

