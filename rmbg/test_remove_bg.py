from transformers import pipeline
from PIL import Image
import torch, requests, io

# === 1. Cihaz seçimi ===
device_id = 0 if torch.cuda.is_available() else -1
print("Device:", "cuda" if device_id==0 else "cpu")

# === 2. Pipeline yükle ===
pipe = pipeline("image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                device=device_id)

# === 3. Görseli oku ===
# image_url = "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg"
# resp = requests.get(image_url, timeout=20)
# img = Image.open(io.BytesIO(resp.content)).convert("RGB")

img = Image.open("/home/gedik/works/RMBG/data/xs-34-koton_1690798971.jpg").convert("RGB")
# === 4. Maske çıkar ===
mask_pil = pipe(img, return_mask=True)  # 0–255 arası gri tonlu maske

# === 5. Alfa uygula ===
rgba = img.copy()
rgba.putalpha(mask_pil)

# === 6. Kaydet ===
out_file = "out_no_bg4.png"
rgba.save(out_file)
print(f"Kaydedildi: {out_file}")
