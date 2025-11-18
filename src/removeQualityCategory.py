"""
Basit ve fonksiyonel görüntü işleme modülü
- Quality Check (PyIQA/MANIQA)
- Fashion Category (FashionCLIP)
- Background Removal (RMBG-1.4)
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False

try:
    from fashion_clip.fashion_clip import FashionCLIP
    FASHIONCLIP_AVAILABLE = True
except ImportError:
    FASHIONCLIP_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# 1. QUALITY CHECK
# ============================================================================

def quality_check(image_path: str, device: str = "cpu") -> str:
    """
    Görüntü kalitesini kontrol eder.
    
    Args:
        image_path: Görüntü dosya yolu
        device: 'cpu' veya 'cuda'
    
    Returns:
        "ACCEPT", "WARN", "REJECT" veya "ERROR"
    """
    if not PYIQA_AVAILABLE or not TORCH_AVAILABLE:
        return "ERROR"
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        maniqa = pyiqa.create_metric('maniqa', device=torch.device(device))
        image = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():
            score = float(maniqa(image))
        
        del maniqa
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if score < 0.4:
            return "REJECT"
        elif score < 0.5:
            return "WARN"
        else:
            return "ACCEPT"
            
    except Exception as e:
        print(f"Quality check error: {e}")
        return "ERROR"


# ============================================================================
# 2. FASHION CATEGORY
# ============================================================================

FASHION_CATEGORIES = [
    "t-shirt", "shirt", "dress", "pants", 
    "skirt", "hoodie", "jacket", "coat"
]

def fashion_category(image_path: str, categories: list = None) -> Dict[str, Any]:
    """
    Görüntüdeki kıyafet kategorisini tahmin eder.
    
    Args:
        image_path: Görüntü dosya yolu
        categories: Kategori listesi (default: FASHION_CATEGORIES)
    
    Returns:
        {"category": str, "score": float, "all_scores": dict} veya "ERROR"
    """
    if not FASHIONCLIP_AVAILABLE:
        return "ERROR"
    
    if categories is None:
        categories = FASHION_CATEGORIES
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        fclip = FashionCLIP("fashion-clip")
        image = Image.open(image_path).convert("RGB")
        
        # Embedding'leri hesapla
        image_emb = fclip.encode_images([image], batch_size=1)[0]
        text_emb = fclip.encode_text(categories, batch_size=len(categories))
        
        # Kosinüs benzerliği
        similarities = np.dot(text_emb, image_emb) / (
            np.linalg.norm(text_emb, axis=1) * np.linalg.norm(image_emb)
        )
        
        best_idx = int(np.argmax(similarities))
        
        result = {
            "category": categories[best_idx],
            "score": float(similarities[best_idx]),
            "all_scores": {cat: float(sim) for cat, sim in zip(categories, similarities)}
        }
        
        del fclip
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"Fashion category error: {e}")
        return "ERROR"


# ============================================================================
# 3. BACKGROUND REMOVAL
# ============================================================================

def remove_background(image_path: str, return_mask: bool = False) -> Optional[Image.Image]:
    """
    Görüntüden arka planı kaldırır (RMBG-1.4 pipeline metodu).
    
    Args:
        image_path: Görüntü dosya yolu
        return_mask: True ise sadece maske döner, False ise şeffaf PNG
    
    Returns:
        PIL Image (RGBA veya L mode) veya None (hata durumunda)
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        print("Transformers or torch not available")
        return None
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Pipeline metodu (test_remove_bg2.py'den)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
            device=device
        )
        
        image = Image.open(image_path).convert("RGB")
        mask = pipe(image, return_mask=True)
        
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_mask:
            return mask
        
        # Şeffaf PNG oluştur
        rgba = image.convert("RGBA")
        rgba.putalpha(mask)
        return rgba
        
    except Exception as e:
        # Pipeline başarısız olursa, direkt import dene
        print(f"Pipeline failed, trying direct import...")
        return _remove_background_direct(image_path, return_mask)


def _remove_background_direct(image_path: str, return_mask: bool = False) -> Optional[Image.Image]:
    """
    Alternatif metod: Doğrudan model import (pipeline çalışmazsa).
    """
    try:
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4",
            trust_remote_code=True
        ).eval().to(device)
        
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)[0]
            mask_array = (output[0] > 0).cpu().numpy().astype(np.uint8) * 255
            mask = Image.fromarray(mask_array).resize(image.size, Image.LANCZOS)
        
        del model, input_tensor, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_mask:
            return mask
        
        rgba = image.convert("RGBA")
        rgba.putalpha(mask)
        return rgba
        
    except Exception as e:
        print(f"Direct method also failed: {e}")
        return None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Module Status")
    print("=" * 60)
    print(f"PyIQA:       {PYIQA_AVAILABLE}")
    print(f"FashionCLIP: {FASHIONCLIP_AVAILABLE}")
    print(f"Transformers: {TRANSFORMERS_AVAILABLE}")
    print(f"Torch:       {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"CUDA:        {torch.cuda.is_available()}")
    
    test_image = "data/28-lc-waikiki_1690125527.jpg"
    
    if not os.path.exists(test_image):
        print(f"\n⚠ Test image not found: {test_image}")
        exit(1)
    
    print("\n" + "=" * 60)
    print(f"Testing: {test_image}")
    print("=" * 60)
    
    # Test 1: Fashion Category
    if FASHIONCLIP_AVAILABLE:
        print("\n1. Fashion Category")
        result = fashion_category(test_image)
        if isinstance(result, dict):
            print(f"   Category: {result['category']}")
            print(f"   Score: {result['score']:.3f}")
        else:
            print(f"   Result: {result}")
    
    # Test 2: Quality Check
    if PYIQA_AVAILABLE:
        print("\n2. Quality Check")
        result = quality_check(test_image, device="cpu")
        print(f"   Result: {result}")
    
    # Test 3: Background Removal
    if TRANSFORMERS_AVAILABLE:
        print("\n3. Background Removal")
        result = remove_background(test_image, return_mask=True)
        if result:
            print(f"   Result: {result.mode} image {result.size}")
            # İsterseniz kaydet: result.save("output_mask.png")
        else:
            print("   Result: ERROR")
    
    print("\n" + "=" * 60)
    print("✓ Tests completed")
    print("=" * 60)
