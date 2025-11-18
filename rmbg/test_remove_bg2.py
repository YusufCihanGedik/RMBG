# rmbg_tool.py
from __future__ import annotations
import argparse, pathlib, sys, io, os
from typing import Optional, Tuple
from PIL import Image, ImageColor, ImageFilter
import numpy as np
import torch
from tqdm import tqdm

# v1.4 için
from transformers import pipeline

# v2.0 için
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

def _device_id():
    return 0 if torch.cuda.is_available() else -1

# -------- v1.4 loader ----------
def load_rmbg14():
    dev = _device_id()
    return pipeline("image-segmentation",
                    model="briaai/RMBG-1.4",
                    trust_remote_code=True,
                    device=dev)

def mask_with_rmbg14(pipe, img: Image.Image) -> Image.Image:
    # 0-255 gri tonlu PIL maske
    return pipe(img, return_mask=True)

# -------- v2.0 loader ----------
class RMBG20:
    def __init__(self, half: bool = True, size: Tuple[int,int]=(1024,1024)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        ).eval().to(self.device)
        self.image_size = size
        self.half = half and self.device == "cuda"
        self.tf = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> Image.Image:
        # Girdi hazırla
        tin = self.tf(img).unsqueeze(0).to(self.device)
        # Yarım hassasiyet hız için
        if self.half:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = self.model(tin)[-1].sigmoid().cpu()
        else:
            pred = self.model(tin)[-1].sigmoid().cpu()
        pred2d = pred[0].squeeze()  # HxW 0..1
        # Orijinal boyuta ölçekle
        mask_pil = transforms.ToPILImage()(pred2d)
        mask_pil = mask_pil.resize(img.size, Image.BILINEAR)
        # 0..255
        return (np.array(mask_pil)*255).astype(np.uint8)
# --------------------------------

def postprocess_mask(mask: Image.Image|np.ndarray,
                     feather: int=0,
                     morph: int=0,
                     threshold: Optional[float]=None) -> Image.Image:
    """
    feather: 0-3 (Gaussian blur benzeri yumuşatma)
    morph:   0-3 (küçük delikleri kapatmak için morfolojik close/open)
    threshold: None -> grayscale alpha; 0.0-1.0 -> ikili maske
    """
    if isinstance(mask, Image.Image):
        m = np.array(mask)
    else:
        m = mask
    m = m.astype(np.uint8)

    # Morphological close+open
    if morph > 0:
        import cv2
        k = max(1, 2*morph+1)
        kernel = np.ones((k,k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    # Feather (blur)
    m_img = Image.fromarray(m)
    for _ in range(feather):
        m_img = m_img.filter(ImageFilter.GaussianBlur(radius=1.2))

    if threshold is not None:
        t = int(max(0,min(1,threshold))*255)
        m_img = m_img.point(lambda p: 255 if p >= t else 0)

    return m_img

def compose_output(img: Image.Image, alpha: Image.Image,
                   bg: Optional[str]) -> Image.Image:
    """
    bg=None -> şeffaf PNG (RGBA)
    bg='solid:#RRGGBB' -> düz renk zemin (RGB)
    """
    if bg is None:
        rgba = img.convert("RGBA")
        rgba.putalpha(alpha)
        return rgba
    elif bg.startswith("solid:"):
        color = ImageColor.getrgb(bg.split("solid:",1)[1])
        bg_img = Image.new("RGB", img.size, color)
        # Alfa ile birleştir
        fg = img.convert("RGBA")
        fg.putalpha(alpha)
        comp = Image.alpha_composite(bg_img.convert("RGBA"), fg)
        return comp.convert("RGB")
    else:
        raise ValueError("Desteklenmeyen bg seçeneği. Örn: solid:#ffffff")

def process_one(model_kind: str,
                model_obj,
                in_path: pathlib.Path,
                out_path: pathlib.Path,
                feather: int,
                morph: int,
                threshold: Optional[float],
                bg: Optional[str]):
    img = Image.open(in_path).convert("RGB")

    if model_kind == "1.4":
        mask = mask_with_rmbg14(model_obj, img)
    else:
        mask = model_obj(img)  # np.uint8 0..255
        mask = Image.fromarray(mask)

    mask_pp = postprocess_mask(mask, feather=feather, morph=morph, threshold=threshold)
    out = compose_output(img, mask_pp, bg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Şeffaf ise PNG, düz zemin ise JPG/PNG serbest
    if out.mode == "RGBA" or bg is None:
        out_path = out_path.with_suffix(".png")
    out.save(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True, help="Dosya veya klasör")
    ap.add_argument("-o","--output", required=True, help="Dosya veya klasör")
    ap.add_argument("--version", choices=["1.4","2.0"], default="1.4",
                    help="Model versiyonu")
    ap.add_argument("--feather", type=int, default=0, help="0-3")
    ap.add_argument("--morph", type=int, default=0, help="0-3")
    ap.add_argument("--threshold", type=float, default=None,
                    help="0.0-1.0 (ikili maske), None=grayscale alpha")
    ap.add_argument("--bg", type=str, default=None,
                    help="None=şeffaf, örn: solid:#ffffff")
    ap.add_argument("--pattern", default="*.jpg,*.jpeg,*.png,*.webp",
                    help="Klasör modu için glob desenleri")
    args = ap.parse_args()

    # Model yükle
    if args.version == "1.4":
        model = load_rmbg14()
    else:
        model = RMBG20(half=True)

    in_p = pathlib.Path(args.input)
    out_p = pathlib.Path(args.output)

    if in_p.is_file():
        out_file = out_p if out_p.suffix else out_p / (in_p.stem + "_rmbg.png")
        p = process_one(args.version, model, in_p, out_file, args.feather, args.morph,
                        args.threshold, args.bg)
        print("OK ->", p)
    else:
        patterns = [p.strip() for p in args.pattern.split(",")]
        files = []
        for pat in patterns:
            files.extend(in_p.rglob(pat))
        if not files:
            print("Girdi klasöründe eşleşen görsel bulunamadı.")
            sys.exit(1)
        for f in tqdm(sorted(files), desc=f"RMBG {args.version}"):
            rel = f.relative_to(in_p)
            out_file = (out_p / rel).with_suffix(".png")
            try:
                process_one(args.version, model, f, out_file, args.feather, args.morph,
                            args.threshold, args.bg)
            except Exception as e:
                print(f"[WARN] {f}: {e}")

if __name__ == "__main__":
    main()
