"""MUSIQ – Multi-Scale Image Quality Transformer Brisque Yöntemi ile görüntü kalitesi değerlendirme"""
import pyiqa
import torch
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modeli oluştur
musiq = pyiqa.create_metric('musiq').to(device)
brisque = pyiqa.create_metric('brisque').to(device)

# Görseli yükle
img = Image.open('data/kalitesiz.jpeg').convert('RGB')

# Skorları hesapla
musiq_score = musiq(img).item()
brisque_score = brisque(img).item()

print('MUSIQ:', musiq_score) #Büyük daha iyi kaliteyi ifade eder
print('BRISQUE:', brisque_score) #Küçük daha iyi kaliteyi ifade eder