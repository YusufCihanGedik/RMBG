import pyiqa
import torch
from PIL import Image
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any

QualityDecision = Literal["ACCEPT", "WARN", "REJECT"]

@dataclass
class QualityResult:
    decision: QualityDecision
    fused_score: float
    musiq_score: float
    maniqa_score: float
    brisque_score: float
    musiq_norm: float
    brisque_norm: float
    message: str

class QualityChecker:
    def __init__(
        self,
        device: str | None = None,
        musiq_weight: float = 0.6,
        brisque_weight: float = 0.4,
        accept_th: float = 0.7,
        warn_th: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.musiq_weight = musiq_weight
        self.brisque_weight = brisque_weight
        self.accept_th = accept_th
        self.warn_th = warn_th

        # pyiqa modellerini yükle
        self.musiq = pyiqa.create_metric('musiq').to(self.device)
        self.brisque = pyiqa.create_metric('brisque').to(self.device)
        self.maniqa = pyiqa.create_metric('maniqa').to(self.device)

    def _normalize_scores(self, musiq_score: float, brisque_score: float):
        """
        Basit normalizasyon:
        - MUSIQ: daha büyük = daha iyi (0-100 varsayımı)
        - BRISQUE: daha küçük = daha iyi (0-100 varsayımı)
        Bu kısmı kendi gerçek skor dağılımına göre güncelleyebilirsin.
        """
        musiq_norm = max(0.0, min(musiq_score / 100.0, 1.0))
        brisque_norm = 1.0 - max(0.0, min(brisque_score / 100.0, 1.0))
        return musiq_norm, brisque_norm

    def analyze_pil(self, img: Image.Image) -> QualityResult:
        # pyiqa, PIL Image ya da tensor alabiliyor; PIL ile devam edelim
        with torch.no_grad():
            musiq_score = float(self.musiq(img).item())
            brisque_score = float(self.brisque(img).item())
            maniqa_score = float(self.maniqa(img).item())
        musiq_norm, brisque_norm = self._normalize_scores(musiq_score, brisque_score)

        fused = (
            self.musiq_weight * musiq_norm +
            self.brisque_weight * brisque_norm
        )

        if fused >= self.accept_th:
            decision: QualityDecision = "ACCEPT"
            msg = "Görsel kalitesi yeterli, devam edebilirsiniz."
        elif fused >= self.warn_th:
            decision = "WARN"
            msg = (
                "Görsel kalitesi sınırda. Çalışacak, ancak daha net ve iyi ışıkta "
                "bir fotoğraf yüklerseniz daha iyi sonuç alırsınız."
            )
        else:
            decision = "REJECT"
            msg = (
                "Görsel kalitesi düşük görünüyor. Lütfen daha net ve iyi ışıkta "
                "yeni bir fotoğraf yükleyiniz."
            )

        return QualityResult(
            decision=decision,
            fused_score=float(fused),
            musiq_score=musiq_score,
            maniqa_score=maniqa_score,
            brisque_score=brisque_score,
            musiq_norm=float(musiq_norm),
            brisque_norm=float(brisque_norm),
            message=msg,
        )

    def analyze_file(self, path: str) -> Dict[str, Any]:
        img = Image.open(path).convert("RGB")
        result = self.analyze_pil(img)
        return asdict(result)


if __name__ == "__main__":
    qc = QualityChecker()

    result = qc.analyze_file("data/kalitesiz.jpeg")
    print(result)

    # Örnek çıktı:
    # {
    #   'decision': 'WARN',
    #   'fused_score': 0.63,
    #   'musiq_score': 58.2,
    #   'brisque_score': 42.1,
    #   'musiq_norm': 0.58,
    #   'brisque_norm': 0.58,
    #   'message': 'Görsel kalitesi sınırda. ...'
    # }

#| Model       | Türü                | Gücü                                                     | Zayıflığı             | Canlı kullanım       |
#| ----------- | ------------------- | -------------------------------------------------------- | --------------------- | -------------------- |
#| **BRISQUE** | Klasik (no-ref)     | Çok hızlı, eğitim gerekmez                               | Bazı hataları kaçırır | Mükemmel (çok hızlı) |
#| **MUSIQ**   | Transformer tabanlı | Çok kararlı, geniş bozulmaları yakalar                   | BRISQUE’den yavaş     | Uygun                |
#| **MANIQA**  | En güçlü AI modeli  | Blur, noise, exposure, compression hepsini süper yakalar | En yavaş              | GPU varsa çok uygun  |
