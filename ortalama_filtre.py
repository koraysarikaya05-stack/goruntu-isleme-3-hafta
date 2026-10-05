import cv2
import numpy as np
import os

# 1. Dosya yollarını belirle
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'resim.jpg')
output_path = os.path.join(script_dir, 'gurultu_temizlenmis.jpg')

def resmi_oku_gri(dosya_yolu):
    try:
        data = np.fromfile(dosya_yolu, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

def resmi_kaydet(dosya_yolu, resim_matrisi):
    try:
        uzanti = os.path.splitext(dosya_yolu)[1]
        basarili, tampon = cv2.imencode(uzanti, resim_matrisi)
        if basarili:
            tampon.tofile(dosya_yolu)
            return True
        return False
    except Exception:
        return False

#  UYGULAMA 
img = resmi_oku_gri(input_path)

if img is None:
    print(f"❌ Resim bulunamadı: {input_path}")
else:
    print("⏳ Konvolüsyon işlemi hesaplanıyor, lütfen bekleyin...")

    # Kenar doldurma (Padding)
    padded = np.pad(img.astype(np.float32), pad_width=1, mode='edge')

    # HIZLI MANUEL KONVOLÜSYON (3x3 Komşulukların Vektörel Toplamı)
    # 3x3'lük matristeki 9 komşu piksel kaydırılarak toplanır ve 9'a bölünür
    toplam = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2]   + padded[2:, 1:-1]   + padded[2:, 2:]
    ) / 9.0

    output_img = np.clip(toplam, 0, 255).astype(np.uint8)

    # Sonucu Kaydet
    if resmi_kaydet(output_path, output_img):
        print("✅ İŞLEM BAŞARIYLA TAMAMLANDI!")
        print("• 3x3 Ortalama filtresi konvolüsyon ile uygulandı.")
        print("• Kaydedilen Dosya: 'gurultu_temizlenmis.jpg'")
    else:
        print("❌ Resim kaydedilirken hata oluştu.")