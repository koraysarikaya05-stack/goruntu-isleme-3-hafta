import cv2
import numpy as np
import os

# 1. Dosya yollarını belirle (Kodun bulunduğu klasör)
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'resim.jpg')
output_path = os.path.join(script_dir, 'histogram_esitlenmis1.jpg')

# Türkçe karakter içeren klasör yollarından resmi güvenle okuma ve kaydetme
def resmi_oku_gri(dosya_yolu):
    try:
        data = np.fromfile(dosya_yolu, dtype=np.uint8)
        # cv2.IMREAD_GRAYSCALE -> Resmi TEK KANALLI (Gri tonlamalı) olarak açar
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

# --- UYGULAMA BAŞLANGICI ---
img = resmi_oku_gri(input_path)

if img is None:
    print(f"❌ Resim bulunamadı: {input_path}")
else:
    yukseklik, genislik = img.shape
    toplam_piksel = yukseklik * genislik

    # -------------------------------------------------------------
    # ADIM 1: 256-DEĞERLİ HISTOGRAM HESABI (Hazır fonksiyon yok)
    # -------------------------------------------------------------
    histogram = [0] * 256
    for i in range(yukseklik):
        for j in range(genislik):
            piksel_degeri = img[i, j]
            histogram[piksel_degeri] += 1

    # -------------------------------------------------------------
    # ADIM 2: KÜMÜLATİF DAĞILIM FONKSİYONU (CDF) (Hazır fonksiyon yok)
    # -------------------------------------------------------------
    cdf = [0] * 256
    birikimli_toplam = 0
    for i in range(256):
        birikimli_toplam += histogram[i]
        cdf[i] = birikimli_toplam

    # -------------------------------------------------------------
    # ADIM 3: HISTOGRAM EŞİTLEME DÖNÜŞÜMÜ (Hazır fonksiyon yok)
    # Formül: T(v) = round( (CDF(v) - CDF_min) / (Toplam_Piksel - CDF_min) * 255 )
    # -------------------------------------------------------------
    
    # Sıfırdan büyük ilk CDF değerini (CDF_min) bul
    cdf_min = 0
    for val in cdf:
        if val > 0:
            cdf_min = val
            break

    # Dönüşüm (Mapping / Lookup Table) tablosunu oluştur
    lut = [0] * 256
    for i in range(256):
        if cdf[i] < cdf_min:
            lut[i] = 0
        else:
            pay = cdf[i] - cdf_min
            payda = toplam_piksel - cdf_min
            lut[i] = int(round((pay / payda) * 255))

    # -------------------------------------------------------------
    # ADIM 4: YENİ RESMİ OLUŞTURMA VE KAYDETME
    # -------------------------------------------------------------
    esitlenmis_img = np.zeros((yukseklik, genislik), dtype=np.uint8)
    for i in range(yukseklik):
        for j in range(genislik):
            esitlenmis_img[i, j] = lut[img[i, j]]

    if resmi_kaydet(output_path, esitlenmis_img):
        print("✅ İŞLEM BAŞARIYLA TAMAMLANDI!")
        print(f"Çıktı resmi kaydedildi: 'histogram_esitlenmis.jpg'")
    else:
        print("❌ Resim kaydedilirken hata oluştu.")