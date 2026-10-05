import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'reim.jpg')
output_path = os.path.join(script_dir, 'reim_kontrastli.jpg')

def resmi_oku(dosya_yolu):
    try:
        data = np.fromfile(dosya_yolu, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except:
        return None

def resmi_kaydet(dosya_yolu, resim_matrisi):
    try:
        uzanti = os.path.splitext(dosya_yolu)[1]
        basarili, tampon = cv2.imencode(uzanti, resim_matrisi)
        if basarili:
            tampon.tofile(dosya_yolu)
            return True
        return False
    except:
        return False

image = resmi_oku(input_path)

if image is not None:
    # YUV uzayına geç
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv_img[:, :, 0].astype(np.float32)

    # Uç noktadaki gürültüleri/yazıları atlamak için %2 ve %98 yüzdelik dilimleri min-max alıyoruz
    p_min, p_max = np.percentile(y_channel, (2, 98))

    print(f"Hesaplanan Min (2. Yüzdelik): {p_min}, Max (98. Yüzdelik): {p_max}")

    # Doğrusal Germe Formülü
    y_stretched = ((y_channel - p_min) / (p_max - p_min)) * 255.0
    yuv_img[:, :, 0] = np.clip(y_stretched, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

    if resmi_kaydet(output_path, result):
        print("Doğrusal Kontrast uygulandı!")