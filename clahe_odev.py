





56import cv2
import numpy as np
import os

# 1. Dosya yollarını belirle
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'resim.jpg')
output_opencv_path = os.path.join(script_dir, 'clahe_opencv.jpg')
output_manual_path = os.path.join(script_dir, 'clahe_manuel.jpg')

# Türkçe karakterli yollardan resmi güvenle okuma ve kaydetme
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


# KISIM 1: OPENCV HAZIR CLAHE FONKSİYONU İLE GERÇEKLEME

def clahe_opencv(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

# KISIM 2: HAZIR FONKSİYON KULLANMADAN MANUEL CLAHE GERÇEKLEME

def clahe_manuel(img, clip_limit=2.0, grid_size=(8, 8)):
    h, w = img.shape
    grid_h, grid_w = grid_size
    
    tile_h = h // grid_h
    tile_w = w // grid_w
    
    # 1. Her kutucuk (tile) için sınırlanmış LUT (Lookup Table) hesapla
    luts = np.zeros((grid_h, grid_w, 256), dtype=np.uint8)
    
    for r in range(grid_h):
        for c in range(grid_w):
            # Kutucuk alanını kes
            y_start, y_end = r * tile_h, (r + 1) * tile_h if r < grid_h - 1 else h
            x_start, x_end = c * tile_w, (c + 1) * tile_w if c < grid_w - 1 else w
            tile = img[y_start:y_end, x_start:x_end]
            
            # A) Manuel Histogram Hesabı
            hist = np.zeros(256, dtype=np.int32)
            for pixel in tile.ravel():
                hist[pixel] += 1
                
            # B) Clip Limit Sınırlaması ve Yeniden Dağıtım
            actual_tile_size = tile.size
            actual_clip = int(clip_limit * (actual_tile_size / 256.0))
            if actual_clip < 1:
                actual_clip = 1
                
            excess = 0
            for i in range(256):
                if hist[i] > actual_clip:
                    excess += hist[i] - actual_clip
                    hist[i] = actual_clip
                    
            # Artan pikselleri 256 bin'e eşit dağıt
            bonus = excess // 256
            leftover = excess % 256
            hist += bonus
            hist[:leftover] += 1
            
            # C) Manuel CDF Hesabı
            cdf = np.zeros(256, dtype=np.float32)
            cumulative = 0
            for i in range(256):
                cumulative += hist[i]
                cdf[i] = cumulative
                
            # D) LUT Dönüşümü (0-255 aralığına ölçekleme)
            cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
            cdf_max = cdf[-1]
            
            if cdf_max != cdf_min:
                lut = np.round(((cdf - cdf_min) / (cdf_max - cdf_min)) * 255.0)
            else:
                lut = np.zeros(256)
                
            luts[r, c] = np.clip(lut, 0, 255).astype(np.uint8)

    # 2. Bilineer İnterpolasyon ile Kutucukları Yumuşakça Birleştirme
    out_img = np.zeros_like(img, dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            # En yakın 4 kutucuk merkezini bul
            gx = (x / tile_w) - 0.5
            gy = (y / tile_h) - 0.5
            
            g_x1 = int(np.floor(gx))
            g_y1 = int(np.floor(gy))
            g_x2 = g_x1 + 1
            g_y2 = g_y1 + 1
            
            # Sınır kontrolleri
            g_x1_c = np.clip(g_x1, 0, grid_w - 1)
            g_x2_c = np.clip(g_x2, 0, grid_w - 1)
            g_y1_c = np.clip(g_y1, 0, grid_h - 1)
            g_y2_c = np.clip(g_y2, 0, grid_h - 1)
            
            fx = gx - g_x1
            fy = gy - g_y1
            
            val = img[y, x]
            
            # 4 komşu kutucuktan gelen değerler
            v11 = luts[g_y1_c, g_x1_c, val]
            v12 = luts[g_y1_c, g_x2_c, val]
            v21 = luts[g_y2_c, g_x1_c, val]
            v22 = luts[g_y2_c, g_x2_c, val]
            
            # Bilineer ağırlıklandırma
            if g_x1 < 0 or g_x2 >= grid_w:
                fx = 0
            if g_y1 < 0 or g_y2 >= grid_h:
                fy = 0
                
            interpolated = (1 - fx) * (1 - fy) * v11 + \
                           fx * (1 - fy) * v12 + \
                           (1 - fx) * fy * v21 + \
                           fx * fy * v22
                           
            out_img[y, x] = int(round(interpolated))
            
    return out_img


# UYGULAMA
img = resmi_oku_gri(input_path)

if img is None:
    print(f"❌ Resim bulunamadı: {input_path}")
else:
    print("✓ Resim yüklendi. CLAHE işlemleri çalıştırılıyor...")

    # 1. Hazır OpenCV CLAHE
    res_opencv = clahe_opencv(img, clip_limit=2.0, tile_grid_size=(8, 8))
    resmi_kaydet(output_opencv_path, res_opencv)
    print("✅ OpenCV hazır CLAHE tamamlandı -> 'clahe_opencv.jpg'")

    # 2. Manuel Kodlanmış CLAHE
    res_manuel = clahe_manuel(img, clip_limit=2.0, grid_size=(8, 8))
    resmi_kaydet(output_manual_path, res_manuel)
    print("✅ Manuel yazılan CLAHE tamamlandı -> 'clahe_manuel.jpg'")