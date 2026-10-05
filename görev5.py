import cv2
import numpy as np

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Görsele Salt-and-Pepper (Tuz ve Biber) gürültüsü ekler.
    """
    noisy_image = image.copy()
    
    # Tuz gürültüsü (Beyaz pikseller - 255)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_image[tuple(coords)] = 255

    # Biber gürültüsü (Siyah pikseller - 0)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_image[tuple(coords)] = 0

    return noisy_image

def custom_median_filter_5x5(image):
    """
    Hazır fonksiyon kullanmadan yazılmış 5x5 Medyan Filtresi.
    """
    # Görsel gri seviye değilse gri seviyeye dönüştürelim
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape
    output = np.zeros((height, width), dtype=np.uint8)
    
    # 5x5 çekirdek boyutu için kenar dolgusu (padding) boyutu: (5 - 1) // 2 = 2
    pad_size = 2
    
    # Görüntünün etrafına kenar değerlerini yansıtarak padding ekleme
    padded_img = np.pad(gray, pad_size, mode='reflect')

    # 5x5 komşuluktaki piksel sayısı: 25
    # Sıralı dizideki medyan elemanın indeksi: 12 (0'dan başladığı için 12. eleman ortadakidir)
    median_index = 12 

    # Her piksel üzerinde 5x5 pencereyi gezdirme
    for i in range(height):
        for j in range(width):
            # 5x5 komşuluk bölgesini kesip alıyoruz
            window = padded_img[i:i + 5, j:j + 5]
            
            # 5x5 matrisi tek boyutlu bir diziye dönüştürüp sıralıyoruz
            sorted_pixels = np.sort(window.ravel())
            
            # Ortadaki (medyan) değeri alıp yeni görüntüye atıyoruz
            output[i, j] = sorted_pixels[median_index]

    return output

# UYGULAMA

# 1. Örnek bir görüntü yükleyin (Veya sıfırdan test görüntüsü oluşturun)
# img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
img = np.full((300, 300), 128, dtype=np.uint8) # Test amaçlı gri bir zemin
cv2.rectangle(img, (50, 50), (250, 250), 200, -1) # İçine bir kare çizelim

# 2. Gürültü Ekleme
noisy_img = add_salt_and_pepper_noise(img, salt_prob=0.03, pepper_prob=0.03)

# 3. YÖNTEM 1: OpenCV Hazır Medyan Filtresi (5x5)
opencv_median = cv2.medianBlur(noisy_img, 5)

# 4. YÖNTEM 2: Özel Yazılmış Medyan Filtresi (5x5)
custom_median = custom_median_filter_5x5(noisy_img)

# 5. Sonuçları Karşılaştırma
# Özel yazılan filtre ile OpenCV çıktısının farkı:
difference = cv2.absdiff(opencv_median, custom_median)

print(f"Hazır ve Özel Filtre Arasındaki Maksimum Piksel Farkı: {np.max(difference)}")

# Görselleri Kaydetme / Gösterme
cv2.imwrite('1_gurultulu_resim.png', noisy_img)
cv2.imwrite('2_opencv_medyan_5x5.png', opencv_median)
cv2.imwrite('3_ozel_medyan_5x5.png', custom_median)