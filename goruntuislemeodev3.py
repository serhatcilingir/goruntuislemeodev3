import cv2
import numpy as np

# Resmi yükleme
resim_yolu = r"C:\Users\Serhat\Downloads\serhatpirinc.jpg"
resim = cv2.imread(resim_yolu)

# Gri seviyeye dönüştürme
griton_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

# Eşikleme yapma
_, esiklenmis_resim = cv2.threshold(griton_resim, 181, 255, cv2.THRESH_BINARY)

# İstenmeyen arka planları temizleme
kernel = np.ones((5, 5), np.uint8)
morfoloji = cv2.morphologyEx(esiklenmis_resim, cv2.MORPH_OPEN, kernel)

# Sayma ve etiketleme
_, etiketler, istatistikler, _ = cv2.connectedComponentsWithStats(morfoloji, connectivity=8)

# Tane sayısını ekrana yazdırma
tane_sayisi = len(istatistikler) - 1  
print(f"Pirinç Tanelerinin Sayısı: {tane_sayisi}")

# Görüntüyü ekrana yazdır
cv2.imshow('orjinal goruntu', resim)
cv2.imshow('esiklenmis goruntu', esiklenmis_resim)
cv2.imshow('arka plani temizlenmis goruntu', morfoloji)
cv2.waitKey(0)
cv2.destroyAllWindows()
