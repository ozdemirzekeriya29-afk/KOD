import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

# Sayfa Ayarları
st.set_page_config(page_title="BİM Asistanı", page_icon="🛒", layout="centered")

st.title("🛒 BİM Ürün Bulucu")
st.write("Ürünün fotoğrafını çek, yapay zeka kodunu bulsun!")

# Klasör kontrolü (Veritabanı)
KLASOR = "urunler"
if not os.path.exists(KLASOR):
    st.error("⚠️ 'urunler' klasörü bulunamadı! GitHub'a resimleri yüklediğinden emin ol.")
    st.stop()

# --- GELİŞMİŞ GÖRÜNTÜ İŞLEME VE EŞLEŞTİRME ---
def akilli_karsilastir(aranan_resim, veritabani_resmi):
    # 1. Griye Çevir
    img1 = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(veritabani_resmi, cv2.COLOR_BGR2GRAY)
    
    # 2. Görüntü İyileştirme (CLAHE) - Karanlık/Parlak ortamlar için
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)
    
    # 3. SIFT Algoritması (Detaylı Tarama)
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0
        
    # 4. Eşleştirici (FLANN)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0
    
    # 5. Eleme (Lowe's Ratio Test)
    iyi_eslesmeler = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            iyi_eslesmeler.append(m)
            
    # 6. Geometrik Doğrulama (RANSAC) - Rastgeleliği önler
    if len(iyi_eslesmeler) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is not None:
            return sum(mask.ravel().tolist()) # Eşleşen nokta sayısı
        else:
            return 0
    else:
        return 0

# --- ARAYÜZ ---
yuklenen_foto = st.file_uploader("📸 Fotoğraf Çek veya Yükle", type=["jpg", "jpeg", "png"])

if yuklenen_foto:
    # Kullanıcının yüklediği resmi işle
    pil_image = Image.open(yuklenen_foto)
    open_cv_image = np.array(pil_image)
    aranan_resim = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    st.image(pil_image, caption="Aranan Ürün", width=200)

    if st.button("🔍 ÜRÜNÜ BUL", type="primary"):
        en_yuksek_skor = 0
        bulunan_urun = None
        bulunan_resim_yolu = None
        
        dosyalar = os.listdir(KLASOR)
        bar = st.progress(0)
        durum_yazisi = st.empty()
        
        # Tüm veritabanını tara
        for i, dosya in enumerate(dosyalar):
            if dosya.endswith((".jpg", ".png", ".jpeg")):
                durum_yazisi.text(f"Taranıyor... {dosya}")
                
                db_path = os.path.join(KLASOR, dosya)
                db_img = cv2.imread(db_path)
                
                if db_img is None: continue
                
                skor = akilli_karsilastir(aranan_resim, db_img)
                
                if skor > en_yuksek_skor:
                    en_yuksek_skor = skor
                    bulunan_urun = dosya.split(".")[0]
                    bulunan_resim_yolu = db_path
            
            bar.progress((i + 1) / len(dosyalar))
            
        durum_yazisi.empty()
        bar.empty()
        
        # --- SONUÇ ---
        ESIK_DEGERI = 10 # En az 10 nokta uyuşmalı (Hata payını azaltmak için)
        
        st.divider()
        if bulunan_urun and en_yuksek_skor >= ESIK_DEGERI:
            st.success(f"✅ BULUNDU! KOD: {bulunan_urun}")
            st.image(bulunan_resim_yolu, caption=f"Katalog Resmi (Güven Skoru: {en_yuksek_skor})")
        else:
            st.error("❌ Eşleşme Bulunamadı.")
            st.info("İpucu: Ürünü daha yakından ve dik bir açıyla çekmeyi dene.")
