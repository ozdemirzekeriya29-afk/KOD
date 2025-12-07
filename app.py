import streamlit as st
import os

# --- HATA DÜZELTİCİ YAMA ---
# Bu kod, uygulamanın hafızaya erişip çökmesini engeller
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
# ---------------------------

# Buradan sonra senin kodların devam etsin...
# import easyocr ...
import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title="Akıllı Arama", layout="centered")

st.title("🕵️‍♂️ Müfettiş Modu: Ürün Arama")
st.info("Bu modül, sadece şekil ve geometri eşleşirse onay verir.")

KLASOR = "urunler"
if not os.path.exists(KLASOR):
    st.error("Ürünler klasörü yok!")
    st.stop()

# --- GELİŞMİŞ KARŞILAŞTIRMA (RANSAC) ---
def akilli_karsilastir(aranan_resim, veritabani_resmi):
    # 1. Griye Çevir
    img1 = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(veritabani_resmi, cv2.COLOR_BGR2GRAY)
    
    # 2. SIFT Motorunu Başlat (Ağır Silah) 
    # ORB yerine SIFT kullanıyoruz
    sift = cv2.SIFT_create()
    
    # Özellikleri ve Parmak İzlerini (Descriptors) Bul
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Eğer hiç nokta bulamazsa çık
    if des1 is None or des2 is None:
        return 0
        
    # 3. Eşleştirme (FLANN tabanlı eşleştirici - SIFT için daha iyidir)
    # Bu ayarlar SIFT'in dilinden anlayan ayarlardır
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50) # Ne kadar yüksekse o kadar detaylı arar
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        # En iyi 2 eşleşmeyi bul (k=2)
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0 # Hata olursa sıfır dön
    
    # 4. Eleme (Lowe's Ratio Test)
    # Çürük elmaları ayıkla
    iyi_eslesmeler = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: # 0.75 yerine 0.7 yaptık (Daha sıkı denetim)
            iyi_eslesmeler.append(m)
            
    # 5. GEOMETRİK DOĞRULAMA (RANSAC)
    # En az 4 sağlam nokta lazım (Geometri kurmak için min. sınır)
    if len(iyi_eslesmeler) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
        
        # Perspektif yamukluğunu kontrol et
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is not None:
            # Sadece kalıba uyanları say
            dogru_sayisi = sum(mask.ravel().tolist())
            return dogru_sayisi
        else:
            return 0
    else:
        return 0
# --- ARAYÜZ ---
col1, col2 = st.columns([1, 1])

with col1:
    yuklenen_foto = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "jpeg", "png"])

if yuklenen_foto:
    # Resmi Hazırla
    pil_image = Image.open(yuklenen_foto)
    # Oryantasyon (dönme) sorununu çözmek için
    open_cv_image = np.array(pil_image)
    # RGB -> BGR
    aranan_resim = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    with col1:
        st.image(pil_image, caption="Senin Yüklediğin", use_column_width=True)

    if st.button("🔍 DETAYLI TARA", type="primary"):
        en_yuksek_skor = 0
        bulunan_urun = None
        bulunan_resim_yolu = None
        
        dosyalar = os.listdir(KLASOR)
        bar = st.progress(0)
        durum = st.empty()
        
        for i, dosya in enumerate(dosyalar):
            durum.text(f"Taranıyor: {dosya}")
            
            # Veritabanı resmini oku
            db_path = os.path.join(KLASOR, dosya)
            db_img = cv2.imread(db_path)
            
            if db_img is None: continue
            
            # Karşılaştır
            skor = akilli_karsilastir(aranan_resim, db_img)
            
            # Skor ne kadar yüksekse o kadar iyi
            if skor > en_yuksek_skor:
                en_yuksek_skor = skor
                bulunan_urun = dosya.split(".")[0]
                bulunan_resim_yolu = db_path
            
            bar.progress((i + 1) / len(dosyalar))
            
        durum.empty()
        
        # --- SONUÇ KARARI ---
        # Eşik Değeri: RANSAC sonrası en az 8-10 sağlam nokta eşleşmeli
        ESIK_DEGERI = 10
        
        with col2:
            st.divider()
            if bulunan_urun and en_yuksek_skor >= ESIK_DEGERI:
                st.success("✅ EŞLEŞME DOĞRULANDI!")
                st.write(f"Kod: **{bulunan_urun}**")
                st.write(f"Güven Skoru: {en_yuksek_skor}")
                st.image(bulunan_resim_yolu, caption="Katalogdaki Hali")
            else:
                st.error("❌ Eşleşme Bulunamadı.")
                if en_yuksek_skor > 0:
                    st.warning(f"En yakın tahmin ({bulunan_urun}) idi ama güven skoru çok düşüktü ({en_yuksek_skor}).")

                st.info("İpucu: Fotoğrafı ürünün tam karşısından ve daha aydınlık çekmeyi dene.")
