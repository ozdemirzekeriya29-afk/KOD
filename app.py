import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

# Ayarlar
st.set_page_config(page_title="BİM Asistanı", layout="centered")
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("🛒 Ürün Tarayıcı")

# --- GELİŞMİŞ GÖRÜNTÜ İŞLEME MOTORU ---
def resmi_hazirla(img):
    # 1. Griye Çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. Gürültü Temizle (Bulanıklık)
    # Market raflarındaki parlamayı azaltır
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3. Kontrastı Artır (CLAHE)
    # Katalog resmi parlak, telefon resmi sönükse bunu eşitler
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_img = clahe.apply(denoised)
    return final_img

def akilli_karsilastir(kullanici_resmi, veritabani_resmi):
    try:
        # Resimleri hazırla
        img1 = resmi_hazirla(kullanici_resmi)
        img2 = resmi_hazirla(veritabani_resmi)
        
        # SIFT Algoritması (En Detaylısı)
        sift = cv2.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None: return 0
        
        # Eşleştirme (FLANN)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Eleme (Lowe's Ratio Test - 0.75 yaptık, biraz esnettik)
        iyi_eslesmeler = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                iyi_eslesmeler.append(m)
        
        # Geometrik Doğrulama (RANSAC)
        # En az 6 nokta geometrik olarak uyuşmalı
        if len(iyi_eslesmeler) > 6:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                match_count = np.sum(mask)
                return match_count
        return 0
    except:
        return 0

# --- ARAYÜZ ---
if not os.path.exists("urunler"):
    st.error("Veritabanı bulunamadı!")
    st.stop()

yuklenen_foto = st.file_uploader("Fotoğraf Çek", type=["jpg", "png", "jpeg"])

if yuklenen_foto:
    # Kullanıcı resmini yükle ve çevir
    user_img_pil = Image.open(yuklenen_foto)
    
    # Oryantasyon düzeltme (Telefondan yan gelmemesi için)
    from PIL import ImageOps
    user_img_pil = ImageOps.exif_transpose(user_img_pil)
    
    user_img = np.array(user_img_pil)
    user_img = cv2.cvtColor(user_img, cv2.COLOR_RGB2BGR) # OpenCV formatına çevir

    st.image(user_img_pil, caption="Aranan", width=200)
    
    if st.button("TARA VE BUL"):
        en_iyi_skor = 0
        en_iyi_urun = None
        
        dosyalar = os.listdir("urunler")
        bar = st.progress(0)
        
        for i, dosya in enumerate(dosyalar):
            path = os.path.join("urunler", dosya)
            db_img = cv2.imread(path)
            
            if db_img is not None:
                puan = akilli_karsilastir(user_img, db_img)
                
                # En yüksek puanı tut
                if puan > en_iyi_skor:
                    en_iyi_skor = puan
                    en_iyi_urun = dosya.split(".")[0]
            
            bar.progress((i + 1) / len(dosyalar))
            
        bar.empty()
        
        # SONUÇ EKRANI
        st.divider()
        # Eşik Değeri: En az 10 sağlam nokta bulmalı
        if en_iyi_urun and en_iyi_skor >= 10:
            st.success(f"✅ BULUNDU!\nKod: {en_iyi_urun}")
            st.write(f"Eşleşme Puanı: {en_iyi_skor}")
            st.image(f"urunler/{en_iyi_urun}.jpg", width=150)
        else:
            st.error("❌ Ürün bulunamadı.")
            st.info("İpucu: Sadece ürünün kendisine odaklanın, parlamayı engelleyin.")
            if en_iyi_urun:
                st.write(f"En yakın tahmin: {en_iyi_urun} (Puan: {en_iyi_skor} - Çok düşük)")
