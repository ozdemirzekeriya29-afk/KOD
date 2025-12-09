import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

# 1. SAYFA AYARLARI
st.set_page_config(page_title="BİM Asistanı", page_icon="🛒", layout="centered")

# 2. GİZLEME KODU (CSS + JAVASCRIPT KOMBİNASYONU) 💣
# Hem stil ile gizliyoruz hem de JavaScript ile siliyoruz.

gizleme_kodu = """
    <style>
        /* CSS İLE GİZLEME */
        footer {visibility: hidden !important; display: none !important;}
        header {visibility: hidden !important; display: none !important;}
        #MainMenu {visibility: hidden !important; display: none !important;}
        [data-testid="stFooter"] {display: none !important;}
        .stAppDeployButton {display: none !important;}
        
        /* İçeriği yukarı çek */
        .block-container {
            padding-top: 0rem !important;
            margin-top: -3rem !important;
        }
    </style>
    
    <script>
        // JAVASCRIPT İLE SİLME (GARANTİ YÖNTEM)
        // Bu kod her yarım saniyede bir o yazıyı kontrol eder ve varsa siler.
        setInterval(function() {
            var footer = document.querySelector("footer");
            if(footer) { footer.remove(); }
            
            var header = document.querySelector("header");
            if(header) { header.remove(); }
            
            var mainMenu = document.querySelector("#MainMenu");
            if(mainMenu) { mainMenu.remove(); }
        }, 100);
    </script>
"""
# Javascript'i sayfaya gömüyoruz (Height 0 yaparak görünmez yapıyoruz)
components.html(gizleme_kodu, height=0, width=0)


# 3. UYGULAMA İÇERİĞİ
st.title("🛒 Ürün Bulucu")
st.write("Ürünün fotoğrafını çek, yapay zeka kodunu bulsun!")

# --- KLASÖR KONTROLÜ ---
KLASOR = "urunler"
if not os.path.exists(KLASOR):
    # Eğer klasör yoksa hata verme, sessizce oluştur (Hata mesajı görünmemesi için)
    try:
        os.makedirs(KLASOR)
    except:
        st.error("Veritabanı hatası!")

# --- GÖRÜNTÜ İŞLEME MOTORU ---
def akilli_karsilastir(aranan_resim, veritabani_resmi):
    try:
        img1 = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(veritabani_resmi, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img1 = clahe.apply(img1)
        img2 = clahe.apply(img2)
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None: return 0
            
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        iyi_eslesmeler = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                iyi_eslesmeler.append(m)
                
        if len(iyi_eslesmeler) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                return sum(mask.ravel().tolist())
    except:
        return 0
    return 0

# --- ARAYÜZ ---
yuklenen_foto = st.file_uploader("📸 Ürün Fotoğrafı", type=["jpg", "jpeg", "png"])

if yuklenen_foto:
    pil_image = Image.open(yuklenen_foto)
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 3:
        aranan_resim = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    else:
        aranan_resim = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
    
    st.image(pil_image, caption="Aranan", width=200)

    if st.button("TARA VE BUL", type="primary"):
        en_yuksek_skor = 0
        bulunan_urun = None
        bulunan_resim_yolu = None
        
        dosyalar = os.listdir(KLASOR)
        bar = st.progress(0)
        
        for i, dosya in enumerate(dosyalar):
            if dosya.endswith((".jpg", ".png", ".jpeg")):
                db_path = os.path.join(KLASOR, dosya)
                db_img = cv2.imread(db_path)
                if db_img is None: continue
                skor = akilli_karsilastir(aranan_resim, db_img)
                if skor > en_yuksek_skor:
                    en_yuksek_skor = skor
                    bulunan_urun = dosya.split(".")[0]
                    bulunan_resim_yolu = db_path
            bar.progress((i + 1) / len(dosyalar))
        bar.empty()
        
        ESIK_DEGERI = 6
        st.divider()
        if bulunan_urun and en_yuksek_skor >= ESIK_DEGERI:
            st.success(f"✅ BULUNDU! KOD: **{bulunan_urun}**")
            st.image(bulunan_resim_yolu, caption="Katalog Kaydı", width=200)
        else:
            st.error("❌ Eşleşme Bulunamadı.")
            if en_yuksek_skor > 0:
                st.warning(f"En yakın: {bulunan_urun} (Puan: {en_yuksek_skor})")
