import streamlit as st
import random
import os
import pandas as pd
from PIL import Image
import time
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# CSS Styling
st.markdown("""
<style>
button[kind="secondary"] {
    font-size: 18px !important;
    padding: 12px 0px !important;
    border-radius: 8px !important;
}
.progress-bar {
    height: 10px;
    background-color: #ddd;
    border-radius: 5px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    background-color: #007bff;
    width: 0%;
    transition: width 0.5s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Inisialisasi Session State
if "index" not in st.session_state:
    st.session_state.index = 0
if "skor" not in st.session_state:
    st.session_state.skor = 0
if "jawab_dulu" not in st.session_state:
    st.session_state.jawab_dulu = True
if "show_penjelasan" not in st.session_state:
    st.session_state.show_penjelasan = False
if "kategori" not in st.session_state:
    st.session_state.kategori = None
if "level" not in st.session_state:
    st.session_state.level = "Mudah"
if "acak_level" not in st.session_state:
    st.session_state.acak_level = False
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# Load Dataset Soal
try:
    data_df = pd.read_csv("dataset_game.csv")
except FileNotFoundError:
    st.error("File dataset_game.csv tidak ditemukan.")
    st.stop()

# Warna Tema Berdasarkan Kategori
warna_kategori = {
    "Buah": "#FFD700",
    "Hewan": "#8B4513",
    "Sayur": "#228B22",
    "Alat": "#8B7500",
    "Transportasi": "#008080"
}

# Sidebar Pengaturan
st.sidebar.title("🎮 Pengaturan Game")
level_opsi = ["Mudah", "Sedang"]
st.session_state.level = st.sidebar.selectbox("Pilih Level:", level_opsi)
st.session_state.acak_level = st.sidebar.checkbox("Acak Level")

# Menu Utama
st.markdown("<h1 style='text-align: center; color: #FF9933;'>📚 Belajar Seru</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Pilih kategori untuk mulai belajar!</h3>", unsafe_allow_html=True)

kategori_opsi = ["Buah", "Hewan", "Sayur", "Alat", "Transportasi"]

# 📸 TAMPILAN AWAL DENGAN GAMBAR
try:
    image_path = os.path.join("data", "tampilan.jpg")
    if os.path.exists(image_path):
        st.image(image_path, caption="Selamat datang!", use_container_width=True)
    else:
        st.warning("Gambar tampilan tidak ditemukan di folder /data.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat gambar: {e}")

def tampilkan_kategori():
    baris1 = kategori_opsi[:4]
    baris2 = kategori_opsi[4:]
    for baris in [baris1, baris2]:
        cols = st.columns(len(baris), gap="medium")
        for i, kategori in enumerate(baris):
            with cols[i]:
                if st.button(kategori, use_container_width=True):
                    st.session_state.kategori = kategori
                    st.session_state.index = 0
                    st.session_state.skor = 0
                    st.session_state.jawab_dulu = True
                    st.session_state.show_penjelasan = False
                    st.rerun()

tampilkan_kategori()

# === Fungsi Latih Model Machine Learning ===
def latih_model(df):
    df['kategori_enc'] = df['kategori'].astype('category').cat.codes
    df['level_enc'] = df['level'].astype('category').cat.codes
    X = df[['kategori_enc', 'level_enc', 'jumlah_benar_sebelumnya', 'lama_waktu_jawab']]
    y = df['benar']
    if len(y.unique()) < 2:
        return None, 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    prediksi = model.predict(X_test)
    akurasi = accuracy_score(y_test, prediksi)
    return model, akurasi

# === Jika Kategori Sudah Dipilih ===
if st.session_state.kategori:
    data_kategori = data_df[data_df["kategori"] == st.session_state.kategori].reset_index(drop=True)
    if not st.session_state.acak_level:
        data_kategori = data_kategori[data_kategori["level"] == st.session_state.level].reset_index(drop=True)
    if len(data_kategori) == 0:
        st.warning("Belum ada soal untuk kategori dan level ini.")
        st.stop()
    total_soal = len(data_kategori)
    soal = data_kategori.iloc[st.session_state.index]
    gambar_path = os.path.join("data", soal["file"])
    jawaban = soal["jawaban"].upper()
    penjelasan = soal["penjelasan"]
    warna = warna_kategori.get(st.session_state.kategori, "#008080")

    st.markdown(f"<div style='background-color:{warna}; padding:10px; border-radius:8px;'>",
                unsafe_allow_html=True)
    st.markdown(f"### 📷 Susun Kata dari Gambar - {st.session_state.kategori}")
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        image = Image.open(gambar_path)
        st.image(image, caption="Apa nama objek ini?", width=300)
    except FileNotFoundError:
        st.error(f"Gambar '{soal['file']}' tidak ditemukan di folder /data.")
        st.stop()

    # 🔊 Mainkan audio pertanyaan
    def mainkan_audio(file_audio):
        try:
            with open(os.path.join("audio", file_audio), "rb") as f:
                audio_bytes = f.read()
            st.markdown(f"""
                <audio controls autoplay muted>
                    <source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3">
                    Browser tidak mendukung audio.
                </audio>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"File audio '{file_audio}' tidak ditemukan.")

    mainkan_audio("pertanyaan.mp3")

    huruf_acak = list(jawaban)
    random.shuffle(huruf_acak)
    st.write("### 🧩 Susun Huruf Ini:")
    st.write(" ".join(huruf_acak))

    start_time = time.time()
    jawaban_user = st.text_input("Masukkan jawaban kamu:", key="input_jawaban").upper()

    if st.button("🔍 Periksa Jawaban") and st.session_state.jawab_dulu:
        end_time = time.time()
        waktu_jawab = round(end_time - start_time, 2)
        benar = (jawaban_user == jawaban)
        if benar:
            st.markdown("🎉 <span style='color:green; font-size:24px;'>✅ Benar sekali!</span>", unsafe_allow_html=True)
            st.session_state.skor += 1
            audio_file = "benar.mp3"
        else:
            st.markdown("😢 <span style='color:red; font-size:24px;'>❌ Coba lagi ya!</span>", unsafe_allow_html=True)
            audio_file = "salah.mp3"

        # Mainkan audio hasil jawaban
        try:
            with open(os.path.join("audio", audio_file), "rb") as f:
                audio_bytes = f.read()
            st.markdown(f"""
                <audio id="validasiAudio" controls autoplay>
                    <source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3">
                    Browser tidak mendukung audio.
                </audio>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"File audio '{audio_file}' tidak ditemukan.")

        time.sleep(2)
        st.session_state.riwayat.append({
            "kategori": st.session_state.kategori,
            "level": st.session_state.level,
            "jumlah_benar_sebelumnya": st.session_state.skor,
            "lama_waktu_jawab": waktu_jawab,
            "benar": 1 if benar else 0
        })
        st.session_state.jawab_dulu = False
        st.session_state.show_penjelasan = True
        st.rerun()

    if st.session_state.show_penjelasan:
        st.info(f"📘 {penjelasan}")
        if st.button("🔊 Dengarkan Penjelasan"):
            penjelasan_audio_path = os.path.join("audio", f"{soal['jawaban'].lower()}_penjelasan.mp3")
            try:
                with open(penjelasan_audio_path, "rb") as f:
                    audio_bytes = f.read()
            except FileNotFoundError:
                try:
                    with open(os.path.join("audio", "penjelasan_template.mp3"), "rb") as f:
                        audio_bytes = f.read()
                except FileNotFoundError:
                    st.error("File penjelasan default juga tidak ditemukan.")
                    st.stop()
            st.markdown(f"""
                <audio controls autoplay>
                    <source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3">
                    Browser tidak mendukung audio.
                </audio>
            """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Ulang Soal"):
            st.session_state.jawab_dulu = True
            st.session_state.show_penjelasan = False
    with col2:
        if not st.session_state.jawab_dulu and st.button("➡️ Soal Berikutnya"):
            st.session_state.index = (st.session_state.index + 1) % total_soal
            st.session_state.jawab_dulu = True
            st.session_state.show_penjelasan = False
    with col3:
        if st.button("🔁 Mulai Ulang"):
            st.session_state.index = 0
            st.session_state.skor = 0
            st.session_state.jawab_dulu = True
            st.session_state.show_penjelasan = False

    st.write(f"📁 Soal {st.session_state.index + 1} dari {total_soal}")
    st.sidebar.markdown("### 🎯 Skor")
    st.sidebar.write(f"Benar: {st.session_state.skor} / {st.session_state.index + 1}")

    # Tampilan skor akhir jika semua soal sudah selesai
    if st.session_state.index + 1 == total_soal and not st.session_state.jawab_dulu:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center; color:#4CAF50;'>🎉 Quiz Selesai!</h2>", unsafe_allow_html=True)

        # Progress Bar
        progress = int((st.session_state.skor / total_soal) * 100)
        st.markdown(f"""
            <div class="progress-bar">
                <div class="progress-bar-fill" style="width: {progress}%;"></div>
            </div>
        """, unsafe_allow_html=True)

        # Tampilan Skor Akhir
        st.markdown(f"<h3 style='text-align:center;'>Skor: {st.session_state.skor}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Benar: {st.session_state.skor}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Salah: {total_soal - st.session_state.skor}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Akurasi: {progress}%</p>", unsafe_allow_html=True)

        # Tombol Ulangi Game
        if st.button("🔁 Ulangi Game"):
            st.session_state.index = 0
            st.session_state.skor = 0
            st.session_state.jawab_dulu = True
            st.session_state.show_penjelasan = False
            st.rerun()

    # === Prediksi Soal Selanjutnya Menggunakan RandomForestClassifier ===
    df_riwayat = pd.DataFrame(st.session_state.riwayat)
    if len(df_riwayat) > 5:
        model, akurasi = latih_model(df_riwayat)
        if model:
            kategori_enc = df_riwayat['kategori'].astype('category').cat.codes.max()
            level_enc = df_riwayat['level'].astype('category').cat.codes.max()
            fitur_baru = [[kategori_enc, level_enc, st.session_state.skor, 15]]
            prediksi = model.predict(fitur_baru)
            if prediksi[0] == 1:
                st.success("🎯 Kamu kemungkinan besar akan benar di soal berikutnya!")
            else:
                st.info("⚠️ Soal selanjutnya mungkin agak sulit... tetap semangat!")

            fig, ax = plt.subplots(figsize=(10, 6))
            from sklearn.tree import plot_tree
            plot_tree(model.estimators_[0], feature_names=['Kategori', 'Level', 'Skor', 'Waktu'],
                      class_names=["Salah", "Benar"], filled=True, fontsize=8, ax=ax)
            st.pyplot(fig)
            st.caption(f"Akurasi model saat ini: {round(akurasi * 100, 2)}%")
