import streamlit as st

st.title("Prediksi Kategori Obesitas")
st.write("Silakan lengkapi data diri Anda untuk mengetahui kategori obesitas.")

# Input numerik
age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=25)
height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.7)
weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=70)
fcvc = st.slider("Frekuensi makan sayur per minggu", min_value=0, max_value=10, value=2)
ncp = st.slider("Jumlah makan per hari", min_value=1, max_value=10, value=3)
ch2o = st.slider("Konsumsi air per hari (liter)", min_value=0, max_value=5, value=2)
faf = st.slider("Frekuensi aktivitas fisik per minggu", min_value=0, max_value=7, value=2)
tue = st.slider("Waktu layar per hari (jam)", min_value=0, max_value=5, value=2)

# Input kategorikal
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
favc = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"])
smoke = st.selectbox("Apakah Anda perokok?", ["yes", "no"])
calc = st.selectbox("Seberapa sering konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
caec = st.selectbox("Seberapa sering ngemil di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"])
scc = st.selectbox("Apakah Anda mencatat kalori yang dikonsumsi?", ["yes", "no"])

# Tombol prediksi
if st.button("Prediksi"):
    # Di sini Anda akan memproses input dan menjalankan model
    st.success("Input berhasil disimpan! Silakan lanjutkan ke proses prediksi.")
