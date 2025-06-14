import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.title("Prediksi Kategori Obesitas")
st.write("Silakan lengkapi data diri Anda untuk mengetahui kategori obesitas.")

# Load model dan scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("obesity_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

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
if st.button("Prediksi Sekarang"):
    # Di sini Anda akan memproses input dan menjalankan model
    st.success("Input berhasil disimpan! Silakan lanjutkan ke proses prediksi.")



def preprocess_input(data):
    # Mapping kategorikal
    gender_map = {"Male": 0, "Female": 1}
    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    favc_map = {"no": 0, "yes": 1}
    smoke_map = {"no": 0, "yes": 1}
    scc_map = {"no": 0, "yes": 1}
    caec_map = {"Sometimes": 0, "Frequently": 1, "Always": 2, "no": 3}
    mtrans_map = {
        "Public_Transportation": 0,
        "Automobile": 1,
        "Walking": 2,
        "Motorbike": 3,
        "Bike": 4
    }

    # Encode data
    data['Gender'] = gender_map.get(data['Gender'], -1)
    data['CALC'] = calc_map.get(data['CALC'], -1)
    data['FAVC'] = favc_map.get(data['FAVC'], -1)
    data['SMOKE'] = smoke_map.get(data['SMOKE'], -1)
    data['SCC'] = scc_map.get(data['SCC'], -1)
    data['CAEC'] = caec_map.get(data['CAEC'], -1)
    data['MTRANS'] = mtrans_map.get(data['MTRANS'], -1)

    # Normalisasi fitur numerik menggunakan scaler yang telah dimuat
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    data[numerical_features] = scaler.transform([data[numerical_features]])

    return data


if st.button("Lihat Hasil Prediksi"):
    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'CALC': [calc],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'SCC': [scc],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'family_history_with_overweight': [family_history],
        'FAF': [faf],
        'TUE': [tue],
        'CAEC': [caec],
        'MTRANS': [mtrans]
    })

    # Proses input
    processed_data = preprocess_input(input_data)

    # Lakukan prediksi
    prediction = model.predict(processed_data)[0]

    # Decode hasil prediksi
    categories = {
        0: "Underweight",
        1: "Normal Weight",
        2: "Overweight Level I",
        3: "Overweight Level II",
        4: "Obesity Type I",
        5: "Obesity Type II",
        6: "Obesity Type III"
    }
    result = categories.get(prediction, "Tidak Diketahui")

    # Tampilkan hasil
    st.success(f"Prediksi Kategori Obesitas: {result}")
