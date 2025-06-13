import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Muat model dan scaler
@st.cache_resource
def load_model():
    return joblib.load('obesity_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

try:
    model = load_model()
    scaler = load_scaler()
except Exception as e:
    st.error("Gagal memuat model atau scaler. Pastikan file tersedia.")
    st.stop()

# Mapping untuk encoding
gender_map = {"Male": 0, "Female": 1}
calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
favc_map = {"no": 0, "yes": 1}
smoke_map = {"no": 0, "yes": 1}
scc_map = {"no": 0, "yes": 1}
caec_map = {"no": 3, "Sometimes": 0, "Frequently": 1, "Always": 2}
mtrans_map = {
    "Public_Transportation": 0,
    "Automobile": 1,
    "Walking": 2,
    "Motorbike": 3,
    "Bike": 4
}

# UI Aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Silakan lengkapi data diri Anda untuk mengetahui kategori obesitas.")

# Form Input
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.7)
        weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=70)
        calc = st.selectbox("Seberapa sering konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
        favc = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"])
        fcvc = st.slider("Frekuensi makan sayur per minggu", min_value=0, max_value=10, value=2)
        ncp = st.slider("Jumlah makan per hari", min_value=1, max_value=10, value=3)

    with col2:
        scc = st.selectbox("Apakah Anda mencatat kalori yang dikonsumsi?", ["yes", "no"])
        smoke = st.selectbox("Apakah Anda perokok?", ["yes", "no"])
        ch2o = st.slider("Konsumsi air per hari (liter)", min_value=0, max_value=5, value=2)
        family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"])
        faf = st.slider("Frekuensi aktivitas fisik per minggu", min_value=0, max_value=7, value=2)
        tue = st.slider("Waktu layar/hari (jam)", min_value=0, max_value=5, value=2)
        caec = st.selectbox("Seberapa sering ngemil di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

    submit_button = st.form_submit_button("Lihat Hasil Prediksi")

if submit_button:
    # Encode input
    encoded_data = {
        'Age': [age],
        'Gender': [gender_map[gender]],
        'Height': [height],
        'Weight': [weight],
        'CALC': [calc_map[calc]],
        'FAVC': [favc_map[favc]],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'SCC': [scc_map[scc]],
        'SMOKE': [smoke_map[smoke]],
        'CH2O': [ch2o],
        'family_history_with_overweight': [1 if family_history == "yes" else 0],
        'FAF': [faf],
        'TUE': [tue],
        'CAEC': [caec_map[caec]],
        'MTRANS': [mtrans_map[mtrans]]
    }

    input_df = pd.DataFrame(encoded_data)

    # Normalisasi fitur numerik
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Prediksi
    prediction = model.predict(input_df)[0]

    # Decode hasil
    categories = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Overweight_Level_I",
        3: "Overweight_Level_II",
        4: "Obesity_Type_I",
        5: "Obesity_Type_II",
        6: "Obesity_Type_III"
    }
    result = categories.get(prediction, "Kategori tidak dikenali")

    # Tampilkan hasil
    st.success(f"Prediksi Kategori Obesitas: **{result}**")
