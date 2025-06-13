import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Judul Aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Silakan masukkan data pengguna untuk memprediksi kategori obesitas.")

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Mapping untuk kolom kategorikal
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

# Input Pengguna
age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=25)
gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.7)
weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=70)

calories = st.selectbox("Konsumsi kalori tinggi?", options=list(calc_map.keys()))
favorite_high_calorie = st.selectbox("Mengonsumsi makanan tinggi kalori?", options=["yes", "no"])
frequency_of_vegetables = st.slider("Frekuensi makan sayur (skala 0–3)", min_value=0, max_value=3, value=2)
number_of_meals = st.slider("Jumlah makan dalam sehari", min_value=1, max_value=10, value=3)
snacks = st.slider("Snacking antar waktu makan", min_value=0, max_value=3, value=2)
smoke = st.selectbox("Perokok?", options=["yes", "no"])
consumption_of_water = st.slider("Konsumsi air per hari (liter)", min_value=0, max_value=5, value=2)
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", options=["yes", "no"])
physical_activity_frequency = st.slider("Frekuensi aktivitas fisik (0–3)", min_value=0, max_value=3, value=1)
time_using_technology = st.slider("Waktu menggunakan teknologi (jam/hari)", min_value=0, max_value=5, value=2)
eating_between_meals = st.selectbox("Sering makan di antara jam makan?", options=list(caec_map.keys()))
transportation = st.selectbox("Jenis transportasi utama", options=list(mtrans_map.keys()))

# Encode input ke nilai numerik
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_map[gender]],
    'Height': [height],
    'Weight': [weight],
    'CALC': [calc_map[calories]],
    'FAVC': [favc_map[favorite_high_calorie]],
    'FCVC': [frequency_of_vegetables],
    'NCP': [number_of_meals],
    'CAEC': [caec_map[eating_between_meals]],
    'SMOKE': [smoke_map[smoke]],
    'CH2O': [consumption_of_water],
    'family_history_with_overweight': [favc_map[family_history]],
    'FAF': [physical_activity_frequency],
    'TUE': [time_using_technology],
    'MTRANS': [mtrans_map[transportation]],
})

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)[0]
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
    st.success(f"Prediksi Kategori Obesitas: **{result}**")
