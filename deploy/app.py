import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load model and encoder
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_model()

# Preprocessing input
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

    # Normalisasi fitur numerik
    scaler = StandardScaler()
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    data[numerical_features] = scaler.fit_transform(data[numerical_features].values.reshape(1, -1))

    return data

# Main app
st.title("Prediksi Kategori Obesitas")
st.write("Silakan lengkapi data diri Anda untuk mengetahui kategori obesitas.")

# Input data
age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=25, key="age_input")
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], key="gender_select")
height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.7, key="height_input")
weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=70, key="weight_input")
calc = st.selectbox("Seberapa sering konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"], key="calc_select")
favc = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"], key="favc_select")
fcvc = st.slider("Frekuensi makan sayur per minggu", min_value=0, max_value=10, value=2, key="fcvc_slider")
ncp = st.slider("Jumlah makan per hari", min_value=1, max_value=10, value=3, key="ncp_slider")
scc = st.selectbox("Apakah Anda mencatat kalori yang dikonsumsi?", ["yes", "no"], key="scc_select")
smoke = st.selectbox("Apakah Anda perokok?", ["yes", "no"], key="smoke_select")
ch2o = st.slider("Konsumsi air per hari (liter)", min_value=0, max_value=5, value=2, key="ch2o_slider")
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"], key="family_history_select")
faf = st.slider("Frekuensi aktivitas fisik per minggu", min_value=0, max_value=7, value=2, key="faf_slider")
tue = st.slider("Waktu layar/hari (jam)", min_value=0, max_value=5, value=2, key="tue_slider")
caec = st.selectbox("Seberapa sering ngemil di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"], key="caec_select")
mtrans = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"], key="mtrans_select")

# Tombol prediksi
if st.button("Prediksi"):
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
    input_data = preprocess_input(input_data)

    # Lakukan prediksi
    prediction = model.predict(input_data)[0]

    # Decode hasil prediksi
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

    # Tampilkan hasil prediksi
    st.success(f"Prediksi Kategori Obesitas: **{result}**")
