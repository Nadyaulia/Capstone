import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
st.write("File dalam direktori sekarang:", os.listdir())


# Load model
model = joblib.load("deploy/model_obesitas.pkl")  # Pastikan model ini sudah tersedia
scaler = joblib.load("deploy/scaler.pkl")


# Judul aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Silakan lengkapi data diri Anda untuk mengetahui kategori obesitas.")

# Form untuk input user
with st.form("form_prediksi"):
    age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=25, key="usia")
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], key="gender")
    height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.7, key="tinggi")
    weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=70, key="berat")
    favc = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"], key="favc")
    fcvc = st.slider("Frekuensi makan sayur per minggu", min_value=0, max_value=10, value=2, key="fcvc")
    ncp = st.slider("Jumlah makan per hari", min_value=1, max_value=10, value=3, key="ncp")
    ch2o = st.slider("Konsumsi air per hari (liter)", min_value=0, max_value=5, value=2, key="ch2o")
    faf = st.slider("Frekuensi aktivitas fisik per minggu", min_value=0, max_value=7, value=2, key="faf")
    tue = st.slider("Waktu layar per hari (jam)", min_value=0, max_value=5, value=2, key="tue")
    smoke = st.selectbox("Apakah Anda perokok?", ["yes", "no"], key="smoke")
    calc = st.selectbox("Seberapa sering konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"], key="calc")
    caec = st.selectbox("Seberapa sering ngemil di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"], key="caec")
    mtrans = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"], key="mtrans")
    family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"], key="family")
    scc = st.selectbox("Apakah Anda mencatat kalori yang dikonsumsi?", ["yes", "no"], key="scc")

    submitted = st.form_submit_button("Prediksi Sekarang")

# Preprocessing fungsi
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
    data['Gender'] = data['Gender'].map(gender_map)
    data['CALC'] = data['CALC'].map(calc_map)
    data['FAVC'] = data['FAVC'].map(favc_map)
    data['SMOKE'] = data['SMOKE'].map(smoke_map)
    data['SCC'] = data['SCC'].map(scc_map)
    data['CAEC'] = data['CAEC'].map(caec_map)
    data['MTRANS'] = data['MTRANS'].map(mtrans_map)

    # Normalisasi
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    data[num_cols] = scaler.transform(data[num_cols])

    return data


# Jika tombol ditekan
if submitted:
    # Buat DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue,
        'SMOKE': smoke,
        'SCC': scc,
        'CAEC': caec,
        'CALC': calc,
        'MTRANS': mtrans,
        'family_history_with_overweight': family_history
    }])

    input_encoded = preprocess_input(input_data)

    prediction = model.predict(input_encoded)[0]
    label_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Overweight_Level_I",
        3: "Overweight_Level_II",
        4: "Obesity_Type_I",
        5: "Obesity_Type_II",
        6: "Obesity_Type_III"
    }
    result = label_map.get(prediction, "Tidak diketahui")
    st.success(f"Hasil Prediksi: **{result}**")
