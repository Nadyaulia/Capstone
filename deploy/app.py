import streamlit as st
import pandas as pd
import joblib
import os

st.title("Prediksi Kategori Obesitas")

# Debug: tampilkan file yang tersedia
st.write("📁 File dalam direktori sekarang:", os.listdir())

# Coba load model dan scaler dengan try-except
try:
    model = joblib.load("obesity_model.pkl")
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")

try:
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Scaler berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat scaler: {e}")

st.write("Silakan lengkapi data diri Anda untuk mengetahui kategori obesitas.")

# Form input pengguna
with st.form("form_prediksi"):
    age = st.number_input("Usia (tahun)", 1, 120, 25)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    height = st.number_input("Tinggi Badan (meter)", 0.5, 2.5, 1.7)
    weight = st.number_input("Berat Badan (kg)", 20, 200, 70)
    favc = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"])
    fcvc = st.slider("Frekuensi makan sayur per minggu", 0, 10, 2)
    ncp = st.slider("Jumlah makan per hari", 1, 10, 3)
    ch2o = st.slider("Konsumsi air per hari (liter)", 0, 5, 2)
    faf = st.slider("Frekuensi aktivitas fisik per minggu", 0, 7, 2)
    tue = st.slider("Waktu layar per hari (jam)", 0, 5, 2)
    smoke = st.selectbox("Apakah Anda perokok?", ["yes", "no"])
    calc = st.selectbox("Seberapa sering konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
    caec = st.selectbox("Seberapa sering ngemil di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
    family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"])
    scc = st.selectbox("Apakah Anda mencatat kalori yang dikonsumsi?", ["yes", "no"])

    submitted = st.form_submit_button("Prediksi Sekarang")

# Fungsi preprocessing input
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

    # Apply mapping
    data['Gender'] = data['Gender'].map(gender_map)
    data['CALC'] = data['CALC'].map(calc_map)
    data['FAVC'] = data['FAVC'].map(favc_map)
    data['SMOKE'] = data['SMOKE'].map(smoke_map)
    data['SCC'] = data['SCC'].map(scc_map)
    data['CAEC'] = data['CAEC'].map(caec_map)
    data['MTRANS'] = data['MTRANS'].map(mtrans_map)

    # Normalisasi numerik
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    data[num_cols] = scaler.transform(data[num_cols])

    return data

# Proses prediksi
if submitted:
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

    try:
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
        st.success(f"🎯 Hasil Prediksi: **{result}**")
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
