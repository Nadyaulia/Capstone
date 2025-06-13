import streamlit as st
import pandas as pd
import pickle

# Judul aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Masukkan data pengguna untuk memprediksi kategori obesitas.")

# Load model dan encoder
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_model()

# Fungsi untuk preprocessing input
def preprocess_input(data):
    # Ubah Gender ke numerik
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    # Encode CAEC, MTRANS, dll. sesuai dengan LabelEncoder saat training
    return data

# Input pengguna
age = st.number_input("Umur", min_value=10, max_value=100, value=25)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
family_history = st.selectbox("Riwayat Keluarga Obesitas", ["no", "yes"])
favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori (FAVC)", ["no", "yes"])
fcvc = st.slider("Frekuensi Konsumsi Sayuran (FCVC)", 1, 3, 2)
ncp = st.slider("Jumlah Makan Per Hari (NCP)", 1, 4, 3)
smoke = st.selectbox("Perokok (SMOKE)", ["no", "yes"])
ch2o = st.slider("Konsumsi Air Per Hari (CH2O)", 1, 3, 2)
faf = st.slider("Aktivitas Fisik Frekuensi (FAF)", 0, 3, 1)
tue = st.slider("Waktu Layar Harian (TUE)", 0, 5, 2)
caec = st.selectbox("Kebiasaan Ngemil di Antar Waktu (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi Utama", ["Public_Transportation", "Walking", "Motorbike", "Automobile", "Bike"])

# Simpan input ke dalam DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Height': [height],
    'Weight': [weight],
    'family_history_with_overweight': [family_history],
    'FAVC': [favc],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'SMOKE': [smoke],
    'CH20': [ch2o],
    'FAF': [faf],
    'TUE': [tue],
    'CAEC': [caec],
    'MTRANS': [mtrans]
})

# Preprocessing
input_data_encoded = input_data.copy()
input_data_encoded['Gender'] = input_data_encoded['Gender'].map({'Male': 0, 'Female': 1})
input_data_encoded['family_history_with_overweight'] = input_data_encoded['family_history_with_overweight'].map({'no': 0, 'yes': 1})
input_data_encoded['FAVC'] = input_data_encoded['FAVC'].map({'no': 0, 'yes': 1})
input_data_encoded['SMOKE'] = input_data_encoded['SMOKE'].map({'no': 0, 'yes': 1})

# One-hot encoding untuk CAEC dan MTRANS (pastikan sesuai saat training)
input_data_encoded = pd.get_dummies(input_data_encoded, columns=['CAEC', 'MTRANS'], drop_first=True)

# Pastikan urutan fitur sama seperti saat pelatihan
required_columns = ['Age', 'Gender', 'Height', 'Weight', 'family_history_with_overweight',
                    'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH20', 'FAF', 'TUE',
                    'CAEC_Sometimes', 'CAEC_Frequently', 'CAEC_Always',
                    'MTRANS_Walking', 'MTRANS_Motorbike', 'MTRANS_Automobile', 'MTRANS_Bike']

# Tambahkan kolom dummy jika tidak ada
for col in required_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Urutkan ulang kolom
input_data_encoded = input_data_encoded[required_columns]

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data_encoded)
    result = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Prediksi Kategori Obesitas: **{result}**")
