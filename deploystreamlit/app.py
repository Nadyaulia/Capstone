import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Debugging: Cek isi direktori saat ini
st.write("Files in current directory:", os.listdir())

# Fungsi untuk memuat model dan scaler
def load_model_and_scaler():
    try:
        # Memuat model
        model_path = "obesity_model.pkl"
        model = joblib.load(model_path)
        
        # Memuat scaler
        scaler_path = "scaler.pkl"
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file ada di direktori yang benar.")
        return None, None

# Main Streamlit app
def main():
    st.title("Prediksi Kategori Obesitas")
    st.write("Silakan lengkapi data diri Anda untuk mengetahui kategori obesitas.")

    # Input fields
    age = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=25)
    height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.70)
    weight = st.number_input("Berat Badan (kg)", min_value=1, max_value=200, value=70)
    vegetables_per_week = st.slider("Frekuensi makan sayur per minggu", 0, 10, 2)
    meals_per_day = st.slider("Jumlah makan per hari", 1, 10, 3)
    snacks_between_meals = st.selectbox("Seberapa sering ngemil di antara waktu makan?", ["no", "sometimes", "always"])
    primary_transport = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Motorbike", "Bike", "Walking", "Car"])
    family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"])
    track_calories = st.selectbox("Apakah Anda mencatat kalori yang dikonsumsi?", ["yes", "no"])

    # Button to submit the form
    if st.button("Prediksi Sekarang"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'FAVC': [vegetables_per_week],
            'NCP': [meals_per_day],
            'CAEC': [snacks_between_meals],
            'MTRANS': [primary_transport],
            'family_history_with_overweight': [family_history],
            'CALC': [track_calories]
        })

        # Encode categorical variables
        le = LabelEncoder()
        input_data['CAEC'] = le.fit_transform(input_data['CAEC'])
        input_data['MTRANS'] = le.fit_transform(input_data['MTRANS'])
        input_data['family_history_with_overweight'] = le.fit_transform(input_data['family_history_with_overweight'])
        input_data['CALC'] = le.fit_transform(input_data['CALC'])

        # Load model and scaler
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            return

        # Scale input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display the result
        st.subheader("Hasil Prediksi")
        st.write(f"Kategori Obesitas: {prediction[0]}")

# Run the app
if __name__ == "__main__":
    main()
