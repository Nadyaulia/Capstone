
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Gunakan path relatif untuk memastikan file ditemukan
DATA_PATH = "ObesityDataSet.csv"

def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

# Preprocess the data
def preprocess_data(data):
    # Handle missing values, encode categorical variables, etc.
    # Example:
    data = data.dropna()  # Handle missing values
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['family_history_with_overweight'] = le.fit_transform(data['family_history_with_overweight'])
    data['FAVC'] = le.fit_transform(data['FAVC'])
    data['CAEC'] = le.fit_transform(data['CAEC'])
    data['SMOKE'] = le.fit_transform(data['SMOKE'])
    data['SCC'] = le.fit_transform(data['SCC'])
    data['NCP'] = le.fit_transform(data['NCP'])
    data['CH2O'] = le.fit_transform(data['CH2O'])
    data['CALC'] = le.fit_transform(data['CALC'])
    data['MTRANS'] = le.fit_transform(data['MTRANS'])
    data['NObeyesdad'] = le.fit_transform(data['NObeyesdad'])
    return data

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Make predictions
def predict_obesity(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

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

        # Load and preprocess data
        data = load_data()
        preprocessed_data = preprocess_data(data)

        # Train the model
        X = preprocessed_data.drop(columns=['NObeyesdad'])
        y = preprocessed_data['NObeyesdad']
        model, scaler = train_model(X, y)

        # Make prediction
        prediction = predict_obesity(model, scaler, input_data.values)

        # Display the result
        st.subheader("Hasil Prediksi")
        st.write(f"Kategori Obesitas: {prediction[0]}")

# Run the app
if __name__ == "__main__":
    main()
