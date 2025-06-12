import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set judul aplikasi
st.set_page_config(page_title="Aplikasi EDA Capstone")
st.title('Proses Exploratory Data Analysis (EDA)')

st.write("Aplikasi ini menyajikan hasil analisis data dari proyek Capstone.")

# --- Bagian Pemuatan Data ---
st.subheader('1. Pemuatan Data')
# Asumsi file data ada di folder 'data'
DATA_PATH = 'ObesityDataSet.csv' # Ganti dengan nama file data Anda

@st.cache_data # Dekorator ini akan menyimpan data dalam cache untuk performa
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File data tidak ditemukan di {path}. Pastikan file ada di repositori Anda.")
        return None

df = load_data(DATA_PATH)

if df is not None:
    st.write("Data berhasil dimuat. Berikut 5 baris pertama:")
    st.dataframe(df.head())

    st.write(f"Ukuran Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    st.write("Informasi Kolom:")
    #st.dataframe(df.info()) # st.dataframe() bisa menampilkan output df.info() dengan sedikit tweaking atau gunakan st.text()

    # Untuk menampilkan df.info() dengan baik di Streamlit:
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


    # --- Bagian Analisis Statistik Deskriptif ---
    st.subheader('2. Statistik Deskriptif')
    st.dataframe(df.describe())

    # --- Bagian Visualisasi (Contoh) ---
    st.subheader('3. Visualisasi Data')

    st.write("Distribusi Kolom Numerik (Contoh: 'Age')")
    # Pastikan kolom ini ada di data Anda
    if 'Age' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax)
        ax.set_title('Distribusi Umur')
        st.pyplot(fig)
    else:
        st.warning("Kolom 'Age' tidak ditemukan untuk visualisasi.")

   # st.write("Scatter Plot (Contoh: 'Age' vs 'Fare')")
    #if 'Age' in df.columns and 'Fare' in df.columns:
    #    fig, ax = plt.subplots()
     #   sns.scatterplot(x='Age', y='Fare', data=df, ax=ax)
    #    ax.set_title('Umur vs Harga Tiket')
     #   st.pyplot(fig)
    else:
        st.warning("Kolom 'Age' atau 'Fare' tidak ditemukan untuk visualisasi.")

    # Tambahkan lebih banyak visualisasi dan analisis Anda di sini
    # Contoh interaktivitas sederhana
    selected_column = st.selectbox(
        'Pilih kolom untuk melihat nilai unik:',
        df.columns
    )
    if selected_column:
        st.write(f"Nilai unik di '{selected_column}':")
        st.write(df[selected_column].unique())

else:
    st.warning("Aplikasi tidak dapat berjalan tanpa data. Mohon periksa file data.")
