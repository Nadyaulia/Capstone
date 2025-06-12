import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set judul aplikasi
st.set_page_config(page_title="Aplikasi EDA Capstone - Obesity Dataset")
st.title('Proses Exploratory Data Analysis (EDA) - Obesity Dataset')

st.write("Aplikasi ini menyajikan hasil analisis data dari proyek Capstone, khususnya pada dataset obesitas.")

# --- Bagian Pemuatan Data ---
st.subheader('1. Pemuatan Data')
# Asumsi file data ada di folder 'data' atau di root jika di-deploy
# Jika Anda meng-uploadnya ke GitHub, pastikan path-nya benar relatif terhadap aplikasi Streamlit Anda.
DATA_PATH = 'deploy/ObesityDataSet.csv' # Sesuaikan path jika file berada di subfolder (misal: 'data/ObesityDataSet.csv')

@st.cache_data # Dekorator ini akan menyimpan data dalam cache untuk performa
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File data tidak ditemukan di {path}. Pastikan file 'ObesityDataSet.csv' ada di lokasi yang benar di repositori Anda.")
        return None

df = load_data(DATA_PATH)

if df is not None:
    st.write("Data berhasil dimuat. Berikut 5 baris pertama:")
    st.dataframe(df.head())

    st.write(f"Ukuran Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    st.write("Informasi Kolom:")
    
    # Untuk menampilkan df.info() dengan baik di Streamlit:
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # --- Bagian Analisis Statistik Deskriptif ---
    st.subheader('2. Statistik Deskriptif')
    st.write("Statistik deskriptif untuk kolom numerik:")
    st.dataframe(df.describe())

    st.write("Statistik deskriptif untuk kolom kategori (nilai unik dan frekuensi):")
    # Identifikasi kolom kategorikal (contoh)
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        st.write(f"**Kolom: {col}**")
        st.write(df[col].value_counts())

    # --- Bagian Visualisasi Data ---
    st.subheader('3. Visualisasi Data')

    # Contoh visualisasi yang umum ada di EDA untuk dataset seperti ini:

    # 3.1 Distribusi Kolom Numerik
    st.write("### Distribusi Kolom Numerik")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        selected_numeric_col = st.selectbox('Pilih kolom numerik untuk melihat distribusi:', numeric_cols)
        if selected_numeric_col:
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df[selected_numeric_col], kde=True, ax=ax_hist)
            ax_hist.set_title(f'Distribusi {selected_numeric_col}')
            ax_hist.set_xlabel(selected_numeric_col)
            ax_hist.set_ylabel('Frekuensi')
            st.pyplot(fig_hist)
    else:
        st.warning("Tidak ada kolom numerik yang ditemukan untuk visualisasi distribusi.")

    # 3.2 Count Plot untuk Kolom Kategorikal
    st.write("### Distribusi Kolom Kategorikal")
    if not categorical_cols.empty:
        selected_categorical_col = st.selectbox('Pilih kolom kategorikal untuk melihat distribusi:', categorical_cols)
        if selected_categorical_col:
            fig_count, ax_count = plt.subplots()
            sns.countplot(y=df[selected_categorical_col], order=df[selected_categorical_col].value_counts().index, ax=ax_count)
            ax_count.set_title(f'Distribusi {selected_categorical_col}')
            ax_count.set_xlabel('Jumlah')
            ax_count.set_ylabel(selected_categorical_col)
            st.pyplot(fig_count)
    else:
        st.warning("Tidak ada kolom kategorikal yang ditemukan untuk visualisasi distribusi.")
    
    # 3.3 Heatmap Korelasi (untuk kolom numerik)
    st.write("### Heatmap Korelasi Antar Kolom Numerik")
    if not numeric_cols.empty and len(numeric_cols) > 1:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title('Heatmap Korelasi')
        st.pyplot(fig_corr)
    else:
        st.warning("Tidak cukup kolom numerik untuk membuat heatmap korelasi.")

    # 3.4 Box Plot untuk melihat outlier dan distribusi per kategori
    st.write("### Box Plot: Distribusi Numerik Berdasarkan Kategori")
    if not numeric_cols.empty and not categorical_cols.empty:
        col1, col2 = st.columns(2)
        with col1:
            box_numeric_col = st.selectbox('Pilih kolom numerik:', numeric_cols, key='box_num')
        with col2:
            box_categorical_col = st.selectbox('Pilih kolom kategorikal:', categorical_cols, key='box_cat')
        
        if box_numeric_col and box_categorical_col:
            fig_box, ax_box = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=df[box_categorical_col], y=df[box_numeric_col], ax=ax_box)
            ax_box.set_title(f'Box Plot {box_numeric_col} berdasarkan {box_categorical_col}')
            ax_box.set_xlabel(box_categorical_col)
            ax_box.set_ylabel(box_numeric_col)
            plt.xticks(rotation=45, ha='right') # Memutar label X agar tidak tumpang tindih
            st.pyplot(fig_box)
    else:
        st.warning("Tidak cukup kolom numerik atau kategorikal untuk membuat Box Plot.")


    # Tambahkan lebih banyak visualisasi dan analisis Anda di sini sesuai dengan notebook Capstone Anda
    
    # Contoh interaktivitas sederhana (sudah ada di kode Anda, ini tetap relevan)
    st.subheader('4. Eksplorasi Kolom')
    selected_column = st.selectbox(
        'Pilih kolom untuk melihat nilai unik dan jumlahnya:',
        df.columns
    )
    if selected_column:
        st.write(f"Nilai unik di '{selected_column}':")
        st.write(df[selected_column].unique())
        st.write(f"Jumlah nilai unik: {df[selected_column].nunique()}")
        if df[selected_column].dtype == 'object': # Untuk kolom kategorikal
            st.write("Frekuensi setiap nilai unik:")
            st.write(df[selected_column].value_counts())

else:
    st.warning("Aplikasi tidak dapat berjalan tanpa data. Mohon periksa file data.")
