import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set judul aplikasi
st.set_page_config(page_title="Aplikasi EDA Capstone - Obesity Dataset")
st.title('Proses Exploratory Data Analysis (EDA) - Obesity Dataset')

st.write("Aplikasi ini menyajikan hasil analisis data dari proyek Capstone. Anda dapat mengunggah dataset Anda sendiri untuk dianalisis.")

# --- Bagian Pemuatan Data ---
st.subheader('1. Pemuatan Data')

uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])

df = None # Inisialisasi df sebagai None

if uploaded_file is not None:
    # Membaca file yang diunggah
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File berhasil diunggah dan dibaca!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")

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
    # Gunakan .select_dtypes(include='object', 'category') untuk robustness
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        for col in categorical_cols:
            st.write(f"*Kolom: {col}*")
            st.write(df[col].value_counts())
    else:
        st.write("Tidak ada kolom kategorikal yang terdeteksi.")


    # --- Bagian Visualisasi Data ---
    st.subheader('3. Visualisasi Data')

    # Dapatkan kolom numerik dan kategorikal setelah data dimuat
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # 3.1 Distribusi Kolom Numerik
    st.write("### Distribusi Kolom Numerik")
    if not numeric_cols.empty:
        selected_numeric_col = st.selectbox('Pilih kolom numerik untuk melihat distribusi:', numeric_cols, key='dist_num_select')
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
        selected_categorical_col = st.selectbox('Pilih kolom kategorikal untuk melihat distribusi:', categorical_cols, key='dist_cat_select')
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
        st.warning("Tidak cukup kolom numerik (minimal 2) untuk membuat heatmap korelasi.")

    # 3.4 Box Plot untuk melihat outlier dan distribusi per kategori
    st.write("### Box Plot: Distribusi Numerik Berdasarkan Kategori")
    if not numeric_cols.empty and not categorical_cols.empty:
        col1, col2 = st.columns(2)
        with col1:
            box_numeric_col = st.selectbox('Pilih kolom numerik:', numeric_cols, key='box_num_select')
        with col2:
            box_categorical_col = st.selectbox('Pilih kolom kategorikal:', categorical_cols, key='box_cat_select')
        
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
        df.columns, key='explore_col_select'
    )
    if selected_column:
        st.write(f"Nilai unik di '{selected_column}':")
        st.write(df[selected_column].unique())
        st.write(f"Jumlah nilai unik: {df[selected_column].nunique()}")
        if df[selected_column].dtype in ['object', 'category']: # Untuk kolom kategorikal
            st.write("Frekuensi setiap nilai unik:")
            st.write(df[selected_column].value_counts())

else:
    st.info("Silakan unggah file CSV Anda untuk memulai proses EDA.")


# --- Bagian 6: Konversi Tipe Data Otomatis ---
if df is not None:
    st.subheader('6. Konversi Tipe Data Otomatis')

    # Menampilkan info tipe data sebelum konversi
    st.write("### Informasi Tipe Data Sebelum Konversi")
    buffer_before = io.StringIO()
    df.info(buf=buffer_before)
    s_before = buffer_before.getvalue()
    st.text(s_before)

    # Membuat salinan dataframe agar tidak merusak data asli
    df_auto_converted = df.copy()

    st.write("### Memulai Konversi Otomatis...")

    for col in df_auto_converted.columns:
        # Jika kolom memiliki hanya sedikit nilai unik, ubah menjadi 'category'
        if df_auto_converted[col].dtype == 'object' and df_auto_converted[col].nunique() / len(df_auto_converted) < 0.05:
            df_auto_converted[col] = df_auto_converted[col].astype('category')
        # Jika kolom numerik dan bisa jadi int tanpa desimal, ubah ke int64
        elif pd.api.types.is_numeric_dtype(df_auto_converted[col]):
            if (df_auto_converted[col] % 1 == 0).all():
                df_auto_converted[col] = pd.to_numeric(df_auto_converted[col], errors='coerce').astype('Int64')
            else:
                df_auto_converted[col] = pd.to_numeric(df_auto_converted[col], errors='coerce').astype('float64')
        # Jika kolom terlihat seperti tanggal
        elif pd.api.types.is_datetime64_any_dtype(df_auto_converted[col]):
            df_auto_converted[col] = pd.to_datetime(df_auto_converted[col], errors='coerce')
    
    st.success("Konversi tipe data otomatis selesai.")

    # Menampilkan info tipe data setelah konversi
    st.write("### Informasi Tipe Data Setelah Konversi Otomatis")
    buffer_after = io.StringIO()
    df_auto_converted.info(buf=buffer_after)
    s_after = buffer_after.getvalue()
    st.text(s_after)

    # Tampilkan preview data hasil konversi
    st.write("### Preview Data Setelah Konversi Otomatis")
    st.dataframe(df_auto_converted.head())

    # Simpan df_auto_converted untuk digunakan di visualisasi selanjutnya jika diperlukan
    df = df_auto_converted
