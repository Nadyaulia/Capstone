import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib # Untuk menyimpan/memuat model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Aplikasi Lengkap Capstone Data Science",
    layout="wide", # Menggunakan layout lebar untuk lebih banyak ruang
    initial_sidebar_state="expanded"
)
st.title('Proses Lengkap Proyek Capstone Data Science')
st.write("Aplikasi interaktif ini memungkinkan Anda mengunggah data, melakukan EDA, preprocessing, pemodelan, evaluasi, dan hyperparameter tuning.")

# --- Inisialisasi State Session ---
# Digunakan untuk menyimpan data dan model agar tetap ada antar rerun
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None


# --- Bagian Pemuatan Data (Selalu ada di awal) ---
st.sidebar.header('1. Pemuatan Data')
uploaded_file = st.sidebar.file_uploader("Unggah file CSV Anda", type=["csv"])

if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.sidebar.success("File berhasil diunggah!")
        st.session_state.processed_df = None # Reset processed_df jika data baru diunggah
        st.session_state.model = None # Reset model juga
        st.session_state.pipeline = None
    except Exception as e:
        st.sidebar.error(f"Terjadi kesalahan saat membaca file: {e}")

if st.session_state.df is None:
    st.info("Silakan unggah file CSV Anda di sidebar untuk memulai.")
else:
    # --- Tab Navigasi ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "⚙️ Preprocessing", "📈 Modeling & Evaluation", "🧪 Hyperparameter Tuning"])

    # --- Tab EDA ---
    with tab1:
        st.header('Exploratory Data Analysis (EDA)')
        st.write("Lihat ringkasan data, statistik deskriptif, dan visualisasi awal.")

        st.subheader('Overview Data')
        st.write("Berikut 5 baris pertama dari data yang diunggah:")
        st.dataframe(st.session_state.df.head())
        st.write(f"Ukuran Data: {st.session_state.df.shape[0]} baris, {st.session_state.df.shape[1]} kolom")
        
        st.write("Informasi Kolom:")
        buffer = io.StringIO()
        st.session_state.df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader('Statistik Deskriptif')
        st.write("Statistik deskriptif untuk kolom numerik:")
        st.dataframe(st.session_state.df.describe())

        st.write("Statistik deskriptif untuk kolom kategorikal:")
        categorical_cols_eda = st.session_state.df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols_eda.empty:
            for col in categorical_cols_eda:
                st.write(f"**Kolom: {col}**")
                st.write(st.session_state.df[col].value_counts())
        else:
            st.write("Tidak ada kolom kategorikal yang terdeteksi.")

        st.subheader('Visualisasi Data')
        numeric_cols_eda = st.session_state.df.select_dtypes(include=np.number).columns
        categorical_cols_eda_viz = st.session_state.df.select_dtypes(include=['object', 'category']).columns

        # 3.1 Distribusi Kolom Numerik
        st.write("#### Distribusi Kolom Numerik (Histogram/KDE)")
        if not numeric_cols_eda.empty:
            selected_numeric_col_eda = st.selectbox('Pilih kolom numerik untuk melihat distribusi:', numeric_cols_eda, key='eda_num_select')
            if selected_numeric_col_eda:
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(st.session_state.df[selected_numeric_col_eda], kde=True, ax=ax_hist)
                ax_hist.set_title(f'Distribusi {selected_numeric_col_eda}')
                ax_hist.set_xlabel(selected_numeric_col_eda)
                ax_hist.set_ylabel('Frekuensi')
                st.pyplot(fig_hist)
        else:
            st.warning("Tidak ada kolom numerik yang ditemukan untuk visualisasi distribusi.")

        # 3.2 Count Plot untuk Kolom Kategorikal
        st.write("#### Distribusi Kolom Kategorikal (Count Plot)")
        if not categorical_cols_eda_viz.empty:
            selected_categorical_col_eda = st.selectbox('Pilih kolom kategorikal untuk melihat distribusi:', categorical_cols_eda_viz, key='eda_cat_select')
            if selected_categorical_col_eda:
                fig_count, ax_count = plt.subplots()
                sns.countplot(y=st.session_state.df[selected_categorical_col_eda], order=st.session_state.df[selected_categorical_col_eda].value_counts().index, ax=ax_count)
                ax_count.set_title(f'Distribusi {selected_categorical_col_eda}')
                ax_count.set_xlabel('Jumlah')
                ax_count.set_ylabel(selected_categorical_col_eda)
                st.pyplot(fig_count)
        else:
            st.warning("Tidak ada kolom kategorikal yang ditemukan untuk visualisasi distribusi.")
        
        # 3.3 Heatmap Korelasi (untuk kolom numerik)
        st.write("#### Heatmap Korelasi Antar Kolom Numerik")
        if not numeric_cols_eda.empty and len(numeric_cols_eda) > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            corr_matrix = st.session_state.df[numeric_cols_eda].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            ax_corr.set_title('Heatmap Korelasi')
            st.pyplot(fig_corr)
        else:
            st.warning("Tidak cukup kolom numerik (minimal 2) untuk membuat heatmap korelasi.")

        # 3.4 Box Plot untuk melihat outlier dan distribusi per kategori
        st.write("#### Box Plot: Distribusi Numerik Berdasarkan Kategori")
        if not numeric_cols_eda.empty and not categorical_cols_eda_viz.empty:
            col1_box, col2_box = st.columns(2)
            with col1_box:
                box_numeric_col_eda = st.selectbox('Pilih kolom numerik:', numeric_cols_eda, key='eda_box_num')
            with col2_box:
                box_categorical_col_eda = st.selectbox('Pilih kolom kategorikal:', categorical_cols_eda_viz, key='eda_box_cat')
            
            if box_numeric_col_eda and box_categorical_col_eda:
                fig_box, ax_box = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=st.session_state.df[box_categorical_col_eda], y=st.session_state.df[box_numeric_col_eda], ax=ax_box)
                ax_box.set_title(f'Box Plot {box_numeric_col_eda} berdasarkan {box_categorical_col_eda}')
                ax_box.set_xlabel(box_categorical_col_eda)
                ax_box.set_ylabel(box_numeric_col_eda)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_box)
        else:
            st.warning("Tidak cukup kolom numerik atau kategorikal untuk membuat Box Plot.")


    # --- Tab Preprocessing ---
    with tab2:
        st.header('Preprocessing Data')
        st.write("Lakukan penanganan missing values, encoding, dan scaling pada data.")

        if st.session_state.df is not None:
            processed_df_temp = st.session_state.df.copy() # Bekerja pada salinan
            
            st.subheader('Missing Values')
            missing_data = processed_df_temp.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if not missing_data.empty:
                st.write("Kolom dengan Missing Values:")
                st.dataframe(missing_data.rename('Jumlah Missing Values'))
                
                missing_strategy = st.radio(
                    "Pilih strategi penanganan missing values:",
                    ('Hapus baris', 'Imputasi'), key='missing_strategy'
                )

                if missing_strategy == 'Hapus baris':
                    initial_rows = processed_df_temp.shape[0]
                    processed_df_temp.dropna(inplace=True)
                    st.success(f"Dihapus {initial_rows - processed_df_temp.shape[0]} baris dengan missing values.")
                elif missing_strategy == 'Imputasi':
                    # Imputasi numerik
                    numeric_cols_impute = processed_df_temp.select_dtypes(include=np.number).columns
                    if not numeric_cols_impute.empty:
                        num_imputer_strategy = st.selectbox(
                            "Strategi imputasi numerik:",
                            ('mean', 'median', 'most_frequent'), key='num_imputer'
                        )
                        imputer_num = SimpleImputer(strategy=num_imputer_strategy)
                        processed_df_temp[numeric_cols_impute] = imputer_num.fit_transform(processed_df_temp[numeric_cols_impute])
                        st.success(f"Missing values numerik diisi dengan {num_imputer_strategy}.")
                    
                    # Imputasi kategorikal
                    categorical_cols_impute = processed_df_temp.select_dtypes(include=['object', 'category']).columns
                    if not categorical_cols_impute.empty:
                        cat_imputer_strategy = st.selectbox(
                            "Strategi imputasi kategorikal:",
                            ('most_frequent',), key='cat_imputer'
                        )
                        imputer_cat = SimpleImputer(strategy=cat_imputer_strategy)
                        # Imputasi kategorikal harus dihandle secara terpisah atau dengan ColumnTransformer
                        for col in categorical_cols_impute:
                            processed_df_temp[col] = imputer_cat.fit_transform(processed_df_temp[[col]]).ravel()
                        st.success(f"Missing values kategorikal diisi dengan {cat_imputer_strategy}.")
            else:
                st.info("Tidak ada missing values dalam dataset.")

            st.subheader('Encoding Kategorikal')
            categorical_cols_encode = processed_df_temp.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols_encode.empty:
                encoding_method = st.radio(
                    "Pilih metode encoding:",
                    ('One-Hot Encoding', 'Label Encoding (untuk target jika klasifikasi)'), key='encoding_method'
                )

                if encoding_method == 'One-Hot Encoding':
                    # Hindari meng-encode kolom target jika sudah diputuskan
                    cols_to_ohe = [col for col in categorical_cols_encode if col != st.session_state.target_column]
                    if cols_to_ohe:
                        processed_df_temp = pd.get_dummies(processed_df_temp, columns=cols_to_ohe, drop_first=True)
                        st.success(f"Kolom {', '.join(cols_to_ohe)} di-encode dengan One-Hot Encoding.")
                    else:
                        st.info("Tidak ada kolom kategorikal yang dapat di-One-Hot Encode (selain target).")
                elif encoding_method == 'Label Encoding (untuk target jika klasifikasi)':
                    st.info("Label Encoding biasanya dilakukan pada kolom target untuk masalah klasifikasi.")
                    # Label encoding akan dilakukan pada tahap pemodelan jika target adalah kategorikal
            else:
                st.info("Tidak ada kolom kategorikal yang perlu di-encode.")
            
            st.subheader('Scaling Numerik')
            numeric_cols_scale = processed_df_temp.select_dtypes(include=np.number).columns
            if not numeric_cols_scale.empty:
                scaling_method = st.radio(
                    "Pilih metode scaling (tidak termasuk kolom target jika numerik):",
                    ('None', 'StandardScaler', 'MinMaxScaler'), key='scaling_method'
                )
                
                if scaling_method == 'StandardScaler':
                    scaler = StandardScaler()
                    cols_to_scale = [col for col in numeric_cols_scale if col != st.session_state.target_column]
                    if cols_to_scale:
                        processed_df_temp[cols_to_scale] = scaler.fit_transform(processed_df_temp[cols_to_scale])
                        st.success(f"Kolom numerik di-scale dengan StandardScaler.")
                    else:
                        st.info("Tidak ada kolom numerik yang dapat di-scale (selain target).")
                elif scaling_method == 'MinMaxScaler':
                    scaler = MinMaxScaler() # Anda perlu import MinMaxScaler
                    cols_to_scale = [col for col in numeric_cols_scale if col != st.session_state.target_column]
                    if cols_to_scale:
                        processed_df_temp[cols_to_scale] = scaler.fit_transform(processed_df_temp[cols_to_scale])
                        st.success(f"Kolom numerik di-scale dengan MinMaxScaler.")
                    else:
                        st.info("Tidak ada kolom numerik yang dapat di-scale (selain target).")
                else:
                    st.info("Tidak ada scaling yang diterapkan.")
            else:
                st.info("Tidak ada kolom numerik yang perlu di-scale.")

            st.session_state.processed_df = processed_df_temp
            st.subheader('Data Setelah Preprocessing')
            if st.session_state.processed_df is not None:
                st.dataframe(st.session_state.processed_df.head())
                st.write(f"Ukuran Data Setelah Preprocessing: {st.session_state.processed_df.shape[0]} baris, {st.session_state.processed_df.shape[1]} kolom")
                buffer_proc = io.StringIO()
                st.session_state.processed_df.info(buf=buffer_proc)
                s_proc = buffer_proc.getvalue()
                st.text(s_proc)
        else:
            st.warning("Silakan unggah data terlebih dahulu di tab 'EDA'.")


    # --- Tab Modeling & Evaluation ---
    with tab3:
        st.header('Modeling & Evaluation')
        st.write("Pilih target dan fitur, latih model, dan evaluasi performanya.")

        if st.session_state.processed_df is not None:
            available_columns = st.session_state.processed_df.columns.tolist()

            st.subheader('Pilih Target dan Fitur')
            st.session_state.target_column = st.selectbox(
                'Pilih kolom target (y):',
                available_columns,
                key='target_col_select'
            )

            if st.session_state.target_column:
                # Kolom yang tersisa adalah kandidat fitur
                feature_candidates = [col for col in available_columns if col != st.session_state.target_column]
                st.session_state.feature_columns = st.multiselect(
                    'Pilih kolom fitur (X):',
                    feature_candidates,
                    default=feature_candidates, # Default pilih semua
                    key='feature_cols_select'
                )

                if st.session_state.feature_columns:
                    X = st.session_state.processed_df[st.session_state.feature_columns]
                    y = st.session_state.processed_df[st.session_state.target_column]

                    # Periksa apakah target perlu Label Encoding
                    if y.dtype in ['object', 'category']:
                        st.info(f"Kolom target '{st.session_state.target_column}' adalah kategorikal. Melakukan Label Encoding.")
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        st.session_state.label_encoder_classes = le.classes_ # Simpan kelas untuk nanti

                    # Split data
                    test_size = st.slider('Ukuran data test (persentase):', 0.1, 0.5, 0.2, 0.05)
                    random_state = st.number_input('Random State untuk split data:', min_value=0, value=42)
                    
                    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = \
                        train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'int' else None)
                    
                    st.write(f"Ukuran data training: {st.session_state.X_train.shape[0]} baris")
                    st.write(f"Ukuran data testing: {st.session_state.X_test.shape[0]} baris")

                    st.subheader('Pilih dan Latih Model')
                    model_type = st.selectbox(
                        'Pilih jenis model:',
                        ('Random Forest Classifier', 'Logistic Regression'),
                        key='model_type_select'
                    )

                    if model_type == 'Random Forest Classifier':
                        n_estimators = st.slider('Jumlah Estimator (n_estimators):', 10, 500, 100, 10)
                        max_depth = st.slider('Kedalaman Maksimum (max_depth):', 1, 30, 10, 1)
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                    elif model_type == 'Logistic Regression':
                        C_val = st.slider('Inverse of regularization strength (C):', 0.01, 10.0, 1.0, 0.01)
                        model = LogisticRegression(C=C_val, random_state=random_state, solver='liblinear') # solver='liblinear' cocok untuk dataset kecil

                    if st.button('Latih Model', key='train_model_btn'):
                        st.write("Melatih model...")
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.model = model
                        st.success(f"Model {model_type} berhasil dilatih!")

                        st.subheader('Evaluasi Model')
                        if st.session_state.model:
                            y_pred = st.session_state.model.predict(st.session_state.X_test)

                            st.write(f"Akurasi: {accuracy_score(st.session_state.y_test, y_pred):.4f}")
                            st.write(f"Precision: {precision_score(st.session_state.y_test, y_pred, average='weighted'):.4f}")
                            st.write(f"Recall: {recall_score(st.session_state.y_test, y_pred, average='weighted'):.4f}")
                            st.write(f"F1-Score: {f1_score(st.session_state.y_test, y_pred, average='weighted'):.4f}")

                            st.write("Confusion Matrix:")
                            cm = confusion_matrix(st.session_state.y_test, y_pred)
                            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                        xticklabels=st.session_state.label_encoder_classes if 'label_encoder_classes' in st.session_state else ['Class 0', 'Class 1'],
                                        yticklabels=st.session_state.label_encoder_classes if 'label_encoder_classes' in st.session_state else ['Class 0', 'Class 1'])
                            ax_cm.set_xlabel('Predicted')
                            ax_cm.set_ylabel('True')
                            ax_cm.set_title('Confusion Matrix')
                            st.pyplot(fig_cm)

                            st.write("Classification Report:")
                            report = classification_report(st.session_state.y_test, y_pred,
                                                           target_names=st.session_state.label_encoder_classes if 'label_encoder_classes' in st.session_state else None,
                                                           output_dict=True)
                            st.dataframe(pd.DataFrame(report).T)
                        else:
                            st.warning("Model belum dilatih.")
                else:
                    st.warning("Pilih setidaknya satu kolom fitur.")
            else:
                st.warning("Pilih kolom target terlebih dahulu.")
        else:
            st.warning("Silakan unggah dan proses data terlebih dahulu.")

    # --- Tab Hyperparameter Tuning ---
    with tab4:
        st.header('Hyperparameter Tuning')
        st.write("Tingkatkan performa model dengan menyetel hyperparameter menggunakan GridSearchCV.")

        if st.session_state.model is not None and st.session_state.X_train is not None:
            tuning_model_type = st.session_state.model.__class__.__name__

            st.write(f"Melakukan tuning untuk model: **{tuning_model_type}**")

            param_grid = {}
            if tuning_model_type == 'RandomForestClassifier':
                st.subheader("Parameter Grid untuk Random Forest")
                n_estimators_tuned = st.text_input('n_estimators (comma-separated):', '50, 100, 200')
                max_depth_tuned = st.text_input('max_depth (comma-separated, use None for no limit):', '5, 10, None')
                min_samples_split_tuned = st.text_input('min_samples_split (comma-separated):', '2, 5, 10')

                try:
                    param_grid['n_estimators'] = [int(x.strip()) for x in n_estimators_tuned.split(',')]
                    param_grid['max_depth'] = [int(x.strip()) if x.strip() != 'None' else None for x in max_depth_tuned.split(',')]
                    param_grid['min_samples_split'] = [int(x.strip()) for x in min_samples_split_tuned.split(',')]
                except ValueError:
                    st.error("Format input parameter grid salah. Pastikan format angka dipisahkan koma.")
                    param_grid = {} # Reset untuk mencegah error

            elif tuning_model_type == 'LogisticRegression':
                st.subheader("Parameter Grid untuk Logistic Regression")
                C_tuned = st.text_input('C (Inverse of regularization strength, comma-separated):', '0.1, 1.0, 10.0')
                solver_tuned = st.text_input('solver (comma-separated):', 'liblinear, lbfgs')

                try:
                    param_grid['C'] = [float(x.strip()) for x in C_tuned.split(',')]
                    param_grid['solver'] = [x.strip() for x in solver_tuned.split(',')]
                except ValueError:
                    st.error("Format input parameter grid salah. Pastikan format angka dipisahkan koma.")
                    param_grid = {} # Reset untuk mencegah error

            if param_grid:
                st.write("Parameter Grid yang akan digunakan:")
                st.json(param_grid)

                if st.button('Mulai Hyperparameter Tuning (GridSearchCV)', key='tune_model_btn'):
                    with st.spinner('Melakukan tuning... Ini mungkin memerlukan waktu.'):
                        if tuning_model_type == 'RandomForestClassifier':
                            estimator = RandomForestClassifier(random_state=42)
                        elif tuning_model_type == 'LogisticRegression':
                            estimator = LogisticRegression(random_state=42, max_iter=1000) # Tingkatkan max_iter untuk konvergensi

                        grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
                        grid_search.fit(st.session_state.X_train, st.session_state.y_train)

                        st.session_state.best_model = grid_search.best_estimator_
                        st.session_state.best_params = grid_search.best_params_
                        st.session_state.best_score = grid_search.best_score_

                        st.success("Tuning selesai!")
                        st.write(f"**Model Terbaik:** {st.session_state.best_model}")
                        st.write(f"**Parameter Terbaik:** {st.session_state.best_params}")
                        st.write(f"**Skor Akurasi Terbaik (Cross-Validation):** {st.session_state.best_score:.4f}")

                        # Evaluasi model terbaik pada data test
                        y_pred_tuned = st.session_state.best_model.predict(st.session_state.X_test)
                        st.write(f"**Akurasi Model Terbaik pada Data Test:** {accuracy_score(st.session_state.y_test, y_pred_tuned):.4f}")

                        st.write("Confusion Matrix (Model Terbaik):")
                        cm_tuned = confusion_matrix(st.session_state.y_test, y_pred_tuned)
                        fig_cm_tuned, ax_cm_tuned = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', ax=ax_cm_tuned,
                                    xticklabels=st.session_state.label_encoder_classes if 'label_encoder_classes' in st.session_state else ['Class 0', 'Class 1'],
                                    yticklabels=st.session_state.label_encoder_classes if 'label_encoder_classes' in st.session_state else ['Class 0', 'Class 1'])
                        ax_cm_tuned.set_xlabel('Predicted')
                        ax_cm_tuned.set_ylabel('True')
                        ax_cm_tuned.set_title('Confusion Matrix (Tuned Model)')
                        st.pyplot(fig_cm_tuned)

                        st.write("Classification Report (Model Terbaik):")
                        report_tuned = classification_report(st.session_state.y_test, y_pred_tuned,
                                                            target_names=st.session_state.label_encoder_classes if 'label_encoder_classes' in st.session_state else None,
                                                            output_dict=True)
                        st.dataframe(pd.DataFrame(report_tuned).T)

                        # Opsi untuk menyimpan model terbaik
                        if st.button('Simpan Model Terbaik (.pkl)', key='save_tuned_model_btn'):
                            model_filename = f"best_{tuning_model_type.lower().replace(' ', '_')}_model.pkl"
                            joblib.dump(st.session_state.best_model, model_filename)
                            st.download_button(
                                label="Unduh Model Terbaik",
                                data=open(model_filename, "rb").read(),
                                file_name=model_filename,
                                mime="application/octet-stream"
                            )
                            st.success(f"Model terbaik disimpan sebagai `{model_filename}`.")
            else:
                st.warning("Definisi parameter grid kosong atau salah. Silakan periksa input Anda.")

        else:
            st.warning("Silakan latih model terlebih dahulu di tab 'Modeling & Evaluation' untuk memulai tuning.")
