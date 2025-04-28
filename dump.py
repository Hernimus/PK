# Menampilkan 5 baris pertama menggunakan Streamlit
    st.subheader("5 Baris Pertama DataFrame:")
    st.dataframe(data.head())  # Menampilkan 5 baris pertama
    st.subheader("Statistik deskriptif")
    st.dataframe(data.describe())
    st.markdown("---")

    st.header("Penanganan Missing Value")
    st.dataframe(data.isnull().sum().to_frame(name='Missing Values'))
    st.write("Jumlah missing value pada dataset: ", data.isnull().sum().sum())
    st.write("Hasil pengecekan missing value menunjukkan bahwa tidak ada nilai yang hilang di seluruh kolom, sehingga data dalam kondisi lengkap.")
    st.markdown("---")

  
    st.header("Mapping Label Kategorikal agar lebih informatif")
    # Salin data untuk transformasi (mapping + normalisasi)
    data_processed = data.copy()

    gender_map = {0: "Male", 1: "Female"}
    ethnicity_map = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}
    parental_education_map = {0: "None", 1: "High School", 2: "Some College", 3: "Bachelor's", 4: "Higher"}
    parental_support_map = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}
    extracurricular_map = {0: "No", 1: "Yes"}
    sports_map = {0: "No", 1: "Yes"}
    music_map = {0: "No", 1: "Yes"}
    volunteering_map = {0: "No", 1: "Yes"}
    gradeclass_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    tutoring_map = {0: 'No', 1: 'Yes'}

    # Terapkan mapping
    data_processed['Gender'] = data_processed['Gender'].map(gender_map)
    data_processed['Ethnicity'] = data_processed['Ethnicity'].map(ethnicity_map)
    data_processed['ParentalEducation'] = data_processed['ParentalEducation'].map(parental_education_map)
    data_processed['ParentalSupport'] = data_processed['ParentalSupport'].map(parental_support_map)
    data_processed['Extracurricular'] = data_processed['Extracurricular'].map(extracurricular_map)
    data_processed['Sports'] = data_processed['Sports'].map(sports_map)
    data_processed['Music'] = data_processed['Music'].map(music_map)
    data_processed['Volunteering'] = data_processed['Volunteering'].map(volunteering_map)
    data_processed['GradeClass'] = data_processed['GradeClass'].map(gradeclass_map)
    data_processed['Tutoring'] = data_processed['Tutoring'].map(tutoring_map)

    st.write(data_processed)

    # Simpan data yang sudah diproses ke session_state
    st.session_state.data_processed = data_processed
    st.write("Data diproses dan disimpan dalam session_state.")

    st.header("Normalisasi atau diskretisasi variabel")
    data_processed.columns = data_processed.columns.str.strip()

    # Daftar kolom numerik yang akan dinormalisasi
    features_to_scale = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']

    # Hapus titik dan ubah ke angka
    for col in features_to_scale:
        data_processed[col] = data_processed[col].astype(str).str.replace('.', '', regex=False)
        data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')

    # Daftar kolom numerik yang akan dinormalisasi
    features_to_scale = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']

    # Inisialisasi scaler dan salin data untuk versi ternormalisasi
    data_scaled = data_processed.copy()
    scaler = MinMaxScaler()

    # Menerapkan normalisasi hanya pada fitur numerik
    data_scaled[features_to_scale] = scaler.fit_transform(data_scaled[features_to_scale])

    st.write(data_scaled.head())

    # Simpan data yang sudah diproses ke session_state
    st.session_state.data_scaled = data_scaled
    st.write("Data scaled dan disimpan dalam session_state.")

    st.markdown("---")



    st.header("Analisis korelasi")
    # Salin dataframe
    data_encoded = data_scaled.copy()

    # Ubah kolom kategorikal menjadi numerik
    for col in data_encoded.select_dtypes(include='object').columns:
        data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])

    # Hitung korelasi untuk semua kolom (tanpa StudentID) dengan metode Pearson
    correlation_matrix_all = data_encoded.drop(columns=['StudentID']).corr(method='pearson')

    # Menampilkan DataFrame korelasi
    st.subheader("Matriks Korelasi Variable:")
    st.write(correlation_matrix_all)

    # Visualisasi korelasi dengan heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriks Korelasi Seluruh Variabel (Termasuk Kategorikal)")

    # Menampilkan heatmap ke dalam Streamlit
    st.pyplot(plt)

    # Korelasi antar variabel numerik
    correlation_matrix = data_scaled.drop(columns=['StudentID']).select_dtypes(include=['float64']).corr(method='pearson')

    # Menampilkan DataFrame korelasi
    st.subheader("Matriks Korelasi Antar Variabel Numerik:")
    st.write(correlation_matrix)

    # Visualisasi matriks korelasi dengan heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriks Korelasi antar Variabel Numerik")

    # Menampilkan heatmap ke dalam Streamlit
    st.pyplot(plt)
    st.markdown("---")

    st.header("Visualisasi distribusi data")
    # Scatter plot interaktif untuk dua variabel numerik
    fig1 = px.scatter(data_scaled, x='Age', y='GPA', color='GradeClass',
                    title="Hubungan antara Age dan GPA",
                    labels={'Age': 'Usia', 'GPA': 'Nilai GPA'},
                    color_continuous_scale='Viridis')

    # Pairplot interaktif antara beberapa variabel
    fig2 = px.scatter_matrix(data_scaled, dimensions=['Age', 'StudyTimeWeekly', 'Absences', 'GPA'],
                            color='GradeClass', title="Matrix Hubungan Antar Variabel",
                            labels={'Age': 'Usia', 'StudyTimeWeekly': 'Waktu Belajar', 'Absences': 'Absensi', 'GPA': 'Nilai GPA'})


    # Menampilkan scatter plot
    st.plotly_chart(fig1)

    # Menampilkan pairplot
    st.plotly_chart(fig2)

    st.subheader("Visualisasi distribusi data untuk variabel numerik dan kategorikal")

    # Hitung jumlah kolom yang akan diplot
    num_columns = len(data_scaled.columns)

    # Loop melalui data dengan step 5 (untuk 5 plot per baris)
    for i in range(0, num_columns, 3):
        # Ambil 5 kolom sekaligus
        cols_to_plot = data_scaled.columns[i:i+3]
        num_plots = len(cols_to_plot)
        
        # Buat figure dengan subplots 1 baris x 5 kolom
        fig, axes = plt.subplots(1, num_plots, figsize=(10, 4))
        
        # Jika hanya 1 plot, axes bukan array jadi kita konversi ke list
        if num_plots == 1:
            axes = [axes]
        
        for j, col in enumerate(cols_to_plot):
            ax = axes[j]
            
            # Cek apakah variabel kategorikal atau numerik
            if data_scaled[col].nunique() <= 10 and data_scaled[col].dtype == 'object':
                # Variabel kategorikal
                sns.countplot(x=data_scaled[col], palette='Set2', ax=ax)
                ax.set_title(f"Distribusi {col}")
                ax.tick_params(axis='x', rotation=45)
                
                # Menambahkan nilai pada masing-masing bar
                for p in ax.patches:
                    height = p.get_height()
                    if height > 0:
                        ax.text(p.get_x() + p.get_width() / 2., height + 0.1, 
                            f'{int(height)}', ha="center", fontsize=9)
            
            elif data_scaled[col].dtype in ['float64', 'int64']:
                # Variabel numerik
                sns.histplot(data_scaled[col], kde=True, color='skyblue', bins=20, ax=ax)
                ax.set_title(f"Distribusi {col}")
                
                # Menambahkan nilai pada histogram
                for p in ax.patches:
                    height = p.get_height()
                    if height > 0:
                        ax.text(p.get_x() + p.get_width() / 2., height + 0.1, 
                            f'{int(height)}', ha='center', fontsize=5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Tutup figure untuk menghemat memori