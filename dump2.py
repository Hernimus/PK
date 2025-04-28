import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from streamlit_option_menu import option_menu
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Model Penalaran Probabilistik",
    page_icon="📊",
    layout="wide"
)


with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Bayesian Network", "Metode Tambahan", "Fitur Wajib", "Fitur Tambahan"],
    )

if selected == "Bayesian Network":

    # Mengakses data yang telah diproses dari session_state
    if 'data_processed' in st.session_state:
        data_processed = st.session_state.data_processed
        st.write("Data Precossed:", data_processed)
    else:
        st.write("Data belum diproses.")

    # Cek apakah data_processed sudah ada
    if 'data_processed' not in globals() or data_processed.empty:
        st.error("🚨 **ERROR: Data belum diproses!**")
        st.warning("Silakan ke **'Halaman Preprocessing'** terlebih dahulu.")
        
    if st.button("**Ke Halaman Preprocessing**"):
        st.session_state['auto_select_preprocessing'] = True  # set flag
        st.switch_page("pages/1_🗃_Dataset and Variable.py")

    else:
        # Salin data yang sudah diproses
        data = data_processed.copy()
    

    # --- Normalisasi numerik ---
    scaler = MinMaxScaler()
    numerical_cols = data_processed.select_dtypes(include='number').columns
    data_scaled = pd.DataFrame(scaler.fit_transform(data_processed[numerical_cols]), columns=numerical_cols)

    # Gabungkan kolom kategorikal
    categorical_cols = data_processed.select_dtypes(exclude='number')
    data_scaled = pd.concat([data_scaled, categorical_cols.reset_index(drop=True)], axis=1)

    # --- Diskretisasi numerik ---
    def safe_qcut(col, q):
        col = pd.to_numeric(col, errors='coerce')
        unique_vals = col.nunique(dropna=True)
        if unique_vals < 2:
            return pd.Series([0] * len(col))
        bins = min(q, unique_vals)
        try:
            return pd.qcut(col, q=bins, labels=False, duplicates='drop')
        except ValueError:
            return pd.Series([0] * len(col))

    data = data_scaled.copy()
    for col, bins in zip(['StudyTimeWeekly', 'GPA', 'Absences', 'Age'], [4, 4, 4, 4]):
        if col in data.columns:
            data[col] = safe_qcut(data[col], bins)
        else:
            raise KeyError(f"Kolom '{col}' tidak ditemukan di dataset. Cek nama kolom!")

    # Drop baris yang mengandung NaN
    data = data.dropna().reset_index(drop=True)

    # --- Label Encoding untuk kolom kategorikal ---
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Konversi semua kolom ke integer
    data = data.astype(int)

    # --- Streamlit Interface ---
    st.title("Model Bayesian Network dan Prediksi")

    # --- Cek distribusi hasil diskretisasi ---
    st.subheader("Distribusi Kolom Diskretisasi")
    for col in ['StudyTimeWeekly', 'GPA', 'Absences', 'Age']:
        st.write(f"Distribusi {col}:\n", data[col].value_counts())

    # --- Struktur Bayesian Network ---
    model_bn = BayesianNetwork([
        ('ParentalSupport', 'GPA'),
        ('ParentalEducation', 'GPA'),
        ('StudyTimeWeekly', 'GPA'),
        ('GPA', 'GradeClass'),
        ('Tutoring', 'GradeClass'),
        ('Gender', 'GradeClass')
    ])

    # --- Estimasi parameter CPT ---
    model_bn.fit(data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)

    # --- Validasi CPD ---
    st.subheader("Validasi CPD")
    for cpd in model_bn.get_cpds():
        st.write(cpd)
        assert cpd.is_valid_cpd(), f"CPD untuk {cpd.variable} tidak valid!"

    # --- Inferensi ---
    infer_bn = VariableElimination(model_bn)

    # Ambil 1 sample dari data untuk dijadikan evidence
    sample = data[['ParentalSupport', 'StudyTimeWeekly', 'Tutoring']].sample(1).iloc[0]
    # Konversi semua np.int64 jadi int biasa
    evidence = {col: int(val) for col, val in sample.items()}
    st.subheader("Evidence digunakan")
    st.write(evidence)


    # Prediksi GPA
    q1 = infer_bn.map_query(
        variables=['GPA'],
        evidence={'ParentalSupport': int(sample['ParentalSupport']), 'StudyTimeWeekly': int(sample['StudyTimeWeekly'])}
    )
    st.subheader("Prediksi GPA")
    st.write("Prediksi GPA:", q1['GPA'])

    # Prediksi GradeClass
    q2 = infer_bn.map_query(
        variables=['GradeClass'],
        evidence={'GPA': q1['GPA'], 'Tutoring': int(sample['Tutoring'])}
    )
    st.subheader("Prediksi GradeClass")
    st.write("Prediksi GradeClass:", q2['GradeClass'])

    # --- Debug tambahan ---
    st.subheader("Nilai Unik")
    st.write("Nilai unik StudyTimeWeekly:", data['StudyTimeWeekly'].unique())
    st.write("Nilai unik ParentalSupport:", data['ParentalSupport'].unique())
    st.write("Nilai unik GPA:", data['GPA'].unique())



if selected == "Metode Tambahan":
    st.subheader("Naive Bayes Classifier")
     # Mengakses data yang telah diproses dari session_state
    if 'data_scaled' in st.session_state:
        data_scaled = st.session_state.data_scaled
        st.write("Data scaled:", data_scaled)
    else:
        st.write("Data belum diproses.")

    # Cek apakah data_scaled sudah ada
    if 'data_scaled' not in globals() or data_scaled.empty:
        st.error("🚨 **ERROR: Data belum diproses!**")
        st.warning("Silakan ke **'Halaman Preprocessing'** terlebih dahulu.")
        
    if st.button("**Ke Halaman Preprocessing**"):
        st.session_state['auto_select_preprocessing'] = True  # set flag
        st.switch_page("pages/1_🗃_Dataset and Variable.py")

    else:
        # Salin data yang sudah diproses
        df_encoded_ap = data_scaled.copy()

    # Label encoding kolom kategorikal
    for col in df_encoded_ap.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded_ap[col] = le.fit_transform(df_encoded_ap[col])

    # Pisahkan fitur dan target
    X = df_encoded_ap.drop(columns=['GradeClass'])
    y = df_encoded_ap['GradeClass']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat dan latih model Gaussian Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Prediksi
    y_pred_nb = nb_model.predict(X_test)

    # Evaluasi
    accuracy_nb = accuracy_score(y_test, y_pred_nb)

    st.write("Akurasi Naive Bayes:", accuracy_nb)

    st.markdown("---")

    st.subheader("Gaussian Mixture Models")
    st.write("df_encoded_ap")
    st.write(df_encoded_ap)
    # Data hasil preprocessing sebelumnya (df_encoded_ap)
    X_cluster = df_encoded_ap.drop(columns=['GradeClass'])
    st.write("x_cluster", X_cluster)
    # Inisialisasi dan latih Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_cluster)

    # Prediksi cluster untuk masing-masing mahasiswa
    clusters = gmm.predict(X_cluster)

    # Tambahkan hasil cluster ke dataframe
    df_encoded_ap['Cluster'] = clusters

    # Visualisasi 2D pakai PCA biar gampang lihat klusternya
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)
    df_encoded_ap['PCA1'] = X_pca[:, 0]
    df_encoded_ap['PCA2'] = X_pca[:, 1]

    # Plot hasil clustering
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_encoded_ap, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=60)
    plt.title('Clustering Mahasiswa dengan Gaussian Mixture Model')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    st.pyplot(plt)

    cluster_summary = df_encoded_ap.groupby('Cluster')['GPA'].mean()
    st.write("Rata-rata GPA per Cluster:")
    st.write(cluster_summary)