import streamlit as st
import pandas as pd


from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="HOME",
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

    # --- Gunakan data_processed yang sudah strip kolomnya ---
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
    model_bn = DiscreteBayesianNetwork([
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
