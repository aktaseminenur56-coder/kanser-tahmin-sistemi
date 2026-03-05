import streamlit as st
import joblib
import numpy as np

# Modelleri yükle
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Meme Kanseri Teşhis Sistemi")

# Kaggle modelindeki 30 özelliğin tam listesi
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.info("Lütfen tümör özelliklerini giriniz:")
user_inputs = []

# Giriş alanlarını 3 sütuna bölerek daha düzenli gösterelim
cols = st.columns(3)
for i, name in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(f"{name}", value=0.0, format="%.4f")
        user_inputs.append(val)

if st.button("Analiz Et"):
    # Veriyi modele uygun formata getir
    input_array = np.array([user_inputs])
    
    # Kaggle'da yaptığın gibi: Önce scaler ile ölçeklendir
    scaled_input = scaler.transform(input_array)
    
    # Tahmin yap
    prediction = model.predict(scaled_input)
    
    if prediction[0] == 1:
        st.error("Tahmin Sonucu: Kötü Huylu (Malignant) ⚠️")
    else:
        st.success("Tahmin Sonucu: İyi Huylu (Benign) ✅")
