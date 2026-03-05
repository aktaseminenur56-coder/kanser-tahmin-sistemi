import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Modelleri yükle
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Kanser Tahmin Sistemi", layout="wide")
st.title("🩺 Meme Kanseri Teşhis Tahmin Sistemi")
st.write("Lütfen laboratuvar sonuçlarını giriniz.")

# Özellik isimleri
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

inputs = []
col1, col2 = st.columns(2)
for i, name in enumerate(feature_names):
    with col1 if i < 15 else col2:
        val = st.number_input(f"{name}", value=0.0)
        inputs.append(val)

if st.button("Tahmin Et"):
    data = np.array([inputs])
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 0:
        st.success("Sonuç: İyi Huylu (Benign) ✅")
    else:
        st.error("Sonuç: Kötü Huylu (Malignant) ⚠️")
