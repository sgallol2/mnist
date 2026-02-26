import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuración inicial
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

@st.cache_data
def load_data():
    # Cargamos una versión reducida de MNIST para que el despliegue sea rápido
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data / 255.0  # Normalización
    y = mnist.target.astype(int)
    # Usamos un subconjunto para velocidad en Streamlit Cloud
    return X[:10000], y[:10000]

st.title("🔢 Clasificador de Dígitos MNIST")
st.markdown("Este aplicativo permite entrenar diferentes modelos para reconocer números escritos a mano.")

# --- SIDEBAR: CONFIGURACIÓN ---
st.sidebar.header("1. Configuración del Modelo")
model_type = st.sidebar.selectbox(
    "Selecciona el Algoritmo",
    ("Random Forest", "K-Nearest Neighbors", "Logistic Regression")
)

test_size = st.sidebar.slider("Tamaño del set de prueba (%)", 10, 40, 20) / 100

# --- CARGA Y DIVISIÓN DE DATOS ---
with st.spinner('Cargando dataset MNIST...'):
    X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# --- ENTRENAMIENTO ---
if model_type == "Random Forest":
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
elif model_type == "K-Nearest Neighbors":
    clf = KNeighborsClassifier(n_neighbors=5)
else:
    clf = LogisticRegression(max_iter=100)

with st.spinner(f'Entrenando {model_type}...'):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

# --- MÉTRICAS ---
st.header("📊 Desempeño del Modelo")
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(label="Precisión (Accuracy)", value=f"{acc:.2%}")
    st.write(f"**Modelo:** {model_type}")
    st.write(f"**Imágenes de entrenamiento:** {len(X_train)}")
    st.write(f"**Imágenes de prueba:** {len(X_test)}")

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)

st.divider()

# --- PRUEBA DE USUARIO (VALIDACIÓN) ---
st.header("🧪 Validación con imágenes del Dataset")
st.write("Selecciona un índice de la base de datos de prueba para ver la imagen y la predicción.")

index_to_test = st.number_input("Índice de la imagen (0 - 999)", min_value=0, max_value=999, value=10)

col_img, col_pred = st.columns(2)

img_valid = X_test[index_to_test].reshape(28, 28)
label_real = y_test[index_to_test]
label_pred = clf.predict([X_test[index_to_test]])[0]

with col_img:
    st.subheader("🖼️ Imagen del dígito")
    fig2, ax2 = plt.subplots()
    ax2.imshow(img_valid, cmap='gray')
    ax2.axis('off')
    st.pyplot(fig2)

with col_pred:
    st.subheader("🔍 Resultado")
    st.write(f"**Valor Real:** {label_real}")
    if label_real == label_pred:
        st.success(f"**Predicción del Modelo:** {label_pred}")
    else:
        st.error(f"**Predicción del Modelo:** {label_pred}")
    
    # Probabilidades
    probs = clf.predict_proba([X_test[index_to_test]])[0]
    st.bar_chart(pd.DataFrame(probs, index=list(range(10)), columns=["Probabilidad"]))
