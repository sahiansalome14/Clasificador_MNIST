import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="MNIST Classifier", layout="wide")

st.title("🔢 Clasificador de Dígitos MNIST")
st.write("Selecciona un modelo, entrena y visualiza la predicción.")

# 1. Carga de datos
@st.cache_data
def load_data():
    mnist = datasets.load_digits() # Versión ligera de MNIST para despliegue rápido
    return mnist

mnist = load_data()
X, y = mnist.data, mnist.target

# 2. Barra lateral para Configuración
st.sidebar.header("Configuración del Modelo")
classifier_name = st.sidebar.selectbox(
    "Selecciona el Clasificador",
    ("SVM", "Random Forest", "KNN")
)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    return clf

# 3. Entrenamiento y Métricas
clf = get_classifier(classifier_name, params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### Modelo Actual: **{classifier_name}**")
st.metric("Precisión (Accuracy)", f"{acc:.2%}")

# 4. Pruebas y Visualización
st.divider()
st.subheader("🧪 Prueba de Validación")
sample_index = st.number_input("Elige un índice de la base de datos para probar (0-1796):", 
                               min_value=0, max_value=len(X)-1, value=10)

col1, col2 = st.columns(2)

with col1:
    st.write("**Imagen de Entrada:**")
    fig, ax = plt.subplots()
    ax.imshow(mnist.images[sample_index], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

with col2:
    prediction = clf.predict(X[sample_index].reshape(1, -1))
    st.write("**Resultado de la Clasificación:**")
    st.success(f"El modelo predice que el dígito es:  # {prediction[0]}")
    st.write(f"Valor real: {y[sample_index]}")
