import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():

    st.title("TAller Streamlit")
    st.header("Esto es un encabezado")
    st.subheader("Esto es un sub-encabezado")
    st.text("Esto es un texto")

    nombre = "Andres Cabrales"

    st.text(f"Mi nombre es {nombre}")

    st.success("Exito")
    st.warning("Correccion")
    st.info("Azul")

    np.random.seed(42)

    X = np.random.rand(100, 1) * 10
    y = 3 * X + 8 + np.random.rand(100, 1) * 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.title("Regresion Lineal")
    st.write("Proyecto Streamlit")

    opcion = st.selectbox("Seleccione el tipo de visualización", ["Dispersión", "Línea de regresión"])

    if st.checkbox("Mostrar coeficientes de la regresión lineal"):
        st.write(f"Coeficiente: {model.coef_[0][0]:.2f}")
        st.write(f"Coeficiente intersección: {model.intercept_[0]:.2f}")
        st.write(f"Error medio cuadrático: {mse:.2f}")

    data_range = st.slider("Seleccione el rango que quiere evaluar", 0, len(X_test) - 1, (10, 90))
    x_display = X_test[data_range[0]:data_range[1]]
    y_display = y_test[data_range[0]:data_range[1]]
    y_pred_display = y_pred[data_range[0]:data_range[1]]

    st.write(f"Datos mostrados: {len(x_display)} puntos")
    st.write(f"Rango de X: {x_display.min():.2f} a {x_display.max():.2f}")
    st.write(f"Rango de Y: {y_display.min():.2f} a {y_display.max():.2f}")

    fig, ax = plt.subplots()

    if opcion == "Dispersión":
        ax.scatter(x_display, y_display, color='blue', label="Datos de prueba")
        ax.set_title("Diagrama de Dispersión")
    elif opcion == "Línea de regresión":
        ax.scatter(x_display, y_display, color='blue', label="Datos de prueba")
        ax.plot(x_display, y_pred_display, color='red', linewidth=2, label="Línea de regresión")
        ax.set_title("Línea de Regresión")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()