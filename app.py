
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FreshMind", page_icon="🥗", layout="wide")

@st.cache_resource
def cargar():
    modelo = joblib.load("modelo_urgencia.pkl")
    le = joblib.load("label_encoder.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    opciones = joblib.load("opciones_app.pkl")
    return modelo, le, columnas, opciones

modelo, le, columnas, opciones = cargar()

# Inventario en memoria de sesión
if "inventario" not in st.session_state:
    st.session_state.inventario = []

st.title("🥗 FreshMind")
st.caption("Predicción de urgencia real de consumo — TFM UCM 2026")

pagina = st.sidebar.radio("Navegación",
    ["Añadir producto", "Semáforo de urgencia", "Recetas sugeridas"])

if pagina == "Añadir producto":
    st.header("Añadir producto a tu despensa")
    col1, col2 = st.columns(2)
    with col1:
        nombre = st.text_input("Nombre del producto", "Yogur")
        categoria = st.selectbox("Categoría", opciones["categorias"])
    with col2:
        estado = st.selectbox("¿Está abierto?", opciones["estados"])
        lugar = st.selectbox("¿Dónde lo guardas?", opciones["lugares"])

    if st.button("Calcular urgencia", type="primary"):
        fila = pd.DataFrame([{"categoria_es": categoria, "estado": estado,
                              "lugar": lugar, "referencia": "compra"}])
        X = pd.get_dummies(fila).reindex(columns=columnas, fill_value=0)
        pred = le.inverse_transform(modelo.predict(X))[0]

        st.session_state.inventario.append(
            {"Producto": nombre, "Categoria": categoria,
             "Estado": estado, "Lugar": lugar, "Urgencia": pred})

        colores = {"rojo": "🔴 Consúmelo ya (1-2 días)",
                   "naranja": "🟠 Esta semana (3-7 días)",
                   "verde": "🟢 Tienes tiempo (+7 días)"}
        st.success(f"**{nombre}**: {colores[pred]}")


elif pagina == "Semáforo de urgencia":
    st.header("Tu despensa ahora mismo")
    if not st.session_state.inventario:
        st.info("Todavía no has añadido productos. Ve a 'Añadir producto'.")
    else:
        df = pd.DataFrame(st.session_state.inventario)
        orden = {"rojo": 0, "naranja": 1, "verde": 2}
        df = df.sort_values("Urgencia", key=lambda c: c.map(orden))

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Urgentes", (df["Urgencia"] == "rojo").sum())
        c2.metric("🟠 Esta semana", (df["Urgencia"] == "naranja").sum())
        c3.metric("🟢 Con tiempo", (df["Urgencia"] == "verde").sum())

        st.dataframe(df, use_container_width=True)

        ahorro = (df["Urgencia"] != "verde").sum() * 2.5
        st.success(f"Consumiendo lo urgente evitas tirar unos {ahorro:.2f} € "
                   f"y unos {ahorro * 1.8:.1f} kg de CO2.")
        st.caption("Estimación basada en un coste medio de 2,5 €/producto "
                   "y 1,8 kg CO2 por euro de alimento desperdiciado.")

elif pagina == "Recetas sugeridas":
    st.header("Recetas con lo más urgente")
    if not st.session_state.inventario:
        st.info("Añade productos para recibir sugerencias.")
    else:
        df = pd.DataFrame(st.session_state.inventario)
        urgentes = df[df["Urgencia"].isin(["rojo", "naranja"])]
        if urgentes.empty:
            st.success("No tienes nada urgente. ¡Buen trabajo!")
        else:
            st.write("Prioriza estos ingredientes:")
            for _, fila in urgentes.iterrows():
                st.write(f"- **{fila['Producto']}** ({fila['Categoria']})")
            st.info("El motor semántico de recetas con Word2Vec sobre RecipeNLG "
                    "se describe en la memoria como línea de desarrollo.")
