"""
FreshMind - Prediccion de urgencia real de consumo y recomendacion de recetas
TFM - Master Data Science, Big Data y Business Analytics - UCM 2026
Alumna: Daniela Melendez

Modulo 1 (Machine Learning): XGBoost predice la urgencia (rojo/naranja/verde)
Modulo 2 (Text Mining): Word2Vec sobre RecipeNLG recomienda recetas
Modulo 3 (Productivizar modelos): esta aplicacion Streamlit

Nota de productivizacion: la app NO instala gensim. Solo carga los vectores
de los ingredientes que puede necesitar (fichero de 0,2 MB), lo que reduce
las dependencias en produccion y acelera el arranque.
"""

import ast
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="FreshMind", page_icon="🥗", layout="wide")

# Pesos de urgencia: un producto rojo influye 3 veces mas que uno verde
PESO_URGENCIA = {"rojo": 3, "naranja": 2, "verde": 1}
ETIQUETA = {"rojo": "🔴 Consúmelo ya (1-2 días)",
            "naranja": "🟠 Esta semana (3-7 días)",
            "verde": "🟢 Tienes tiempo (+7 días)"}
ORDEN = {"rojo": 0, "naranja": 1, "verde": 2}


# ---------------------------------------------------------------- carga datos
@st.cache_resource
def cargar_modelo():
    """Modelo del Modulo 1 y objetos necesarios para predecir."""
    modelo = joblib.load("modelo_urgencia.pkl")
    le = joblib.load("label_encoder.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    opciones = joblib.load("opciones_app.pkl")
    return modelo, le, columnas, opciones


@st.cache_data
def cargar_recetas():
    """Artefactos del Modulo 2 generados en el notebook 02."""
    productos = pd.read_parquet("productos_app.parquet")
    vocab = pd.read_parquet("vocab_app.parquet")["token"].tolist()
    vectores = np.load("vectores_tokens.npy")
    recetas = pd.read_parquet("recetas_app.parquet")
    matriz = np.load("matriz_app.npy")          # ya viene normalizada
    return productos, vocab, vectores, recetas, matriz


modelo, le, columnas, opciones = cargar_modelo()
productos, vocab, vectores, recetas, matriz = cargar_recetas()

idx_token = {t: i for i, t in enumerate(vocab)}
token_de = dict(zip(productos["nombre_es"], productos["token"]))
categoria_de = dict(zip(productos["nombre_es"], productos["categoria_es"]))


# ------------------------------------------------------------ motor de recetas
def vector_inventario(tokens, pesos):
    """Media ponderada de los vectores: desplaza la consulta hacia lo urgente."""
    filas, ws = [], []
    for t, p in zip(tokens, pesos):
        if t in idx_token:
            filas.append(vectores[idx_token[t]])
            ws.append(p)
    if not filas:
        return None
    return np.average(filas, axis=0, weights=ws)


def recomendar(inventario, top=5, max_solape=0.5):
    """inventario: lista de dicts con Producto y Urgencia. Devuelve top recetas.

    Ordena por numero de ingredientes urgentes usados y, en caso de empate,
    por similitud coseno. Despues filtra por diversidad (indice de Jaccard)
    para no devolver cinco variantes del mismo plato.
    """
    tokens = [token_de.get(p["Producto"]) for p in inventario]
    pesos = [PESO_URGENCIA[p["Urgencia"]] for p in inventario]
    pares = [(t, w) for t, w in zip(tokens, pesos) if t]
    if not pares:
        return None

    v = vector_inventario([t for t, _ in pares], [w for _, w in pares])
    if v is None:
        return None
    sim = matriz @ (v / np.linalg.norm(v))

    urgentes = {token_de.get(p["Producto"]) for p in inventario
                if p["Urgencia"] == "rojo"}

    res = recetas[["title", "ingredientes", "directions"]].copy()
    res["similitud"] = sim
    res["usa_urgentes"] = res["ingredientes"].apply(
        lambda l: len(urgentes & set(l)))
    res = res.sort_values(["usa_urgentes", "similitud"],
                          ascending=False).head(200)

    elegidas = []
    for _, r in res.iterrows():
        s = set(r["ingredientes"])
        repetida = any(
            len(s & set(e["ingredientes"])) / len(s | set(e["ingredientes"]))
            >= max_solape for e in elegidas)
        if not repetida:
            elegidas.append(r)
        if len(elegidas) == top:
            break
    return pd.DataFrame(elegidas)


def formatear_pasos(texto):
    """La columna directions viene como texto con formato de lista."""
    try:
        return ast.literal_eval(texto)
    except Exception:
        return [str(texto)]


# ------------------------------------------------------------------ interfaz
if "inventario" not in st.session_state:
    st.session_state.inventario = []

st.title("🥗 FreshMind")
st.caption("Predicción de urgencia real de consumo y recetas contra el "
           "desperdicio — TFM UCM 2026")

pagina = st.sidebar.radio(
    "Navegación",
    ["1. Añadir productos", "2. Semáforo de urgencia", "3. Recetas sugeridas"])

st.sidebar.divider()
st.sidebar.metric("Productos en tu despensa", len(st.session_state.inventario))
if st.sidebar.button("Vaciar despensa"):
    st.session_state.inventario = []
    st.rerun()


# --------------------------------------------------------- pagina 1: anadir
if pagina.startswith("1"):
    st.header("Añade lo que tienes en casa")
    st.write("El modelo no mira la fecha de la etiqueta: calcula cuánto dura "
             "**de verdad** según si está abierto y dónde lo guardas.")

    col1, col2 = st.columns(2)
    with col1:
        nombre = st.selectbox("Producto",
                              sorted(productos["nombre_es"].tolist()))
        st.caption(f"Categoría: {categoria_de[nombre]}")
    with col2:
        estado = st.selectbox("¿Está abierto?", opciones["estados"])
        lugar = st.selectbox("¿Dónde lo guardas?", opciones["lugares"])

    if st.button("Calcular urgencia", type="primary"):
        fila = pd.DataFrame([{"categoria_es": categoria_de[nombre],
                              "estado": estado, "lugar": lugar,
                              "referencia": "compra"}])
        X = pd.get_dummies(fila).reindex(columns=columnas, fill_value=0)
        pred = le.inverse_transform(modelo.predict(X))[0]

        st.session_state.inventario.append(
            {"Producto": nombre, "Categoria": categoria_de[nombre],
             "Estado": estado, "Lugar": lugar, "Urgencia": pred})
        st.success(f"**{nombre}** → {ETIQUETA[pred]}")

    st.divider()
    if st.button("Cargar despensa de ejemplo"):
        ejemplos = [("chicken", "cerrado", "nevera"),
                    ("greens", "abierto", "nevera"),
                    ("yogurt", "abierto", "nevera"),
                    ("white rice", "cerrado", "despensa")]
        st.session_state.inventario = []
        for tok, est, lug in ejemplos:
            fila_p = productos[productos["token"] == tok]
            if not len(fila_p):
                continue
            nom = fila_p.iloc[0]["nombre_es"]
            fila = pd.DataFrame([{"categoria_es": categoria_de[nom],
                                  "estado": est, "lugar": lug,
                                  "referencia": "compra"}])
            X = pd.get_dummies(fila).reindex(columns=columnas, fill_value=0)
            pred = le.inverse_transform(modelo.predict(X))[0]
            st.session_state.inventario.append(
                {"Producto": nom, "Categoria": categoria_de[nom],
                 "Estado": est, "Lugar": lug, "Urgencia": pred})
        st.rerun()


# ------------------------------------------------------- pagina 2: semaforo
elif pagina.startswith("2"):
    st.header("Tu despensa ahora mismo")
    if not st.session_state.inventario:
        st.info("Todavía no has añadido productos. Ve a '1. Añadir productos'.")
    else:
        df = pd.DataFrame(st.session_state.inventario)
        df = df.sort_values("Urgencia", key=lambda c: c.map(ORDEN))

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Urgentes", int((df["Urgencia"] == "rojo").sum()))
        c2.metric("🟠 Esta semana", int((df["Urgencia"] == "naranja").sum()))
        c3.metric("🟢 Con tiempo", int((df["Urgencia"] == "verde").sum()))

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Impacto estimado")
        en_riesgo = int((df["Urgencia"] != "verde").sum())
        ahorro = en_riesgo * 2.5
        i1, i2 = st.columns(2)
        i1.metric("Euros que evitas tirar", f"{ahorro:.2f} €")
        i2.metric("CO₂ evitado", f"{ahorro * 1.8:.1f} kg")
        st.caption("Estimación con un coste medio de 2,5 € por producto y "
                   "1,8 kg de CO₂ por euro de alimento desperdiciado. "
                   "Cifras orientativas, detalladas en la memoria.")


# -------------------------------------------------------- pagina 3: recetas
else:
    st.header("Cocina primero lo que caduca antes")
    if not st.session_state.inventario:
        st.info("Añade productos para recibir sugerencias.")
    else:
        df = pd.DataFrame(st.session_state.inventario)
        urgentes = df[df["Urgencia"] == "rojo"]["Producto"].tolist()
        if urgentes:
            st.write("Prioridad máxima: " + ", ".join(f"**{u}**" for u in urgentes))

        top = recomendar(st.session_state.inventario)
        if top is None or top.empty:
            st.warning("No se pudo enlazar ningún producto con el recetario.")
        else:
            for _, r in top.iterrows():
                with st.expander(
                        f"{r['title']}  ·  usa {int(r['usa_urgentes'])} "
                        f"ingrediente(s) urgente(s)"):
                    st.write("**Ingredientes:** " + ", ".join(r["ingredientes"]))
                    st.write("**Preparación:**")
                    for i, paso in enumerate(formatear_pasos(r["directions"]), 1):
                        st.write(f"{i}. {paso}")
            st.caption("Las recetas proceden de RecipeNLG y se muestran en "
                       "inglés; la interfaz y el inventario funcionan en "
                       "español mediante la traducción oficial del USDA "
                       "FoodKeeper. Uso académico no comercial.")
