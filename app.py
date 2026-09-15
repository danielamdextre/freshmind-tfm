"""
FreshMind v2 - Urgencia real de consumo, recetas y ciclo de aprendizaje
TFM - Master Data Science, Big Data y Business Analytics - UCM 2026
Alumna: Daniela Melendez

Modulo 1 (Machine Learning): urgencia de consumo.
    - Producto en catalogo FoodKeeper -> dias de vida util de la fuente oficial.
    - Producto fuera de catalogo      -> modelo v2 (texto del nombre + contexto).
    - Si no existe modelo v2          -> modelo v1 (solo contexto), del Notebook 01.
Modulo 2 (Text Mining): Word2Vec sobre RecipeNLG recomienda recetas que
    aprovechan lo mas urgente y penalizan los ingredientes que faltan.
Modulo 3 (Productivizar): esta app. La despensa persiste en despensa.json y
    cada salida de inventario (cocinado / consumido / tirado) queda en
    registro_consumo.csv: el dataset propio con el que el modelo podra
    reentrenarse en el futuro.
"""

import ast
import json
import os
import uuid
from datetime import date, timedelta
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FreshMind", page_icon="🥗", layout="wide")

# ----------------------------------------------------------------- constantes
FICHERO_DESPENSA = "despensa.json"
FICHERO_REGISTRO = "registro_consumo.csv"

PESO_URGENCIA = {"rojo": 3, "naranja": 2, "verde": 1, "caducado": 0}
ORDEN = {"caducado": 0, "rojo": 1, "naranja": 2, "verde": 3}
ETIQUETA = {"caducado": "⚫ Revisar: ha superado su vida útil",
            "rojo": "🔴 Consúmelo ya (1-2 días)",
            "naranja": "🟠 Esta semana (3-7 días)",
            "verde": "🟢 Tienes tiempo (+7 días)"}
# Cuando solo tenemos la clase del modelo, usamos el punto medio del tramo
DIAS_POR_CLASE = {"rojo": 2, "naranja": 5, "verde": 14}
PREGUNTA_FECHA = {"abierto": "¿Cuándo lo abriste?",
                  "cerrado": "¿Cuándo lo compraste?",
                  "descongelado": "¿Cuándo lo descongelaste?"}
OTRO = "Otro producto (escribirlo)"
# Streamlit >= 1.49 sustituye use_container_width por width="stretch"
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
ANCHO = {"width": "stretch"} if _v >= (1, 49) else {"use_container_width": True}


# --------------------------------------------------------------- carga datos
@st.cache_resource
def cargar_modelos():
    """Modelo v1 (siempre) y v2 (si se ha generado con el Notebook 03)."""
    v1 = {"modelo": joblib.load("modelo_urgencia.pkl"),
          "le": joblib.load("label_encoder.pkl"),
          "columnas": joblib.load("columnas_modelo.pkl")}
    opciones = joblib.load("opciones_app.pkl")
    v2 = None
    if os.path.exists("modelo_v2.pkl") and os.path.exists("label_encoder_v2.pkl"):
        v2 = {"modelo": joblib.load("modelo_v2.pkl"),
              "le": joblib.load("label_encoder_v2.pkl")}
    return v1, v2, opciones


@st.cache_data
def cargar_datos():
    """Catalogo, vida util FoodKeeper y artefactos del motor de recetas."""
    productos = pd.read_parquet("productos_app.parquet")
    vocab = pd.read_parquet("vocab_app.parquet")["token"].tolist()
    vectores = np.load("vectores_tokens.npy")
    recetas = pd.read_parquet("recetas_app.parquet")
    matriz = np.load("matriz_app.npy")          # ya normalizada
    vida = (pd.read_parquet("vida_util_app.parquet")
            if os.path.exists("vida_util_app.parquet") else None)
    return productos, vocab, vectores, recetas, matriz, vida


v1, v2, opciones = cargar_modelos()
productos, vocab, vectores, recetas, matriz, vida = cargar_datos()

idx_token = {t: i for i, t in enumerate(vocab)}
token_de = dict(zip(productos["nombre_es"], productos["token"]))
categoria_de = dict(zip(productos["nombre_es"], productos["categoria_es"]))
CATALOGO = sorted(productos["nombre_es"].tolist())
catalogo_min = {n.lower(): n for n in CATALOGO}


# ------------------------------------------------------------- persistencia
def cargar_despensa():
    if os.path.exists(FICHERO_DESPENSA):
        try:
            with open(FICHERO_DESPENSA, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def guardar_despensa():
    with open(FICHERO_DESPENSA, "w", encoding="utf-8") as f:
        json.dump(st.session_state.inventario, f, ensure_ascii=False, indent=1)


def registrar_salida(item, resultado, via):
    """Cada salida del inventario es una observacion real: producto, contexto,
    dias que llevaba y que paso con el. Es el dataset que FoodKeeper no tiene."""
    fila = {"fecha": date.today().isoformat(),
            "producto": item["producto"], "categoria": item["categoria"],
            "estado": item["estado"], "lugar": item["lugar"],
            "dias_vida_util_estimados": item.get("dias_vida_util"),
            "dias_transcurridos": dias_transcurridos(item),
            "urgencia_en_ese_momento": calcular(item)["urgencia"],
            "resultado": resultado,          # consumido | tirado
            "via": via}                      # receta | manual | caducado
    pd.DataFrame([fila]).to_csv(FICHERO_REGISTRO, mode="a", index=False,
                                header=not os.path.exists(FICHERO_REGISTRO))


def quitar(item_id, resultado=None, via="manual"):
    for it in st.session_state.inventario:
        if it["id"] == item_id and resultado:
            registrar_salida(it, resultado, via)
    st.session_state.inventario = [
        it for it in st.session_state.inventario if it["id"] != item_id]
    guardar_despensa()


# ---------------------------------------------------- modulo 1: vida util
def vida_util_foodkeeper(nombre, estado, lugar):
    """Dias segun FoodKeeper para ese escenario, o None si no hay dato.
    Prefiere la referencia 'compra' (desde la fecha de compra) por ser la
    mas realista para el hogar; si no existe, usa la que haya."""
    if vida is None:
        return None
    f = vida[(vida["nombre_es"] == nombre) & (vida["estado"] == estado)
             & (vida["lugar"] == lugar)]
    if f.empty:
        return None
    if (f["referencia"] == "compra").any():
        f = f[f["referencia"] == "compra"]
    return float(f["dias_vida_util"].min())


def predecir_clase(nombre, categoria, estado, lugar):
    """Urgencia predicha y nombre del modelo usado."""
    fila = pd.DataFrame([{"nombre_es": nombre, "categoria_es": categoria,
                          "estado": estado, "lugar": lugar,
                          "referencia": "compra"}])
    if v2 is not None:
        pred = v2["le"].inverse_transform(v2["modelo"].predict(fila))[0]
        return pred, "Modelo v2 (texto + contexto)"
    X = pd.get_dummies(fila.drop(columns="nombre_es")).reindex(
        columns=v1["columnas"], fill_value=0)
    pred = v1["le"].inverse_transform(v1["modelo"].predict(X))[0]
    return pred, "Modelo v1 (contexto)"


def resolver_producto(texto):
    """Texto libre -> producto del catalogo si se parece lo suficiente."""
    if texto in catalogo_min:
        return catalogo_min[texto]
    cerca = get_close_matches(texto.lower(), list(catalogo_min), n=1, cutoff=0.75)
    return catalogo_min[cerca[0]] if cerca else None


def crear_item(producto, nombre_catalogo, categoria, estado, lugar,
               fecha_ref, fecha_caducidad):
    dias, origen, clase = None, None, None
    if nombre_catalogo:
        dias = vida_util_foodkeeper(nombre_catalogo, estado, lugar)
        if dias is not None:
            origen = "USDA FoodKeeper"
    if dias is None:
        clase, origen = predecir_clase(producto, categoria, estado, lugar)
        dias = DIAS_POR_CLASE[clase]
    return {"id": uuid.uuid4().hex[:8],
            "producto": producto, "nombre_catalogo": nombre_catalogo,
            "categoria": categoria, "token": token_de.get(nombre_catalogo),
            "estado": estado, "lugar": lugar,
            "fecha_ref": fecha_ref.isoformat(),
            "fecha_caducidad": fecha_caducidad.isoformat() if fecha_caducidad else None,
            "dias_vida_util": dias, "origen": origen, "clase_modelo": clase}


# ------------------------------------------------- urgencia que decae
def dias_transcurridos(item):
    return (date.today() - date.fromisoformat(item["fecha_ref"])).days


def calcular(item):
    """Dias restantes, fecha limite y urgencia HOY. Se recalcula en cada
    visita: un producto verde acaba en rojo si pasa el tiempo."""
    restantes = item["dias_vida_util"] - dias_transcurridos(item)
    if item.get("fecha_caducidad"):
        por_etiqueta = (date.fromisoformat(item["fecha_caducidad"]) - date.today()).days
        restantes = min(restantes, por_etiqueta)   # criterio conservador
    restantes = int(np.floor(restantes))
    if restantes < 0:
        urg = "caducado"
    elif restantes <= 2:
        urg = "rojo"
    elif restantes <= 7:
        urg = "naranja"
    else:
        urg = "verde"
    return {"restantes": restantes, "urgencia": urg,
            "limite": date.today() + timedelta(days=max(restantes, 0))}


def tabla_despensa():
    filas = []
    for it in st.session_state.inventario:
        c = calcular(it)
        filas.append({"id": it["id"], "Producto": it["producto"],
                      "Dónde": it["lugar"], "Estado": it["estado"],
                      "Días restantes": c["restantes"],
                      "Consumir antes de": c["limite"].strftime("%d/%m/%Y"),
                      "Urgencia": c["urgencia"], "Fuente": it["origen"]})
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values(["Urgencia", "Días restantes"],
                            key=lambda col: col.map(ORDEN) if col.name == "Urgencia" else col)
    return df


# ----------------------------------------------------- modulo 2: recetas
def coincide(token, ingrediente):
    """'chicken' casa con 'chicken' y con 'chicken breasts'; no con 'rice vinegar'."""
    return ingrediente == token or (
        len(token) > 3 and token in ingrediente.split())


def recomendar(top=5, max_solape=0.5):
    """Recetas que usan mas productos urgentes y exigen comprar menos.

    1) Consulta Word2Vec: media de vectores ponderada por urgencia.
    2) Puntuacion = suma de pesos de urgencia de los productos que la receta
       usa; desempate por ingredientes que faltan (menos es mejor) y coseno.
    3) Diversidad: descarta recetas con Jaccard >= max_solape con una elegida.
    """
    activos = [(it, calcular(it)["urgencia"]) for it in st.session_state.inventario
               if it.get("token") and calcular(it)["urgencia"] != "caducado"]
    if not activos:
        return None
    pesos = {}                      # un token puede repetirse: nos quedamos con el mas urgente
    for it, u in activos:
        pesos[it["token"]] = max(pesos.get(it["token"], 0), PESO_URGENCIA[u])
    tokens = list(pesos)
    filas = [vectores[idx_token[t]] for t in tokens if t in idx_token]
    ws = [pesos[t] for t in tokens if t in idx_token]
    if not filas:
        return None
    v = np.average(filas, axis=0, weights=ws)
    sim = matriz @ (v / np.linalg.norm(v))

    res = recetas[["title", "ingredientes", "directions"]].copy()
    res["similitud"] = sim

    def analizar(ings):
        usados = [t for t in tokens if any(coincide(t, i) for i in ings)]
        puntos = sum(pesos[t] for t in usados)
        urgentes = sum(1 for t in usados if pesos[t] >= 2)
        return pd.Series([usados, puntos, urgentes, len(ings) - len(usados)])

    # Preseleccion por similitud para no analizar 25.000 recetas
    cand = res.nlargest(1500, "similitud").copy()
    cand[["usados", "puntos", "n_urgentes", "faltan"]] = cand["ingredientes"].apply(analizar)
    cand = cand[cand["puntos"] > 0]
    cand = cand.sort_values(["puntos", "faltan", "similitud"],
                            ascending=[False, True, False]).head(200)

    elegidas = []
    for _, r in cand.iterrows():
        s = set(r["ingredientes"])
        repetida = any(len(s & set(e["ingredientes"])) / len(s | set(e["ingredientes"]))
                       >= max_solape for e in elegidas)
        if not repetida:
            elegidas.append(r)
        if len(elegidas) == top:
            break
    return pd.DataFrame(elegidas)


def formatear_pasos(texto):
    try:
        return ast.literal_eval(texto)
    except Exception:
        return [str(texto)]


def cocinada(receta):
    """La receta es la salida del inventario: los productos usados se
    registran como consumidos y desaparecen de la despensa."""
    usados = set(receta["usados"])
    for it in list(st.session_state.inventario):
        if it.get("token") in usados:
            quitar(it["id"], resultado="consumido", via="receta")


# ------------------------------------------------------------------ interfaz
if "inventario" not in st.session_state:
    st.session_state.inventario = cargar_despensa()

st.title("🥗 FreshMind")
st.caption("Urgencia real de consumo, recetas contra el desperdicio y un "
           "registro que aprende de lo que pasa en tu cocina. TFM UCM 2026")

pagina = st.sidebar.radio("Navegación", ["1. Añadir productos",
                                         "2. Tu despensa hoy",
                                         "3. Qué cocinar",
                                         "4. Registro y datos"])
st.sidebar.divider()
n_items = len(st.session_state.inventario)
n_urg = sum(calcular(it)["urgencia"] in ("rojo", "caducado")
            for it in st.session_state.inventario)
st.sidebar.metric("Productos en tu despensa", n_items,
                  delta=f"{n_urg} urgentes" if n_urg else None, delta_color="inverse")
st.sidebar.caption("Modelo activo: " + ("v2 texto + contexto" if v2 else "v1 contexto")
                   + (" · vida útil FoodKeeper" if vida is not None else ""))
if st.sidebar.button("Vaciar despensa"):
    st.session_state.inventario = []
    guardar_despensa()
    st.rerun()


# --------------------------------------------------------- pagina 1: anadir
if pagina.startswith("1"):
    st.header("Añade lo que tienes en casa")
    st.write("La fecha de la etiqueta es solo una parte. FreshMind calcula cuánto "
             "dura **de verdad** según si está abierto, dónde lo guardas y "
             "desde cuándo, y recalcula la urgencia cada día.")

    col1, col2 = st.columns(2)
    with col1:
        eleccion = st.selectbox("Producto", [OTRO] + CATALOGO, index=1)
        if eleccion == OTRO:
            texto = st.text_input("Escribe el producto", placeholder="p. ej. yogur griego de cabra")
            reconocido = resolver_producto(texto.strip()) if texto.strip() else None
            if reconocido:
                st.caption(f"Lo hemos reconocido en el catálogo como **{reconocido}**.")
                nombre_catalogo, producto = reconocido, reconocido
                categoria = categoria_de[reconocido]
            else:
                nombre_catalogo, producto = None, texto.strip()
                categoria = st.selectbox("¿Qué tipo de alimento es?", opciones["categorias"])
                if texto.strip():
                    st.caption("No está en el catálogo: el modelo estimará su vida útil "
                               "a partir del nombre y el tipo de alimento.")
        else:
            nombre_catalogo, producto = eleccion, eleccion
            categoria = categoria_de[eleccion]
            st.caption(f"Categoría: {categoria}")
    with col2:
        estado = st.selectbox("¿Está abierto?", opciones["estados"], index=1)
        lugar = st.selectbox("¿Dónde lo guardas?", opciones["lugares"], index=2)
        fecha_ref = st.date_input(PREGUNTA_FECHA[estado], value=date.today(),
                                  max_value=date.today(), format="DD/MM/YYYY")
        con_cad = st.checkbox("Tiene fecha de caducidad o consumo preferente impresa")
        fecha_cad = st.date_input("Fecha impresa en el envase", value=date.today() + timedelta(days=7),
                                  format="DD/MM/YYYY") if con_cad else None

    if st.button("Añadir a la despensa", type="primary", disabled=not producto):
        item = crear_item(producto, nombre_catalogo, categoria, estado, lugar,
                          fecha_ref, fecha_cad)
        st.session_state.inventario.append(item)
        guardar_despensa()
        c = calcular(item)
        st.success(f"**{producto}** → {ETIQUETA[c['urgencia']]}  \n"
                   f"Vida útil estimada: {item['dias_vida_util']:.0f} días "
                   f"({item['origen']}). Han pasado {dias_transcurridos(item)} días. "
                   f"Consumir antes del **{c['limite'].strftime('%d/%m/%Y')}**.")

    st.divider()
    if st.button("Cargar despensa de ejemplo"):
        hoy = date.today()
        ejemplos = [("chicken", "cerrado", "nevera", hoy - timedelta(days=1), None),
                    ("greens", "abierto", "nevera", hoy - timedelta(days=2), None),
                    ("yogurt", "abierto", "nevera", hoy - timedelta(days=5), hoy + timedelta(days=10)),
                    ("white rice", "cerrado", "despensa", hoy - timedelta(days=30), None),
                    ("eggs", "cerrado", "nevera", hoy - timedelta(days=20), hoy + timedelta(days=3))]
        st.session_state.inventario = []
        for tok, est, lug, f_ref, f_cad in ejemplos:
            fila = productos[productos["token"] == tok]
            if fila.empty:
                continue
            nom = fila.iloc[0]["nombre_es"]
            st.session_state.inventario.append(
                crear_item(nom, nom, categoria_de[nom], est, lug, f_ref, f_cad))
        guardar_despensa()
        st.rerun()


# ------------------------------------------------------- pagina 2: despensa
elif pagina.startswith("2"):
    st.header("Tu despensa hoy")
    if not st.session_state.inventario:
        st.info("Todavía no has añadido productos. Empieza en '1. Añadir productos'.")
    else:
        df = tabla_despensa()
        c0, c1, c2, c3 = st.columns(4)
        c0.metric("⚫ Revisar", int((df["Urgencia"] == "caducado").sum()))
        c1.metric("🔴 Urgentes", int((df["Urgencia"] == "rojo").sum()))
        c2.metric("🟠 Esta semana", int((df["Urgencia"] == "naranja").sum()))
        c3.metric("🟢 Con tiempo", int((df["Urgencia"] == "verde").sum()))

        st.dataframe(df.drop(columns="id"), hide_index=True, **ANCHO)

        st.subheader("Gestionar productos")
        st.caption("Marca qué pasó con cada producto: es lo que permite que el "
                   "sistema aprenda cuánto dura la comida en tu casa, no en un laboratorio.")
        for _, fila in df.iterrows():
            a, b, c, d = st.columns([4, 1.2, 1.2, 1.2])
            a.write(f"{ETIQUETA[fila['Urgencia']].split(' ')[0]} **{fila['Producto']}** · "
                    f"{fila['Días restantes']} días · {fila['Dónde']}, {fila['Estado']}")
            if b.button("Consumido", key=f"c_{fila['id']}"):
                quitar(fila["id"], "consumido"); st.rerun()
            if c.button("Tirado", key=f"t_{fila['id']}"):
                quitar(fila["id"], "tirado", via="caducado" if fila["Urgencia"] == "caducado" else "manual")
                st.rerun()
            if d.button("Quitar", key=f"q_{fila['id']}", help="Eliminar sin registrar"):
                quitar(fila["id"]); st.rerun()

        st.subheader("Valor en riesgo esta semana")
        en_riesgo = int(df["Urgencia"].isin(["rojo", "naranja", "caducado"]).sum())
        r1, r2 = st.columns(2)
        r1.metric("Productos que pueden acabar en la basura", en_riesgo)
        r2.metric("Valor estimado", f"{en_riesgo * 2.5:.2f} €")
        st.caption("Estimación orientativa con un coste medio de 2,5 € por producto. "
                   "El ahorro real se mide en '4. Registro y datos' con lo que "
                   "efectivamente se consume o se tira.")


# -------------------------------------------------------- pagina 3: recetas
elif pagina.startswith("3"):
    st.header("Cocina primero lo que caduca antes")
    if not st.session_state.inventario:
        st.info("Añade productos para recibir sugerencias.")
    else:
        df = tabla_despensa()
        urgentes = df[df["Urgencia"] == "rojo"]["Producto"].tolist()
        if urgentes:
            st.write("Prioridad máxima: " + ", ".join(f"**{u}**" for u in urgentes))
        sin_token = [it["producto"] for it in st.session_state.inventario if not it.get("token")]
        if sin_token:
            st.caption("Sin enlace al recetario (fuera del catálogo): " + ", ".join(sin_token))

        top = recomendar()
        if top is None or top.empty:
            st.warning("No se pudo enlazar ningún producto con el recetario.")
        else:
            for i, (_, r) in enumerate(top.iterrows()):
                titulo = (f"{r['title']}  ·  usa {len(r['usados'])} de tus productos "
                          f"({int(r['n_urgentes'])} urgentes) · te faltan {int(r['faltan'])}")
                with st.expander(titulo, expanded=(i == 0)):
                    ings = [f"**{ing}**" if any(coincide(t, ing) for t in r["usados"]) else ing
                            for ing in r["ingredientes"]]
                    st.write("**Ingredientes:** " + ", ".join(ings))
                    st.write("**Preparación:**")
                    for k, paso in enumerate(formatear_pasos(r["directions"]), 1):
                        st.write(f"{k}. {paso}")
                    if st.button("La he cocinado", key=f"cook_{i}",
                                 help="Los productos usados salen de la despensa y se registran como consumidos"):
                        cocinada(r)
                        st.success("Registrado. Los ingredientes usados ya no están en tu despensa.")
                        st.rerun()
            st.caption("Recetas de RecipeNLG (en inglés, uso académico no comercial). "
                       "En negrita, los ingredientes que ya tienes.")


# ------------------------------------------------------- pagina 4: registro
else:
    st.header("Registro y datos")
    st.write("Cada producto que sale de la despensa deja una observación: qué era, "
             "cómo se guardaba, cuántos días llevaba y si se **consumió o se tiró**. "
             "FoodKeeper dice cuánto dura la comida en teoría; este registro dice "
             "cuánto duró en tu casa. Es el dato con el que el modelo puede reentrenarse.")
    if not os.path.exists(FICHERO_REGISTRO):
        st.info("Aún no hay registros. Marca productos como consumidos o tirados en "
                "'2. Tu despensa hoy' o cocina una receta en '3. Qué cocinar'.")
    else:
        reg = pd.read_csv(FICHERO_REGISTRO)
        k1, k2, k3 = st.columns(3)
        k1.metric("Salidas registradas", len(reg))
        k2.metric("Consumidos", int((reg["resultado"] == "consumido").sum()))
        k3.metric("Tirados", int((reg["resultado"] == "tirado").sum()))
        if len(reg):
            tasa = (reg["resultado"] == "tirado").mean() * 100
            st.metric("Tasa de desperdicio real", f"{tasa:.0f} %",
                      help="Porcentaje de salidas que acabaron en la basura")
        st.dataframe(reg.sort_values("fecha", ascending=False), hide_index=True, **ANCHO)
        st.download_button("Descargar registro (CSV)", reg.to_csv(index=False),
                           file_name="registro_consumo.csv", mime="text/csv")
        if st.button("Borrar registro"):
            os.remove(FICHERO_REGISTRO)
            st.rerun()
