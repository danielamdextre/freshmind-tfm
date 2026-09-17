# FreshMind — TFM UCM 2026

Predicción de la urgencia real de consumo de alimentos y recomendación de recetas contra el desperdicio.
Máster en Data Science, Big Data y Business Analytics (UCM). Alumna: Daniela Natalia  Melendez Dextre.

## Qué hace la app
1. **Añadir productos**: producto (catálogo USDA FoodKeeper o texto libre), si está abierto, dónde se guarda,
   desde cuándo y, opcionalmente, la fecha impresa en el envase.
2. **Tu despensa hoy**: días restantes y semáforo de urgencia recalculados cada día.
   Cada producto puede marcarse como consumido o tirado.
3. **Qué cocinar**: recetas (Word2Vec sobre RecipeNLG) que aprovechan lo más urgente y exigen comprar menos.
   "La he cocinado" descuenta los ingredientes de la despensa.
4. **Registro y datos**: cada salida de inventario queda registrada (producto, contexto, días, resultado).
   Es el dataset propio con el que el modelo puede reentrenarse.

## Cómo se estima la vida útil
| Caso | Fuente |
|---|---|
| Producto en catálogo y escenario documentado | USDA FoodKeeper (`vida_util_app.parquet`) |
| Producto fuera de catálogo o escenario sin dato | Modelo v2: TF-IDF del nombre + contexto (`modelo_v2.pkl`, Notebook 03) |
| Si no existe el modelo v2 | Modelo v1: XGBoost sobre contexto (`modelo_urgencia.pkl`, Notebook 01) |

## Estructura
```
app.py                     Aplicación Streamlit
requirements.txt           Dependencias con versiones fijadas
notebooks/
  01_EDA_FreshMind.ipynb   Datos, EDA y modelo v1 (Módulo 1)
  02_Fresh_Mind.ipynb      Word2Vec y motor de recetas (Módulo 2)
  03_Modelo_v2_FreshMind.ipynb  Modelo v2 con texto, baselines y validación leave-product-out
modelo_urgencia.pkl, label_encoder.pkl, columnas_modelo.pkl, opciones_app.pkl   Modelo v1
modelo_v2.pkl, label_encoder_v2.pkl                                            Modelo v2
vida_util_app.parquet      Vida útil FoodKeeper por producto y escenario
productos_app.parquet, vocab_app.parquet, vectores_tokens.npy                  Puente catálogo -> recetario
recetas_app.parquet, matriz_app.npy                                            25.000 recetas RecipeNLG vectorizadas
```

## Ejecutar en local
```
pip install -r requirements.txt
streamlit run app.py
```

## Datos y licencias
- USDA FoodKeeper (CC0 1.0): vida útil de alimentos, versiones en inglés y español.
- RecipeNLG (uso académico no comercial): recetas.
- Open Food Facts (ODbL): exploración de productos españoles en el Notebook 01.
