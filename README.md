# 🍅 TomatoMeter AI

TomatoMeter AI es una aplicación web interactiva construida con Streamlit que permite analizar el sentimiento de reseñas de películas. Descubre si una crítica es positiva o negativa utilizando una selección de modelos de inteligencia artificial.

https://github.com/user-attachments/assets/9e57f605-3aa2-463d-9b97-160b62ff2690

## ✨ Características Principales

- **Análisis de Sentimiento:** Clasifica automáticamente las reseñas de películas como positivas o negativas.
- **Múltiples Modelos de IA:** Elige entre cuatro modelos diferentes, cada uno con sus propias fortalezas:
    - Un modelo Transformer (rápido y eficiente).
    - Regresión Logística (el más veloz).
    - FLAN-T5 (un modelo avanzado de Google).
    - ChatGPT (análisis premium con explicaciones).
- **Dos Modos de Análisis:**
    1.  **Análisis Individual:** Pega una reseña y obtén un resultado al instante.
    2.  **Análisis por Lotes:** Sube un archivo (CSV o Excel) con cientos de reseñas y procésalas todas a la vez.
- **Visualización de Resultados:** Obtén métricas claras como la confianza del modelo y un resumen de los resultados en el análisis por lotes.
- **Descarga de Resultados:** Exporta los resultados del análisis por lotes a un archivo CSV.

## 🚀 Cómo Usar la Aplicación

1.  **Instalación y Ejecución:**
    - Clona el repositorio.
    - Instala las dependencias (se recomienda usar `uv`):
      ```bash
      # Asumiendo que las dependencias están en un pyproject.toml o requirements.txt
      uv pip install -e .
      ```
    - Ejecuta la aplicación con Streamlit:
      ```bash
      streamlit run app/app.py
      ```

2.  **Selecciona un Modelo:**
    En la parte superior, elige uno de los cuatro modelos de IA disponibles en el menú desplegable. Cada modelo tiene una etiqueta que indica su perfil (🚀 Rápido, ⚡ El más rápido, 🔄 Avanzado, 🌟 Premium).

3.  **Realiza un Análisis:**

    - **Para una sola reseña (Pestaña "📝 Single Review"):**
        1.  Escribe o pega el texto de la reseña en el área de texto.
        2.  Haz clic en el botón "🔍 Analyze Sentiment".
        3.  El resultado (sentimiento y confianza) aparecerá debajo.

    - **Para múltiples reseñas (Pestaña "📊 Batch Analysis"):**
        1.  Arrastra y suelta un archivo **CSV** o **Excel** que contenga tus reseñas. El archivo debe tener una columna con el texto de las críticas.
        2.  La aplicación cargará el archivo y te mostrará una vista previa.
        3.  Selecciona la columna que contiene el texto de las reseñas del menú desplegable.
        4.  Haz clic en el botón "🚀 Analyze...".
        5.  Una vez completado el análisis, verás un resumen de los resultados y podrás descargar un archivo CSV con las predicciones detalladas.

## 🏗️ Estructura del Proyecto

El proyecto está organizado en los siguientes módulos principales:

```
/
├── app/                # Contiene la lógica principal de la aplicación Streamlit.
│   ├── app.py          # Archivo principal que define la interfaz de usuario y el flujo de la app.
│   ├── predicitions_batch.py # Funciones para el análisis de reseñas en lote.
│   ├── settings.py     # Almacena configuraciones y constantes (ej. nombres de modelos).
│   └── utils.py        # Funciones de utilidad (ej. para cargar los modelos).
│
├── data/               # Almacena datos, como el CSV de ejemplo para pruebas.
│   └── sample.csv
│
├── model/              # Directorio destinado a guardar los modelos entrenados.
│
├── notebooks/          # Jupyter Notebooks para experimentación y desarrollo.
│   └── Text-Classification.ipynb
│
├── main.py             # Punto de entrada principal (potencial).
├── pyproject.toml      # Define los metadatos y dependencias del proyecto.
└── requirements.txt    # Lista de dependencias para facilitar la instalación.
```
