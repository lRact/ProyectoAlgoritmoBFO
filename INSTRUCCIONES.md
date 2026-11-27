# Sistema de Traducción Automática - Optimización con BFO (Español a Chino)

Este repositorio contiene los experimentos para la optimización de hiperparámetros de traducción mediante el algoritmo bioinspirado de Alimentación Bacteriana (BFO).

# Librerias necesarias:
- scikit-learn
- pandas
- numpy
- torch
- transformers
- matplotlib
- deep-translator
- gradio

## 1. Conjunto de Datos Autorizado
* **Nombre:** News-Commentary v16 (es-zh)
* **Fuente:** OPUS
* **Ruta original:** `https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/es-zh.txt.zip`
* Los datos ya procesados se generaran congelados en la carpeta `datos_evaluacion/`.

## 2. Entorno y Dependencias
Ejecutado con aceleración CUDA en Python 3.x (Python 3.14 recomendado). Instalar dependencias con:<br>
`pip install pandas torch transformers sacrebleu scikit-learn matplotlib gradio deep-translator`<br>
Se recomienda general un entorno virtual (.venv) para la ejecucion del proyecto.

Versiones clave:
* transformers (Hugging Face)
* sacrebleu (Tokenización: "zh")
* modelo: `facebook/nllb-200-distilled-600M`

## 3. Orden de Ejecución para Reproducibilidad
Para regenerar las tablas y gráficas del artículo, ejecute los scripts ubicados en la misma carpeta de proyecto en el siguiente orden:

1. `main.py` Descarga el dataset crudo y genera los csv necesarios en `datos_evaluacion/`.
2. `experimento_1.py`: Evalúa el modelo base NLLB-600M con los datos de prueba de `datos_evaluacion/prueba.csv`.
3. `experimento_2.py`: Ejecuta el optimizador BFO (3 semillas). Genera la gráfica de convergencia.
4. `experimento_3.py`: Traduce el conjunto de prueba con los mejores parámetros y calcula las diferencias.
5. `experimento_4.py`: Extrae oraciones automaticamente para análisis manual.
6. `experimento_5.py`:<br>
6.1. Genera un subconjunto de oraciones en el archivo `para_google.txt`.<br>
6.2. Posteriormente, se debera de traducir y guardar cada oracion del archivo generado `para_google.txt` en otro archivo `de_google.txt` en el mismo directorio. Puede ser con un traductor a eleccion (Google Translate o DeepL recomendados).<br>
6.3. Una vez generado el archivo `de_google.txt` con las 15 oraciones traducidas al idioma destino. Se debera de ejecutar nuevamente `experimento5.py` para generar la tabla comparativa correspondiente.
7. `interfaz.py`: Lanza la aplicación interactiva web en localhost. Permite introducir oraciones manualmente y traducirlas con el modelo y parametros optimizados.