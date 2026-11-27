import os
import urllib.request
import zipfile
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

## Asignar tamaños de csvs para evitar overlifting
tamano_prueba = 300
tamano_validacion = 200
tamano_train = 2500

def preparar_corpus():
    os.makedirs("datos_evaluacion", exist_ok=True)
    os.makedirs("corpus", exist_ok=True)

    ## Descargar corpus:
    # Nombre: News-Commentary v16 (es-zh)
    # Fuente: OPUS
    # Licencia: CC-BY
    url = "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/es-zh.txt.zip"
    zip_path = "corpus/corpus.zip"

    if not os.path.exists(zip_path):
        print("Descargando corpus...")
        urllib.request.urlretrieve(url, zip_path)

    ## Descomprimir
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("Extrayendo datos del corpus...")
        zip_ref.extractall("corpus")

    ## Capturamos los archivos extraídos
    print("Obteniendo datos del corpus...")
    archivo_es = glob.glob("corpus/*.es")[0]
    archivo_zh = glob.glob("corpus/*.zh*")[0]

    ## Leer archivos de texto plano
    with open(archivo_es, 'r', encoding='utf-8') as f_es, open(archivo_zh, 'r', encoding='utf-8') as f_zh:
        es_sentences = f_es.read().splitlines()
        zh_sentences = f_zh.read().splitlines()

    df = pd.DataFrame({'es': es_sentences, 'zh': zh_sentences})
    cantidad_original = len(df)

    ## Limpiar datos
    print("Limpiando datos nulos y/o duplicados...")
    df = df.dropna().drop_duplicates()
    cantidad_limpia = len(df)

    print(f"Cantidad original: {cantidad_original} pares")
    print(f"Cantidad limpia: {cantidad_limpia} pares")

    ## Dividir train, test y split (80, 10, 10)
    print("Dividiendo datos...")
    ## Originalmente se generan +10k pares en los csv, por lo que se tuvo que limitar
    resto_df, test_df = train_test_split(df, test_size = tamano_prueba - 1, random_state = 42)
    train_df, val_df = train_test_split(resto_df, train_size = tamano_train - 1, test_size = tamano_validacion - 1, random_state = 42)

    print(f"Exportando en archivos csv en {zip_path}")
    train_df.to_csv("datos_evaluacion/train.csv", index=False)
    val_df.to_csv("datos_evaluacion/validacion.csv", index=False)
    test_df.to_csv("datos_evaluacion/prueba.csv", index=False)

    print("Prueba congelada en 'datos_evaluacion/prueba.csv'")
    print("\nPares de ejemplo:")
    print(test_df.head(5).to_string(index=False))

    print("Done!")


if __name__ == "__main__":
    preparar_corpus()