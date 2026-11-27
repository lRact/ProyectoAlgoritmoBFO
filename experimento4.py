import pandas as pd


def experimento_4():
    print("Recopilando traducciones")
    # Cargar los archivos generados en los experimentos anteriores
    df_base = pd.read_csv("registros/traducciones_base.csv")
    df_opt = pd.read_csv("registros/traducciones_optimizadas.csv")

    # Unir ambas tablas asegurando que correspondan a la misma oración
    df_completo = pd.merge(df_base, df_opt[['es', 'traduccion_opt']], on='es')

    # Calcular longitud de la oración en español (por número de palabras)
    df_completo['longitud'] = df_completo['es'].apply(lambda x: len(str(x).split()))

    # Ordenar por longitud para asegurar "distintas longitudes o estructuras"
    df_completo = df_completo.sort_values(by='longitud').reset_index(drop=True)

    # Seleccionar 5 índices distribuidos (1 corta, 3 medianas, 1 larga)
    indices = [
        0,
        len(df_completo) // 4,
        len(df_completo) // 2,
        3 * len(df_completo) // 4,
        len(df_completo) - 1
    ]
    ejemplos = df_completo.iloc[indices]

    # Mostrar resultados listos para el reporte
    for i, (_, row) in enumerate(ejemplos.iterrows(), 1):
        print(f"\n================ EJEMPLO {i} ================")
        print(f"Original (ES)    : {row['es']}")
        print(f"Referencia (ZH)  : {row['referencia']}")
        print(f"Base (NLLB)      : {row['traduccion_base']}")
        print(f"Optimizada (BFO) : {row['traduccion_opt']}")


if __name__ == "__main__":
    experimento_4()