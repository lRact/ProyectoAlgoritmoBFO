import pandas as pd
import os
from sacrebleu.metrics import BLEU, CHRF
from datetime import date


def paso_1_preparar():
    # Tomamos solo las primeras 15 oraciones (el subconjunto permitido)
    df_prueba = pd.read_csv("datos_evaluacion/prueba.csv").head(15)

    with open("para_google.txt", "w", encoding="utf-8") as f:
        for oracion in df_prueba['es']:
            f.write(oracion + "\n")

    print("¡Listo! Se creó el archivo 'para_google.txt'.")
    print("-> Ábrelo, copia todo el texto y pégalo en Google Translate o DeepL.")
    print("-> Copia la traducción en chino y guárdala en un archivo nuevo llamado 'de_google.txt'.")
    print("-> Cuando lo tengas, vuelve a correr este script para ver la tabla final.")


def paso_2_evaluar():
    if not os.path.exists("de_google.txt"):
        return  # Si no existe, asume que estamos en el Paso 1

    with open("de_google.txt", "r", encoding="utf-8") as f:
        trad_ext = [line.strip() for line in f.readlines() if line.strip()]

    if len(trad_ext) != 15:
        print(f"Error: Hay {len(trad_ext)} líneas en 'de_google.txt' pero se esperaban 15. Revisa los saltos de línea.")
        return

    print("Calculando métricas en el subconjunto de 15 oraciones...")

    # Cargamos las referencias y las traducciones de nuestros modelos (solo las primeras 15)
    df_prueba = pd.read_csv("datos_evaluacion/prueba.csv").head(15)
    df_base = pd.read_csv("registros/traducciones_base.csv").head(15)
    df_opt = pd.read_csv("registros/traducciones_optimizadas.csv").head(15)

    refs = df_prueba['zh'].tolist()
    trad_base = df_base['traduccion_base'].tolist()
    trad_opt = df_opt['traduccion_opt'].tolist()

    bleu = BLEU(tokenize="zh")
    chrf = CHRF()

    tabla_final = pd.DataFrame([
        {"Sistema": "Modelo Base (NLLB)",
         "BLEU": round(bleu.corpus_score(trad_base, [refs]).score, 2),
         "chrF": round(chrf.corpus_score(trad_base, [refs]).score, 2)},

        {"Sistema": "Modelo Optimizado (BFO)",
         "BLEU": round(bleu.corpus_score(trad_opt, [refs]).score, 2),
         "chrF": round(chrf.corpus_score(trad_opt, [refs]).score, 2)},

        {"Sistema": "Google Translate/DeepL",
         "BLEU": round(bleu.corpus_score(trad_ext, [refs]).score, 2),
         "chrF": round(chrf.corpus_score(trad_ext, [refs]).score, 2)}
    ])

    tabla_final.to_csv("tablas/comparativa_sistemas_subconjunto.csv", index=False)

    print(f"\n================ RESULTADOS EXPERIMENTO 5 ================")
    print(f"Evaluación realizada sobre subconjunto de 15 oraciones")
    print(f"Fecha de consulta: {date.today()}")
    print("\n" + tabla_final.to_markdown(index=False))
    print("==========================================================")


if __name__ == "__main__":
    if not os.path.exists("de_google.txt"):
        paso_1_preparar()
    else:
        paso_2_evaluar()