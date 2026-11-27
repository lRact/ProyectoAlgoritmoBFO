import pandas as pd
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU, CHRF


def experimento_3(mejor_num_beams, mejor_length_penalty):
    df_prueba = pd.read_csv("datos_evaluacion/prueba.csv")
    textos_es = df_prueba['es'].tolist()
    referencias_zh = df_prueba['zh'].tolist()

    ## Cargar resultados previos
    df_base = pd.read_csv("tablas/experimento_1.csv")
    df_bfo = pd.read_csv("registros/bfo_resultados.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    target_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")
    tokenizer.src_lang = "spa_Latn"

    print("Traduciendo conjunto de prueba con hiperparámetros optimizados...")
    traducciones_opt = []
    start_time = time.time()

    for texto in textos_es:
        inputs = tokenizer(texto, return_tensors="pt").to(device)
        tokens_traducidos = modelo.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=128,
            num_beams=mejor_num_beams,
            length_penalty=mejor_length_penalty
        )
        texto_traducido = tokenizer.batch_decode(tokens_traducidos, skip_special_tokens=True)[0]
        traducciones_opt.append(texto_traducido)

    tiempo_opt = time.time() - start_time

    pd.DataFrame({
        "es": textos_es, "referencia": referencias_zh, "traduccion_opt": traducciones_opt
    }).to_csv("registros/traducciones_optimizadas.csv", index=False)

    # Métricas
    bleu = BLEU(tokenize="zh")
    chrf = CHRF()

    b_score_opt = bleu.corpus_score(traducciones_opt, [referencias_zh]).score
    c_score_opt = chrf.corpus_score(traducciones_opt, [referencias_zh]).score

    ## Comparativa Base vs Optimizada
    b_base = df_base.iloc[0]["BLEU"]
    c_base = df_base.iloc[0]["chrF"]
    t_base = df_base.iloc[0]["Tiempo de inferencia (s)"]

    tabla_comparativa = pd.DataFrame([
        {"Configuración": "Base", "BLEU": b_base, "chrF": c_base, "Tiempo": t_base},
        {"Configuración": "Optimizada", "BLEU": round(b_score_opt, 2), "chrF": round(c_score_opt, 2),
         "Tiempo": round(tiempo_opt, 2)},
        {"Configuración": "Diferencia", "BLEU": round(abs(b_score_opt - b_base), 2),
         "chrF": round(abs(c_score_opt - c_base), 2), "Tiempo": round(abs(tiempo_opt - t_base), 2)}
    ])

    ## Resumen de Validación
    mejores_f_val = df_bfo.groupby(["Ejecucion", "Semilla"])["Objetivo"].min().reset_index()
    mejores_f_val["Mejor chrF en validación"] = 100.0 - mejores_f_val["Objetivo"]

    ## Calcular tiempo de este experimento basado en el tiempo de ejecucion del experimento anterior (1 hora, 43 minutos)
    tiempo_total_real = 6180
    tiempo_por_ejecucion = round(tiempo_total_real / len(mejores_f_val), 2)

    tabla_ejecuciones = pd.DataFrame({
        "Ejecución": mejores_f_val["Ejecucion"],
        "Semilla": mejores_f_val["Semilla"],
        "Mejor chrF en validación": mejores_f_val["Mejor chrF en validación"].round(2),
        "Iteraciones": 10,
        "Duración (s)": tiempo_por_ejecucion  # Estimado de los 103 minutos totales
    })

    promedio_chrf = mejores_f_val["Mejor chrF en validación"].mean()
    desviacion_chrf = mejores_f_val["Mejor chrF en validación"].std()

    tabla_comparativa.to_csv("tablas/comparativa_base_optimizada.csv", index=False)
    tabla_ejecuciones.to_csv("tablas/resumen_ejecuciones_val.csv", index=False)

    print("\n--- Tabla Comparativa (Prueba) ---")
    print(tabla_comparativa.to_markdown(index=False))
    print("\n--- Resumen de Ejecuciones (Validación) ---")
    print(tabla_ejecuciones.to_markdown(index=False))
    print(f"\nPromedio mejor chrF (Validación): {round(promedio_chrf, 2)}")
    print(f"Desviación estándar chrF (Validación): {round(desviacion_chrf, 2)}")


if __name__ == "__main__":
    ## Ingresar los mejores resultados en el experimento anterior
    experimento_3(mejor_num_beams=8, mejor_length_penalty=1.4)