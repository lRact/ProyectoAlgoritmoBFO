import os
import time
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU, CHRF


def experimento_1():
    ## Asegurar la estructura de carpetas requerida
    os.makedirs("registros", exist_ok=True)
    os.makedirs("tablas", exist_ok=True)

    ## Cargar datos de prueba previamente generados en main
    print("Cargando datos de prueba...")
    df_prueba = pd.read_csv("datos_evaluacion/prueba.csv")
    textos_es = df_prueba['es'].tolist()
    referencias_zh = df_prueba['zh'].tolist()

    ## Verificar si existe una GPU compatible (Usara CPU en caso de que no)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Configuración computacional: Ejecutando en {device.upper()}")

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print("Generando traducciones (cronometrando inferencia)...")
    start_time = time.time()

    traducciones_base = []
    ## NLLB usa códigos BCP-47 específicos: spa_Latn (Español) y zho_Hans (Chino simplificado)
    tokenizer.src_lang = "spa_Latn"
    target_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")

    ## Inferencia (procesando uno por uno para medir un baseline crudo sin batching agresivo)
    for texto in textos_es:
        inputs = tokenizer(texto, return_tensors="pt").to(device)
        tokens_traducidos = modelo.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=128
        )
        texto_traducido = tokenizer.batch_decode(tokens_traducidos, skip_special_tokens=True)[0]
        traducciones_base.append(texto_traducido)

    tiempo_inferencia = time.time() - start_time

    print("Guardando traducciones en orden...")
    df_resultados = pd.DataFrame({
        "es": textos_es,
        "referencia": referencias_zh,
        "traduccion_base": traducciones_base
    })
    df_resultados.to_csv("registros/traducciones_base.csv", index=False)

    print("Calculando métricas SacreBLEU...")
    ## Es vital instanciar BLEU con el tokenizador en chino
    bleu = BLEU(tokenize="zh")
    chrf = CHRF()

    b_score = bleu.corpus_score(traducciones_base, [referencias_zh])
    c_score = chrf.corpus_score(traducciones_base, [referencias_zh])

    ## Construir la tabla del experimento
    tabla_exp1 = pd.DataFrame([{
        "Modelo": model_name,
        "BLEU": round(b_score.score, 2),
        "chrF": round(c_score.score, 2),
        "Tiempo de inferencia (s)": round(tiempo_inferencia, 2)
    }])

    print("\n--- Resultados del experimento 1 ---")
    print(tabla_exp1.to_markdown(index=False))
    print(f"Firma BLEU: {bleu.get_signature()}")
    print(f"Firma chrF: {chrf.get_signature()}")

    tabla_exp1.to_csv("tablas/experimento_1.csv", index=False)
    print("\nFertig!")


if __name__ == "__main__":
    experimento_1()