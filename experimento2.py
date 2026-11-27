import os
import random
import copy
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import CHRF


class Bacteria:
    def __init__(self, limites):
        ## Elegimos num_beams (int) y length_penalty (float)
        self.num_beams = random.randint(limites['num_beams'][0], limites['num_beams'][1])
        self.length_penalty = random.uniform(limites['length_penalty'][0], limites['length_penalty'][1])
        self.f_actual = float('inf')
        self.salud = 0.0

    def obtener_parametros(self):
        return {'num_beams': self.num_beams, 'length_penalty': round(self.length_penalty, 2)}

    def mutar(self, limites):
        ## Damos un pequeño salto en la búsqueda
        self.num_beams += random.choice([-1, 0, 1])
        self.length_penalty += random.uniform(-0.2, 0.2)

        ## Restringimos a los límites válidos
        self.num_beams = max(limites['num_beams'][0], min(limites['num_beams'][1], self.num_beams))
        self.length_penalty = max(limites['length_penalty'][0], min(limites['length_penalty'][1], self.length_penalty))


def evaluar_configuracion(parametros, modelo, tokenizer, textos_val, referencias_val, device):
    traducciones = []
    target_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")
    tokenizer.src_lang = "spa_Latn"

    ## Procesar en pequeños lotes para que la GPU no se sature
    batch_size = 16
    for i in range(0, len(textos_val), batch_size):
        lote_textos = textos_val[i:i + batch_size]
        inputs = tokenizer(lote_textos, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            tokens = modelo.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=128,
                num_beams=parametros['num_beams'],
                length_penalty=parametros['length_penalty']
            )
        traducciones.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))

    score_chrf = CHRF().corpus_score(traducciones, [referencias_val]).score
    return 100.0 - score_chrf  # Minimizar f(x) equivale a maximizar chrF


def ejecutar_bfo_una_vez(semilla, ejecucion_id, modelo, tokenizer, textos_val, referencias_val, device):
    random.seed(semilla)
    limites = {'num_beams': (2, 8), 'length_penalty': (0.6, 1.4)}
    poblacion = [Bacteria(limites) for _ in range(5)]  # 5 soluciones por iteración

    registros = []
    mejor_f_acumulado = float('inf')

    for it in range(1, 11):  # 10 iteraciones
        print(f"  [Ejecución {ejecucion_id}] Iteración {it}/10...")

        ## Quimiotaxis
        for i, bacteria in enumerate(poblacion):
            params = bacteria.obtener_parametros()
            f_val = evaluar_configuracion(params, modelo, tokenizer, textos_val, referencias_val, device)

            bacteria.f_actual = f_val
            bacteria.salud += f_val

            if f_val < mejor_f_acumulado:
                mejor_f_acumulado = f_val

            registros.append({
                "Ejecucion": ejecucion_id,
                "Semilla": semilla,
                "Iteracion": it,
                "Solucion": i + 1,
                "Parametros": str(params),
                "Objetivo": round(f_val, 4),
                "Mejor_acumulado": round(mejor_f_acumulado, 4)
            })

        ## Reproducción
        poblacion.sort(key=lambda b: b.salud)
        mitad = len(poblacion) // 2
        nuevos = [copy.deepcopy(b) for b in poblacion[:mitad]]
        poblacion = poblacion[:mitad] + nuevos
        while len(poblacion) < 5:  # Rellenar si es impar
            poblacion.append(copy.deepcopy(poblacion[0]))

        ## Dispersión
        for i in range(len(poblacion)):
            if random.random() < 0.1:
                poblacion[i] = Bacteria(limites)
            else:
                poblacion[i].mutar(limites)

    return pd.DataFrame(registros)


def experimento_2():
    os.makedirs("registros", exist_ok=True)
    os.makedirs("graficas", exist_ok=True)

    print("Cargando validación y modelo (Esto tomara un buen rato)...")
    df_val = pd.read_csv("datos_evaluacion/validacion.csv")
    textos_val = df_val['es'].tolist()
    referencias_val = df_val['zh'].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    semillas = [42, 123, 999]  # Tres semillas aleatorias distintas
    dfs_resultados = []

    start_total = time.time()
    for idx, semilla in enumerate(semillas, 1):
        print(f"\n--- Iniciando Ejecución {idx}/3 (Semilla: {semilla}) ---")
        df_res = ejecutar_bfo_una_vez(semilla, idx, modelo, tokenizer, textos_val, referencias_val, device)
        dfs_resultados.append(df_res)

    print(f"\nOptimización terminada en {round(time.time() - start_total, 2)} segundos!")

    ## Unificar y guardar el CSV general
    df_final = pd.concat(dfs_resultados, ignore_index=True)
    ruta_csv = "registros/bfo_resultados.csv"
    df_final.to_csv(ruta_csv, index=False)
    print(f"Registros guardados en: {ruta_csv}")

    ## Crear la gráfica de convergencia
    plt.figure(figsize=(10, 6))
    for idx in range(1, 4):
        df_ejec = df_final[df_final["Ejecucion"] == idx]
        mejores_por_iteracion = df_ejec.groupby("Iteracion")["Mejor_acumulado"].min()
        plt.plot(mejores_por_iteracion.index, mejores_por_iteracion.values, marker='o', label=f'Ejecución {idx}')

    plt.title("Convergencia del BFO - Función Objetivo vs Iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor Valor Acumulado (f(x) = 100 - chrF)")
    plt.legend()
    plt.grid(True)

    ruta_grafica = "graficas/convergencia_bfo.png"
    plt.savefig(ruta_grafica)
    print(f"Gráfica generada en: {ruta_grafica}")

    ## Seleccionar la mejor configuración global
    mejor_fila = df_final.loc[df_final["Objetivo"].idxmin()]
    print("\nMejor configuracion encontrada")
    print(mejor_fila[["Parametros", "Objetivo", "Ejecucion", "Iteracion"]])


if __name__ == "__main__":
    experimento_2()