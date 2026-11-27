import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("Cargando el traductor optimizado. Esto tomará unos segundos...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
modelo = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

target_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")
tokenizer.src_lang = "spa_Latn"

# Parametros que ganaron en el BFO
MEJOR_NUM_BEAMS = 8
MEJOR_LENGTH_PENALTY = 1.4


def traducir_texto(texto_espanol):
    if not texto_espanol.strip():
        return "Chino", "", "Sin configuración"

    inputs = tokenizer(texto_espanol, return_tensors="pt").to(device)

    # Inferencia con la configuración óptima del BFO
    tokens_traducidos = modelo.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        max_length=128,
        num_beams=MEJOR_NUM_BEAMS,
        length_penalty=MEJOR_LENGTH_PENALTY
    )

    texto_traducido = tokenizer.batch_decode(tokens_traducidos, skip_special_tokens=True)[0]
    config_usada = f"Algoritmo: BFO | Modelo: NLLB-600M | num_beams: {MEJOR_NUM_BEAMS} | length_penalty: {MEJOR_LENGTH_PENALTY}"

    return "Chino (Simplificado)", texto_traducido, config_usada


# Construcción de la interfaz web
app = gr.Interface(
    fn=traducir_texto,
    inputs=gr.Textbox(lines=4, label="Texto en Español", placeholder="Escribe tu texto aquí..."),
    outputs=[
        gr.Textbox(label="Idioma de Destino"),
        gr.Textbox(label="Traducción"),
        gr.Textbox(label="Configuración Usada")
    ],
    title="Traductor Bioinspirado (ES -> ZH)",
    description="Sistema de traducción automática optimizado mediante el algoritmo de Alimentación Bacteriana (BFO).",
    theme="huggingface",
    allow_flagging="never"
)

if __name__ == "__main__":
    print("¡Interfaz lista! Abre el enlace local en tu navegador (Brave).")
    app.launch()