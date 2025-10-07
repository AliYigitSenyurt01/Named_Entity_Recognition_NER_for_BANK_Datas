
# Ner Modeli İçin Gradio Arayüz Tasarımı
import gradio as gr
import pandas as pd
import html
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch


MODEL_ID = "C:\\Users\\90532\\Desktop\\Staj Dosyaları\\Ziraat Staj\\NER\\Ner proje\\bert-turkish-ner-final\\yeni_veri3"
DEVICE = 0 if torch.cuda.is_available() else -1

# model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
ner = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=DEVICE
)

# Renk paleti
LABEL_COLORS = {
    "PERSON":   "#4E79A7",  # mavi
    "ORG":      "#F28E2B",  # turuncu
    "DATE":     "#59A14F",  # yeşil
    "TIME":     "#FF33A6",  # pembe
    "IBAN":     "#E15759",  # kırmızı
    "AMOUNT":   "#EDC948",  # sarı
    "CURRENCY": "#B07AA1",  # mor
    "TCKN":     "#76B7B2",  # teal
    "CARD_NO":  "#9C755F",  # kahverengi
    "PLATE":    "#FF9DA7",  # açık pembe
    "ACCOUNT_NO": "#33B2FF",  # açık mavi
    "ADDRESS":  "#8CD17D",  # açık yeşil
    "LANDLINE": "#F39C12",  # turuncu-sarı
    "EMAIL":    "#7F7F7F",  # gri
    "PHONE":    "#17BECF",  # cyan
}

def _non_overlapping(ents):
    """Örtüşenleri basitçe eler: en yüksek skorlu olan kalır."""
    ents = sorted(ents, key=lambda e: (e["start"], -(e["score"])))
    result = []
    last_end = -1
    for e in ents:
        if e["start"] >= last_end:  
            result.append(e)
            last_end = e["end"]
    return result

def _highlight_html(text, ents):
    ents = _non_overlapping(ents)
    ents = sorted(ents, key=lambda e: e["start"])
    cur = 0
    out = []
    for e in ents:
        out.append(html.escape(text[cur:e["start"]]))
        label = e["entity_group"]
        bg = LABEL_COLORS.get(label, "#e9d5ff")
        span_text = html.escape(text[e["start"]:e["end"]])
        title = f'{label} | score={e["score"]:.3f}'
        out.append(
            f'<span style="background:{bg}; padding:0.15rem 0.25rem; border-radius:0.35rem;" title="{title}">{span_text}</span>'
        )
        cur = e["end"]
    out.append(html.escape(text[cur:]))

    legend_bits = []
    for k, v in LABEL_COLORS.items():
        legend_bits.append(f'<span style="background:{v}; padding:0.15rem 0.35rem; border-radius:0.35rem; margin-right:6px;">{k}</span>')
    legend = '<div style="margin-top:10px; opacity:0.85;">' + " ".join(legend_bits) + "</div>"
    return "<div style='line-height:1.8; font-size:16px;'>" + "".join(out) + "</div>" + legend

def predict(text):
    text = (text or "").strip()
    if not text:
        return "<i>Metin gir.</i>", pd.DataFrame(columns=["label", "text", "start", "end", "score"])
    preds = ner(text)
    # tablo
    rows = []
    for p in preds:
        rows.append({
            "label": p["entity_group"],
            "text": p["word"],
            "start": p["start"],
            "end": p["end"],
            "score": round(float(p["score"]), 4),
        })
    df = pd.DataFrame(rows, columns=["label", "text", "start", "end", "score"])

    html_out = _highlight_html(text, preds)
    return html_out, df


with gr.Blocks() as demo:
    gr.Markdown("## Ziraat Teknoloji A.S. Türkçe NER MODELİ")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Girdi Metni", lines=14, placeholder="Metni buraya yaz...")
            btn = gr.Button("Çalıştır", variant="primary")
        with gr.Column(scale=1):
            out_html = gr.HTML(label="Vurgulu Çıktı")
            out_txt = gr.Textbox(label="Varlık Listesi (JSON)", lines=14, show_copy_button=True)

    btn.click(predict, inputs=inp, outputs=[out_html, out_txt])
    inp.submit(predict, inputs=inp, outputs=[out_html, out_txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

