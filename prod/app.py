"""
app.py — Interfaz Streamlit para detección de muelas del juicio (Exp_G8b).
"""

import base64
from pathlib import Path
import base64
from pathlib import Path
import streamlit as st

import streamlit as st

from utils import (
    CONF_DEFAULT,
    IOU_DEFAULT,
    bgr_to_pil,
    load_model,
    load_sahi_model,
    pil_to_bytes,
    postprocess_results,
    preprocess_image,
    run_inference,
    run_inference_sahi,
    summarize_detections,
)

# ── Página ────────────────────────────────────────────────────────────────────


# ── Helper logo ───────────────────────────────────────────────────────────────
def get_logo_b64(path: str) -> str | None:
    # Buscar relativo al archivo app.py, no al cwd
    base = Path(__file__).parent.parent
    p = base / path
    if not p.exists():
        p = base / "src" / "dentex_logo.png"
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None

# ── Página ────────────────────────────────────────────────────────────────────
logo_b64 = get_logo_b64("src/dentex_logo.png")
st.set_page_config(
    page_title="WisdomScan — Detección de Muelas del Juicio",
    page_icon=f"data:image/png;base64,{logo_b64}" if logo_b64 else "🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Fondo blanco */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #1a1a2e;
}
[data-testid="stSidebar"] {
    background-color: #fafafa;
    border-right: 2px solid #ff6b2b;
}
[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background-color: #fff8f5;
    border: 1.5px solid #ff6b2b;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    color: #ff6b2b !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #555 !important;
    font-weight: 500 !important;
}

/* Botones */
.stButton > button, .stDownloadButton > button {
    background-color: #ff6b2b;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #e55a1f;
}

/* Toggle */
[data-testid="stToggle"] { accent-color: #ff6b2b; }

/* Tabla */
[data-testid="stDataFrame"] {
    border: 1.5px solid #ffe0d0;
    border-radius: 8px;
}

/* Headers */
h1, h2, h3, h4 { color: #1a1a2e !important; }

/* Divider */
hr { border-color: #ffe0d0 !important; border-width: 1.5px !important; }

/* Info */
[data-testid="stInfo"] {
    background-color: #fff8f5;
    border-left: 4px solid #ff6b2b;
    color: #1a1a2e;
}
[data-testid="stWarning"] {
    border-left: 4px solid #ff6b2b;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #fff8f5;
    border: 2px dashed #ff6b2b;
    border-radius: 10px;
}

/* Caption */
.stCaption { color: #999 !important; }

/* Accent line bajo el header */
.accent-bar {
    height: 3px;
    background: linear-gradient(90deg, #ff6b2b, #ffaa80);
    border-radius: 2px;
    margin: 8px 0 20px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helper logo ───────────────────────────────────────────────────────────────

def get_logo_b64(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent.parent / "src" / "dentex_logo.png"
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    logo_b64 = get_logo_b64("src/dentex_logo.png")
    if logo_b64:
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'style="width:100%; max-width:200px; margin: 0 auto 16px auto; display:block;">',
            unsafe_allow_html=True,
)

    st.markdown("## ⚙️ Parámetros")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Umbral de confianza",
        min_value=0.10, max_value=0.90,
        value=CONF_DEFAULT, step=0.05,
        help="Confianza mínima para reportar una detección. Default: 0.25",
    )
    iou_thresh = st.slider(
        "Umbral IoU (NMS)",
        min_value=0.10, max_value=0.90,
        value=IOU_DEFAULT, step=0.05,
        help="Supresión no máxima — elimina detecciones solapadas.",
    )

    st.markdown("---")

    use_sahi = st.toggle(
        "🔲 SAHI — tiling",
        value=False,
        help="Divide la imagen en tiles 640×640 solapados. Mejora detección en "
             "radiografías grandes. ~3-5s extra.",
    )
    enhance_contrast = st.toggle(
        "🔆 CLAHE — normalizar contraste",
        value=False,
        help="Ecualización adaptativa de histograma. Útil en radiografías oscuras.",
    )

    if use_sahi:
        st.info("SAHI activo", icon="🔲")
    if enhance_contrast:
        st.info("CLAHE activo", icon="🔆")

    st.markdown("---")
    st.markdown("### Modelo")
    st.markdown("""
| Campo | Valor |
|-------|-------|
| Experimento | Exp_G8b |
| Arquitectura | YOLOv8m |
| Backbone | COCO pretrain |
| Dataset | DENTEX + ExAn-MTM |
| Clase detectada | `impacted` |
| mAP50 (test) | **0.992** |
| Precisión | 0.980 |
| Recall | 0.999 |
""")

    st.markdown("---")
    st.caption("Redes Neuronales · UTN FRM")
    st.caption("Docente: Ing. Pablo Marinozi")


# ── Modelo ────────────────────────────────────────────────────────────────────

model      = load_model()
sahi_model = load_sahi_model()


# ── Header ────────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 7])

with col_title:
    st.markdown("""
    <h1 style='margin-bottom:2px; color:#1a1a2e;'>WisdomScan</h1>
    <p style='color:#888; font-size:1.05rem; margin-top:0;'>
    Detección automática de muelas del juicio impactadas · YOLOv8m
    </p>
    <div style='height:3px; background:linear-gradient(90deg,#ff6b2b,#ffaa80);
    border-radius:2px; margin-top:6px;'></div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# ── Upload ────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "webp"],
    help="Resolución recomendada: ≥ 800px de ancho.",
)

if uploaded_file is None:
    st.markdown("""
<div style='text-align:center; padding:56px 0; color:#bbb;'>
  <div style='font-size:3.5rem;'>🦷</div>
  <div style='font-size:1.15rem; margin-top:10px; color:#555; font-weight:500;'>
    Subí una radiografía panorámica para comenzar
  </div>
  <div style='font-size:0.9rem; margin-top:6px; color:#aaa;'>
    El modelo detectará muelas del juicio impactadas automáticamente
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── Inferencia ────────────────────────────────────────────────────────────────

modo = "SAHI (tiling)" if use_sahi else "Estándar"

with st.spinner(f"Analizando radiografía — modo {modo}…"):
    img_bgr = preprocess_image(uploaded_file, enhance_contrast=enhance_contrast)

    if use_sahi:
        detections = run_inference_sahi(sahi_model, img_bgr, conf=conf_thresh, iou=iou_thresh)
    else:
        detections = run_inference(model, img_bgr, conf=conf_thresh, iou=iou_thresh)

    img_result_bgr = postprocess_results(img_bgr, detections)
    img_result_pil = bgr_to_pil(img_result_bgr)
    img_orig_pil   = bgr_to_pil(img_bgr)

summary = summarize_detections(detections)


# ── Métricas ──────────────────────────────────────────────────────────────────

st.markdown("### Resultado del análisis")

col1, col2 = st.columns(2)
col1.metric("🦷 Muelas impactadas", summary["impacted"])
col2.metric("📐 Modo", modo)

st.markdown("---")


# ── Imágenes ──────────────────────────────────────────────────────────────────

col_orig, col_res = st.columns(2)
with col_orig:
    st.markdown("#### Radiografía original")
    st.image(img_orig_pil, use_column_width=True)
with col_res:
    st.markdown("#### Detecciones")
    st.image(img_result_pil, use_column_width=True)


# ── Tabla ─────────────────────────────────────────────────────────────────────

if summary["impacted"] > 0:
    st.markdown("###Detalle de detecciones")
    rows = []
    for i, det in enumerate(summary["details"], 1):
        if det.cls_name != "impacted" and det.cls_id != 1:
            continue
        w = det.x2 - det.x1
        h = det.y2 - det.y1
        rows.append({
            "#":          i,
            "Confianza":  f"{det.conf:.1%}",
            "Posición":   "Superior" if det.y1 < img_bgr.shape[0] // 2 else "Inferior",
            "Ancho (px)": w,
            "Alto (px)":  h,
            "Área (px²)": w * h,
            "x1": det.x1, "y1": det.y1,
            "x2": det.x2, "y2": det.y2,
        })
    if rows:
        st.dataframe(rows, use_container_width=True)
else:
    st.warning(
        f"No se detectaron muelas impactadas con conf ≥ {conf_thresh:.0%}. "
        "Probá bajar el umbral o activar SAHI.",
        icon="⚠️",
    )


# ── Descarga ──────────────────────────────────────────────────────────────────

st.markdown("---")
col_dl, col_info = st.columns([2, 5])
with col_dl:
    st.download_button(
        label="Descargar resultado",
        data=pil_to_bytes(img_result_pil, fmt="PNG"),
        file_name="wisdomscan_resultado.png",
        mime="image/png",
        use_container_width=True,
    )