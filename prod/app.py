"""
app.py — Interfaz Streamlit para detección de muelas del juicio (Exp_G7).

Responsabilidades:
  - Layout y navegación de la UI
  - Recepción del archivo subido por el usuario
  - Coordinación entre utils.py y la pantalla
  - Presentación de resultados y métricas
"""

import streamlit as st
from PIL import Image

from utils import (
    CONF_DEFAULT,
    IOU_DEFAULT,
    bgr_to_pil,
    load_model,
    pil_to_bytes,
    postprocess_results,
    preprocess_image,
    run_inference,
    summarize_detections,
)

# ── Configuración de la página ───────────────────────────────────────────────

st.set_page_config(
    page_title="Detección de Muelas del Juicio",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🦷 Parámetros")
    st.markdown("---")

    conf_thresh = st.slider(
        "Umbral de confianza",
        min_value=0.10,
        max_value=0.90,
        value=CONF_DEFAULT,
        step=0.05,
        help="Mínima confianza para reportar una detección. "
             "Valor usado en validación/test: 0.25",
    )

    iou_thresh = st.slider(
        "Umbral IoU (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=IOU_DEFAULT,
        step=0.05,
        help="Umbral de supresión no máxima.",
    )

    st.markdown("---")
    st.markdown("### Acerca del modelo")
    st.markdown(
        """
        **Experimento:** Exp_G7  
        **Arquitectura:** YOLOv8m  
        **Backbone:** COCO  
        **Dataset:** DENTEX + ExAn-MTM  
        **Clases:**  
        - 🟢 `erupted` — muela erupcionada  
        - 🔵 `impacted` — muela impactada  
        
        **Métricas en test (Exp_G7):**  
        | Clase | mAP50 |
        |-------|-------|
        | erupted | 0.444 |
        | impacted | 0.905 |
        | all | 0.756 |
        """
    )

    st.markdown("---")
    st.caption(
        "Materia: Redes Neuronales · UTN FRM  \n"
        "Docente: Ing. Pablo Marinozi"
    )

# ── Carga del modelo (cacheado) ──────────────────────────────────────────────

model = load_model()

# ── Encabezado principal ─────────────────────────────────────────────────────

st.title("🦷 Detección de Muelas del Juicio")
st.markdown(
    "Subí una radiografía panorámica (OPG) y el modelo YOLOv8 "
    "**Exp_G7** detectará las muelas del juicio **erupcionadas** e **impactadas**."
)
st.markdown("---")

# ── Upload ───────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Seleccioná una imagen (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Radiografía panorámica dental en cualquier resolución.",
)

if uploaded_file is None:
    st.info("⬆️ Subí una imagen para comenzar.")
    st.stop()

# ── Pipeline de inferencia ───────────────────────────────────────────────────

with st.spinner("Procesando imagen…"):
    # Preprocesamiento (idéntico a validación/test — ver utils.py)
    img_bgr = preprocess_image(uploaded_file)

    # Inferencia
    detections = run_inference(model, img_bgr, conf=conf_thresh, iou=iou_thresh)

    # Postprocesamiento — dibujo de bounding boxes
    img_result_bgr = postprocess_results(img_bgr, detections)
    img_result_pil = bgr_to_pil(img_result_bgr)
    img_orig_pil   = bgr_to_pil(img_bgr)

# ── Resumen de detecciones ───────────────────────────────────────────────────

summary = summarize_detections(detections)

col_tot, col_eru, col_imp = st.columns(3)
col_tot.metric("Total detectadas", summary["total"])
col_eru.metric("🟢 Erupcionadas",   summary["erupted"])
col_imp.metric("🔵 Impactadas",     summary["impacted"])

st.markdown("---")

# ── Visualización lado a lado ────────────────────────────────────────────────

col_orig, col_res = st.columns(2)

with col_orig:
    st.subheader("Original")
    st.image(img_orig_pil, use_column_width=True)

with col_res:
    st.subheader("Detecciones")
    st.image(img_result_pil, use_column_width=True)

# ── Tabla de detecciones ─────────────────────────────────────────────────────

if summary["total"] > 0:
    st.markdown("### Detalle de detecciones")

    rows = []
    for i, det in enumerate(summary["details"], 1):
        rows.append(
            {
                "#": i,
                "Clase": det.cls_name,
                "Confianza": f"{det.conf:.3f}",
                "x1": det.x1,
                "y1": det.y1,
                "x2": det.x2,
                "y2": det.y2,
            }
        )

    st.dataframe(rows, use_container_width=True)
else:
    st.warning(
        f"No se detectaron muelas del juicio con conf ≥ {conf_thresh:.2f}. "
        "Probá bajando el umbral de confianza en el panel izquierdo."
    )

# ── Descarga del resultado ───────────────────────────────────────────────────

st.markdown("---")
img_bytes = pil_to_bytes(img_result_pil, fmt="PNG")

st.download_button(
    label="⬇️ Descargar imagen con detecciones",
    data=img_bytes,
    file_name="deteccion_muelas_g7.png",
    mime="image/png",
)