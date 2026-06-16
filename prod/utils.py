"""
utils.py — Funciones auxiliares para la app de detección de muelas del juicio.

Responsabilidades:
  - Carga cacheada del modelo YOLOv8 (load_model)
  - Preprocesamiento de la imagen de entrada (preprocess_image)
  - Postprocesamiento de la salida YOLO (postprocess_results)
  - Dibujo de bounding boxes (draw_detections)

IMPORTANTE: el preprocesamiento es idéntico al de entrenamiento/validación:
  - imgsz=640, normalización interna de Ultralytics
  - NO se aplica ninguna transformación manual antes de llamar a model.predict()
  - YOLO hace internamente: resize + pad a 640×640, /255, inversión canal
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Constantes ───────────────────────────────────────────────────────────────

CLASSES = {0: "erupted", 1: "impacted"}

# Colores BGR para cada clase
COLORS = {
    0: (34, 197, 94),   # verde — erupted
    1: (0, 80, 220),    # azul  — impacted
}

# Parámetros de inferencia — deben coincidir con validación/test del notebook
CONF_DEFAULT  = 0.25
IOU_DEFAULT   = 0.45
IMGSZ         = 640     # mismo que TRAIN_KWARGS / model.val()

# URL de descarga del modelo (Google Drive o Hugging Face)
# Si el modelo está en el repo (< 100 MB) se usa la ruta local automáticamente.
MODEL_GDRIVE_URL = (
    "https://drive.google.com/uc?export=download&id=TU_FILE_ID_AQUI"
)
MODEL_LOCAL_PATH = Path("prod/model/best.pt")
MODEL_FALLBACK_PATH = Path("model/best.pt")


# ── Carga del modelo ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo YOLOv8 (Exp_G7)…")
def load_model(model_path: str | None = None) -> YOLO:
    """
    Carga el modelo YOLOv8 desde disco o lo descarga si no existe.
    Cacheado con @st.cache_resource — se ejecuta una sola vez por sesión.

    Args:
        model_path: ruta explícita al .pt. Si es None, se busca automáticamente.

    Returns:
        Instancia de YOLO lista para inferencia.
    """
    # 1. Ruta explícita (pasada por el llamador)
    if model_path:
        pt = Path(model_path)
        if pt.exists():
            return YOLO(str(pt))

    # 2. Rutas locales estándar del repositorio
    for candidate in [MODEL_LOCAL_PATH, MODEL_FALLBACK_PATH, Path("best.pt")]:
        if candidate.exists():
            return YOLO(str(candidate))

    # 3. Descarga desde Drive (estrategia documentada en README para archivos > límite hosting)
    _download_model(MODEL_LOCAL_PATH)
    return YOLO(str(MODEL_LOCAL_PATH))


def _download_model(dest: Path) -> None:
    """Descarga best.pt desde Google Drive si no existe localmente."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Descargando pesos del modelo desde Drive…"):
        r = requests.get(MODEL_GDRIVE_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


# ── Preprocesamiento ─────────────────────────────────────────────────────────

def preprocess_image(uploaded_file) -> np.ndarray:
    """
    Convierte el archivo subido por el usuario en un array BGR (H×W×3).

    El preprocesamiento es IDÉNTICO al de validación/test en el notebook:
      - No hay resize manual ni normalización.
      - Ultralytics hace internamente resize+pad a imgsz=640 y /255.
      - Solo se convierte de PIL/bytes a numpy BGR para que cv2 pueda dibujar
        los resultados sobre la imagen original.

    Args:
        uploaded_file: objeto UploadedFile de Streamlit.

    Returns:
        np.ndarray BGR con la imagen en su resolución original.
    """
    pil_img = Image.open(uploaded_file).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


# ── Inferencia ───────────────────────────────────────────────────────────────

class Detection(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_id: int
    cls_name: str


def run_inference(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float = CONF_DEFAULT,
    iou: float = IOU_DEFAULT,
) -> list[Detection]:
    """
    Ejecuta inferencia YOLOv8 sobre la imagen.

    Parámetros idénticos al notebook (Exp_G7, sección de inferencia):
      imgsz=640, conf=0.25

    Args:
        model:   modelo YOLO cargado.
        img_bgr: imagen BGR numpy.
        conf:    umbral de confianza.
        iou:     umbral NMS IoU.

    Returns:
        Lista de Detection con las predicciones.
    """
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        imgsz=IMGSZ,
        verbose=False,
    )[0]

    detections: list[Detection] = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id   = int(box.cls.item())
        cls_name = CLASSES.get(cls_id, f"cls_{cls_id}")
        conf_val = float(box.conf.item())
        detections.append(Detection(x1, y1, x2, y2, conf_val, cls_id, cls_name))

    return detections


# ── Postprocesamiento / visualización ────────────────────────────────────────

def postprocess_results(
    img_bgr: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """
    Dibuja bounding boxes y etiquetas sobre una copia de la imagen.

    Args:
        img_bgr:    imagen original BGR.
        detections: lista de Detection del paso de inferencia.

    Returns:
        Imagen BGR con bounding boxes dibujados.
    """
    img_draw = img_bgr.copy()

    for det in detections:
        color = COLORS.get(det.cls_id, (200, 200, 200))
        label = f"{det.cls_name} {det.conf:.2f}"

        # Bounding box
        cv2.rectangle(img_draw, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        # Fondo del texto
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            img_draw,
            (det.x1, det.y1 - th - 6),
            (det.x1 + tw + 4, det.y1),
            color,
            -1,
        )

        # Texto
        cv2.putText(
            img_draw,
            label,
            (det.x1 + 2, det.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return img_draw


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convierte numpy BGR a PIL RGB para mostrar en Streamlit."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serializa PIL Image a bytes para el botón de descarga."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def summarize_detections(detections: list[Detection]) -> dict:
    """
    Genera un resumen de las detecciones por clase.

    Returns:
        dict con claves 'erupted', 'impacted', 'total' y lista 'details'.
    """
    erupted  = [d for d in detections if d.cls_id == 0]
    impacted = [d for d in detections if d.cls_id == 1]
    return {
        "total":    len(detections),
        "erupted":  len(erupted),
        "impacted": len(impacted),
        "details":  detections,
    }