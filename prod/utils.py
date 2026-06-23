"""
utils.py — Funciones auxiliares para la app de detección de muelas del juicio.

Responsabilidades:
  - Carga cacheada del modelo YOLOv8 (load_model)
  - Preprocesamiento de la imagen de entrada (preprocess_image)
  - Inferencia estándar sobre imagen completa
  - Postprocesamiento / dibujo de bounding boxes

IMPORTANTE: el preprocesamiento es idéntico al de entrenamiento/validación:
  - imgsz=640, normalización interna de Ultralytics
  - NO se aplica ninguna transformación manual antes de llamar a predict()
  - YOLO hace internamente: resize + pad a 640×640, /255
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Constantes ───────────────────────────────────────────────────────────────

CLASSES = {0: "impacted"}

COLORS = {
    0: (34, 197, 94),   # verde — impacted
}

CONF_DEFAULT = 0.15
IOU_DEFAULT  = 0.45
IMGSZ        = 640

MODEL_LOCAL_PATH    = Path(__file__).parent / "model" / "best.pt"
MODEL_FALLBACK_PATH = Path("dev/model/best.pt")


# ── Carga del modelo ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo YOLOv8 (Exp_G8b)…")
def load_model(model_path: str | None = None) -> YOLO:
    """
    Carga el modelo YOLOv8 desde disco.
    Cacheado con @st.cache_resource — se ejecuta una sola vez por sesión.
    """
    if model_path:
        pt = Path(model_path)
        if pt.exists():
            return YOLO(str(pt))

    for candidate in [MODEL_LOCAL_PATH, MODEL_FALLBACK_PATH, Path("best.pt")]:
        if candidate.exists():
            return YOLO(str(candidate))

    raise FileNotFoundError(
        "No se encontró best.pt. "
        "Verificar que prod/model/best.pt esté commiteado en el repo."
    )


# ── Preprocesamiento ─────────────────────────────────────────────────────────

def preprocess_image(uploaded_file) -> np.ndarray:
    """
    Convierte el archivo subido a BGR numpy.
    G8b fue entrenado con imágenes ~900px — YOLO hace el resize a 640px internamente.
    """
    pil_img = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Detección ─────────────────────────────────────────────────────────────────

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
    """Inferencia estándar sobre la imagen completa."""
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
    """Dibuja bounding boxes y etiquetas sobre una copia de la imagen."""
    img_draw = img_bgr.copy()

    for det in detections:
        color = COLORS.get(det.cls_id, (200, 200, 200))
        label = f"{det.cls_name} {det.conf:.2f}"

        cv2.rectangle(img_draw, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            img_draw,
            (det.x1, det.y1 - th - 6),
            (det.x1 + tw + 4, det.y1),
            color, -1,
        )
        cv2.putText(
            img_draw, label, (det.x1 + 2, det.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            lineType=cv2.LINE_AA,
        )

    return img_draw


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def summarize_detections(detections: list[Detection]) -> dict:
    impacted = [d for d in detections if d.cls_name == "impacted"]
    return {
        "total":    len(detections),
        "impacted": len(impacted),
        "details":  detections,
    }