"""
utils.py — Funciones auxiliares para la app de detección de muelas del juicio.

Responsabilidades:
  - Carga cacheada del modelo YOLOv8 (load_model)
  - Preprocesamiento de la imagen de entrada (preprocess_image)
  - Inferencia estándar y con SAHI (tiling) para imágenes grandes
  - Postprocesamiento / dibujo de bounding boxes

IMPORTANTE: el preprocesamiento es idéntico al de entrenamiento/validación:
  - imgsz=640, normalización interna de Ultralytics
  - NO se aplica ninguna transformación manual antes de llamar a predict()
  - YOLO hace internamente: resize + pad a 640×640, /255

SAHI (Slicing Aided Hyper Inference):
  - Divide la imagen en tiles solapados de 640×640
  - Corre el modelo en cada tile
  - Reensambla con NMS global
  - Mejora la detección en radiografías grandes donde las muelas
    quedan muy pequeñas relativas al frame completo
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
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# ── Constantes ───────────────────────────────────────────────────────────────

# Si usás G8:
CLASSES = {0: "impacted"}

COLORS = {
    0: (34, 197, 94),   # verde — erupted
    1: (0, 80, 220),    # azul  — impacted
}

CONF_DEFAULT    = 0.25
IOU_DEFAULT     = 0.45
IMGSZ           = 640

# Parámetros SAHI — tile de 640×640 con 20% de overlap
SAHI_SLICE_H    = 640
SAHI_SLICE_W    = 640
SAHI_OVERLAP    = 0.2   # 20% solapamiento entre tiles

MODEL_GDRIVE_URL    = "https://drive.google.com/uc?export=download&id=TU_FILE_ID_AQUI"
MODEL_LOCAL_PATH    = Path("prod/model/best.pt")
MODEL_FALLBACK_PATH = Path("model/best.pt")


# ── Carga del modelo ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo YOLOv8 (Exp_G7)…")
def load_model(model_path: str | None = None) -> YOLO:
    """
    Carga el modelo YOLOv8 desde disco o lo descarga si no existe.
    Cacheado con @st.cache_resource — se ejecuta una sola vez por sesión.
    """
    if model_path:
        pt = Path(model_path)
        if pt.exists():
            return YOLO(str(pt))

    for candidate in [MODEL_LOCAL_PATH, MODEL_FALLBACK_PATH, Path("best.pt")]:
        if candidate.exists():
            return YOLO(str(candidate))

    _download_model(MODEL_LOCAL_PATH)
    return YOLO(str(MODEL_LOCAL_PATH))


@st.cache_resource(show_spinner="Cargando modelo SAHI…")
def load_sahi_model(model_path: str | None = None) -> AutoDetectionModel:
    """
    Carga el modelo envuelto en AutoDetectionModel de SAHI.
    Necesario para get_sliced_prediction().
    Cacheado por separado para no recargar en cada interacción.
    """
    pt = None
    if model_path:
        pt = Path(model_path)
    if pt is None or not pt.exists():
        for candidate in [MODEL_LOCAL_PATH, MODEL_FALLBACK_PATH, Path("best.pt")]:
            if candidate.exists():
                pt = candidate
                break

    if pt is None:
        _download_model(MODEL_LOCAL_PATH)
        pt = MODEL_LOCAL_PATH

    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(pt),
        confidence_threshold=CONF_DEFAULT,
        device="cpu",
    )


def _download_model(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Descargando pesos del modelo desde Drive…"):
        r = requests.get(MODEL_GDRIVE_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


# ── Preprocesamiento ─────────────────────────────────────────────────────────

def preprocess_image(uploaded_file, enhance_contrast: bool = False) -> np.ndarray:
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # Upscale al tamaño del dataset DENTEX si la imagen es más chica
    TARGET_W = 2800
    TARGET_H = 1316
    w, h = pil_img.size
    
    if w < TARGET_W or h < TARGET_H:
        # Escalar manteniendo aspect ratio hasta que entre en el target
        scale = min(TARGET_W / w, TARGET_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    if enhance_contrast:
        bgr = _apply_clahe(bgr)
    
    return bgr

def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) en LAB.
    Mejora el contraste local sin saturar zonas ya brillantes.
    clipLimit=2.0 y tileGridSize=(8,8) son valores estándar para RX dentales.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ── Detección estándar (imagen completa) ─────────────────────────────────────

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
    Inferencia estándar sobre la imagen completa (sin tiling).
    Parámetros idénticos al notebook: conf=0.25, imgsz=640.
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


# ── Detección con SAHI (tiling) ───────────────────────────────────────────────

def run_inference_sahi(
    sahi_model: AutoDetectionModel,
    img_bgr: np.ndarray,
    conf: float = CONF_DEFAULT,
    iou: float = IOU_DEFAULT,
) -> list[Detection]:
    """
    Inferencia con SAHI: divide la imagen en tiles 640×640 con 20% overlap,
    corre el modelo en cada tile y reensambla con NMS global.

    Mejora la detección en radiografías panorámicas grandes donde las muelas
    superiores quedan muy pequeñas relativas al frame completo.

    Args:
        sahi_model: modelo cargado con load_sahi_model()
        img_bgr:    imagen BGR numpy en resolución original
        conf:       umbral de confianza
        iou:        umbral NMS para el ensamblado de tiles

    Returns:
        Lista de Detection con coordenadas en el espacio de la imagen original.
    """
    # SAHI trabaja con PIL RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Actualizar umbral de confianza del modelo SAHI
    sahi_model.confidence_threshold = conf

    result = get_sliced_prediction(
        image=pil_img,
        detection_model=sahi_model,
        slice_height=SAHI_SLICE_H,
        slice_width=SAHI_SLICE_W,
        overlap_height_ratio=SAHI_OVERLAP,
        overlap_width_ratio=SAHI_OVERLAP,
        postprocess_type="NMM",         # Non-Maximum Merging — más suave que NMS duro
        postprocess_match_threshold=iou,
        verbose=0,
    )

    detections: list[Detection] = []
    for obj in result.object_prediction_list:
        bbox     = obj.bbox
        x1, y1   = int(bbox.minx), int(bbox.miny)
        x2, y2   = int(bbox.maxx), int(bbox.maxy)
        cls_id   = obj.category.id
        cls_name = CLASSES.get(cls_id, obj.category.name)
        conf_val = float(obj.score.value)
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
    erupted  = [d for d in detections if d.cls_id == 0]
    impacted = [d for d in detections if d.cls_id == 1]
    return {
        "total":    len(detections),
        "erupted":  len(erupted),
        "impacted": len(impacted),
        "details":  detections,
    }