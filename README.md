# Detección de Muelas del Juicio en Radiografías Panorámicas

Proyecto final de la materia **Redes Neuronales** — Universidad Tecnológica Nacional.  
Docente a cargo: **Ing. Pablo Marinozi**

---

## Descripción

Este proyecto implementa un sistema de detección automática de muelas del juicio (terceros molares) en radiografías panorámicas dentales, clasificándolas en dos estados clínicos:

- **erupted** — erupcionada, posición normal
- **impacted** — retenida, requiere intervención quirúrgica

La tarea es detección de objetos (bounding box + clase). El modelo localiza cada muela del juicio presente en la radiografía y la clasifica. Una imagen puede contener hasta 4 muelas en distintos estados simultáneamente.

La aplicación final (Streamlit) permite al odontólogo subir una radiografía y obtener las detecciones superpuestas sobre la imagen original.

---

## Dataset

El proyecto utiliza el dataset **DENTEX** (Dental Enumeration and Diagnosis on Panoramic X-rays), presentado en MICCAI 2023.

- Paper: [arXiv:2305.19112](https://arxiv.org/abs/2305.19112)
- Dataset: [Hugging Face — ibrahimhamamci/DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
- Licencia: CC-BY-NC-SA 4.0 (no comercial, con atribución)

Ver `data/README.md` para instrucciones de descarga.

---

## Arquitectura del modelo

Se utiliza **YOLOv8s** con fine-tuning desde pesos preentrenados en COCO.  
El baseline del paper (HierarchicalDet con DiffusionDet) fue evaluado y descartado por complejidad de implementación — ver `dev/02_entrenamiento.ipynb` para el análisis comparativo.

---

## Estructura del repositorio

```
.
├── data/
│   ├── README.md          # Instrucciones de descarga del dataset
│   ├── download.py        # Script de descarga automática desde HuggingFace
│   ├── train.csv          # Split de entrenamiento (rutas + etiquetas)
│   ├── val.csv            # Split de validación
│   └── test.csv           # Split de test
├── dev/
│   ├── 01_dataset_exploration.ipynb   # Entrega 1: EDA y preparación
│   └── 02_dataset_pytorch.ipynb       # Entrega 2: Dataset PyTorch + augmentation
├── prod/
│   └── app.py             # Aplicación Streamlit (entrega final)
├── requirements.txt
└── README.md
```

---

## Cómo ejecutar

### Opción A — Google Colab (recomendado)

1. Abrir el notebook en Colab haciendo clic en el badge de cada notebook
2. Ejecutar todas las celdas — la descarga del dataset se hace automáticamente desde HuggingFace
3. No se necesita Drive ni configuración adicional

### Opción B — Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/dentex-wisdom-teeth.git
cd dentex-wisdom-teeth

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Descargar el dataset
python data/download.py

# 4. Abrir los notebooks
jupyter notebook dev/
```

El script `data/download.py` descarga automáticamente el dataset desde HuggingFace en la carpeta `data/raw/`. No se requieren pasos manuales.

---

## Reproducibilidad

Todos los experimentos usan `seed=42`. El particionado train/val/test está fijado en los archivos `data/train.csv`, `data/val.csv` y `data/test.csv`, que se versionan en el repositorio. Clonar el repo y correr el script de descarga es suficiente para reproducir exactamente los mismos resultados.

---

## Resultados

*(Se completará en entregas posteriores)*

| Métrica | Valor |
|---------|-------|
| mAP@50  | — |
| mAP@50-95 | — |
| AR | — |

---

## Requisitos

Ver `requirements.txt`. Principales dependencias:

- `ultralytics` — YOLOv8
- `torch` / `torchvision`
- `huggingface_hub` — descarga del dataset
- `scikit-learn` — split estratificado
- `streamlit` — aplicación web (entrega final)
