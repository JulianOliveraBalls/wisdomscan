# Dataset — DENTEX

## Fuente

**DENTEX: Dental Enumeration and Diagnosis on Panoramic X-rays**  
MICCAI 2023 — Ibrahim Ethem Hamamci et al.

- Paper: [arXiv:2305.19112](https://arxiv.org/abs/2305.19112)  
- Dataset: [https://huggingface.co/datasets/ibrahimhamamci/DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)  
- GitHub del paper: [https://github.com/ibrahimethemhamamci/DENTEX](https://github.com/ibrahimethemhamamci/DENTEX)  
- Licencia: **CC-BY-NC-SA 4.0** (uso no comercial, con atribución obligatoria)

---

## Cómo descargar

### Automático (recomendado)

```bash
python data/download.py
```

El script descarga el dataset completo desde HuggingFace y lo organiza en `data/raw/`.  
Requiere ~3 GB de espacio en disco y conexión a internet.

### Manual

Ir a [https://huggingface.co/datasets/ibrahimhamamci/DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX), descargar los archivos y colocarlos en `data/raw/dentex_raw/`.

---

## Estructura esperada tras la descarga

```
data/
├── raw/
│   └── dentex_raw/
│       └── DENTEX/
│           └── training_data/
│               └── quadrant-enumeration-disease/
│                   ├── train_quadrant_enumeration_disease.json
│                   └── xrays/
│                       ├── train_1.png
│                       ├── train_2.png
│                       └── ...
├── processed/         # Generado por los notebooks
│   └── yolo_dataset/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── dataset.yaml
├── train.csv          # Split versionado en Git
├── val.csv
├── test.csv
├── download.py
└── README.md
```

---

## Por qué las imágenes no están en el repositorio

Git no está diseñado para archivos binarios grandes. Las imágenes del dataset pesan ~2.7 GB en total, y GitHub tiene un límite de 100 MB por archivo. Por eso las imágenes se descargan desde HuggingFace en tiempo de ejecución y están excluidas del repositorio via `.gitignore`.

Lo que sí se versiona son los **CSVs de splits** (`train.csv`, `val.csv`, `test.csv`), que son livianos y garantizan que todos trabajen con exactamente el mismo particionado (mismas imágenes en train/val/test).

---

## Descripción del dataset

El dataset (c) — el que usamos — contiene 1005 radiografías panorámicas completamente anotadas con:

- `category_id_1` — cuadrante FDI (1-4)
- `category_id_2` — posición del diente en el cuadrante (1-8). El **7** es el tercer molar (muela del juicio)
- `category_id_3` — diagnóstico (0=impacted, 1=caries, 2=caries profunda, 3=lesión periapical)

De las 705 imágenes disponibles públicamente, **440 tienen muela del juicio anotada** y son las que usamos para entrenar el detector binario erupted/impacted.

Split utilizado: **70% train / 5% val / 25% test** (estratificado, seed=42).
