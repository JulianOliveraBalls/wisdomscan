# Dataset — Origen y descarga

Este proyecto usa dos datasets públicos de radiografías panorámicas dentales con anotaciones de muelas del juicio. Ambos son de acceso libre bajo licencia CC BY 4.0.

---

## Dataset 1 — DENTEX MICCAI 2023

**Descripción:** Dataset oficial del challenge MICCAI 2023 para detección de enfermedades dentales. Contiene 1.005 radiografías panorámicas con anotaciones jerárquicas en formato COCO JSON: cuadrante → número FDI → patología. Se usan las imágenes con anotaciones de tercer molar (`category_id_2 == 7`) con estado impactado (`category_id_3 == 0`).

**Fuente:** https://huggingface.co/datasets/ibrahimhamamci/DENTEX  
**Licencia:** CC BY 4.0  
**Tamaño:** ~1.8 GB (imágenes PNG, ~2800×1316px)

### Descarga automática (desde el notebook)

La descarga está implementada en `data/01_dataset_preparation.ipynb` y se ejecuta automáticamente al correr el notebook en Colab:

```python
from huggingface_hub import hf_hub_download as hf_dl
import zipfile, os

DENTEX_DIR = Path('/content/dentex-wisdom-teeth/data/raw/dentex_raw')

for fname, cname in [('DENTEX/training_data.zip',   'training_data'),
                      ('DENTEX/validation_data.zip', 'validation_data')]:
    zp = hf_dl(repo_id='ibrahimhamamci/DENTEX', repo_type='dataset',
               filename=fname, local_dir=str(DENTEX_DIR))
    with zipfile.ZipFile(zp, 'r') as z:
        z.extractall(str(DENTEX_DIR / 'DENTEX'))
    os.remove(zp)
```

### Descarga manual

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='ibrahimhamamci/DENTEX', repo_type='dataset',
                  local_dir='data/raw/dentex_raw')
"
```

---

## Dataset 2 — ExAn-MTM (Expert-Annotated Mandibular Third Molar)

**Descripción:** 973 radiografías panorámicas con anotaciones de muelas del juicio en formato YOLO TXT (`clase cx cy w h` normalizado). Las imágenes ya vienen en resolución reducida (~885×430px), representativa de equipos de menor resolución.

**Fuente labels:** https://www.kaggle.com/datasets/ikyd26/expert-annotated-mandibular-third-molar-dataset  
**Fuente imágenes:** https://doi.org/10.6084/m9.figshare.21621705  
**Licencia:** CC BY 4.0  
**Tamaño:** ~1.6 GB (imágenes JPEG)

### Descarga automática (desde el notebook)

La descarga también está implementada en `data/01_dataset_preparation.ipynb`. Requiere un token de Kaggle:

1. Crear cuenta en https://www.kaggle.com
2. Ir a **Settings → API → Create New Token** — descarga `kaggle.json`
3. Reemplazar `KAGGLE_TOKEN` en la celda del notebook con el valor del token

```python
import kaggle

# Labels desde Kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    'ikyd26/expert-annotated-mandibular-third-molar-dataset',
    path='data/raw/exan_mtm', unzip=True)

# Imágenes desde Figshare (~1.6 GB)
import requests
with requests.get('https://ndownloader.figshare.com/files/38322366',
                  stream=True) as r:
    with open('data/raw/exan_mtm/dental_dataset.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
```

### Descarga manual

- **Labels:** descargar desde https://www.kaggle.com/datasets/ikyd26/expert-annotated-mandibular-third-molar-dataset y extraer en `data/raw/exan_mtm/`
- **Imágenes:** descargar desde https://figshare.com/articles/dataset/21621705 y extraer en `data/raw/exan_mtm/`

---

## Estructura resultante tras la descarga

```
data/raw/
├── dentex_raw/
│   └── DENTEX/
│       ├── training_data/
│       │   ├── xrays/          # imágenes PNG
│       │   └── train_disease_data.json
│       └── validation_data/
│           ├── xrays/
│           └── val_disease_data.json
└── exan_mtm/
    └── ExAn-MTM dataset/
        ├── train/
        │   ├── images/         # imágenes JPG
        │   └── labels/         # archivos YOLO TXT
        └── valid/
            ├── images/
            └── labels/
```

Una vez descargados, correr el resto del notebook `01_dataset_preparation.ipynb` para generar los datasets procesados en `data/processed/`.