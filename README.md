<p align="center">
  <img src="src/dentex_logo.png" width="180" alt="WisdomScan logo">
</p>

# WisdomScan

Detección automática de muelas del juicio impactadas en radiografías panorámicas dentales, usando YOLOv8m fine-tuneado sobre el dataset DENTEX MICCAI 2023 y ExAn-MTM (Expert-Annotated Mandibular Third Molar)

**🔗 App desplegada:** https://wisdomscan.streamlit.app/

---

## Descripción

Las muelas del juicio impactadas (terceros molares retenidos) son una de las patologías dentales más frecuentes y pueden causar dolor, infección y daño a dientes adyacentes si no se detectan a tiempo. El diagnóstico tradicional requiere un especialista que interprete radiografías panorámicas manualmente.

WisdomScan permite subir una radiografía panorámica (OPG) y obtener en segundos la localización de las muelas impactadas con bounding boxes y nivel de confianza, sin necesidad de software especializado.

### Modelo

El modelo de producción: YOLOv8m fine-tuneado en una cadena de 3 etapas sobre los datasets DENTEX y ExAn-MTM, con imágenes downscaleadas a ~900px para igualar la escala de imágenes reales de entrada.

| Métrica | Valor |
|---------|-------|
| mAP50 (test) | **0.992** |
| Precision | 0.980 |
| Recall | 0.999 |
| F1 | 0.989 |

### Funcionalidades

- Detección de muelas impactadas con bounding boxes sobre la radiografía
- Tabla de detecciones con confianza, posición (superior/inferior) y dimensiones
- Modo SAHI (tiling) para radiografías de alta resolución
- CLAHE para normalización de contraste en imágenes oscuras
- Descarga del resultado anotado

---

## Integrantes

| Nombre |
|--------|
| Anselmi, Matías |
| De Coninck, Ramiro |
| Eraso, Leandro |
| Mazurán, Clara |
| Olivera Balls, Julián |

**Materia:** Redes Neuronales — UTN Facultad Regional Mendoza  
**Docente:** Ing. Pablo Marinozi

---

## Datasets

| Dataset | Fuente | Licencia |
|---------|--------|----------|
| DENTEX MICCAI 2023 | [HuggingFace — ibrahimhamamci/DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) | CC BY 4.0 |
| ExAn-MTM | [Kaggle](https://www.kaggle.com/datasets/ikyd26/expert-annotated-mandibular-third-molar-dataset) + [Figshare](https://doi.org/10.6084/m9.figshare.21621705) | CC BY 4.0 |

---

## Estructura del repositorio

```
wisdomscan/
├── data/
│   ├── 01_dataset_preparation.ipynb
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── dev/
│   ├── 02_dataset_pytorch.ipynb
│   └── 03_experiments.ipynb
├── prod/
│   ├── model/
│   │   └── best.pt           # pesos del modelo de producción
│   ├── app.py                # interfaz Streamlit
│   └── utils.py              # funciones auxiliares
├── src/
│   └── dentex_logo.png
├── .gitignore
├── .python-version           # Python 3.11
├── packages.txt              # dependencias del sistema (Streamlit Cloud)
├── README.md
└── requirements.txt
```

---

## Correr localmente

### Requisitos

- Python 3.11
- Git

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/JulianOliveraBalls/wisdomscan.git
cd wisdomscan

# 2. Crear entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Correr la app
streamlit run prod/app.py
```

La app queda disponible en http://localhost:8501

### Requisitos del sistema (Linux)

Si corrés en Linux y cv2 falla, instalá:

```bash
sudo apt-get install libgl1
```

---

## Reproducir los experimentos

Para reproducir el pipeline completo desde cero, correr los notebooks en orden en Google Colab:

```
data/01_dataset_preparation.ipynb   — descarga y prepara los datasets
dev/02_dataset_pytorch.ipynb        — Dataset, DataLoaders y augmentation
dev/03_experiments.ipynb            — entrenamientos y evaluación
```
