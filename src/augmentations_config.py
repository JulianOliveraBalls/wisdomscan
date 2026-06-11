"""
src/augmentations_config.py
Configuración de augmentations para detección de muelas del juicio.

Uso:
    import sys; sys.path.insert(0, str(REPO_ROOT / 'src'))
    from augmentations_config import AUGMENTATIONS, BASE_TRANSFORM, MEAN, STD

Las augmentations SOLO se aplican al train set.
Val y test usan siempre BASE_TRANSFORM.

Clases:
    0: erupted  — muela erupcionada normalmente
    1: impacted — muela retenida / impactada

Justificación clínica por transformación:
    HorizontalFlip   → el arco dental es bilateralmente simétrico
    ColorJitter      → variabilidad entre equipos de distintos hospitales
    RandomAffine     → variación de posicionamiento del paciente en el equipo
    RandomErasing    → simula artefactos metálicos (empastes, brackets, implantes)
    GaussianBlur     → variabilidad de resolución entre equipos
    VerticalFlip     → EXCLUIDO: anatómicamente incorrecto
    mosaic/mixup     → EXCLUIDO: mezclar anatomías distintas no es válido en Rx
"""

import torchvision.transforms as T

# ── Estadísticas de normalización ────────────────────────────────────────────
# Se usan las estadísticas de ImageNet porque YOLOv8 fue preentrenado con ellas.
# Las panorámicas son grises (R=G=B) pero se cargan como RGB.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Transform base (val y test siempre usan este) ─────────────────────────────
BASE_TRANSFORM = T.Compose([
    T.Resize((640, 640)),   # resolución nativa de YOLOv8 — necesaria para objetos pequeños
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# ── 6 estrategias de augmentation (solo train) ────────────────────────────────
AUGMENTATIONS = {

    # A: referencia limpia — sin augmentation adicional
    'A_baseline': T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ]),

    # B: solo flip horizontal — simetría bilateral del arco dental
    'B_flips': T.Compose([
        T.Resize((640, 640)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ]),

    # C: variabilidad de contraste + rotación leve
    'C_color': T.Compose([
        T.Resize((640, 640)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.RandomAffine(degrees=5),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ]),

    # D: deformaciones geométricas + artefactos metálicos
    'D_geometric': T.Compose([
        T.Resize((640, 640)),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.10), value='random'),
    ]),

    # E: combinación completa de B + C + D
    'E_full': T.Compose([
        T.Resize((640, 640)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.RandomErasing(p=0.10, scale=(0.02, 0.10), value='random'),
    ]),

    # F: E_full + blur para variabilidad de resolución entre equipos
    'F_mixaug': T.Compose([
        T.Resize((640, 640)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.15),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.RandomErasing(p=0.10, scale=(0.02, 0.10), value='random'),
    ]),
}