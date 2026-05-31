"""
download.py — Descarga automática del dataset DENTEX desde HuggingFace.

Uso:
    python data/download.py

El dataset se guarda en data/raw/dentex_raw/.
Requiere ~3 GB de espacio en disco.
"""

import sys
from pathlib import Path

# ── Dependencias ──────────────────────────────────────────────────
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub no está instalado.")
    print("        Ejecutá: pip install huggingface_hub")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────
# Si se corre desde la raíz del repo: data/raw/
# Si se corre desde data/: raw/
SCRIPT_DIR   = Path(__file__).parent.resolve()
RAW_DIR      = SCRIPT_DIR / 'raw' / 'dentex_raw'
REPO_ID      = 'ibrahimhamamci/DENTEX'

def is_real_file(p):
    try:
        return p.resolve().exists() and p.stat().st_size > 0
    except (OSError, FileNotFoundError):
        return False

def check_existing():
    """Verifica si el dataset ya está descargado."""
    if not RAW_DIR.exists():
        return False
    json_files = [f for f in RAW_DIR.rglob('*.json') if is_real_file(f)]
    return len(json_files) > 0

def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if check_existing():
        json_files = [f for f in RAW_DIR.rglob('*.json') if is_real_file(f)]
        print(f"[OK] Dataset ya descargado ({len(json_files)} JSONs en {RAW_DIR})")
        print("     Para re-descargar, borrar la carpeta data/raw/ y volver a correr.")
        return

    print(f"[INFO] Descargando DENTEX desde HuggingFace...")
    print(f"       Destino: {RAW_DIR}")
    print(f"       Tamaño estimado: ~3 GB — puede tardar varios minutos.")
    print()

    snapshot_download(
        repo_id=REPO_ID,
        repo_type='dataset',
        local_dir=str(RAW_DIR),
        ignore_patterns=['*.git*'],
    )

    # Verificar descarga
    json_files = [f for f in RAW_DIR.rglob('*.json') if is_real_file(f)]
    if json_files:
        print()
        print(f"[OK] Descarga completada — {len(json_files)} JSONs encontrados")

        # Buscar carpeta de imágenes
        xrays_dirs = [d for d in RAW_DIR.rglob('xrays') if d.is_dir()]
        for d in xrays_dirs:
            imgs = list(d.glob('*.png')) + list(d.glob('*.jpg'))
            print(f"     {d.relative_to(RAW_DIR)}/  →  {len(imgs)} imágenes")
    else:
        print("[ERROR] La descarga no encontró JSONs válidos.")
        print(f"        Verificar manualmente: {RAW_DIR}")
        sys.exit(1)

    # Verificar imágenes de validación (zip separado)
    val_zip = RAW_DIR / 'DENTEX' / 'validation_data.zip'
    if val_zip.exists() and is_real_file(val_zip):
        import zipfile
        print(f"\n[INFO] Descomprimiendo validation_data.zip...")
        with zipfile.ZipFile(val_zip, 'r') as z:
            z.extractall(val_zip.parent)
        print(f"[OK] Validación descomprimida")

    print()
    print("[OK] Dataset listo. Podés correr los notebooks en dev/")

if __name__ == '__main__':
    download()
