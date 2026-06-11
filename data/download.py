#!/usr/bin/env python3
"""
data/download.py
Descarga automática de DENTEX MICCAI 2023 + ExAn-MTM (imágenes desde Figshare).

Uso:
    python data/download.py

Genera:
    data/raw/dentex_raw/   <- DENTEX
    data/raw/exan_mtm/     <- ExAn-MTM labels + imágenes UESB
"""

import os, sys, json, shutil, zipfile, subprocess
from pathlib import Path

# ── Paths (relativos a la raíz del repo) ─────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
DATA_DIR    = REPO_ROOT / 'data'
RAW_DIR     = DATA_DIR / 'raw'
DENTEX_DIR  = RAW_DIR / 'dentex_raw'
EXAN_DIR    = RAW_DIR / 'exan_mtm'
for d in [DENTEX_DIR, EXAN_DIR]: d.mkdir(parents=True, exist_ok=True)

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *pkgs, '-q'])

def log(msg, level='INFO'):
    icons = {'INFO':'[INFO]','OK':'[OK]  ','WARN':'[WARN]','ERR':'[ERR] ','DATA':'[DATA]'}
    print(f'{icons.get(level,"[INFO]")} {msg}')

def is_real_file(p):
    try: return Path(p).resolve().exists() and Path(p).stat().st_size > 0
    except: return False


# ════════════════════════════════════════════════════════════════════════
# 1. DENTEX MICCAI 2023
# ════════════════════════════════════════════════════════════════════════
def download_dentex():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pip_install('huggingface_hub'); from huggingface_hub import hf_hub_download

    json_files = [f for f in DENTEX_DIR.rglob('*.json') if is_real_file(f)]
    if any('disease' in f.name.lower() for f in json_files):
        log('DENTEX ya descargado', 'OK'); return

    log('Descargando DENTEX desde HuggingFace...', 'WARN')
    for fname, cname in [
        ('DENTEX/training_data.zip',   'training_data'),
        ('DENTEX/validation_data.zip', 'validation_data'),
    ]:
        check = DENTEX_DIR / 'DENTEX' / cname
        if check.exists() and len(list(check.rglob('*.png'))) > 0:
            log(f'{cname} ya existe', 'OK'); continue
        zp = hf_hub_download(
            repo_id='ibrahimhamamci/DENTEX', repo_type='dataset',
            filename=fname, local_dir=str(DENTEX_DIR))
        with zipfile.ZipFile(zp, 'r') as z:
            z.extractall(str(DENTEX_DIR / 'DENTEX'))
        os.remove(zp)
        log(f'{fname} extraído', 'OK')

    log('DENTEX listo', 'OK')


# ════════════════════════════════════════════════════════════════════════
# 2. ExAn-MTM — labels desde Kaggle, imágenes desde Figshare
# ════════════════════════════════════════════════════════════════════════
def download_exan_mtm():
    import requests

    EXAN_BASE      = EXAN_DIR / 'ExAn-MTM dataset'
    EXAN_TRAIN_IMG = EXAN_BASE / 'train' / 'images'
    EXAN_TRAIN_LBL = EXAN_BASE / 'train' / 'labels'
    EXAN_VAL_IMG   = EXAN_BASE / 'valid' / 'images'
    EXAN_VAL_LBL   = EXAN_BASE / 'valid' / 'labels'

    n_imgs = len(list(EXAN_TRAIN_IMG.glob('*.jpg'))) if EXAN_TRAIN_IMG.exists() else 0
    if n_imgs >= 875:
        log(f'ExAn-MTM ya descargado: {n_imgs} imgs train', 'OK'); return

    # ── Labels desde Kaggle ───────────────────────────────────────────
    KAGGLE_TOKEN = 'KGAT_9541105c6a2fec68be43345f171bc2c9'
    os.environ['KAGGLE_API_TOKEN'] = KAGGLE_TOKEN
    token_path = Path.home() / '.kaggle' / 'access_token'
    token_path.parent.mkdir(exist_ok=True)
    token_path.write_text(KAGGLE_TOKEN); token_path.chmod(0o600)

    try: import kaggle as _k
    except ImportError: pip_install('kaggle')

    n_labels = len(list(EXAN_TRAIN_LBL.glob('*.txt'))) if EXAN_TRAIN_LBL.exists() else 0
    if n_labels < 100:
        log('Descargando labels ExAn-MTM desde Kaggle...', 'WARN')
        r = subprocess.run(
            ['python', '-m', 'kaggle', 'datasets', 'download',
             '-d', 'ikyd26/expert-annotated-mandibular-third-molar-dataset',
             '--path', str(EXAN_DIR), '--unzip'],
            capture_output=True, text=True,
            env={**os.environ, 'KAGGLE_API_TOKEN': KAGGLE_TOKEN})
        if r.returncode != 0:
            log(f'Error Kaggle: {r.stderr[:300]}', 'ERR'); return
        log('Labels descargados', 'OK')

    # ── Imágenes desde Figshare ───────────────────────────────────────
    # Los labels de Kaggle NO incluyen imágenes.
    # Las imágenes son el dataset UESB, publicado en:
    # https://doi.org/10.6084/m9.figshare.21621705
    EXAN_TRAIN_IMG.mkdir(parents=True, exist_ok=True)
    EXAN_VAL_IMG.mkdir(parents=True, exist_ok=True)
    ZIP_PATH = EXAN_DIR / 'dental_dataset.zip'

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 1e9:
        log('Descargando imágenes UESB desde Figshare (1.6 GB)...', 'WARN')
        FIGSHARE_URL = 'https://ndownloader.figshare.com/files/38322366'
        with requests.get(FIGSHARE_URL, stream=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            with open(ZIP_PATH, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (200*1024*1024) == 0:
                        log(f'  {downloaded/1e6:.0f} / {total/1e6:.0f} MB', 'DATA')
        log(f'ZIP descargado: {ZIP_PATH.stat().st_size/1e9:.2f} GB', 'OK')

    log('Extrayendo imágenes...', 'WARN')
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_names = z.namelist()
        imgs_tr = {Path(n).stem: n for n in all_names
                   if 'Dataset and code/train/images' in n
                   and n.lower().endswith(('.jpg', '.png'))}
        imgs_val = {Path(n).stem: n for n in all_names
                    if 'Dataset and code/test/images' in n
                    and n.lower().endswith(('.jpg', '.png'))}

        extr_tr = extr_val = 0
        for stem in [p.stem for p in EXAN_TRAIN_LBL.glob('*.txt')]:
            dst = EXAN_TRAIN_IMG / f'{stem}.jpg'
            if not dst.exists() and stem in imgs_tr:
                dst.write_bytes(z.read(imgs_tr[stem])); extr_tr += 1
        for stem in [p.stem for p in EXAN_VAL_LBL.glob('*.txt')]:
            src_key = imgs_tr.get(stem) or imgs_val.get(stem)
            dst = EXAN_VAL_IMG / f'{stem}.jpg'
            if not dst.exists() and src_key:
                dst.write_bytes(z.read(src_key)); extr_val += 1

    log(f'ExAn-MTM extraído: {extr_tr} train, {extr_val} val', 'OK')


# ════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log('=== Descarga de datasets ===', 'INFO')
    download_dentex()
    download_exan_mtm()
    log('=== Todo listo ===', 'OK')
    log(f'  DENTEX:  {DENTEX_DIR}', 'DATA')
    log(f'  ExAn-MTM: {EXAN_DIR}', 'DATA')