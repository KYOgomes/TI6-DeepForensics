# -*- coding: utf-8 -*-
"""
DeepForensics — Backend HTTP (Flask)
======================================
Expõe o pipeline de detector_unificado.py como uma API REST consumida
pelo front-end estático em index.html.

Endpoints:
  GET  /api/health
  GET  /api/info
  POST /api/analyze               (multipart: image=<file>)
  POST /api/batch                 (multipart: images=<file>*, mode=seq|par)
  POST /api/dataset               (multipart: au=<file>*, sp=<file>*)
  POST /api/benchmark             (multipart: images=<file>*)
  POST /api/dataset-local         (json: {au_dir, sp_dir, max_per_class?})
  POST /api/benchmark-local       (json: {dir, max?})

Como rodar:
  python Algorithm/app.py
  (servidor sobe em http://127.0.0.1:5000)
"""

import os
import sys
import tempfile
import shutil
import time
import io
from glob import glob

# Reconfigura stdout/stderr para UTF-8 no Windows
# (o detector usa caracteres como → e acentos nos prints de log)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Importa o pipeline
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from detector_unificado import (
    analisar_para_api,
    analisar_lote_api,
    avaliar_dataset_api,
    benchmark_paralelo_api,
    benchmark_analises_api,
    _analisar_leve,
    L1, L2, L3, L4,
)
import detector_ml
from multiprocessing import cpu_count, freeze_support

# Front-end é a raiz do repositório
app = Flask(__name__, static_folder=ROOT, static_url_path='')
CORS(app)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
MAX_BATCH   = 60   # imagens por requisição em lote
MAX_LOCAL   = 200  # imagens por classe ao usar dataset local


def _ext_ok(name):
    return os.path.splitext(name.lower())[1] in ALLOWED_EXT


def _save_uploaded(files, tmpdir):
    """Salva arquivos multipart no tmpdir, devolve lista de paths válidos."""
    paths = []
    for f in files:
        if not f or not f.filename:
            continue
        if not _ext_ok(f.filename):
            continue
        safe = os.path.basename(f.filename).replace('\\', '_').replace('/', '_')
        dst  = os.path.join(tmpdir, safe)
        # evita colisão de nome
        i = 1
        base, ext = os.path.splitext(dst)
        while os.path.exists(dst):
            dst = f"{base}__{i}{ext}"
            i += 1
        f.save(dst)
        paths.append(dst)
    return paths


# ─────────────────────────────────────────────
# Estáticos: index.html e assets
# ─────────────────────────────────────────────
@app.route('/')
def root():
    return send_from_directory(ROOT, 'index.html')


# ─────────────────────────────────────────────
# Info / health
# ─────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'ok': True, 'service': 'DeepForensics API'})


def _resolve_dataset_dir(base):
    """Se a pasta principal estiver vazia mas houver Dataset/<X>/sample/, usa a sample."""
    if not os.path.isdir(base):
        return base
    has_imgs = any(
        f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))
        for f in os.listdir(base)
        if os.path.isfile(os.path.join(base, f))
    )
    if has_imgs:
        return base
    sample = os.path.join(base, 'sample')
    if os.path.isdir(sample):
        return sample
    return base


@app.route('/api/info')
def info():
    au_base = os.path.join(ROOT, 'Dataset', 'Au')
    sp_base = os.path.join(ROOT, 'Dataset', 'Tp')
    return jsonify({
        'ok'        : True,
        'cpu_count' : cpu_count(),
        'limiares'  : {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4},
        'max_batch' : MAX_BATCH,
        'max_local' : MAX_LOCAL,
        'dataset_local': {
            'au': _resolve_dataset_dir(au_base),
            'sp': _resolve_dataset_dir(sp_base),
            'au_exists': os.path.isdir(au_base),
            'sp_exists': os.path.isdir(sp_base),
        },
    })


# ─────────────────────────────────────────────
# Benchmark CP — roda sobre o que o usuário enviou (não sobre o dataset):
#   • imagem única → distribui as 4 análises (VP/ELA/Ruído/Escala) entre workers
#   • lote         → distribui as N imagens enviadas entre workers
# ─────────────────────────────────────────────
def _bench_lote(paths):
    """Benchmark de CP distribuindo as imagens enviadas entre os workers."""
    cfg = sorted(set([1, 2, 4, cpu_count()]))
    cfg = [w for w in cfg if w <= max(len(paths), 1)] or [1]
    bm = benchmark_paralelo_api(paths, configs_workers=cfg, incluir_fraca=True)
    if bm and bm.get('ok'):
        bm['titulo'] = f"Paralelizacao das {len(paths)} imagens enviadas"
    return bm


# ─────────────────────────────────────────────
# Modo 1 — imagem única (pipeline completo + visuais + benchmark das 4 análises)
# ─────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze_single():
    f = request.files.get('image')
    if not f or not f.filename:
        return jsonify({'ok': False, 'erro': 'envie image=<arquivo>'}), 400
    if not _ext_ok(f.filename):
        return jsonify({'ok': False, 'erro': 'formato nao suportado'}), 400

    tmpdir = tempfile.mkdtemp(prefix='df_')
    try:
        safe = os.path.basename(f.filename)
        dst  = os.path.join(tmpdir, safe)
        f.save(dst)

        t_ini = time.perf_counter()
        payload = analisar_para_api(dst, include_visuals=True)
        payload['tempo'] = round(time.perf_counter() - t_ini, 3)
        bm = benchmark_analises_api(dst)   # paraleliza as 4 análises desta imagem
        if bm and bm.get('ok'):
            payload['benchmark'] = bm
        return jsonify(payload)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────
# Modo 2 — lote (várias imagens enviadas)
#   mode = 'seq' (1 worker) ou 'par' (cpu_count())
# ─────────────────────────────────────────────
@app.route('/api/batch', methods=['POST'])
def analyze_batch():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'ok': False, 'erro': 'envie images=<arquivo>*'}), 400

    mode = (request.form.get('mode') or 'par').lower()
    workers = 1 if mode == 'seq' else cpu_count()

    tmpdir = tempfile.mkdtemp(prefix='df_lote_')
    try:
        paths = _save_uploaded(files, tmpdir)[:MAX_BATCH]
        if not paths:
            return jsonify({'ok': False, 'erro': 'nenhum arquivo valido'}), 400

        result = analisar_lote_api(paths, workers=workers)
        result['modo']    = mode
        result['limite']  = MAX_BATCH
        bm = _bench_lote(paths)            # distribui as N imagens enviadas
        if bm and bm.get('ok'):
            result['benchmark'] = bm
        return jsonify(result)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────
# Modo 3 — dataset rotulado (uploads em au[] e sp[])
# ─────────────────────────────────────────────
@app.route('/api/dataset', methods=['POST'])
def analyze_dataset():
    au_files = request.files.getlist('au')
    sp_files = request.files.getlist('sp')
    if not au_files and not sp_files:
        return jsonify({'ok': False, 'erro': 'envie au=<arquivo>* e/ou sp=<arquivo>*'}), 400

    mode = (request.form.get('mode') or 'par').lower()
    workers = 1 if mode == 'seq' else cpu_count()

    tmpdir = tempfile.mkdtemp(prefix='df_ds_')
    try:
        au_paths = _save_uploaded(au_files, tmpdir)[:MAX_BATCH]
        sp_paths = _save_uploaded(sp_files, tmpdir)[:MAX_BATCH]
        if not (au_paths or sp_paths):
            return jsonify({'ok': False, 'erro': 'nenhum arquivo valido'}), 400

        result = avaliar_dataset_api(au_paths, sp_paths, workers=workers)
        result['modo'] = mode
        return jsonify(result)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────
# Modo 3b — dataset rotulado usando pastas locais (sem upload)
#   Body JSON: {au_dir?, sp_dir?, max_per_class?}
#   Padrão: Dataset/Au e Dataset/Tp
# ─────────────────────────────────────────────
@app.route('/api/dataset-local', methods=['POST'])
def analyze_dataset_local():
    data = request.get_json(silent=True) or {}
    au_dir = data.get('au_dir') or _resolve_dataset_dir(os.path.join(ROOT, 'Dataset', 'Au'))
    sp_dir = data.get('sp_dir') or _resolve_dataset_dir(os.path.join(ROOT, 'Dataset', 'Tp'))
    max_per_class = int(data.get('max_per_class') or 40)
    max_per_class = min(max_per_class, MAX_LOCAL)

    if not (os.path.isdir(au_dir) and os.path.isdir(sp_dir)):
        return jsonify({'ok': False, 'erro': 'pastas nao encontradas',
                        'au_dir': au_dir, 'sp_dir': sp_dir}), 400

    def _list_imgs(p):
        out = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp'):
            out.extend(glob(os.path.join(p, ext)))
        return sorted(out)[:max_per_class]

    au_paths = _list_imgs(au_dir)
    sp_paths = _list_imgs(sp_dir)
    if not (au_paths or sp_paths):
        return jsonify({'ok': False, 'erro': 'pastas vazias'}), 400

    mode = (data.get('mode') or 'par').lower()
    workers = 1 if mode == 'seq' else cpu_count()

    result = avaliar_dataset_api(au_paths, sp_paths, workers=workers)
    result['modo']         = mode
    result['max_per_class'] = max_per_class
    result['au_dir']       = au_dir
    result['sp_dir']       = sp_dir
    return jsonify(result)


# ─────────────────────────────────────────────
# Benchmark CP — upload
# ─────────────────────────────────────────────
@app.route('/api/benchmark', methods=['POST'])
def benchmark_upload():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'ok': False, 'erro': 'envie images=<arquivo>*'}), 400

    tmpdir = tempfile.mkdtemp(prefix='df_bench_')
    try:
        paths = _save_uploaded(files, tmpdir)[:MAX_BATCH]
        if not paths:
            return jsonify({'ok': False, 'erro': 'nenhum arquivo valido'}), 400

        configs = sorted(set([1, 2, 4, cpu_count()]))
        result = benchmark_paralelo_api(paths, configs_workers=configs,
                                        incluir_fraca=True)
        return jsonify(result)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────
# Benchmark CP — pasta local
# ─────────────────────────────────────────────
@app.route('/api/benchmark-local', methods=['POST'])
def benchmark_local():
    data = request.get_json(silent=True) or {}
    pasta = data.get('dir') or _resolve_dataset_dir(os.path.join(ROOT, 'Dataset', 'Au'))
    n_max = int(data.get('max') or 30)
    n_max = min(n_max, MAX_LOCAL)

    if not os.path.isdir(pasta):
        return jsonify({'ok': False, 'erro': 'pasta nao encontrada', 'dir': pasta}), 400

    paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp'):
        paths.extend(glob(os.path.join(pasta, ext)))
    paths = sorted(paths)[:n_max]

    if not paths:
        return jsonify({'ok': False, 'erro': 'pasta sem imagens'}), 400

    configs = sorted(set([1, 2, 4, cpu_count()]))
    result = benchmark_paralelo_api(paths, configs_workers=configs,
                                    incluir_fraca=True)
    result['dir'] = pasta
    result['max'] = n_max
    return jsonify(result)


# ─────────────────────────────────────────────
# ML — status do modelo Random Forest
# ─────────────────────────────────────────────
@app.route('/api/ml/status')
def ml_status():
    return jsonify(detector_ml.status_modelo())


# ─────────────────────────────────────────────
# ML — treino usando o dataset local (gera/atualiza o .pkl)
#   Body JSON: {au_dir?, sp_dir?, max_per_class?, n_estimators?}
#   Padrão: Dataset/Au e Dataset/Tp
# ─────────────────────────────────────────────
@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    data = request.get_json(silent=True) or {}
    au_dir = data.get('au_dir')
    sp_dir = data.get('sp_dir')
    max_per_class = data.get('max_per_class')
    if max_per_class is not None:
        max_per_class = min(int(max_per_class), MAX_LOCAL)
    n_estimators = int(data.get('n_estimators') or 200)

    result = detector_ml.treinar(
        au_dir=au_dir, sp_dir=sp_dir,
        max_per_class=max_per_class, n_estimators=n_estimators)
    return jsonify(result), (200 if result.get('ok') else 400)


# ─────────────────────────────────────────────
# ML — predição de UMA imagem por upload (Random Forest)
#   multipart: image=<file>
# ─────────────────────────────────────────────
@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    if not detector_ml.modelo_existe():
        return jsonify({'ok': False, 'erro': 'modelo nao treinado — chame POST /api/ml/train'}), 400

    f = request.files.get('image')
    if not f or not f.filename:
        return jsonify({'ok': False, 'erro': 'envie image=<arquivo>'}), 400
    if not _ext_ok(f.filename):
        return jsonify({'ok': False, 'erro': 'formato nao suportado'}), 400

    tmpdir = tempfile.mkdtemp(prefix='df_ml_')
    try:
        dst = os.path.join(tmpdir, os.path.basename(f.filename))
        f.save(dst)

        t_ini = time.perf_counter()
        r = _analisar_leve(dst)               # roda VP+ELA+Ruído+Escala (4 scores)
        if 'erro' in r:
            return jsonify({'ok': False, 'erro': r['erro']}), 400
        pred = detector_ml.prever_de_resultado(r)
        if pred is None:
            return jsonify({'ok': False, 'erro': 'falha ao carregar modelo'}), 500

        return jsonify({
            'ok'         : True,
            'arquivo'    : os.path.basename(f.filename),
            'tempo'      : round(time.perf_counter() - t_ini, 3),
            'scores'     : {
                'vp'    : float(r['score_vp']),
                'ela'   : float(r['score_ela']),
                'ruido' : float(r['score_ruido']),
                'escala': float(r['score_escala']),
            },
            **pred,
        })
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────
# Servir exemplos / assets
# ─────────────────────────────────────────────
@app.route('/<path:filename>')
def static_files(filename):
    if filename in ('app.py', 'detector_unificado.py', 'detector_ml.py'):
        return ('forbidden', 403)
    full = os.path.join(ROOT, filename)
    if os.path.isfile(full):
        return send_from_directory(ROOT, filename)
    return ('not found', 404)


if __name__ == '__main__':
    freeze_support()  # necessário no Windows p/ multiprocessing
    port = int(os.environ.get('PORT', 5000))
    print(f"\n  DeepForensics API em http://127.0.0.1:{port}")
    print(f"  ROOT estaticos: {ROOT}")
    print(f"  CPUs disponiveis: {cpu_count()}\n")
    # Preferimos um servidor WSGI real (waitress): o servidor de
    # desenvolvimento do Werkzeug trunca respostas grandes no Windows
    # (as análises devolvem visuais base64 de vários MB). Fallback p/ app.run.
    try:
        from waitress import serve
        print("  Servidor: waitress (WSGI)\n")
        serve(app, host='127.0.0.1', port=port, threads=8)
    except ImportError:
        print("  Servidor: Werkzeug (dev) — instale 'waitress' p/ respostas grandes\n")
        app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, threaded=True)
