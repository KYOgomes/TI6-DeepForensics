# -*- coding: utf-8 -*-
"""
Detector ML — Random Forest sobre as 4 análises
================================================
Em vez de combinar VP + ELA + Ruído + Escala por uma soma de pesos fixos,
este módulo TREINA um Random Forest que aprende a fronteira de decisão a
partir dos 4 scores de cada análise.

Fluxo:
  1) Para cada imagem do dataset roda o pipeline (detector_unificado) e
     extrai o vetor de features = [score_vp, score_ela, score_ruido, score_escala].
  2) Treina um RandomForestClassifier com labels  Au=0 (autêntica)  Sp=1 (manipulada).
  3) Salva o modelo em disco (.pkl) junto com nomes das features e métricas.
  4) Predição: dada uma imagem nova (upload), extrai os 4 scores e o modelo
     devolve a classe (0/1) + probabilidade.

Treino offline (gera o .pkl a partir do Dataset embutido):
  python Algorithm/detector_ml.py --treinar
  python Algorithm/detector_ml.py --treinar --au Dataset/Au --sp Dataset/Tp --n 200

A predição de uma imagem nova é feita por UPLOAD via API (app.py), não pelo terminal.
"""

import os
import sys
import time
import datetime
from glob import glob

import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# Reaproveita TODO o pipeline já existente (rodando sem dados visuais → leve/paralelo)
from detector_unificado import _analisar_leve, calcular_status

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────
# Apenas os 4 scores das análises são usados como features.
FEATURES   = ['score_vp', 'score_ela', 'score_ruido', 'score_escala']
MODEL_PATH = os.path.join(HERE, 'modelo_rf.pkl')

EXTS = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp')

# cache do modelo carregado (evita reler o .pkl a cada predição)
_MODELO_CACHE = None
_MODELO_MTIME = None


# ══════════════════════════════════════════════
# EXTRAÇÃO DE FEATURES
# ══════════════════════════════════════════════

def vetor_de_resultado(r):
    """Recebe o dict de _analisar_leve / analisar_para_api e devolve o vetor [4]."""
    return [float(r['score_vp']), float(r['score_ela']),
            float(r['score_ruido']), float(r['score_escala'])]


def _features_de_path(path):
    """Top-level (para multiprocessing): roda o pipeline e devolve (vetor, ok)."""
    r = _analisar_leve(path)
    if 'erro' in r:
        return None
    return vetor_de_resultado(r)


def _listar_imagens(pasta, limite=None):
    out = []
    for ext in EXTS:
        out.extend(glob(os.path.join(pasta, ext)))
    out = sorted(out)
    return out[:limite] if limite else out


def _resolver_dir(base):
    """Se a pasta estiver vazia mas houver <base>/sample/, usa a sample."""
    if not os.path.isdir(base):
        return base
    tem_img = any(f.lower().endswith(tuple(e.strip('*') for e in EXTS))
                  for f in os.listdir(base)
                  if os.path.isfile(os.path.join(base, f)))
    if tem_img:
        return base
    sample = os.path.join(base, 'sample')
    return sample if os.path.isdir(sample) else base


def extrair_matriz(au_paths, sp_paths, workers=None):
    """
    Roda o pipeline em todas as imagens e monta (X, y).
      X = matriz [n_amostras, 4]   y = [0]*n_au + [1]*n_sp
    Descarta imagens que falharem ao abrir.
    """
    from multiprocessing import Pool, cpu_count
    if workers is None:
        workers = cpu_count()

    todas  = list(au_paths) + list(sp_paths)
    labels = [0] * len(au_paths) + [1] * len(sp_paths)
    if not todas:
        raise ValueError('nenhuma imagem para extrair features')

    if workers > 1 and len(todas) > 1:
        with Pool(workers) as pool:
            vetores = pool.map(_features_de_path, todas)
    else:
        vetores = [_features_de_path(p) for p in todas]

    X, y = [], []
    for vet, lab in zip(vetores, labels):
        if vet is not None:
            X.append(vet)
            y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ══════════════════════════════════════════════
# TREINO
# ══════════════════════════════════════════════

def treinar(au_dir=None, sp_dir=None, max_per_class=None,
            n_estimators=200, test_size=0.25, workers=None,
            salvar_em=MODEL_PATH):
    """
    Treina o Random Forest e salva em disco. Devolve dict de métricas (JSON-friendly).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix)

    au_dir = _resolver_dir(au_dir or os.path.join(ROOT, 'Dataset', 'Au'))
    sp_dir = _resolver_dir(sp_dir or os.path.join(ROOT, 'Dataset', 'Tp'))

    au_paths = _listar_imagens(au_dir, max_per_class)
    sp_paths = _listar_imagens(sp_dir, max_per_class)
    if not au_paths or not sp_paths:
        return {'ok': False, 'erro': 'precisa de imagens em AMBAS as classes',
                'au_dir': au_dir, 'sp_dir': sp_dir,
                'n_au': len(au_paths), 'n_sp': len(sp_paths)}

    t_ini = time.perf_counter()
    X, y = extrair_matriz(au_paths, sp_paths, workers=workers)
    t_feat = time.perf_counter() - t_ini

    if len(np.unique(y)) < 2:
        return {'ok': False, 'erro': 'features de apenas uma classe sobraram'}

    # split estratificado (mantém proporção Au/Sp em treino e teste)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        class_weight='balanced',   # robustez se Au/Sp ficarem desbalanceados
        random_state=42,
        n_jobs=-1)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

    # validação cruzada no conjunto completo (mais estável em dataset pequeno)
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits >= 2:
        cv_acc = cross_val_score(clf, X, y, cv=n_splits, scoring='accuracy')
        cv_f1  = cross_val_score(clf, X, y, cv=n_splits, scoring='f1')
        cv = {'folds': int(n_splits),
              'acuracia_media': float(cv_acc.mean()),
              'acuracia_std':   float(cv_acc.std()),
              'f1_media':       float(cv_f1.mean()),
              'f1_std':         float(cv_f1.std())}
    else:
        cv = None

    bundle = {
        'modelo':   clf,
        'features': FEATURES,
        'meta': {
            'criado_em':    datetime.datetime.now().isoformat(timespec='seconds'),
            'n_au':         int((y == 0).sum()),
            'n_sp':         int((y == 1).sum()),
            'n_estimators': int(n_estimators),
            'test_size':    float(test_size),
            'tempo_features': round(t_feat, 2),
            'classes':      {'0': 'autentica (Au)', '1': 'manipulada (Sp)'},
        },
    }
    joblib.dump(bundle, salvar_em)

    return {
        'ok'        : True,
        'modelo'    : salvar_em,
        'features'  : FEATURES,
        'n_au'      : int((y == 0).sum()),
        'n_sp'      : int((y == 1).sum()),
        'n_treino'  : int(len(y_tr)),
        'n_teste'   : int(len(y_te)),
        'tempo_features': round(t_feat, 2),
        'metricas_teste': {
            'acuracia': float(accuracy_score(y_te, y_pred)),
            'precisao': float(precision_score(y_te, y_pred, zero_division=0)),
            'recall'  : float(recall_score(y_te, y_pred, zero_division=0)),
            'f1'      : float(f1_score(y_te, y_pred, zero_division=0)),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        },
        'cross_val' : cv,
        'importancias': {f: float(imp) for f, imp
                         in zip(FEATURES, clf.feature_importances_)},
    }


# ══════════════════════════════════════════════
# CARREGAR MODELO + PREDIÇÃO
# ══════════════════════════════════════════════

def modelo_existe(path=MODEL_PATH):
    return os.path.isfile(path)


def carregar_modelo(path=MODEL_PATH):
    """Carrega (com cache) o bundle salvo. Recarrega se o arquivo mudou."""
    global _MODELO_CACHE, _MODELO_MTIME
    if not os.path.isfile(path):
        return None
    mtime = os.path.getmtime(path)
    if _MODELO_CACHE is None or _MODELO_MTIME != mtime:
        _MODELO_CACHE = joblib.load(path)
        _MODELO_MTIME = mtime
    return _MODELO_CACHE


def prever_de_resultado(r, path=MODEL_PATH):
    """
    Recebe um dict com os 4 scores (saída de _analisar_leve/_analisar_core) e
    devolve a predição do RF. Retorna None se não houver modelo treinado.
    """
    bundle = carregar_modelo(path)
    if bundle is None:
        return None
    clf  = bundle['modelo']
    vet  = np.array([vetor_de_resultado(r)], dtype=np.float32)
    classe = int(clf.predict(vet)[0])

    # índice da classe 1 (manipulada) em predict_proba
    classes = list(clf.classes_)
    prob_manip = float(clf.predict_proba(vet)[0][classes.index(1)]) \
        if 1 in classes else float(classe)

    # score 0–100 no MESMO sentido do pipeline antigo: 100 = consistente/autêntica
    score = round((1.0 - prob_manip) * 100.0, 2)
    return {
        'classe'        : classe,                 # 0 = autêntica, 1 = manipulada
        'rotulo'        : 'manipulada' if classe == 1 else 'autentica',
        'prob_manipulada': round(prob_manip, 4),
        'prob_autentica': round(1.0 - prob_manip, 4),
        'score'         : score,
        'status'        : calcular_status(score),
        'fonte'         : 'random_forest',
    }


def status_modelo(path=MODEL_PATH):
    """Metadados do modelo (para a API)."""
    if not modelo_existe(path):
        return {'ok': True, 'treinado': False, 'modelo': path}
    bundle = carregar_modelo(path)
    return {'ok': True, 'treinado': True, 'modelo': path,
            'features': bundle.get('features', FEATURES),
            'meta': bundle.get('meta', {})}


# ══════════════════════════════════════════════
# CLI — treino offline (gera o .pkl)
# ══════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    import io
    from multiprocessing import freeze_support
    freeze_support()

    # console do Windows pode não suportar emoji/acentos → força UTF-8
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

    ap = argparse.ArgumentParser(description='Treino do Random Forest (4 scores → Au/Sp)')
    ap.add_argument('--treinar', action='store_true', help='Treina e salva o modelo')
    ap.add_argument('--au', default=None, help='Pasta de imagens autênticas (Au)')
    ap.add_argument('--sp', default=None, help='Pasta de imagens manipuladas (Sp/Tp)')
    ap.add_argument('--max', type=int, default=None, help='Máx. imagens por classe')
    ap.add_argument('--n',   type=int, default=200, help='n_estimators do RF')
    ap.add_argument('--out', default=MODEL_PATH, help='Caminho do .pkl de saída')
    args = ap.parse_args()

    if args.treinar:
        print('\n🌲 Treinando Random Forest (features = 4 scores)...\n')
        res = treinar(au_dir=args.au, sp_dir=args.sp, max_per_class=args.max,
                      n_estimators=args.n, salvar_em=args.out)
        if not res.get('ok'):
            print('❌', res.get('erro'), '|', {k: v for k, v in res.items() if k != 'ok'})
            sys.exit(1)
        m = res['metricas_teste']
        print(f"  Au={res['n_au']}  Sp={res['n_sp']}  "
              f"(treino={res['n_treino']}, teste={res['n_teste']})")
        print(f"  features extraídas em {res['tempo_features']}s\n")
        print(f"  {'='*50}")
        print(f"  Acurácia : {m['acuracia']*100:5.2f}%")
        print(f"  Precisão : {m['precisao']*100:5.2f}%")
        print(f"  Recall   : {m['recall']*100:5.2f}%")
        print(f"  F1-Score : {m['f1']*100:5.2f}%")
        print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
        if res.get('cross_val'):
            cv = res['cross_val']
            print(f"  CV({cv['folds']} folds): acc={cv['acuracia_media']*100:.2f}%"
                  f" ±{cv['acuracia_std']*100:.2f}  f1={cv['f1_media']*100:.2f}%")
        print(f"  {'='*50}")
        print('  Importância das features:')
        for f, imp in sorted(res['importancias'].items(),
                             key=lambda kv: kv[1], reverse=True):
            print(f'    {f:<14}: {imp*100:5.1f}%')
        print(f"\n✅ Modelo salvo em: {res['modelo']}\n")
    else:
        print(__doc__)
        ap.print_help()
