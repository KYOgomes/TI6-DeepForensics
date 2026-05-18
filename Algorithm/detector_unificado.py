# -*- coding: utf-8 -*-
"""
Detecção de Adulteração de Imagens — Script UNIFICADO v5
=========================================================
Combina quatro abordagens em um único pipeline adaptativo:

  • Vanishing Points (VP)  — consistência geométrica de perspectiva
  • ELA (Error Level Analysis) — artefatos de compressão/reprocessamento
  • Análise de Ruído        — inconsistências de sensor entre regiões
  • Análise de Escala/Blob  — proporções e tamanhos relativos de objetos

Classificação em 5 níveis (score 0–100):
   0 – 20  → MANIPULADA              (evidência forte de adulteração)
  21 – 40  → ALTA CHANCE DE MANIPULAÇÃO  (sinais relevantes detectados)
  41 – 60  → INCONCLUSIVA            (análise não determinante)
  61 – 80  → CONSISTÊNCIA MÉDIA      (poucos indícios, mas não isento)
  81 – 100 → CONSISTENTE             (sem sinais de adulteração)

Modos de uso:
  1) Imagem única:
       python detector_unificado.py Fotos/img2.jpg

  2) Dataset rotulado (Au + Sp) com F1/acurácia + métricas de CP:
       python detector_unificado.py --dataset --au Dataset/Au --sp Dataset/Tp
       python detector_unificado.py --dataset --au Au --sp Sp

  3) Pasta sem rótulo (análise em lote) + métricas de CP:
       python detector_unificado.py --pasta dataset

Computação Paralela (modos 2 e 3):
  - Executa PRIMEIRO sequencialmente (T1 = baseline)
  - Depois com Pool de 1, 2, 4 e cpu_count() workers (TN)
  - Calcula Speedup = T1/TN, Eficiência = Speedup/N, Overhead = TN*N - T1
  - Gera dashboard de 4 painéis com curvas de escalabilidade forte e fraca

O pipeline ajusta pesos automaticamente conforme a imagem:
  - Poucas linhas (retratos)           → peso maior para ELA + Ruído
  - Muitas linhas (arquitetura / cena) → peso maior para VP
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image
import io
import sys
import os
import time
import argparse
from glob import glob
from multiprocessing import Pool, cpu_count

# ─────────────────────────────────────────────
# ⚙️  CONFIGURAÇÃO GLOBAL
# ─────────────────────────────────────────────
OUTPUT_UNICA  = 'Resultado_Unico.jpg'
OUTPUT_AVALIA = 'Resultado_Dataset.jpg'
OUTPUT_LOTE   = 'Resultado_Lote.jpg'
OUTPUT_CP     = 'Metricas_cp.jpg'

# Limiares dos 5 níveis de classificação
L1 = 20   # ≤ L1              → MANIPULADA
L2 = 40   # L1 < score ≤ L2  → ALTA CHANCE DE MANIPULAÇÃO
L3 = 60  # L2 < score ≤ L3  → INCONCLUSIVA
L4 = 80   # L3 < score ≤ L4  → CONSISTÊNCIA MÉDIA
          # score > L4        → CONSISTENTE

# Pesos BASE (ajustados dinamicamente conforme a imagem)
PESO_VP_BASE     = 0.25
PESO_ELA_BASE    = 0.35
PESO_RUIDO_BASE  = 0.25
PESO_ESCALA_BASE = 0.15

ELA_QUALITY = 75
# ──────────────────────────────────────────────


# ══════════════════════════════════════════════
# BLOCO 1 — VANISHING POINTS
# ══════════════════════════════════════════════

def detect_lines(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines   = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                               minLineLength=30, maxLineGap=10)
    return edges, lines


def line_intersection(l1, l2):
    x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    return (x1+t*(x2-x1), y1+t*(y2-y1))


def estimate_vanishing_point(lines, n_iter=1000):
    if lines is None or len(lines) < 2:
        return None
    flat = [l[0] for l in lines]; pts = []
    np.random.seed(42)
    for _ in range(min(n_iter, len(flat)*(len(flat)-1)//2)):
        i,j = np.random.choice(len(flat), 2, replace=False)
        p = line_intersection(flat[i], flat[j])
        if p and abs(p[0]) < 15000 and abs(p[1]) < 15000:
            pts.append(p)
    return np.median(np.array(pts), axis=0) if len(pts) >= 3 else None


def angle_divergence(line, vp):
    x1,y1,x2,y2 = line
    al = np.arctan2(y2-y1, x2-x1)
    av = np.arctan2(vp[1]-(y1+y2)/2, vp[0]-(x1+x2)/2)
    return np.degrees(min(abs(al-av), np.pi-abs(al-av)))


def analisar_vp(img):
    _, lines = detect_lines(img)
    n = len(lines) if lines is not None else 0
    if lines is None or n < 5:
        return 50, None, [], n
    vp = estimate_vanishing_point(lines)
    if vp is None:
        return 50, None, [], n
    divs  = [angle_divergence(l[0], vp) for l in lines]
    score = round(max(0.0, min(100.0, 100.0 - np.mean(divs) * 2.2)), 2)
    return score, vp, divs, n


def montar_visual_vp(img_rgb, lines_raw, vp, divs):
    img_lines = img_rgb.copy()
    heatmap   = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    if lines_raw is not None and vp is not None:
        for line in lines_raw:
            x1,y1,x2,y2 = line[0]
            t = min(1.0, angle_divergence(line[0], vp) / 45.0)
            cv2.line(img_lines,(x1,y1),(x2,y2),(int(255*t),int(255*(1-t)),0),1)
            cv2.line(heatmap,  (x1,y1),(x2,y2), t, 3)
        vx,vy = int(vp[0]),int(vp[1])
        h,w = img_rgb.shape[:2]
        if 0<=vx<w and 0<=vy<h:
            cv2.circle(img_lines,(vx,vy),12,(255,200,0),-1)
            cv2.putText(img_lines,'VP',(vx+14,vy),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(255,200,0),2)
    return img_lines, cv2.GaussianBlur(heatmap,(21,21),0)


# ══════════════════════════════════════════════
# BLOCO 2 — ELA (Error Level Analysis)
# ══════════════════════════════════════════════

def analisar_ela(image_path, quality=ELA_QUALITY):
    original_pil = Image.open(image_path).convert('RGB')

    buf1 = io.BytesIO()
    original_pil.save(buf1, format='JPEG', quality=95)
    buf1.seek(0)
    baseline = Image.open(buf1).convert('RGB')

    buf2 = io.BytesIO()
    baseline.save(buf2, format='JPEG', quality=quality)
    buf2.seek(0)
    recomprimida = Image.open(buf2).convert('RGB')

    base_arr   = np.array(baseline,      dtype=np.float32)
    recomp_arr = np.array(recomprimida,  dtype=np.float32)

    ela_arr  = np.abs(base_arr - recomp_arr)
    ela_gray = ela_arr.mean(axis=2)
    ela_vis  = np.clip(ela_gray * 10, 0, 255).astype(np.uint8)

    h, w   = ela_gray.shape
    block  = 32
    mean_map    = np.zeros_like(ela_gray)
    std_map     = np.zeros_like(ela_gray)
    block_means = []

    for y in range(0, h, block):
        for x in range(0, w, block):
            blk = ela_gray[y:y+block, x:x+block]
            m, s = blk.mean(), blk.std()
            mean_map[y:y+block, x:x+block] = m
            std_map [y:y+block, x:x+block] = s
            block_means.append(m)

    media_global = np.mean(block_means)
    std_global   = np.std(block_means)

    penalidade_media = min(50, (media_global / 12.0) * 50)
    penalidade_std   = min(50, (std_global   /  6.0) * 50)
    score_ela = round(max(0.0, 100.0 - penalidade_media - penalidade_std), 2)

    return ela_vis, score_ela, std_map, media_global, std_global


# ══════════════════════════════════════════════
# BLOCO 3 — ANÁLISE DE RUÍDO
# ══════════════════════════════════════════════

def analisar_ruido(img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    suavizada= cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    noise    = np.abs(gray - suavizada)

    h, w   = noise.shape
    block  = 32
    var_map    = np.zeros_like(noise)
    block_vars = []

    for y in range(0, h, block):
        for x in range(0, w, block):
            blk = noise[y:y+block, x:x+block]
            v   = blk.var()
            var_map[y:y+block, x:x+block] = v
            block_vars.append(v)

    media_var = np.mean(block_vars)
    std_var   = np.std(block_vars)
    cv        = std_var / (media_var + 1e-6)

    score_ruido = round(max(0.0, min(100.0, 100.0 - (cv / 1.2) * 100)), 2)

    noise_vis = np.clip(noise * 5, 0, 255).astype(np.uint8)
    return noise_vis, score_ruido, var_map, cv


# ══════════════════════════════════════════════
# BLOCO 4 — ANÁLISE DE ESCALA / BLOB
# ══════════════════════════════════════════════

def analisar_escala(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    min_area = h_img * w_img * 0.003
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        blobs.append({'contour':cnt,'area':area,
                      'bbox':(x,y,w,h),'cy':y+h/2,'altura':h})
    blobs = sorted(blobs, key=lambda b: b['area'], reverse=True)[:10]

    score_escala = 100.0
    anomalias    = []
    vis = img.copy()

    if len(blobs) >= 2:
        for b in blobs:
            b['dist_rel'] = 1.0 - (b['cy'] / h_img)
        pen_total = 0.0; n_pares = 0
        for i in range(len(blobs)):
            for j in range(i+1, len(blobs)):
                bi,bj = blobs[i],blobs[j]
                if bj['altura'] < 1 or bi['altura'] < 1:
                    continue
                razao_real     = bi['altura'] / bj['altura']
                d_i = max(0.01, 1-bi['dist_rel'])
                d_j = max(0.01, 1-bj['dist_rel'])
                razao_esperada = d_j / d_i
                if razao_esperada > 0 and razao_real > 0:
                    div = abs(np.log(razao_real) - np.log(razao_esperada))
                    pen_total += div
                    n_pares   += 1
                    if div > 1.0:
                        anomalias.append((i,j,div))
        if n_pares > 0:
            score_escala = max(0.0, 100.0 - (pen_total/n_pares/1.5)*100)

    detalhes = []
    for idx, b in enumerate(blobs):
        x,y,w,h = b['bbox']
        anomalo = any(idx in (a[0],a[1]) for a in anomalias)
        cor     = (0,0,255) if anomalo else (0,255,0)
        cv2.rectangle(vis,(x,y),(x+w,y+h),cor,2)
        cv2.putText(vis,f"Obj{idx+1}",(x,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,cor,2)
        detalhes.append(f"  Obj{idx+1}: h={h}px  "
                        f"{'SUSPEITO' if anomalo else 'ok'}")

    vis_rgb      = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    n_anomalias_graves = sum(1 for a in anomalias if a[2] > 2.0)
    return round(score_escala,2), vis_rgb, blobs, anomalias, detalhes, n_anomalias_graves


# ══════════════════════════════════════════════
# BLOCO 5 — PESOS ADAPTATIVOS
# ══════════════════════════════════════════════

def calcular_pesos(n_linhas, n_blobs, n_anomalias_graves, score_ruido=100):
    w_vp     = PESO_VP_BASE
    w_ela    = PESO_ELA_BASE
    w_ruido  = PESO_RUIDO_BASE
    w_escala = PESO_ESCALA_BASE

    if n_linhas < 20:
        delta = 0.15
        w_vp    -= delta
        w_ela   += delta * 0.6
        w_ruido += delta * 0.4
    elif n_linhas > 60:
        delta = 0.10
        w_vp    += delta
        w_ela   -= delta * 0.6
        w_ruido -= delta * 0.4

    w_vp    = max(0.05, w_vp)
    w_ela   = max(0.05, w_ela)
    w_ruido = max(0.05, w_ruido)
    w_escala= max(0.05, w_escala)

    total = w_vp + w_ela + w_ruido + w_escala
    return (round(w_vp/total, 3), round(w_ela/total, 3),
            round(w_ruido/total, 3), round(w_escala/total, 3))


# ══════════════════════════════════════════════
# BLOCO 6 — CORE: analisa UMA imagem
# ══════════════════════════════════════════════

def calcular_status(score):
    if score <= L1: return 'MANIPULADA'
    if score <= L2: return 'ALTA CHANCE DE MANIPULACAO'
    if score <= L3: return 'INCONCLUSIVA'
    if score <= L4: return 'CONSISTENCIA MEDIA'
    return 'CONSISTENTE'

def label_status(score):
    if score <= L1: return 'MANIPULADA                  🚨'
    if score <= L2: return 'ALTA CHANCE DE MANIPULACAO  ⛔'
    if score <= L3: return 'INCONCLUSIVA                ⚠️'
    if score <= L4: return 'CONSISTENCIA MEDIA          🔎'
    return 'CONSISTENTE                 ✅'


def _analisar_core(image_path):
    """Pipeline completo: VP + ELA + Ruído + Escala com pesos adaptativos."""
    img = cv2.imread(image_path)
    if img is None:
        return {'path':image_path,'erro':'não encontrada','score':50,
                'score_vp':50,'score_ela':50,'score_ruido':50,'score_escala':50,
                'status':label_status(50),'n_linhas':0,'n_blobs':0,'n_anomalias':0,
                'peso_vp':0.25,'peso_ela':0.35,'peso_ruido':0.25,'peso_escala':0.15}

    score_vp, vp, divs, n_linhas = analisar_vp(img)
    _, lines_raw = detect_lines(img)

    ela_vis, score_ela, std_map, ela_mean, ela_std = analisar_ela(image_path)

    noise_vis, score_ruido, var_map, cv_ruido = analisar_ruido(img)

    score_escala, vis_escala, blobs, anomalias, detalhes, n_anom_graves = \
        analisar_escala(img)

    w_vp, w_ela, w_ruido, w_escala = calcular_pesos(
        n_linhas, len(blobs), n_anom_graves, score_ruido)

    score_final = round(
        w_vp    * score_vp    +
        w_ela   * score_ela   +
        w_ruido * score_ruido +
        w_escala* score_escala, 2)

    return {
        'path':          image_path,
        'score':         score_final,
        'score_vp':      score_vp,
        'score_ela':     score_ela,
        'score_ruido':   score_ruido,
        'score_escala':  score_escala,
        'status':        label_status(score_final),
        'n_linhas':      n_linhas,
        'n_blobs':       len(blobs),
        'n_anomalias':   len(anomalias),
        'n_anom_graves': n_anom_graves,
        'ela_mean':      round(float(ela_mean),2),
        'ela_std':       round(float(ela_std),2),
        'cv_ruido':      round(float(cv_ruido),3),
        'peso_vp':       w_vp,
        'peso_ela':      w_ela,
        'peso_ruido':    w_ruido,
        'peso_escala':   w_escala,
        'detalhes':      detalhes,
        '_img':          img,
        '_vp':           vp,
        '_divs':         divs,
        '_lines_raw':    lines_raw,
        '_ela_vis':      ela_vis,
        '_std_map':      std_map,
        '_noise_vis':    noise_vis,
        '_var_map':      var_map,
        '_vis_escala':   vis_escala,
    }


def _analisar_leve(path):
    """Versão sem dados visuais — usada no Pool (multiprocessing)."""
    r = _analisar_core(path)
    for k in ['_img','_vp','_divs','_lines_raw','_ela_vis',
              '_std_map','_noise_vis','_var_map','_vis_escala']:
        r.pop(k, None)
    return r


# ══════════════════════════════════════════════
# HELPERS API (JSON-friendly, sem matplotlib)
# ══════════════════════════════════════════════

def _np_to_b64(arr, cmap=None):
    """Converte array numpy (grayscale ou BGR) em PNG base64 (data URL)."""
    import base64
    if arr is None:
        return None
    a = arr
    if a.dtype != np.uint8:
        amin, amax = float(a.min()), float(a.max())
        if amax - amin < 1e-6:
            a = np.zeros_like(a, dtype=np.uint8)
        else:
            a = ((a - amin) / (amax - amin) * 255).astype(np.uint8)
    if cmap == 'hot':
        a = cv2.applyColorMap(a, cv2.COLORMAP_HOT)
    elif cmap == 'inferno':
        a = cv2.applyColorMap(a, cv2.COLORMAP_INFERNO)
    elif cmap == 'gray' and a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode('.png', a)
    if not ok:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def analisar_para_api(image_path, include_visuals=True):
    """
    Roda o pipeline completo e devolve dict JSON-serializável,
    com (opcionalmente) imagens em base64 prontas para enviar ao front.
    """
    r = _analisar_core(image_path)
    if 'erro' in r:
        return {'ok': False, 'erro': r['erro']}

    img       = r.pop('_img')
    vp        = r.pop('_vp')
    divs      = r.pop('_divs')
    lines_raw = r.pop('_lines_raw')
    ela_vis   = r.pop('_ela_vis')
    std_map   = r.pop('_std_map')
    noise_vis = r.pop('_noise_vis')
    var_map   = r.pop('_var_map')
    vis_esc   = r.pop('_vis_escala')

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lines_rgb, heatmap_vp = montar_visual_vp(img_rgb, lines_raw, vp, divs)
    img_lines_bgr = cv2.cvtColor(img_lines_rgb, cv2.COLOR_RGB2BGR)
    vis_esc_bgr   = cv2.cvtColor(vis_esc, cv2.COLOR_RGB2BGR)

    payload = {
        'ok'           : True,
        'arquivo'      : os.path.basename(image_path),
        'largura'      : int(w),
        'altura'       : int(h),
        'score'        : float(r['score']),
        'status'       : calcular_status(r['score']),
        'status_label' : r['status'],
        'componentes'  : {
            'vp'    : {'score': float(r['score_vp']),    'peso': float(r['peso_vp'])},
            'ela'   : {'score': float(r['score_ela']),   'peso': float(r['peso_ela'])},
            'ruido' : {'score': float(r['score_ruido']), 'peso': float(r['peso_ruido'])},
            'escala': {'score': float(r['score_escala']),'peso': float(r['peso_escala'])},
        },
        'metricas'     : {
            'n_linhas'     : int(r['n_linhas']),
            'n_blobs'      : int(r['n_blobs']),
            'n_anomalias'  : int(r['n_anomalias']),
            'n_anom_graves': int(r['n_anom_graves']),
            'ela_mean'     : float(r['ela_mean']),
            'ela_std'      : float(r['ela_std']),
            'cv_ruido'     : float(r['cv_ruido']),
            'vp_coords'    : (None if vp is None else [float(vp[0]), float(vp[1])]),
            'desvio_medio' : (round(float(np.mean(divs)), 2) if divs else None),
        },
        'limiares'     : {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4},
        'detalhes'     : r.get('detalhes', []),
    }

    if include_visuals:
        payload['visuais'] = {
            'original'  : _np_to_b64(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)),
            'vp_linhas' : _np_to_b64(img_lines_bgr),
            'vp_heatmap': _np_to_b64((heatmap_vp * 255).clip(0,255).astype(np.uint8), cmap='hot'),
            'ela'       : _np_to_b64(ela_vis, cmap='hot'),
            'ela_std'   : _np_to_b64(std_map, cmap='inferno'),
            'ruido'     : _np_to_b64(noise_vis, cmap='gray'),
            'ruido_var' : _np_to_b64(var_map, cmap='hot'),
            'escala'    : _np_to_b64(vis_esc_bgr),
        }
    return payload


def benchmark_paralelo_api(imagens, configs_workers=None, incluir_fraca=True):
    """Wrapper de benchmark que devolve dict JSON-serializável."""
    if not imagens:
        return {'ok': False, 'erro': 'nenhuma imagem'}
    if configs_workers is None:
        configs_workers = sorted(set([1, 2, 4, cpu_count()]))

    metricas_forte, T1 = _benchmark_paralelo(imagens, configs_workers)
    metricas_fraca = (_benchmark_escalabilidade_fraca(imagens, configs_workers)
                      if incluir_fraca else [])

    return {
        'ok'            : True,
        'n_imagens'     : len(imagens),
        'cpu_count'     : cpu_count(),
        'configs'       : configs_workers,
        'T1'            : float(T1),
        'forte'         : [{k: (float(v) if isinstance(v,(int,float)) else v)
                            for k, v in m.items()} for m in metricas_forte],
        'fraca'         : [{k: (float(v) if isinstance(v,(int,float)) else v)
                            for k, v in m.items()} for m in metricas_fraca],
    }


def avaliar_dataset_api(au_paths, sp_paths, workers=None):
    """
    Avalia um dataset rotulado (Au=0, Sp=1) e devolve métricas + scores.
    Usa Pool com `workers` (default = cpu_count()).
    """
    if workers is None:
        workers = cpu_count()
    todas  = list(au_paths) + list(sp_paths)
    labels = [0]*len(au_paths) + [1]*len(sp_paths)

    if not todas:
        return {'ok': False, 'erro': 'nenhuma imagem'}

    t_ini = time.perf_counter()
    if workers > 1 and len(todas) > 1:
        with Pool(workers) as pool:
            resultados = pool.map(_analisar_leve, todas)
    else:
        resultados = [_analisar_leve(p) for p in todas]
    elapsed = time.perf_counter() - t_ini

    preds  = [1 if r['score'] <= L2 else 0 for r in resultados]
    scores = [float(r['score']) for r in resultados]

    tp = sum(1 for r,p in zip(labels,preds) if r==1 and p==1)
    tn = sum(1 for r,p in zip(labels,preds) if r==0 and p==0)
    fp = sum(1 for r,p in zip(labels,preds) if r==0 and p==1)
    fn = sum(1 for r,p in zip(labels,preds) if r==1 and p==0)

    total    = tp+tn+fp+fn
    acuracia = (tp+tn)/total if total else 0
    precisao = tp/(tp+fp)    if (tp+fp) else 0
    recall   = tp/(tp+fn)    if (tp+fn) else 0
    f1       = 2*precisao*recall/(precisao+recall) if (precisao+recall) else 0

    niveis = {'MANIPULADA':0,'ALTA CHANCE DE MANIPULACAO':0,
              'INCONCLUSIVA':0,'CONSISTENCIA MEDIA':0,'CONSISTENTE':0}
    for r in resultados:
        k = calcular_status(r['score'])
        if k in niveis: niveis[k] += 1

    return {
        'ok'        : True,
        'n_au'      : len(au_paths),
        'n_sp'      : len(sp_paths),
        'workers'   : workers,
        'tempo'     : float(elapsed),
        'metricas'  : {
            'acuracia' : float(acuracia),
            'precisao' : float(precisao),
            'recall'   : float(recall),
            'f1'       : float(f1),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        },
        'limiar_pos': L2,
        'distribuicao': niveis,
        'au_scores' : scores[:len(au_paths)],
        'sp_scores' : scores[len(au_paths):],
        'resultados': [{
            'arquivo': os.path.basename(r['path']),
            'score'  : float(r['score']),
            'status' : calcular_status(r['score']),
            'score_vp'    : float(r['score_vp']),
            'score_ela'   : float(r['score_ela']),
            'score_ruido' : float(r['score_ruido']),
            'score_escala': float(r['score_escala']),
        } for r in resultados],
    }


def analisar_lote_api(paths, workers=None):
    """Versão sem rótulos — distribuição + ranking de suspeitas."""
    if workers is None:
        workers = cpu_count()
    if not paths:
        return {'ok': False, 'erro': 'nenhuma imagem'}

    t_ini = time.perf_counter()
    if workers > 1 and len(paths) > 1:
        with Pool(workers) as pool:
            resultados = pool.map(_analisar_leve, paths)
    else:
        resultados = [_analisar_leve(p) for p in paths]
    elapsed = time.perf_counter() - t_ini

    niveis = {'MANIPULADA':0,'ALTA CHANCE DE MANIPULACAO':0,
              'INCONCLUSIVA':0,'CONSISTENCIA MEDIA':0,'CONSISTENTE':0}
    for r in resultados:
        k = calcular_status(r['score'])
        if k in niveis: niveis[k] += 1

    return {
        'ok'        : True,
        'n_imagens' : len(paths),
        'workers'   : workers,
        'tempo'     : float(elapsed),
        'distribuicao': niveis,
        'resultados': [{
            'arquivo'     : os.path.basename(r['path']),
            'score'       : float(r['score']),
            'status'      : calcular_status(r['score']),
            'score_vp'    : float(r['score_vp']),
            'score_ela'   : float(r['score_ela']),
            'score_ruido' : float(r['score_ruido']),
            'score_escala': float(r['score_escala']),
        } for r in resultados],
    }


# ══════════════════════════════════════════════
# BLOCO 7 — MODO 1: IMAGEM ÚNICA (visual completo)
# ══════════════════════════════════════════════

def modo_imagem_unica(image_path, out=OUTPUT_UNICA):
    print(f"\n🔍 Analisando: {image_path}\n")

    r = _analisar_core(image_path)
    if 'erro' in r:
        print(f"❌ {r['erro']}"); return

    img       = r.pop('_img')
    vp        = r.pop('_vp')
    divs      = r.pop('_divs')
    lines_raw = r.pop('_lines_raw')
    ela_vis   = r.pop('_ela_vis')
    std_map   = r.pop('_std_map')
    noise_vis = r.pop('_noise_vis')
    var_map   = r.pop('_var_map')
    vis_esc   = r.pop('_vis_escala')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lines, heatmap_vp = montar_visual_vp(img_rgb, lines_raw, vp, divs)

    print("="*65)
    print(f"  SCORE FINAL : {r['score']} / 100")
    print(f"  STATUS      : {r['status']}")
    print("="*65)
    print(f"  ├─ VP     ({r['peso_vp']*100:.0f}%): {r['score_vp']}/100"
          f"  ({r['n_linhas']} linhas)")
    print(f"  ├─ ELA    ({r['peso_ela']*100:.0f}%): {r['score_ela']}/100"
          f"  (médio={r['ela_mean']:.2f}, desvio={r['ela_std']:.2f})")
    print(f"  ├─ Ruído  ({r['peso_ruido']*100:.0f}%): {r['score_ruido']}/100"
          f"  (CV={r['cv_ruido']:.3f})")
    print(f"  └─ Escala ({r['peso_escala']*100:.0f}%): {r['score_escala']}/100"
          f"  ({r['n_blobs']} blobs, {r['n_anomalias']} anomalia(s)"
          f", {r['n_anom_graves']} grave(s))")
    print("="*65)
    print(f"   0 – 20  → MANIPULADA")
    print(f"  21 – 40  → ALTA CHANCE DE MANIPULACAO")
    print(f"  41 – 60  → INCONCLUSIVA")
    print(f"  61 – 80  → CONSISTENCIA MEDIA")
    print(f"  81 – 100 → CONSISTENTE")
    print("="*65)
    if r.get('detalhes'):
        print("\n  Objetos detectados (análise de escala):")
        for d in r['detalhes']:
            print(d)
    print()

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#1a1a2e')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.08)

    titulo = (
        f"Score Final: {r['score']}/100  |  {r['status']}\n"
        f"VP:{r['score_vp']}({r['peso_vp']*100:.0f}%)  "
        f"ELA:{r['score_ela']}({r['peso_ela']*100:.0f}%)  "
        f"Ruido:{r['score_ruido']}({r['peso_ruido']*100:.0f}%)  "
        f"Escala:{r['score_escala']}({r['peso_escala']*100:.0f}%)"
    )
    fig.suptitle(titulo, fontsize=12, fontweight='bold', color='white', y=0.995)

    def add(pos, data, title, cmap=None, vmin=None, vmax=None, cbar=False):
        ax = fig.add_subplot(pos); ax.set_facecolor('#1a1a2e')
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color='white', fontsize=9, pad=5); ax.axis('off')
        if cbar:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    add(gs[0,0], img_rgb, 'Original')
    ax_l = fig.add_subplot(gs[0,1])
    ax_l.imshow(img_lines); ax_l.axis('off')
    ax_l.set_title(f'VP — Linhas ({r["n_linhas"]})\nScore: {r["score_vp"]}/100',
                   color='white', fontsize=9, pad=5)
    ax_l.legend(handles=[mpatches.Patch(color='green', label='Consistente'),
                          mpatches.Patch(color='red',   label='Suspeita')],
                loc='lower right', fontsize=7, facecolor='#1a1a2e', labelcolor='white')
    add(gs[0,2], heatmap_vp, 'VP — Divergencia angular',
        cmap='hot', vmin=0, vmax=1, cbar=True)

    add(gs[1,0], ela_vis,
        f'ELA — Mapa de erro\nScore: {r["score_ela"]}/100  (medio={r["ela_mean"]:.1f})',
        cmap='hot')
    add(gs[1,1], std_map,
        f'ELA — Inconsistencia regional\n(desvio entre blocos={r["ela_std"]:.2f})',
        cmap='inferno')

    add(gs[2,0], noise_vis,
        f'Ruido — Extracao\nScore: {r["score_ruido"]}/100  (CV={r["cv_ruido"]:.3f})',
        cmap='gray')
    add(gs[2,1], var_map,
        'Ruido — Variancia local\n(brilhante = ruido inconsistente)',
        cmap='hot')

    CORES_ZONA = ['#c0392b', '#e67e22', '#f1c40f', '#2980b9', '#27ae60']
    LABELS_ZONA = [
        (L1/2,          'MANIPULADA',          '#c0392b'),
        ((L1+L2)/2,     'ALTA CHANCE',          '#e67e22'),
        ((L2+L3)/2,     'INCONCLUSIVA',         '#f1c40f'),
        ((L3+L4)/2,     'CONSIST.\nMEDIA',      '#2980b9'),
        ((L4+100)/2,    'CONSISTENTE',          '#27ae60'),
    ]
    LIMITES = [0, L1, L2, L3, L4, 100]

    ax_sc = fig.add_subplot(gs[1:, 2]); ax_sc.set_facecolor('#0f3460')
    ax_sc.set_xlim(0, 100); ax_sc.set_ylim(0, 1)

    for i, cor in enumerate(CORES_ZONA):
        ax_sc.axvspan(LIMITES[i], LIMITES[i+1], color=cor, alpha=0.30)
    for lim in [L1, L2, L3, L4]:
        ax_sc.axvline(lim, color='white', linewidth=0.8, linestyle=':', alpha=0.5)

    for s, lbl, cor in [
        (r['score_vp'],    'VP',     '#74b9ff'),
        (r['score_ela'],   'ELA',    '#fd79a8'),
        (r['score_ruido'], 'Ruido',  '#fdcb6e'),
        (r['score_escala'],'Escala', '#a29bfe'),
    ]:
        ax_sc.axvline(s, color=cor, linewidth=1.5, linestyle='--', alpha=0.80)
        ax_sc.text(s, 0.48, lbl, ha='center', color=cor, fontsize=7)

    ax_sc.axvline(r['score'], color='white', linewidth=3)
    ax_sc.text(r['score'], 0.82, f"{r['score']}",
               ha='center', color='white', fontsize=22, fontweight='bold')

    for x, txt, cor in LABELS_ZONA:
        ax_sc.text(x, 0.18, txt, ha='center', color=cor,
                   fontsize=7, fontweight='bold', va='center')

    ax_sc.set_title('Score Final (5 niveis)', color='white', fontsize=10, pad=6)
    ax_sc.axis('off')

    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"🎨 Visual salvo em: {out}")
    print("✅ Análise completa!\n")


# ══════════════════════════════════════════════
# BLOCO 8 — COMPUTAÇÃO PARALELA: benchmark
# ══════════════════════════════════════════════

def _benchmark_paralelo(imagens, configs_workers):
    """
    Roda o pipeline em `imagens` com cada configuração de workers.
    Retorna lista de dicts com métricas por configuração.

    Passo 1: sequencial puro (T1) — sem Pool, sem overhead de fork.
    Passo 2: Pool com N workers para cada N em configs_workers.
    Passo 3: calcula speedup, eficiência e overhead.
    """
    metricas = []

    # ── Passo 1: T1 — sequencial puro ─────────
    print(f"\n  [CP] Rodando sequencial (baseline T1)...")
    t_ini = time.perf_counter()
    for path in imagens:
        _analisar_leve(path)
    T1 = time.perf_counter() - t_ini
    print(f"       T1 = {T1:.3f}s  ({len(imagens)} imagens)")

    metricas.append({
        'workers':    1,
        'tempo':      T1,
        'speedup':    1.0,
        'eficiencia': 1.0,
        'overhead':   0.0,
        'label':      'Sequencial\n(baseline)',
    })

    # ── Passo 2: paralelo com N workers ───────
    for n in configs_workers:
        if n <= 1:
            continue  # já medimos o caso sequencial acima
        print(f"  [CP] Pool com {n} workers...")
        t_ini = time.perf_counter()
        with Pool(n) as pool:
            pool.map(_analisar_leve, imagens)
        TN = time.perf_counter() - t_ini

        # ── Passo 3: métricas ─────────────────
        speedup    = T1 / TN if TN > 0 else 1.0
        eficiencia = speedup / n
        overhead   = max(0.0, TN * n - T1)

        metricas.append({
            'workers':    n,
            'tempo':      TN,
            'speedup':    round(speedup, 3),
            'eficiencia': round(eficiencia, 3),
            'overhead':   round(overhead, 3),
            'label':      f'{n} workers',
        })
        print(f"       TN={TN:.3f}s  speedup={speedup:.2f}x  "
              f"eficiência={eficiencia:.2f}  overhead={overhead:.3f}s")

    return metricas, T1


def _benchmark_escalabilidade_fraca(imagens_base, configs_workers):
    """
    Escalabilidade fraca: cada worker recebe 1 imagem.
    Mede se o tempo permanece constante ao aumentar workers + carga.
    Ideal: tempo constante (eficiência = 1.0 para todos os N).
    """
    print(f"\n  [CP] Escalabilidade Fraca...")
    resultados_fraca = []
    T1_fraca = None

    for n in configs_workers:
        # Carga proporcional: n imagens para n workers
        imgs_fraca = (imagens_base * ((n // len(imagens_base)) + 1))[:n]

        if n == 1:
            t_ini = time.perf_counter()
            for p in imgs_fraca:
                _analisar_leve(p)
            TN = time.perf_counter() - t_ini
            T1_fraca = TN
        else:
            t_ini = time.perf_counter()
            with Pool(n) as pool:
                pool.map(_analisar_leve, imgs_fraca)
            TN = time.perf_counter() - t_ini

        ef = (T1_fraca / TN) if T1_fraca and TN > 0 else 1.0
        resultados_fraca.append({'workers': n, 'tempo': TN, 'eficiencia': round(ef, 3)})
        print(f"       {n} workers × {len(imgs_fraca)} imgs → {TN:.3f}s  "
              f"(eficiência fraca: {ef:.3f})")

    return resultados_fraca


def _salvar_dashboard_cp(metricas_forte, metricas_fraca, n_imagens, out=OUTPUT_CP):
    """
    Dashboard de 4 painéis:
      1. Speedup real vs ideal (Lei de Amdahl)
      2. Eficiência por workers
      3. Tempo absoluto por configuração
      4. Overhead de comunicação entre processos
    """
    BG    = '#1a1a2e'
    PANEL = '#0f3460'
    AZUL  = '#74b9ff'
    VERDE = '#55efc4'
    VERM  = '#e17055'
    AMAR  = '#fdcb6e'
    ROXO  = '#a29bfe'

    workers_forte = [m['workers']    for m in metricas_forte]
    speedups      = [m['speedup']    for m in metricas_forte]
    eficiencias   = [m['eficiencia'] for m in metricas_forte]
    tempos        = [m['tempo']      for m in metricas_forte]
    overheads     = [m['overhead']   for m in metricas_forte]
    labels        = [m['label']      for m in metricas_forte]

    workers_fraca = [m['workers']    for m in metricas_fraca]
    ef_fraca      = [m['eficiencia'] for m in metricas_fraca]

    # Speedup ideal (Lei de Amdahl com fração paralela estimada = 0.9)
    f_par = 0.90
    speedup_ideal    = [n for n in workers_forte]
    speedup_amdahl   = [1 / ((1 - f_par) + f_par / n) for n in workers_forte]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f'Dashboard — Computação Paralela\n'
        f'{n_imagens} imagens  |  máx. {cpu_count()} núcleos disponíveis',
        color='white', fontsize=14, fontweight='bold'
    )

    # ── Painel 1: Speedup ─────────────────────
    ax = axes[0, 0]; ax.set_facecolor(PANEL)
    ax.plot(workers_forte, speedup_ideal,  '--', color='#636e72',
            linewidth=1.5, label='Ideal (linear)')
    ax.plot(workers_forte, speedup_amdahl, '--', color=AMAR,
            linewidth=1.5, label=f'Amdahl (f={f_par})')
    ax.plot(workers_forte, speedups, 'o-',  color=AZUL,
            linewidth=2.5, markersize=8, label='Speedup real')
    for x, y in zip(workers_forte, speedups):
        ax.annotate(f'{y:.2f}x', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center', color=AZUL, fontsize=9)
    ax.set_title('Escalabilidade Forte — Speedup', color='white', fontsize=11)
    ax.set_xlabel('Workers', color='white'); ax.set_ylabel('Speedup', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor=PANEL, labelcolor='white', fontsize=9)
    ax.set_xticks(workers_forte)
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    # ── Painel 2: Eficiência forte + fraca ────
    ax = axes[0, 1]; ax.set_facecolor(PANEL)
    ax.axhline(1.0, color='#636e72', linewidth=1, linestyle='--', label='Ideal (1.0)')
    ax.plot(workers_forte, eficiencias, 's-', color=VERDE,
            linewidth=2.5, markersize=8, label='Efic. forte')
    ax.plot(workers_fraca, ef_fraca,    '^-', color=ROXO,
            linewidth=2.5, markersize=8, label='Efic. fraca')
    for x, y in zip(workers_forte, eficiencias):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center', color=VERDE, fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_title('Eficiência por Workers\n(forte × fraca)', color='white', fontsize=11)
    ax.set_xlabel('Workers', color='white'); ax.set_ylabel('Eficiência', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor=PANEL, labelcolor='white', fontsize=9)
    ax.set_xticks(workers_forte)
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    # ── Painel 3: Tempo absoluto ───────────────
    ax = axes[1, 0]; ax.set_facecolor(PANEL)
    cores_barras = [VERM if i == 0 else AZUL for i in range(len(workers_forte))]
    bars = ax.bar(range(len(workers_forte)), tempos, color=cores_barras,
                  edgecolor='#ffffff22', linewidth=0.8)
    ax.set_xticks(range(len(workers_forte)))
    ax.set_xticklabels(labels, color='white', fontsize=8)
    for bar, t in zip(bars, tempos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{t:.2f}s', ha='center', color='white', fontsize=9, fontweight='bold')
    ax.set_title('Tempo de Execução por Configuração', color='white', fontsize=11)
    ax.set_ylabel('Tempo (s)', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    # ── Painel 4: Overhead ────────────────────
    ax = axes[1, 1]; ax.set_facecolor(PANEL)
    ax.axhline(0.0, color='#636e72', linewidth=1, linestyle='--')
    ax.bar(range(len(workers_forte)), overheads, color=AMAR,
           edgecolor='#ffffff22', linewidth=0.8)
    ax.set_xticks(range(len(workers_forte)))
    ax.set_xticklabels(labels, color='white', fontsize=8)
    for i, (oh, w) in enumerate(zip(overheads, workers_forte)):
        ax.text(i, oh + max(overheads)*0.02 if max(overheads) > 0 else 0.01,
                f'{oh:.3f}s', ha='center', color='white', fontsize=9)
    ax.set_title('Overhead de Comunicação\n(fork + serialização de dados)',
                 color='white', fontsize=11)
    ax.set_ylabel('Overhead (s)', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊 Dashboard CP salvo em: {out}")


def _imprimir_tabela_cp(metricas_forte, metricas_fraca):
    """Imprime tabela resumo no terminal."""
    print(f"\n{'='*70}")
    print("  MÉTRICAS DE COMPUTAÇÃO PARALELA — ESCALABILIDADE FORTE")
    print(f"{'='*70}")
    print(f"  {'Workers':<12} {'Tempo (s)':<14} {'Speedup':<12} "
          f"{'Eficiência':<14} {'Overhead (s)'}")
    print(f"  {'-'*64}")
    for m in metricas_forte:
        print(f"  {m['label']:<12} {m['tempo']:<14.3f} {m['speedup']:<12.3f} "
              f"{m['eficiencia']:<14.3f} {m['overhead']:.3f}")

    print(f"\n{'='*70}")
    print("  MÉTRICAS DE COMPUTAÇÃO PARALELA — ESCALABILIDADE FRACA")
    print(f"{'='*70}")
    print(f"  {'Workers':<12} {'Imgs':<10} {'Tempo (s)':<14} {'Eficiência'}")
    print(f"  {'-'*46}")
    for m in metricas_fraca:
        print(f"  {m['workers']:<12} {m['workers']:<10} "
              f"{m['tempo']:<14.3f} {m['eficiencia']:.3f}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════
# BLOCO 9 — MODO 2: DATASET Au + Sp
# ══════════════════════════════════════════════

def modo_dataset(pasta_au='Au', pasta_sp='Sp'):
    au_imgs = glob(os.path.join(pasta_au,'*.jpg')) + \
              glob(os.path.join(pasta_au,'*.png'))
    sp_imgs = glob(os.path.join(pasta_sp,'*.jpg')) + \
              glob(os.path.join(pasta_sp,'*.png'))

    if not au_imgs and not sp_imgs:
        print(f"❌ Nenhuma imagem em '{pasta_au}' ou '{pasta_sp}'"); return

    todas  = au_imgs + sp_imgs
    labels = [0]*len(au_imgs) + [1]*len(sp_imgs)

    print(f"\n📊 MODO DATASET (Au + Sp)")
    print(f"   Autenticas  (Au): {len(au_imgs)} imagens")
    print(f"   Manipuladas (Sp): {len(sp_imgs)} imagens")
    print(f"   Total: {len(todas)} imagens")
    print(f"   Limiares: ≤{L1} MANIP | ≤{L2} ALTA CHANCE | ≤{L3} INCONCL"
          f" | ≤{L4} CONSIST.MEDIA | >{L4} CONSISTENTE")

    # ── Workers: sequencial + 2 + 4 + todos os núcleos
    N_CPU = cpu_count()
    configs_workers = sorted(set([1, 2, 4, N_CPU]))

    print(f"\n⚡ Iniciando benchmark CP com configs: {configs_workers} workers...")

    # Escalabilidade forte (mesma carga total, workers variável)
    metricas_forte, T1 = _benchmark_paralelo(todas, configs_workers)

    # Escalabilidade fraca (carga proporcional ao nº de workers)
    # Usa a lista completa de imagens como base (repete se necessário)
    metricas_fraca = _benchmark_escalabilidade_fraca(todas, configs_workers)

    # Resultados finais (usa a execução com max workers)
    print(f"\n  Coletando resultados com {N_CPU} workers para classificação...")
    with Pool(N_CPU) as pool:
        resultados = pool.map(_analisar_leve, todas)

    # Métricas de modelo
    preds  = [1 if r['score'] <= L2 else 0 for r in resultados]
    scores = [r['score'] for r in resultados]

    tp = sum(1 for r,p in zip(labels,preds) if r==1 and p==1)
    tn = sum(1 for r,p in zip(labels,preds) if r==0 and p==0)
    fp = sum(1 for r,p in zip(labels,preds) if r==0 and p==1)
    fn = sum(1 for r,p in zip(labels,preds) if r==1 and p==0)

    total    = tp+tn+fp+fn
    acuracia = (tp+tn)/total if total else 0
    precisao = tp/(tp+fp)   if (tp+fp) else 0
    recall   = tp/(tp+fn)   if (tp+fn) else 0
    f1       = 2*precisao*recall/(precisao+recall) if (precisao+recall) else 0

    niveis = {'MANIPULADA':0,'ALTA CHANCE DE MANIPULACAO':0,
              'INCONCLUSIVA':0,'CONSISTENCIA MEDIA':0,'CONSISTENTE':0}
    for r in resultados:
        k = calcular_status(r['score'])
        if k in niveis: niveis[k] += 1

    print(f"\n{'='*65}")
    print("         AVALIACAO DO MODELO — Au + Sp")
    print(f"{'='*65}")
    print(f"  Acuracia:  {acuracia*100:.2f}%")
    print(f"  Precisao:  {precisao*100:.2f}%   (positivo = score <= {L2})")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*65}")
    print("  Distribuicao por nivel:")
    for k,v in niveis.items():
        print(f"    {k:<35}: {v}")
    print(f"{'='*65}")

    # Tabela CP no terminal
    _imprimir_tabela_cp(metricas_forte, metricas_fraca)

    # Gráfico de avaliação do modelo
    _salvar_grafico_avaliacao(
        scores[:len(au_imgs)], scores[len(au_imgs):],
        acuracia, precisao, recall, f1,
        tp, tn, fp, fn, OUTPUT_AVALIA,
        titulo='Avaliacao — Dataset Au + Sp  (VP + ELA + Ruido + Escala)'
    )
    print(f"🎨 Grafico modelo salvo em: {OUTPUT_AVALIA}")

    # Dashboard de CP
    _salvar_dashboard_cp(metricas_forte, metricas_fraca, len(todas), out=OUTPUT_CP)


# ══════════════════════════════════════════════
# BLOCO 10 — MODO 3: PASTA SEM RÓTULO
# ══════════════════════════════════════════════

def modo_pasta(pasta):
    imagens  = glob(os.path.join(pasta,'*.jpg'))
    imagens += glob(os.path.join(pasta,'*.png'))

    if not imagens:
        print(f"❌ Nenhuma imagem em '{pasta}'"); return

    print(f"\n📂 MODO PASTA — {len(imagens)} imagens em '{pasta}'")

    N_CPU = cpu_count()
    configs_workers = sorted(set([1, 2, 4, N_CPU]))

    print(f"⚡ Iniciando benchmark CP com configs: {configs_workers} workers...")

    # Escalabilidade forte
    metricas_forte, T1 = _benchmark_paralelo(imagens, configs_workers)

    # Escalabilidade fraca
    metricas_fraca = _benchmark_escalabilidade_fraca(imagens, configs_workers)

    # Resultados finais com max workers
    print(f"\n  Coletando resultados com {N_CPU} workers para classificação...")
    with Pool(N_CPU) as pool:
        resultados = pool.map(_analisar_leve, imagens)

    niveis = {'MANIPULADA':[], 'ALTA CHANCE DE MANIPULACAO':[],
              'INCONCLUSIVA':[], 'CONSISTENCIA MEDIA':[], 'CONSISTENTE':[]}
    for r in resultados:
        k = calcular_status(r['score'])
        if k in niveis:
            niveis[k].append(r)

    print(f"\n{'='*65}")
    print(f"  RESULTADO — {len(imagens)} imagens analisadas")
    print(f"{'='*65}")
    print(f"  🚨 MANIPULADA:              {len(niveis['MANIPULADA'])}")
    print(f"  ⛔ ALTA CHANCE MANIPULACAO: {len(niveis['ALTA CHANCE DE MANIPULACAO'])}")
    print(f"  ⚠️  INCONCLUSIVA:            {len(niveis['INCONCLUSIVA'])}")
    print(f"  🔎 CONSISTENCIA MEDIA:      {len(niveis['CONSISTENCIA MEDIA'])}")
    print(f"  ✅ CONSISTENTE:             {len(niveis['CONSISTENTE'])}")
    print(f"{'='*65}")

    suspeitas = sorted(resultados, key=lambda r: r['score'])[:5]
    if suspeitas:
        print("\n  🔍 Top 5 imagens mais suspeitas:")
        for r in suspeitas:
            print(f"     {r['score']:5.1f}/100 | {calcular_status(r['score']):<30} | "
                  f"{os.path.basename(r['path'])}")

    # Tabela CP no terminal
    _imprimir_tabela_cp(metricas_forte, metricas_fraca)

    # Histograma de scores
    scores = [r['score'] for r in resultados]
    CORES_ZONA = ['#c0392b','#e67e22','#f1c40f','#2980b9','#27ae60']
    LIMITES    = [0, L1, L2, L3, L4, 100]

    fig, ax = plt.subplots(figsize=(13,5))
    fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#0f3460')
    ax.hist(scores, bins=20, color='#74b9ff', edgecolor='white', alpha=0.85, zorder=2)
    for i, cor in enumerate(CORES_ZONA):
        ax.axvspan(LIMITES[i], LIMITES[i+1], color=cor, alpha=0.12, zorder=1)
    for lim in [L1, L2, L3, L4]:
        ax.axvline(lim, color='white', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_title(f'Distribuicao de Scores — {len(imagens)} imagens de "{pasta}"',
                 color='white', fontsize=13)
    ax.set_xlabel('Score (0=manipulada, 100=consistente)', color='white')
    ax.set_ylabel('Quantidade de imagens', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ffffff33')
    plt.tight_layout()
    plt.savefig(OUTPUT_LOTE, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"🎨 Grafico lote salvo em: {OUTPUT_LOTE}")

    # Dashboard de CP
    _salvar_dashboard_cp(metricas_forte, metricas_fraca, len(imagens), out=OUTPUT_CP)


# ══════════════════════════════════════════════
# BLOCO 11 — GRÁFICO DE AVALIAÇÃO (modo dataset)
# ══════════════════════════════════════════════

def _salvar_grafico_avaliacao(au_scores, sp_scores,
                               acuracia, precisao, recall, f1,
                               tp, tn, fp, fn, out, titulo=''):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(titulo, color='white', fontsize=14, fontweight='bold')

    CORES_ZONA = ['#c0392b','#e67e22','#f1c40f','#2980b9','#27ae60']
    LIMITES    = [0, L1, L2, L3, L4, 100]

    axes[0].set_facecolor('#0f3460')
    for i, cor in enumerate(CORES_ZONA):
        axes[0].axvspan(LIMITES[i], LIMITES[i+1], color=cor, alpha=0.12)
    axes[0].hist(au_scores, bins=20, alpha=0.75, color='#2ecc71', label='Autentica')
    axes[0].hist(sp_scores, bins=20, alpha=0.75, color='#e74c3c', label='Manipulada')
    for lim in [L1, L2, L3, L4]:
        axes[0].axvline(lim, color='white', linewidth=0.8, linestyle='--', alpha=0.6)
    axes[0].set_title('Distribuicao de Scores (5 niveis)', color='white')
    axes[0].set_xlabel('Score', color='white')
    axes[0].set_ylabel('Quantidade', color='white')
    axes[0].tick_params(colors='white')
    axes[0].legend(facecolor='#0f3460', labelcolor='white')

    matriz = np.array([[tn,fp],[fn,tp]])
    axes[1].imshow(matriz, cmap='Blues')
    axes[1].set_facecolor('#0f3460')
    for i in range(2):
        for j in range(2):
            axes[1].text(j,i,str(matriz[i,j]),
                         ha='center',va='center',
                         color='white',fontsize=16,fontweight='bold')
    axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
    axes[1].set_xticklabels(['Pred: OK','Pred: MANIP'], color='white')
    axes[1].set_yticklabels(['Real: OK','Real: MANIP'], color='white')
    axes[1].set_title(f'Matriz de Confusao\n(positivo = score <= {L2})', color='white')

    metricas = ['Acuracia','Precisao','Recall','F1-Score']
    valores  = [acuracia, precisao, recall, f1]
    cores    = ['#74b9ff','#fd79a8','#fdcb6e','#55efc4']
    axes[2].set_facecolor('#0f3460')
    bars = axes[2].bar(metricas, [v*100 for v in valores], color=cores)
    axes[2].set_ylim(0,115)
    axes[2].set_title('Metricas (%)', color='white')
    axes[2].tick_params(colors='white')
    for bar, val in zip(bars, valores):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+1.5, f'{val*100:.1f}%',
                     ha='center', color='white', fontsize=10, fontweight='bold')

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor('#ffffff33')
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detector unificado — VP + ELA + Ruído + Escala + métricas CP')
    parser.add_argument('imagem',    nargs='?', default=None,
                        help='Caminho de uma imagem para análise individual')
    parser.add_argument('--dataset', action='store_true',
                        help='Modo dataset: avalia Au + Sp com F1, acurácia e métricas CP')
    parser.add_argument('--au',      default='Au',
                        help='Pasta de imagens autênticas (padrão: Au)')
    parser.add_argument('--sp',      default='Sp',
                        help='Pasta de imagens manipuladas (padrão: Sp)')
    parser.add_argument('--pasta',   default=None,
                        help='Pasta com imagens sem rótulo (análise em lote) + métricas CP')
    parser.add_argument('--out',     default=OUTPUT_UNICA,
                        help=f'Arquivo de saída (padrão: {OUTPUT_UNICA})')

    args = parser.parse_args()

    if args.dataset:
        modo_dataset(args.au, args.sp)
    elif args.pasta:
        modo_pasta(args.pasta)
    elif args.imagem:
        modo_imagem_unica(args.imagem, out=args.out)
    else:
        print(__doc__)
        parser.print_help()