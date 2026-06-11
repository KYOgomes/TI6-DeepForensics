# -*- coding: utf-8 -*-
"""
DeepForensics — Módulo de Computação DISTRIBUÍDA
=================================================
Estende o pipeline VP+ELA+Ruído+Escala para rodar em vários **nós** conectados
por uma rede WireGuard (10.0.0.0/24). Enquanto `detector_unificado.py` paraleliza
entre os **núcleos de uma máquina** (multiprocessing.Pool), aqui o paralelismo é
**entre máquinas**: cada nó analisa uma fatia das imagens e devolve, para cada
uma, se é manipulada ou não.

Arquitetura
-----------
  • Cada nó roda um *worker* HTTP leve (Flask) com:
        GET  /dist/health   → identidade do nó (hostname, núcleos)
        POST /dist/process  → recebe uma fatia de imagens, roda o pipeline e
                              devolve scores + classificação
  • O *coordenador* (a UI, ou o CLI abaixo) lê a lista de nós de `Cd/nodes.json`,
    divide as imagens em fatias balanceadas e envia cada fatia ao seu nó em
    paralelo (ThreadPool). Depois agrega os resultados na ordem original.

Transporte das imagens (escolhido automaticamente, ou forçado):
  • "path"  → envia só o caminho relativo ao repositório. Pressupõe que todos os
              nós têm o mesmo Dataset clonado (git). Rápido e leve — ideal para o
              benchmark do dataset CASIA.
  • "bytes" → envia o conteúdo da imagem em base64. Funciona com qualquer upload,
              mesmo que o nó não tenha o arquivo. Mais tráfego.

Uso
---
  # Em CADA nó (inclusive neste), suba o worker:
      python Algorithm/distributed.py --worker
      python Algorithm/distributed.py --worker --port 5001

  # No coordenador, rode o benchmark distribuído sobre uma pasta:
      python Algorithm/distributed.py --coordinator --pasta Dataset/Au/sample
      python Algorithm/distributed.py --coordinator --pasta Dataset/Tp/sample --bytes

  # Só checar quais nós estão de pé:
      python Algorithm/distributed.py --status

A API REST (app.py) também expõe:
      GET  /api/distributed/nodes
      POST /api/distributed         (multipart images=<file>* | json {dir,max})
"""

import os
import sys
import io
import json
import time
import base64
import tempfile
import argparse
import socket
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import urllib.request
import urllib.error

# Garante import do detector estando em qualquer cwd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from detector_unificado import (
    _analisar_leve,
    _aplicar_rf,
    calcular_status,
    L1, L2, L3, L4,
)
from multiprocessing import Pool, cpu_count

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DOS NÓS
# ─────────────────────────────────────────────
DEFAULT_PORT     = 5001
NODES_CONFIG     = os.path.join(ROOT, 'Cd', 'nodes.json')
DIST_TIMEOUT     = float(os.environ.get('DF_DIST_TIMEOUT', '300'))  # s por requisição
HEALTH_TIMEOUT   = float(os.environ.get('DF_HEALTH_TIMEOUT', '4'))  # s para o /health


def carregar_nos():
    """
    Lê a lista de nós. Ordem de prioridade:
      1. Variável de ambiente DF_NODES="10.0.0.1,10.0.0.6:5001,..."
      2. Arquivo Cd/nodes.json
      3. Fallback: só localhost
    Devolve (lista_de_nos, worker_port). Cada nó = {'name','host','port','url'}.
    """
    port = DEFAULT_PORT
    nodes = []

    env = os.environ.get('DF_NODES', '').strip()
    if env:
        for i, tok in enumerate(env.split(',')):
            tok = tok.strip()
            if not tok:
                continue
            host, _, p = tok.partition(':')
            nodes.append({'name': f'no-{i+1}', 'host': host,
                          'port': int(p) if p else port})
    elif os.path.isfile(NODES_CONFIG):
        try:
            with open(NODES_CONFIG, 'r', encoding='utf-8') as fh:
                cfg = json.load(fh)
            port = int(cfg.get('worker_port', DEFAULT_PORT))
            for i, n in enumerate(cfg.get('nodes', [])):
                nodes.append({'name': n.get('name', f'no-{i+1}'),
                              'host': n['host'],
                              'port': int(n.get('port', port))})
        except Exception as e:
            print(f"[dist] erro lendo {NODES_CONFIG}: {e}")

    if not nodes:
        nodes = [{'name': 'local', 'host': '127.0.0.1', 'port': port}]

    for n in nodes:
        n['url'] = f"http://{n['host']}:{n['port']}"
    return nodes, port


# ─────────────────────────────────────────────
# HELPERS de transporte
# ─────────────────────────────────────────────
def _rel_to_root(path):
    """Converte um caminho absoluto para relativo ao repositório (p/ transporte 'path')."""
    try:
        return os.path.relpath(os.path.abspath(path), ROOT).replace('\\', '/')
    except ValueError:
        return path  # drives diferentes no Windows → manda absoluto mesmo


def _resolve_no_root(path):
    """No worker: resolve um caminho relativo recebido contra o ROOT local."""
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path.replace('/', os.sep))


def _http_post_json(url, payload, timeout):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _http_get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


# ══════════════════════════════════════════════
# LADO WORKER — processa uma fatia de imagens
# ══════════════════════════════════════════════
def _processar_fatia(itens, transport, use_rf, cores):
    """
    Núcleo do worker. `itens` é uma lista de dicts:
        transport='path'  → {'id': int, 'path': 'Dataset/.../x.jpg'}
        transport='bytes' → {'id': int, 'nome': 'x.jpg', 'data': '<base64>'}
    Roda o pipeline (opcionalmente com Pool local de `cores`), aplica o RF e
    devolve a lista de resultados na MESMA ordem de entrada.
    """
    tmpdir = None
    paths, ids, nomes = [], [], []

    if transport == 'bytes':
        tmpdir = tempfile.mkdtemp(prefix='df_dist_')
        for it in itens:
            nome = os.path.basename(it.get('nome', f"img_{it['id']}"))
            dst = os.path.join(tmpdir, f"{it['id']}_{nome}")
            with open(dst, 'wb') as fh:
                fh.write(base64.b64decode(it['data']))
            paths.append(dst); ids.append(it['id']); nomes.append(nome)
    else:  # 'path'
        for it in itens:
            paths.append(_resolve_no_root(it['path']))
            ids.append(it['id'])
            nomes.append(os.path.basename(it['path']))

    try:
        if cores and cores > 1 and len(paths) > 1:
            with Pool(cores) as pool:
                resultados = pool.map(_analisar_leve, paths)
        else:
            resultados = [_analisar_leve(p) for p in paths]

        if use_rf:
            _aplicar_rf(resultados)

        saida = []
        for _id, nome, r in zip(ids, nomes, resultados):
            score = float(r.get('score', 50))
            saida.append({
                'id'          : _id,
                'arquivo'     : nome,
                'score'       : score,
                'status'      : calcular_status(score),
                'manipulada'  : bool(score <= L2),
                'score_vp'    : float(r.get('score_vp', 50)),
                'score_ela'   : float(r.get('score_ela', 50)),
                'score_ruido' : float(r.get('score_ruido', 50)),
                'score_escala': float(r.get('score_escala', 50)),
                'erro'        : r.get('erro'),
            })
        return saida
    finally:
        if tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def criar_worker_app():
    """Cria um Flask app mínimo que expõe os endpoints do worker."""
    from flask import Flask, request, jsonify

    app = Flask('df_worker')

    @app.route('/dist/health')
    def health():
        return jsonify({
            'ok'        : True,
            'node'      : socket.gethostname(),
            'cpu_count' : cpu_count(),
            'role'      : 'worker',
        })

    @app.route('/dist/process', methods=['POST'])
    def process():
        data = request.get_json(silent=True) or {}
        itens     = data.get('imagens', [])
        transport = data.get('transport', 'path')
        use_rf    = bool(data.get('use_rf', True))
        cores     = data.get('cores', 1)
        if cores in (None, 0):
            cores = cpu_count()

        t0 = time.perf_counter()
        try:
            resultados = _processar_fatia(itens, transport, use_rf, int(cores))
        except Exception as e:
            return jsonify({'ok': False, 'erro': str(e),
                            'node': socket.gethostname()}), 500
        elapsed = time.perf_counter() - t0

        return jsonify({
            'ok'           : True,
            'node'         : socket.gethostname(),
            'cpu_count'    : cpu_count(),
            'cores_usados' : int(cores),
            'n'            : len(resultados),
            'tempo'        : elapsed,
            'resultados'   : resultados,
        })

    return app


def run_worker(host='0.0.0.0', port=DEFAULT_PORT):
    """Sobe o worker neste nó (escuta na interface WireGuard)."""
    app = criar_worker_app()
    print(f"\n  [worker] DeepForensics — nó '{socket.gethostname()}'")
    print(f"  [worker] escutando em http://{host}:{port}  ({cpu_count()} núcleos)\n")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


# ══════════════════════════════════════════════
# LADO COORDENADOR — distribui as imagens entre os nós
# ══════════════════════════════════════════════
def checar_nos(nodes=None):
    """Faz /dist/health em cada nó. Devolve a lista com 'online' e infos."""
    if nodes is None:
        nodes, _ = carregar_nos()
    out = []
    for n in nodes:
        info = dict(n)
        try:
            h = _http_get_json(n['url'] + '/dist/health', HEALTH_TIMEOUT)
            info.update({'online': True,
                         'hostname': h.get('node'),
                         'cpu_count': h.get('cpu_count')})
        except Exception as e:
            info.update({'online': False, 'erro': str(e)})
        out.append(info)
    return out


def _chunk_balanceado(itens, k):
    """Divide `itens` em k fatias o mais equilibradas possível (round-robin por blocos)."""
    k = max(1, k)
    n = len(itens)
    base, resto = divmod(n, k)
    fatias, i = [], 0
    for w in range(k):
        tam = base + (1 if w < resto else 0)
        fatias.append(itens[i:i+tam])
        i += tam
    return fatias


def _montar_itens(paths, transport):
    """Cria os itens (com id global) prontos para envio, conforme o transporte."""
    itens = []
    for i, p in enumerate(paths):
        if transport == 'bytes':
            with open(p, 'rb') as fh:
                data = base64.b64encode(fh.read()).decode('ascii')
            itens.append({'id': i, 'nome': os.path.basename(p), 'data': data})
        else:
            itens.append({'id': i, 'path': _rel_to_root(p)})
    return itens


def processar_distribuido(paths, nodes=None, transport='path',
                          use_rf=True, cores_por_no=1):
    """
    Distribui `paths` entre os nós ONLINE e agrega os resultados.
    Devolve dict com:
      'resultados' (ordenado pelo id original), 'por_no' (tempo/contagem de cada nó),
      'tempo' (wall-clock total), 'n_nos', 'distribuicao' (contagem por nível).
    """
    if nodes is None:
        nodes, _ = carregar_nos()
    online = [n for n in checar_nos(nodes) if n.get('online')]
    if not online:
        return {'ok': False, 'erro': 'nenhum nó online'}

    itens_todos = _montar_itens(paths, transport)
    fatias = _chunk_balanceado(itens_todos, len(online))

    def _enviar(no, itens):
        if not itens:
            return {'node': no['name'], 'n': 0, 'tempo': 0.0, 'resultados': []}
        t0 = time.perf_counter()
        try:
            resp = _http_post_json(no['url'] + '/dist/process', {
                'imagens': itens, 'transport': transport,
                'use_rf': use_rf, 'cores': cores_por_no,
            }, DIST_TIMEOUT)
            resp['rtt'] = time.perf_counter() - t0
            resp.setdefault('node', no['name'])
            resp['no_cfg'] = no['name']
            return resp
        except Exception as e:
            return {'ok': False, 'node': no['name'], 'no_cfg': no['name'],
                    'erro': str(e), 'n': len(itens),
                    'tempo': time.perf_counter() - t0, 'resultados': []}

    t_ini = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(online)) as ex:
        respostas = list(ex.map(_enviar, online, fatias))
    elapsed = time.perf_counter() - t_ini

    # Agrega resultados na ordem original
    plano = []
    por_no = []
    for resp in respostas:
        por_no.append({
            'node'     : resp.get('node'),
            'no_cfg'   : resp.get('no_cfg'),
            'cpu_count': resp.get('cpu_count'),
            'n'        : resp.get('n', 0),
            'tempo'    : round(float(resp.get('tempo', 0.0)), 3),
            'rtt'      : round(float(resp.get('rtt', 0.0)), 3),
            'ok'       : resp.get('ok', True),
            'erro'     : resp.get('erro'),
        })
        plano.extend(resp.get('resultados', []))

    plano.sort(key=lambda r: r['id'])

    niveis = {'MANIPULADA': 0, 'ALTA CHANCE DE MANIPULACAO': 0,
              'INCONCLUSIVA': 0, 'CONSISTENCIA MEDIA': 0, 'CONSISTENTE': 0}
    for r in plano:
        k = r.get('status')
        if k in niveis:
            niveis[k] += 1

    return {
        'ok'          : True,
        'n_imagens'   : len(paths),
        'n_nos'       : len(online),
        'transporte'  : transport,
        'cores_por_no': cores_por_no,
        'tempo'       : float(elapsed),
        'distribuicao': niveis,
        'por_no'      : por_no,
        'resultados'  : plano,
    }


def benchmark_distribuido_api(paths, nodes=None, transport='path',
                              cores_por_no=1, use_rf=True):
    """
    Benchmark de escalabilidade ENTRE NÓS (formato compatível com o dashboard de CP).

    Passo 1: baseline T1 — processa TODAS as imagens sequencialmente NESTE nó
             (1 núcleo), sem rede. É o mesmo baseline da CP local.
    Passo 2: para k = 1, 2, ..., nº de nós online → distribui entre os k primeiros
             nós e mede TN. speedup=T1/TN, eficiência=speedup/k, overhead=TN*k - T1.
    Passo 3: faz uma execução final com TODOS os nós para colher a classificação.

    Cada nó usa `cores_por_no` núcleos (default 1 → speedup atribuível só à
    distribuição, comparável ao benchmark de núcleos do detector_unificado).
    """
    if not paths:
        return {'ok': False, 'erro': 'nenhuma imagem'}
    if nodes is None:
        nodes, _ = carregar_nos()

    online = [n for n in checar_nos(nodes) if n.get('online')]
    if not online:
        return {'ok': False, 'erro': 'nenhum nó online', 'nodes': checar_nos(nodes)}

    # ── Passo 0: warm-up — neutraliza cold start (cache de disco, 1ª carga do
    #    cv2/RF nos workers) para o baseline T1 não sair artificialmente lento e
    #    inflar o speedup. Lê os bytes de todas as imagens (popula o cache do SO),
    #    roda 1 análise localmente e manda 1 imagem para cada nó (descartado). ──
    for p in paths:
        try:
            with open(p, 'rb') as fh:
                fh.read()
        except OSError:
            pass
    try:
        _aplicar_rf([_analisar_leve(paths[0])])
    except Exception:
        pass
    processar_distribuido(paths[:len(online)], online, transport, use_rf, cores_por_no)

    # ── Passo 1: baseline sequencial local (T1) ──────────────
    t0 = time.perf_counter()
    base = [_analisar_leve(p) for p in paths]
    if use_rf:
        _aplicar_rf(base)
    T1 = time.perf_counter() - t0

    forte = [{
        'workers': 1, 'nodes': 1, 'tempo': T1,
        'speedup': 1.0, 'eficiencia': 1.0, 'overhead': 0.0,
        'label': 'Sequencial\n(1 nó / baseline)',
    }]

    # ── Passo 2: distribuição com k nós (escalabilidade forte) ──
    detalhe_final = None
    for k in range(1, len(online) + 1):
        subset = online[:k]
        dist = processar_distribuido(paths, subset, transport, use_rf, cores_por_no)
        TN = dist['tempo']
        speedup = T1 / TN if TN > 0 else 1.0
        forte_item = {
            'workers'   : k,
            'nodes'     : k,
            'tempo'     : TN,
            'speedup'   : round(speedup, 3),
            'eficiencia': round(speedup / k, 3),
            'overhead'  : round(max(0.0, TN * k - T1), 3),
            'label'     : '1 nó' if k == 1 else f'{k} nós',
        }
        if k == 1:
            # k=1 já é o "sequencial via rede" — substitui o rótulo, mantém ambos
            forte_item['label'] = '1 nó\n(via rede)'
        forte.append(forte_item)
        if k == len(online):
            detalhe_final = dist

    # ── Distribuição de classificação (da execução com todos os nós) ──
    return {
        'ok'          : True,
        'n_imagens'   : len(paths),
        'cpu_count'   : cpu_count(),
        'n_nos'       : len(online),
        'nodes'       : online,
        'transporte'  : transport,
        'cores_por_no': cores_por_no,
        'T1'          : float(T1),
        'forte'       : [{kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                          for kk, vv in m.items()} for m in forte],
        'fraca'       : [],
        'por_no'      : (detalhe_final or {}).get('por_no', []),
        'distribuicao': (detalhe_final or {}).get('distribuicao', {}),
        'resultados'  : (detalhe_final or {}).get('resultados', []),
        'escopo'      : 'distribuido',
        'unidade'     : 'nos',
        'titulo'      : f'Distribuicao de {len(paths)} imagens entre {len(online)} no(s)',
    }


# ══════════════════════════════════════════════
# DASHBOARD (CLI) — speedup/eficiência entre nós
# ══════════════════════════════════════════════
def salvar_dashboard_dist(bench, out='Metricas_distribuido.jpg'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BG, PANEL = '#1a1a2e', '#0f3460'
    AZUL, VERDE, VERM, AMAR = '#74b9ff', '#55efc4', '#e17055', '#fdcb6e'

    forte = bench['forte']
    xs   = [m['nodes']      for m in forte]
    spd  = [m['speedup']    for m in forte]
    efi  = [m['eficiencia'] for m in forte]
    tmp  = [m['tempo']      for m in forte]
    ovh  = [m['overhead']   for m in forte]
    lbl  = [m['label']      for m in forte]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Dashboard — Computacao Distribuida\n"
                 f"{bench['n_imagens']} imagens  |  {bench['n_nos']} no(s)  |  "
                 f"{bench['cores_por_no']} nucleo(s)/no",
                 color='white', fontsize=14, fontweight='bold')

    # Painel 1: Speedup
    ax = axes[0, 0]; ax.set_facecolor(PANEL)
    ax.plot(xs, xs, '--', color='#636e72', linewidth=1.5, label='Ideal (linear)')
    ax.plot(xs, spd, 'o-', color=AZUL, linewidth=2.5, markersize=8, label='Speedup real')
    for x, y in zip(xs, spd):
        ax.annotate(f'{y:.2f}x', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center', color=AZUL, fontsize=9)
    ax.set_title('Escalabilidade entre nos — Speedup', color='white', fontsize=11)
    ax.set_xlabel('Nos', color='white'); ax.set_ylabel('Speedup', color='white')
    ax.tick_params(colors='white'); ax.set_xticks(xs)
    ax.legend(facecolor=PANEL, labelcolor='white', fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    # Painel 2: Eficiência
    ax = axes[0, 1]; ax.set_facecolor(PANEL)
    ax.axhline(1.0, color='#636e72', linewidth=1, linestyle='--', label='Ideal (1.0)')
    ax.plot(xs, efi, 's-', color=VERDE, linewidth=2.5, markersize=8, label='Eficiencia')
    for x, y in zip(xs, efi):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center', color=VERDE, fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_title('Eficiencia por no', color='white', fontsize=11)
    ax.set_xlabel('Nos', color='white'); ax.set_ylabel('Eficiencia', color='white')
    ax.tick_params(colors='white'); ax.set_xticks(xs)
    ax.legend(facecolor=PANEL, labelcolor='white', fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    # Painel 3: Tempo
    ax = axes[1, 0]; ax.set_facecolor(PANEL)
    cores = [VERM if i == 0 else AZUL for i in range(len(xs))]
    bars = ax.bar(range(len(xs)), tmp, color=cores, edgecolor='#ffffff22')
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(lbl, color='white', fontsize=8)
    for b, t in zip(bars, tmp):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{t:.2f}s',
                ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
    ax.set_title('Tempo por configuracao', color='white', fontsize=11)
    ax.set_ylabel('Tempo (s)', color='white'); ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    # Painel 4: Overhead (rede + serialização)
    ax = axes[1, 1]; ax.set_facecolor(PANEL)
    ax.axhline(0.0, color='#636e72', linewidth=1, linestyle='--')
    ax.bar(range(len(xs)), ovh, color=AMAR, edgecolor='#ffffff22')
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(lbl, color='white', fontsize=8)
    for i, oh in enumerate(ovh):
        ax.text(i, oh, f'{oh:.3f}s', ha='center', va='bottom', color='white', fontsize=9)
    ax.set_title('Overhead de rede\n(latencia + serializacao)', color='white', fontsize=11)
    ax.set_ylabel('Overhead (s)', color='white'); ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#ffffff33')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[dist] Dashboard salvo em: {out}")


def _imprimir_tabela_dist(bench):
    print(f"\n{'='*72}")
    print("  COMPUTACAO DISTRIBUIDA — ESCALABILIDADE ENTRE NOS")
    print(f"{'='*72}")
    print(f"  {'Config':<16} {'Tempo (s)':<14} {'Speedup':<12} "
          f"{'Eficiencia':<14} {'Overhead (s)'}")
    print(f"  {'-'*66}")
    for m in bench['forte']:
        rot = m['label'].replace('\n', ' ')
        print(f"  {rot:<16} {m['tempo']:<14.3f} {m['speedup']:<12.3f} "
              f"{m['eficiencia']:<14.3f} {m['overhead']:.3f}")
    print(f"{'='*72}")
    print("  Trabalho por no (execucao com todos os nos):")
    for p in bench.get('por_no', []):
        st = 'ok' if p.get('ok') else f"ERRO: {p.get('erro')}"
        print(f"    {str(p.get('node')):<22} {p.get('n',0):>3} imgs em "
              f"{p.get('tempo',0):.3f}s (rtt {p.get('rtt',0):.3f}s)  [{st}]")
    d = bench.get('distribuicao', {})
    if d:
        print(f"{'='*72}")
        print("  Classificacao das imagens:")
        for k, v in d.items():
            print(f"    {k:<32}: {v}")
    print(f"{'='*72}\n")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
def _listar_imgs(pasta, n_max=None):
    out = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp'):
        out.extend(glob(os.path.join(pasta, ext)))
    out = sorted(out)
    return out[:n_max] if n_max else out


def main():
    # UTF-8 no Windows (prints com acentos/emoji)
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

    ap = argparse.ArgumentParser(description='DeepForensics — computação distribuída')
    ap.add_argument('--worker', action='store_true', help='Sobe o worker NESTE nó')
    ap.add_argument('--coordinator', action='store_true',
                    help='Roda o benchmark distribuído sobre --pasta')
    ap.add_argument('--status', action='store_true', help='Lista os nós e se estão online')
    ap.add_argument('--host', default='0.0.0.0', help='Host de escuta do worker')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT, help='Porta do worker')
    ap.add_argument('--pasta', default=os.path.join('Dataset', 'Au', 'sample'),
                    help='Pasta de imagens p/ o coordenador')
    ap.add_argument('--max', type=int, default=None, help='Máx. de imagens')
    ap.add_argument('--bytes', action='store_true',
                    help='Transporta bytes (base64) em vez de paths compartilhados')
    ap.add_argument('--cores-por-no', type=int, default=1,
                    help='Núcleos que cada nó usa (default 1 = só paralelismo entre nós)')
    ap.add_argument('--out', default='Metricas_distribuido.jpg', help='Dashboard de saída')
    args = ap.parse_args()

    if args.worker:
        run_worker(args.host, args.port)
        return

    if args.status:
        nodes, _ = carregar_nos()
        print(f"\n  Nós configurados ({len(nodes)}):")
        for n in checar_nos(nodes):
            tag = 'ONLINE ' if n.get('online') else 'OFFLINE'
            extra = (f"{n.get('hostname')} · {n.get('cpu_count')} núcleos"
                     if n.get('online') else n.get('erro', ''))
            print(f"    [{tag}] {n['name']:<14} {n['url']:<26} {extra}")
        print()
        return

    if args.coordinator:
        transport = 'bytes' if args.bytes else 'path'
        paths = _listar_imgs(args.pasta, args.max)
        if not paths:
            print(f"❌ Nenhuma imagem em '{args.pasta}'"); return
        print(f"\n📡 Coordenador — {len(paths)} imagens de '{args.pasta}' "
              f"(transporte={transport})")
        bench = benchmark_distribuido_api(paths, transport=transport,
                                          cores_por_no=args.cores_por_no)
        if not bench.get('ok'):
            print(f"❌ {bench.get('erro')}")
            for n in bench.get('nodes', []):
                if not n.get('online'):
                    print(f"   offline: {n['name']} {n['url']} — {n.get('erro')}")
            return
        _imprimir_tabela_dist(bench)
        try:
            salvar_dashboard_dist(bench, out=args.out)
        except Exception as e:
            print(f"[dist] não foi possível gerar o dashboard: {e}")
        return

    ap.print_help()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
