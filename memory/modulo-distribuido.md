---
name: modulo-distribuido
description: Computação distribuída entre nós WireGuard — cada nó analisa uma fatia das imagens; coordenador agrega e mede speedup entre nós
metadata:
  type: project
---

Módulo de **computação distribuída** (paralelismo ENTRE máquinas, complementar à CP local entre núcleos): [Algorithm/distributed.py](Algorithm/distributed.py). Cada nó analisa uma **fatia das imagens** e devolve, por imagem, se é manipulada (`score <= L2`) ou não.

**Topologia real (confirmada via `sudo wg show` em 2026-06-11):** hub WireGuard = servidor AWS `ip-172-31-13-119` = `10.0.0.1` (endpoint público `54.207.77.13:51820`, public key `3Laur...`). Peers conectados (handshake recente): `10.0.0.2`, `10.0.0.5`, `10.0.0.6`. Peers `10.0.0.3`/`10.0.0.4` offline. **Decisão do usuário:** o **coordenador é a máquina do Felipe `10.0.0.6`** (Windows; roda `app.py`/`--coordinator`, NÃO entra como worker — o baseline T1 sequencial roda nela só como referência); os **workers são os outros 2 peers: `10.0.0.2` e `10.0.0.5`**. [Cd/nodes.json](Cd/nodes.json) lista só .2 e .5. Worker escuta em `0.0.0.0:5001`.

**Rede:** como o coordenador (.6) e os workers (.2/.5) são todos *peers* (não o hub), o tráfego .6↔.2/.5 é roteado pelo servidor .1 → exige `net.ipv4.ip_forward=1` no AWS e `AllowedIPs = 10.0.0.0/24` nos configs dos peers (o do Felipe já tem). Firewall inbound 5001 no wg0 de cada worker.

- **Worker** (cada nó): `python Algorithm/distributed.py --worker` (porta 5001). Flask leve com `GET /dist/health` e `POST /dist/process` (recebe fatia, roda `_analisar_leve` + RF via `_aplicar_rf`, devolve scores+status na ordem de entrada). Usa waitress, fallback app.run.
- **Coordenador**: `--coordinator --pasta <dir>`; lê nós de [Cd/nodes.json](Cd/nodes.json) (ou env `DF_NODES="ip,ip:porta,..."`), divide em fatias balanceadas, envia em paralelo (ThreadPool, urllib stdlib — sem dependência nova), agrega por `id` global. `--status` lista nós online.
- **Transporte**: `path` (default — só o caminho relativo ao repo; pressupõe dataset clonado em todos os nós) ou `bytes` (base64 do conteúdo; para uploads). `_rel_to_root`/`_resolve_no_root` resolvem paths entre máquinas.
- **Benchmark** `benchmark_distribuido_api`: T1 = sequencial 1 núcleo NESTE nó (mesmo baseline da CP local); depois distribui entre k=1..N nós (escalabilidade forte → speedup/eficiência/overhead). `cores_por_no` default 1 (speedup atribuível só à distribuição). Há **warm-up** antes do T1 (lê bytes de todas as imagens + 1 análise local + 1 rodada distribuída descartada) senão o cold start de disco/cv2/RF inflava o speedup (chegava a superlinear ~3x). Dashboard matplotlib: `salvar_dashboard_dist` → `Metricas_distribuido.jpg` (gitignored).
- **API** ([Algorithm/app.py](Algorithm/app.py)): `GET /api/distributed/nodes` (health), `POST /api/distributed` (multipart `images` → bytes; ou json `{dir,max,cores_por_no}` → path). `distributed.py` adicionado à lista de estáticos proibidos.

**Why:** o usuário pediu o mesmo teste de manipulação rodando com paralelismo entre os nós distribuídos, cada nó testando parte das imagens.

**UI integrada:** aba **"Distribuído"** no [index.html](index.html) (mode-tab `data-mode="distributed"`): status dos nós (chips on/off via `GET /api/distributed/nodes`), origem local/upload, máx. imagens e núcleos/nó; seção `#dist-section` com KPIs, trabalho por nó, 4 canvas (speedup/efic/tempo/overhead — gráficos usam `forte.slice(1)` p/ descartar o baseline local; tempo/overhead incluem `seq` de referência), distribuição e tabela por imagem. [Js/api.js](Js/api.js): `distributedNodes/distributedLocal/distributedUpload`. [Js/results.js](Js/results.js): `showDistResult` (reusa `renderKpis/renderDist/rowResult/Charts`). [Js/upload.js](Js/upload.js): wiring + `refreshDistNodes()` ao abrir a aba/no load. CSS: `.dist-node-chip` + canvases no bloco dark.

**How to apply:** validado local com 2 workers (5001/5002) servidos pelo app (`DF_NODES=...`): endpoints, JS (node --check) e shape da resposta OK; path e bytes OK, classificação correta (Tp→MANIPULADA), 2 nós ~1.76x speedup / efic 0.88 após warm-up. Flask/flask-cors/waitress não estavam no Python global desta máquina (deps pesadas do detector sim); instalados durante o teste. Ver [[benchmark-cp-embutido]] e [[detector-ml-random-forest]].
