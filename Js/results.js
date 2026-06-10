/* ============================================================
   DeepForensics — results.js
   Renderização dos resultados (4 modos: single / batch / dataset / benchmark).
   ============================================================ */

const SEC_SINGLE  = document.getElementById('results-section');
const SEC_BATCH   = document.getElementById('batch-section');
const SEC_DATASET = document.getElementById('dataset-section');
const SEC_BENCH   = document.getElementById('bench-section');

const ALL_SECS    = [SEC_SINGLE, SEC_BATCH, SEC_DATASET, SEC_BENCH];

function hideAllResults() {
  ALL_SECS.forEach(s => { if (s) s.style.display = 'none'; });
}
function showSection(sec) {
  hideAllResults();
  if (!sec) return;
  sec.style.display = 'block';
  sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
/* Mostra várias seções juntas (ex.: resultado + benchmark embutido) e
   rola para a primeira. */
function showSections(secs) {
  hideAllResults();
  const visiveis = secs.filter(Boolean);
  visiveis.forEach(s => { s.style.display = 'block'; });
  if (visiveis[0]) visiveis[0].scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Cores e labels das 5 zonas ── */
const ZONES = [
  { max: 20,  label: 'MANIPULADA',                  cls: 'label-danger', barCls: 'bar-red',    color: '#c0392b' },
  { max: 40,  label: 'ALTA CHANCE DE MANIPULAÇÃO',   cls: 'label-danger', barCls: 'bar-red',    color: '#e67e22' },
  { max: 60,  label: 'INCONCLUSIVA',                cls: 'label-warn',   barCls: 'bar-yellow', color: '#f1c40f' },
  { max: 80,  label: 'CONSISTÊNCIA MÉDIA',          cls: 'label-warn',   barCls: 'bar-yellow', color: '#2980b9' },
  { max: 101, label: 'CONSISTENTE',                 cls: 'label-ok',     barCls: 'bar-green',  color: '#27ae60' },
];
function zoneOf(score) {
  for (const z of ZONES) if (score <= z.max) return z;
  return ZONES[ZONES.length - 1];
}

/* ══════════════════════════════════════════════
   MODO 1 — IMAGEM ÚNICA
   ══════════════════════════════════════════════ */
function showSingleResult(filename, payload) {
  document.getElementById('filenameLabel').textContent = filename;
  document.getElementById('procTime').textContent = (payload.tempo != null) ? `${payload.tempo.toFixed(3)} s` : '—';

  const score = payload.score;
  const z = zoneOf(score);

  const numEl   = document.getElementById('scoreNum');
  const barEl   = document.getElementById('scoreBar');
  const labelEl = document.getElementById('scoreLabel');

  numEl.textContent = score.toFixed(1);
  numEl.className   = 'score-number ' + (
    score <= 40 ? 'score-red' :
    score <= 60 ? 'score-yellow' :
    score <= 80 ? 'score-yellow' : 'score-green'
  );
  barEl.style.width = Math.max(2, score) + '%';
  barEl.className   = 'score-bar ' + z.barCls;
  labelEl.textContent = z.label;
  labelEl.className   = 'score-label ' + z.cls;

  /* componentes (4 algoritmos) */
  const c = payload.componentes;
  const comps = [
    { name: 'Vanishing Points', key: 'vp',    color: '#74b9ff' },
    { name: 'ELA',              key: 'ela',   color: '#fd79a8' },
    { name: 'Ruído',            key: 'ruido', color: '#fdcb6e' },
    { name: 'Escala / blobs',   key: 'escala',color: '#a29bfe' },
  ];
  const list = document.getElementById('componentsList');
  list.innerHTML = comps.map(co => {
    const v = c[co.key];
    const pct = (v.peso * 100).toFixed(0);
    const sc  = v.score.toFixed(1);
    return `
      <div class="comp-row">
        <div class="comp-head">
          <span class="comp-name" style="color:${co.color}">${co.name}</span>
          <span class="comp-peso">peso ${pct}%</span>
          <span class="comp-score">${sc}</span>
        </div>
        <div class="comp-bar-wrap">
          <div class="comp-bar" style="width:${Math.max(2, v.score)}%;background:${co.color}"></div>
        </div>
      </div>`;
  }).join('');

  /* métricas técnicas */
  const m = payload.metricas;
  document.getElementById('vpCoords').textContent =
    m.vp_coords ? `(${m.vp_coords[0].toFixed(0)}, ${m.vp_coords[1].toFixed(0)})` : '— sem VP';
  document.getElementById('linesCount').textContent = m.n_linhas;
  document.getElementById('angularDev').textContent = m.desvio_medio != null ? `${m.desvio_medio.toFixed(2)}°` : '—';
  document.getElementById('blobsCount').textContent = m.n_blobs;
  document.getElementById('anomCount').textContent  = `${m.n_anomalias} / ${m.n_anom_graves}`;
  document.getElementById('elaStats').textContent   = `${m.ela_mean.toFixed(2)} / ${m.ela_std.toFixed(2)}`;
  document.getElementById('noiseCv').textContent    = m.cv_ruido.toFixed(3);
  document.getElementById('imgDims').textContent    = `${payload.largura} × ${payload.altura}`;

  /* visuais (base64 vindos do backend) */
  const v = payload.visuais || {};
  const setImg = (id, src) => {
    const el = document.getElementById(id);
    if (el) el.src = src || el.src;
  };
  setImg('imgPreview',  v.original);
  setImg('imgVpLines',  v.vp_linhas);
  setImg('imgEla',      v.ela);
  setImg('imgElaStd',   v.ela_std);
  setImg('imgNoise',    v.ruido);
  setImg('imgScale',    v.escala);

  /* benchmark de computação paralela embutido (medido no dataset local) */
  const bmExtra = renderBenchmarkEmbutido(payload.benchmark);
  showSections([SEC_SINGLE, bmExtra]);
}

/* ══════════════════════════════════════════════
   MODO 2 — LOTE
   ══════════════════════════════════════════════ */
function showBatchResult(payload) {
  document.getElementById('batchLabel').textContent =
    `${payload.n_imagens} imagens · ${payload.workers} worker(s) · ${payload.tempo.toFixed(2)} s`;

  /* KPIs */
  const kpiRoot = document.getElementById('batchKpis');
  const d = payload.distribuicao;
  const scoresOk    = (d['CONSISTENTE'] || 0) + (d['CONSISTENCIA MEDIA'] || 0);
  const scoresMid   = (d['INCONCLUSIVA'] || 0);
  const scoresBad   = (d['MANIPULADA'] || 0) + (d['ALTA CHANCE DE MANIPULACAO'] || 0);

  kpiRoot.innerHTML = renderKpis([
    { label: 'Imagens',         value: payload.n_imagens,                color: '#74b9ff' },
    { label: 'Tempo',           value: `${payload.tempo.toFixed(2)} s`,  color: '#fdcb6e' },
    { label: 'Workers',         value: payload.workers,                  color: '#a29bfe' },
    { label: 'Suspeitas',       value: scoresBad,                         color: '#e74c3c' },
    { label: 'Consistentes',    value: scoresOk,                          color: '#27ae60' },
    { label: 'Inconclusivas',   value: scoresMid,                         color: '#f1c40f' },
  ]);

  /* distribuição */
  document.getElementById('batchDist').innerHTML = renderDist(d, payload.n_imagens);

  /* tabela */
  const tbody = document.querySelector('#batchTable tbody');
  tbody.innerHTML = payload.resultados
    .slice()
    .sort((a, b) => a.score - b.score)
    .map((r, i) => rowResult(i + 1, r))
    .join('');

  /* benchmark de computação paralela embutido (medido no dataset local) */
  const bmExtra = renderBenchmarkEmbutido(payload.benchmark);
  showSections([SEC_BATCH, bmExtra]);
}

/* ══════════════════════════════════════════════
   MODO 3 — DATASET (Au + Sp)
   ══════════════════════════════════════════════ */
function showDatasetResult(payload) {
  document.getElementById('datasetLabel').textContent =
    `${payload.n_au} Au + ${payload.n_sp} Sp · ${payload.workers}w · ${payload.tempo.toFixed(2)} s`;

  const m = payload.metricas;
  document.getElementById('dsLimiar').textContent = payload.limiar_pos;

  /* KPIs */
  document.getElementById('dsMetricsKpis').innerHTML = renderKpis([
    { label: 'Acurácia',     value: `${(m.acuracia*100).toFixed(1)}%`, color: '#74b9ff' },
    { label: 'Precisão',     value: `${(m.precisao*100).toFixed(1)}%`, color: '#fd79a8' },
    { label: 'Recall',       value: `${(m.recall*100).toFixed(1)}%`,    color: '#fdcb6e' },
    { label: 'F1-Score',     value: `${(m.f1*100).toFixed(1)}%`,        color: '#55efc4' },
    { label: 'Tempo',        value: `${payload.tempo.toFixed(2)} s`,    color: '#a29bfe' },
    { label: 'Workers',      value: payload.workers,                    color: '#74b9ff' },
  ]);

  /* matriz de confusão */
  const cm = document.getElementById('dsConfusion');
  cm.querySelector('.cm-tn').textContent = m.tn;
  cm.querySelector('.cm-fp').textContent = m.fp;
  cm.querySelector('.cm-fn').textContent = m.fn;
  cm.querySelector('.cm-tp').textContent = m.tp;
  /* destaca o maior (intensidade visual) */
  const cmMax = Math.max(m.tn, m.fp, m.fn, m.tp) || 1;
  ['cm-tn','cm-fp','cm-fn','cm-tp'].forEach(cls => {
    const cell = cm.querySelector('.' + cls);
    const v = parseInt(cell.textContent, 10) || 0;
    const a = 0.18 + 0.75 * (v / cmMax);
    cell.style.background = `rgba(116,185,255,${a.toFixed(3)})`;
  });

  /* distribuição em 5 zonas */
  document.getElementById('dsDist').innerHTML = renderDist(payload.distribuicao, payload.n_au + payload.n_sp);

  /* histograma Au × Sp */
  Charts.drawHistogram(document.getElementById('dsHist'), [
    { name: 'Au',  color: 'rgba(46,204,113,0.85)', values: payload.au_scores },
    { name: 'Sp',  color: 'rgba(231,76,60,0.85)',  values: payload.sp_scores },
  ]);

  /* barras de métricas */
  Charts.drawBars(document.getElementById('dsMetricBars'), [
    { label: 'Acur.',  value: m.acuracia*100, color: '#74b9ff' },
    { label: 'Prec.',  value: m.precisao*100, color: '#fd79a8' },
    { label: 'Recall', value: m.recall*100,   color: '#fdcb6e' },
    { label: 'F1',     value: m.f1*100,       color: '#55efc4' },
  ], { yMax: 110, yLabel: '%', format: v => v.toFixed(1)+'%' });

  /* top 10 mais suspeitas (menor score) */
  const tbody = document.querySelector('#dsTable tbody');
  tbody.innerHTML = payload.resultados
    .slice()
    .sort((a, b) => a.score - b.score)
    .slice(0, 10)
    .map((r, i) => rowResult(i + 1, r))
    .join('');

  showSection(SEC_DATASET);
}

/* ══════════════════════════════════════════════
   MODO 4 — BENCHMARK CP
   ══════════════════════════════════════════════ */
/* Renderiza o painel de benchmark (KPIs, gráficos e tabelas) sem decidir
   visibilidade da seção — usado tanto isolado quanto embutido. */
function renderBenchmark(payload) {
  document.getElementById('bmLabel').textContent =
    `${payload.n_imagens} imagens · ${payload.cpu_count} CPUs · T1=${payload.T1.toFixed(2)}s`;

  const forte = payload.forte;
  const fraca = payload.fraca;
  const bestSpeed = Math.max(...forte.map(m => m.speedup));
  const bestEffic = Math.max(...forte.map(m => m.eficiencia));
  const bestTime  = Math.min(...forte.map(m => m.tempo));

  document.getElementById('bmKpis').innerHTML = renderKpis([
    { label: 'CPUs',             value: payload.cpu_count,         color: '#74b9ff' },
    { label: 'T1 (seq.)',         value: `${payload.T1.toFixed(2)} s`, color: '#fdcb6e' },
    { label: 'Melhor tempo',      value: `${bestTime.toFixed(2)} s`,    color: '#55efc4' },
    { label: 'Speedup máx.',      value: `${bestSpeed.toFixed(2)}×`,    color: '#a29bfe' },
    { label: 'Eficiência máx.',   value: `${(bestEffic*100).toFixed(1)}%`, color: '#fd79a8' },
    { label: 'Imagens',           value: payload.n_imagens,         color: '#74b9ff' },
  ]);

  const labels = forte.map(m => `${m.workers}w`);
  const ideal  = forte.map(m => m.workers);
  const amdahl = forte.map(m => 1 / ((1 - 0.9) + 0.9 / m.workers));
  const speeds = forte.map(m => m.speedup);

  Charts.drawLines(document.getElementById('bmSpeedup'), labels, [
    { name: 'Ideal',         color: '#bbb',     values: ideal,  dashed: true },
    { name: 'Amdahl (f=0.9)',color: '#fdcb6e',  values: amdahl, dashed: true },
    { name: 'Real',          color: '#74b9ff',  values: speeds },
  ], { yMax: Math.max(...ideal, ...amdahl, ...speeds, 1) * 1.1, yLabel: 'Speedup' });

  const effForte = forte.map(m => m.eficiencia);
  const lblFraca = fraca.map(m => `${m.workers}w`);
  const effFraca = fraca.map(m => m.eficiencia);

  Charts.drawLines(document.getElementById('bmEffic'),
    labels.length >= lblFraca.length ? labels : lblFraca,
    [
      { name: 'Forte', color: '#55efc4', values: effForte },
      { name: 'Fraca', color: '#a29bfe', values: effFraca },
    ], { yMax: 1.3, yLabel: 'Eficiência' });

  Charts.drawBars(document.getElementById('bmTime'),
    forte.map(m => ({ label: `${m.workers}w`, value: m.tempo, color: m.workers === 1 ? '#e17055' : '#74b9ff' })),
    { yLabel: 's', format: v => v.toFixed(2)+'s' });

  Charts.drawBars(document.getElementById('bmOverhead'),
    forte.map(m => ({ label: `${m.workers}w`, value: m.overhead, color: '#fdcb6e' })),
    { yLabel: 's', format: v => v.toFixed(3)+'s' });

  /* tabelas */
  document.querySelector('#bmTable tbody').innerHTML = forte.map(m => `
    <tr>
      <td>${m.workers}</td>
      <td>${m.tempo.toFixed(3)}</td>
      <td>${m.speedup.toFixed(3)}×</td>
      <td>${(m.eficiencia*100).toFixed(1)}%</td>
      <td>${m.overhead.toFixed(3)}</td>
    </tr>`).join('');

  document.querySelector('#bmTableFraca tbody').innerHTML = fraca.map(m => `
    <tr>
      <td>${m.workers}</td>
      <td>${m.workers}</td>
      <td>${m.tempo.toFixed(3)}</td>
      <td>${(m.eficiencia*100).toFixed(1)}%</td>
    </tr>`).join('');
}

/* Benchmark como seção isolada (mantido por compatibilidade). */
function showBenchResult(payload) {
  renderBenchmark(payload);
  const back = document.getElementById('bmBackWrap');
  const note = document.getElementById('bmEmbeddedNote');
  if (back) back.style.display = '';
  if (note) note.style.display = 'none';
  showSection(SEC_BENCH);
}

/* Benchmark EMBUTIDO no resultado de imagem única / lote. Recebe o objeto
   payload.benchmark vindo do backend (medido no dataset local). Devolve a
   seção a exibir junto, ou null se não houver benchmark. */
function renderBenchmarkEmbutido(bm) {
  if (!bm || bm.ok === false || !bm.forte) return null;
  renderBenchmark(bm);
  /* esconde o "Voltar" próprio (o single/lote já tem o seu) e ajusta os textos */
  const back = document.getElementById('bmBackWrap');
  const note = document.getElementById('bmEmbeddedNote');
  if (back) back.style.display = 'none';

  const analises = bm.escopo === 'analises';
  const unidade  = analises ? 'análises' : 'imagens';
  document.getElementById('bmLabel').textContent =
    `${bm.n_imagens} ${unidade} · ${bm.cpu_count} CPUs · T1=${bm.T1.toFixed(3)}s`;
  if (note) {
    note.style.display = '';
    note.textContent = analises
      ? 'Speedup medido paralelizando as 4 análises (VP, ELA, Ruído e Escala) desta imagem em 1/2/4 workers — uma análise por worker.'
      : 'Speedup medido distribuindo as imagens enviadas entre 1, 2, 4 e N workers.';
  }
  return SEC_BENCH;
}

/* ══════════════════════════════════════════════
   HELPERS DE RENDER
   ══════════════════════════════════════════════ */
function renderKpis(items) {
  return items.map(it => `
    <div class="kpi">
      <div class="kpi-value" style="color:${it.color}">${it.value}</div>
      <div class="kpi-label">${it.label}</div>
    </div>`).join('');
}

function renderDist(dist, total) {
  const order = [
    ['MANIPULADA',                   '#c0392b'],
    ['ALTA CHANCE DE MANIPULACAO',    '#e67e22'],
    ['INCONCLUSIVA',                  '#f1c40f'],
    ['CONSISTENCIA MEDIA',            '#2980b9'],
    ['CONSISTENTE',                   '#27ae60'],
  ];
  return order.map(([k, c]) => {
    const v = dist[k] || 0;
    const pct = total ? (v * 100 / total) : 0;
    return `
      <div class="dist-row">
        <span class="dist-label">${k}</span>
        <div class="dist-bar-wrap">
          <div class="dist-bar" style="width:${pct.toFixed(1)}%;background:${c}"></div>
        </div>
        <span class="dist-val">${v} (${pct.toFixed(1)}%)</span>
      </div>`;
  }).join('');
}

function rowResult(idx, r) {
  const z = zoneOf(r.score);
  return `
    <tr>
      <td>${idx}</td>
      <td class="cell-file">${escapeHtml(r.arquivo)}</td>
      <td><span class="score-pill" style="background:${z.color}">${r.score.toFixed(1)}</span></td>
      <td>${r.status}</td>
      <td>${r.score_vp.toFixed(1)}</td>
      <td>${r.score_ela.toFixed(1)}</td>
      <td>${r.score_ruido.toFixed(1)}</td>
      <td>${r.score_escala.toFixed(1)}</td>
    </tr>`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[m]));
}

/* ── Botões "voltar" ── */
document.getElementById('btnAgain')      ?.addEventListener('click', resetUI);
document.getElementById('btnBatchAgain') ?.addEventListener('click', resetUI);
document.getElementById('btnDsAgain')    ?.addEventListener('click', resetUI);
document.getElementById('btnBmAgain')    ?.addEventListener('click', resetUI);

function resetUI() {
  hideAllResults();
  window.scrollTo({ behavior: 'smooth', top: 0 });
}

/* Exporta no escopo global para upload.js usar */
window.UI = {
  showSingleResult, showBatchResult, showDatasetResult, showBenchResult,
  hideAllResults, resetUI,
};
