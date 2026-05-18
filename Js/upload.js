/* ============================================================
   DeepForensics — upload.js
   Lógica de UI: tabs, uploads, chamadas à API.
   ============================================================ */

(function () {

  /* ── Status do backend ── */
  const statusEl = document.getElementById('apiStatus');
  const statusDot = statusEl.querySelector('.api-status-dot');
  const statusTxt = statusEl.querySelector('.api-status-text');

  API.health().then(r => {
    if (r && r.ok !== false) {
      statusDot.style.background = '#27ae60';
      statusTxt.textContent = 'online';
      return API.info().then(info => {
        const cpu = document.getElementById('cpuMax');
        if (cpu && info.cpu_count) cpu.textContent = info.cpu_count;
      });
    }
    throw new Error('offline');
  }).catch(() => {
    statusDot.style.background = '#e74c3c';
    statusTxt.textContent = 'backend offline';
  });

  /* ══════════════════════════════════════
     Tabs de modo
     ══════════════════════════════════════ */
  const tabs   = document.querySelectorAll('[data-mode]');
  const panels = document.querySelectorAll('[data-mode-panel]');
  function activateMode(mode) {
    tabs.forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
    panels.forEach(p => p.classList.toggle('active', p.dataset.modePanel === mode));
    UI.hideAllResults();
  }
  tabs.forEach(t => t.addEventListener('click', () => activateMode(t.dataset.mode)));
  document.querySelectorAll('[data-mode-link]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      activateMode(a.dataset.modeLink);
      document.getElementById('hero').scrollIntoView({ behavior: 'smooth' });
    });
  });

  /* ══════════════════════════════════════
     Loading overlay (com etapas)
     ══════════════════════════════════════ */
  const loadingEl    = document.getElementById('loadingOverlay');
  const loadingText  = document.getElementById('loadingText');
  const stepsEls     = document.querySelectorAll('#loadingSteps .loading-step');

  function showLoading(text) {
    loadingText.textContent = text || 'Processando…';
    stepsEls.forEach(s => s.classList.remove('active', 'done'));
    let i = 0;
    function tick() {
      if (i > 0) { stepsEls[i-1].classList.remove('active'); stepsEls[i-1].classList.add('done'); }
      if (i < stepsEls.length) { stepsEls[i].classList.add('active'); i++; loadingEl._tick = setTimeout(tick, 350); }
    }
    loadingEl.classList.add('active');
    tick();
  }
  function hideLoading() {
    clearTimeout(loadingEl._tick);
    stepsEls.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    setTimeout(() => loadingEl.classList.remove('active'), 120);
  }

  function fail(err) {
    hideLoading();
    console.error(err);
    alert('Falha: ' + (err.message || err));
  }

  /* ══════════════════════════════════════
     MODO 1 — IMAGEM ÚNICA
     ══════════════════════════════════════ */
  const uploadCard = document.getElementById('uploadCard');
  const btnUpload  = document.getElementById('btnUpload');
  const btnExample = document.getElementById('btnExample');
  const fileInput  = document.getElementById('fileInput');

  const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
  const MAX_SIZE_MB   = 20;

  const previewModal    = document.getElementById('previewModal');
  const previewImage    = document.getElementById('previewImage');
  const previewName     = document.getElementById('previewName');
  const btnPreviewOk    = document.getElementById('btnPreviewOk');
  const btnPreviewClose = document.getElementById('btnPreviewClose');

  let pendingFile = null;
  let pendingName = null;

  uploadCard.addEventListener('dragover', e => { e.preventDefault(); uploadCard.classList.add('dragover'); });
  uploadCard.addEventListener('dragleave', () => uploadCard.classList.remove('dragover'));
  uploadCard.addEventListener('drop', e => {
    e.preventDefault(); uploadCard.classList.remove('dragover');
    const f = e.dataTransfer.files[0]; if (f) handleSingleFile(f);
  });
  btnUpload .addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
  uploadCard.addEventListener('click', () => fileInput.click());
  fileInput .addEventListener('change', () => { if (fileInput.files[0]) handleSingleFile(fileInput.files[0]); });

  /* picker de exemplos */
  btnExample.addEventListener('click', e => {
    e.stopPropagation();
    document.getElementById('examplePicker').classList.toggle('active');
  });
  document.addEventListener('click', e => {
    const picker = document.getElementById('examplePicker');
    if (picker && !picker.contains(e.target) && e.target !== btnExample && !btnExample.contains(e.target)) {
      picker.classList.remove('active');
    }
  });

  function handleSingleFile(file) {
    if (!ALLOWED_TYPES.includes(file.type)) { alert('Use JPG ou PNG.'); return; }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) { alert(`Máximo ${MAX_SIZE_MB}MB.`); return; }
    pendingFile = file;
    const reader = new FileReader();
    reader.onload = e => openPreviewModal(e.target.result, file.name);
    reader.readAsDataURL(file);
  }

  window.loadExampleImage = function (path, label) {
    fetch(path)
      .then(res => { if (!res.ok) throw new Error('Exemplo não encontrado'); return res.blob(); })
      .then(blob => {
        const file = new File([blob], label, { type: blob.type || 'image/jpeg' });
        pendingFile = file;
        const reader = new FileReader();
        reader.onload = e => openPreviewModal(e.target.result, label);
        reader.readAsDataURL(blob);
      })
      .catch(() => alert('Imagem de exemplo não encontrada.'));
  };

  function openPreviewModal(dataUrl, name) {
    pendingName = name;
    previewImage.src = dataUrl;
    previewName.textContent = name;
    previewModal.classList.add('active');
  }
  function closePreviewModal() {
    previewModal.classList.remove('active');
    previewImage.src = ''; fileInput.value = '';
  }
  btnPreviewClose.addEventListener('click', closePreviewModal);
  previewModal.addEventListener('click', e => { if (e.target === previewModal) closePreviewModal(); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && previewModal.classList.contains('active')) closePreviewModal();
  });

  btnPreviewOk.addEventListener('click', () => {
    const file = pendingFile;
    const name = pendingName;
    closePreviewModal();
    if (!file) return;
    showLoading('Analisando imagem…');
    API.analyzeSingle(file, name)
      .then(payload => { hideLoading(); UI.showSingleResult(name, payload); })
      .catch(fail);
  });

  /* ══════════════════════════════════════
     MODO 2 — LOTE
     ══════════════════════════════════════ */
  const batchInput   = document.getElementById('batchInput');
  const btnBatchPick = document.getElementById('btnBatchPick');
  const btnBatchRun  = document.getElementById('btnBatchRun');
  const batchHint    = document.getElementById('batchHint');
  let   batchFiles   = [];
  let   batchMode    = 'par';

  document.querySelectorAll('[data-batch-mode]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-batch-mode]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      batchMode = b.dataset.batchMode;
    });
  });

  btnBatchPick.addEventListener('click', () => batchInput.click());
  batchInput.addEventListener('change', () => {
    batchFiles = Array.from(batchInput.files || []);
    batchHint.textContent = batchFiles.length
      ? `${batchFiles.length} imagem(ns) selecionada(s)`
      : 'Nenhuma imagem selecionada';
    btnBatchRun.disabled = batchFiles.length === 0;
  });

  btnBatchRun.addEventListener('click', () => {
    if (!batchFiles.length) return;
    showLoading(`Processando lote (${batchMode === 'par' ? 'paralelo' : 'sequencial'})…`);
    API.analyzeBatch(batchFiles, batchMode)
      .then(payload => { hideLoading(); UI.showBatchResult(payload); })
      .catch(fail);
  });

  /* ══════════════════════════════════════
     MODO 3 — DATASET
     ══════════════════════════════════════ */
  const auInput    = document.getElementById('auInput');
  const spInput    = document.getElementById('spInput');
  const btnAuPick  = document.getElementById('btnAuPick');
  const btnSpPick  = document.getElementById('btnSpPick');
  const auHint     = document.getElementById('auHint');
  const spHint     = document.getElementById('spHint');
  const btnDsRun   = document.getElementById('btnDatasetRun');
  const datasetUploads = document.getElementById('datasetUploads');
  const dsLimitWrap    = document.getElementById('dsLimitWrap');
  const dsLimit        = document.getElementById('dsLimit');

  let dsSource = 'local';
  let dsMode   = 'par';
  let auFiles = []; let spFiles = [];

  document.querySelectorAll('[data-ds-source]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-ds-source]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      dsSource = b.dataset.dsSource;
      datasetUploads.style.display = dsSource === 'upload' ? 'flex' : 'none';
      dsLimitWrap.style.display    = dsSource === 'local'  ? 'flex' : 'none';
    });
  });
  document.querySelectorAll('[data-ds-mode]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-ds-mode]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      dsMode = b.dataset.dsMode;
    });
  });

  btnAuPick.addEventListener('click', () => auInput.click());
  btnSpPick.addEventListener('click', () => spInput.click());
  auInput.addEventListener('change', () => {
    auFiles = Array.from(auInput.files || []);
    auHint.textContent = auFiles.length ? `${auFiles.length} arquivo(s)` : 'Nenhuma imagem';
  });
  spInput.addEventListener('change', () => {
    spFiles = Array.from(spInput.files || []);
    spHint.textContent = spFiles.length ? `${spFiles.length} arquivo(s)` : 'Nenhuma imagem';
  });

  btnDsRun.addEventListener('click', () => {
    showLoading('Avaliando dataset…');
    if (dsSource === 'local') {
      API.evaluateDatasetLocal({ max_per_class: parseInt(dsLimit.value, 10) || 30, mode: dsMode })
        .then(payload => { hideLoading(); UI.showDatasetResult(payload); })
        .catch(fail);
    } else {
      if (!auFiles.length && !spFiles.length) { hideLoading(); alert('Selecione imagens em Au e/ou Sp'); return; }
      API.evaluateDatasetUpload(auFiles, spFiles, dsMode)
        .then(payload => { hideLoading(); UI.showDatasetResult(payload); })
        .catch(fail);
    }
  });

  /* ══════════════════════════════════════
     MODO 4 — BENCHMARK CP
     ══════════════════════════════════════ */
  const bmInput      = document.getElementById('bmInput');
  const btnBmPick    = document.getElementById('btnBmPick');
  const btnBmRun     = document.getElementById('btnBmRun');
  const bmHint       = document.getElementById('bmHint');
  const bmLimitWrap  = document.getElementById('bmLimitWrap');
  const bmUploadWrap = document.getElementById('bmUploadWrap');
  const bmLimit      = document.getElementById('bmLimit');

  let bmSource = 'local';
  let bmFiles  = [];

  document.querySelectorAll('[data-bm-source]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-bm-source]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      bmSource = b.dataset.bmSource;
      bmUploadWrap.style.display = bmSource === 'upload' ? 'block' : 'none';
      bmLimitWrap .style.display = bmSource === 'local'  ? 'flex'  : 'none';
    });
  });

  btnBmPick.addEventListener('click', () => bmInput.click());
  bmInput.addEventListener('change', () => {
    bmFiles = Array.from(bmInput.files || []);
    bmHint.textContent = bmFiles.length ? `${bmFiles.length} arquivo(s)` : 'Nenhuma imagem';
  });

  btnBmRun.addEventListener('click', () => {
    showLoading('Rodando benchmark CP… (pode levar 1–2 min)');
    if (bmSource === 'local') {
      API.benchmarkLocal({ max: parseInt(bmLimit.value, 10) || 20 })
        .then(payload => { hideLoading(); UI.showBenchResult(payload); })
        .catch(fail);
    } else {
      if (!bmFiles.length) { hideLoading(); alert('Selecione imagens'); return; }
      API.benchmarkUpload(bmFiles)
        .then(payload => { hideLoading(); UI.showBenchResult(payload); })
        .catch(fail);
    }
  });

})();
