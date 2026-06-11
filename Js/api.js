/* ============================================================
   DeepForensics — api.js
   Cliente do backend Flask (Algorithm/app.py).
   Tudo é exposto no objeto global  window.API.
   ============================================================ */

(function () {
  const DEFAULT_BASE = '';  // mesma origem (servido pelo Flask)
  const BASE = (window.DF_API_BASE !== undefined) ? window.DF_API_BASE : DEFAULT_BASE;

  async function _json(res) {
    const ct = res.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
      const txt = await res.text();
      throw new Error(`Resposta não-JSON (${res.status}): ${txt.slice(0,200)}`);
    }
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.erro || `HTTP ${res.status}`);
    }
    return data;
  }

  async function info() {
    const r = await fetch(`${BASE}/api/info`);
    return _json(r);
  }

  async function health() {
    try {
      const r = await fetch(`${BASE}/api/health`);
      return await _json(r);
    } catch (e) {
      return { ok: false, erro: e.message };
    }
  }

  /* file: File | Blob */
  async function analyzeSingle(file, filename) {
    const fd = new FormData();
    fd.append('image', file, filename || file.name || 'imagem');
    const r = await fetch(`${BASE}/api/analyze`, { method: 'POST', body: fd });
    return _json(r);
  }

  /* files: File[] · mode: 'par' | 'seq' */
  async function analyzeBatch(files, mode = 'par') {
    const fd = new FormData();
    files.forEach(f => fd.append('images', f, f.name));
    fd.append('mode', mode);
    const r = await fetch(`${BASE}/api/batch`, { method: 'POST', body: fd });
    return _json(r);
  }

  async function evaluateDatasetUpload(auFiles, spFiles, mode = 'par') {
    const fd = new FormData();
    (auFiles || []).forEach(f => fd.append('au', f, f.name));
    (spFiles || []).forEach(f => fd.append('sp', f, f.name));
    fd.append('mode', mode);
    const r = await fetch(`${BASE}/api/dataset`, { method: 'POST', body: fd });
    return _json(r);
  }

  async function evaluateDatasetLocal(opts = {}) {
    const r = await fetch(`${BASE}/api/dataset-local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    });
    return _json(r);
  }

  async function benchmarkUpload(files) {
    const fd = new FormData();
    files.forEach(f => fd.append('images', f, f.name));
    const r = await fetch(`${BASE}/api/benchmark`, { method: 'POST', body: fd });
    return _json(r);
  }

  async function benchmarkLocal(opts = {}) {
    const r = await fetch(`${BASE}/api/benchmark-local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    });
    return _json(r);
  }

  window.API = {
    health, info,
    analyzeSingle,
    analyzeBatch,
    evaluateDatasetUpload, evaluateDatasetLocal,
    benchmarkUpload, benchmarkLocal,
  };
})();
