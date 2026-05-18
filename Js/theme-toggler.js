/* ============================================================
   DeepForensics — theme-toggler.js
   Toggle de tema (sol ↔ lua) com morph SVG via CSS transitions
   + click sound via Web Audio API.
   ============================================================ */

(function () {
  const KEY  = 'df-theme';
  const root = document.documentElement;

  /* ── 1) Inicializa tema antes do paint para evitar flash ── */
  function initialTheme() {
    const saved = localStorage.getItem(KEY);
    if (saved === 'dark' || saved === 'light') return saved;
    try {
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
    } catch (_) {}
    return 'light';
  }
  root.setAttribute('data-theme', initialTheme());

  /* ── 2) Setup do botão depois do DOM pronto ── */
  function setup() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;

    const getIsDark = () => root.getAttribute('data-theme') === 'dark';

    btn.addEventListener('click', () => {
      const next = getIsDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem(KEY, next);
      tick();
      btn.setAttribute('aria-pressed', next === 'dark' ? 'true' : 'false');
    });
    btn.setAttribute('aria-pressed', getIsDark() ? 'true' : 'false');

    /* sincroniza com mudanças do SO se o usuário não fixou um tema */
    try {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      mq.addEventListener('change', e => {
        if (localStorage.getItem(KEY)) return; // usuário fixou — respeitar
        root.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      });
    } catch (_) {}
  }

  /* ── 3) Áudio: clique curto (3.4 kHz com envelope (1-t)^3) ── */
  let _ctx = null, _buf = null, _last = 0;
  function ensureCtx() {
    if (!_ctx) _ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (_ctx.state === 'suspended') _ctx.resume();
    return _ctx;
  }
  function ensureBuf(ac) {
    if (_buf && _buf.sampleRate === ac.sampleRate) return _buf;
    const rate = ac.sampleRate;
    const len  = Math.floor(rate * 0.006);
    const buf  = ac.createBuffer(1, len, rate);
    const ch   = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
      const t = i / len;
      const sine  = Math.sin(2 * Math.PI * 3400 * t);
      const noise = Math.random() * 2 - 1;
      ch[i] = (sine * 0.6 + noise * 0.4) * Math.pow(1 - t, 3);
    }
    _buf = buf;
    return buf;
  }
  function tick() {
    const now = performance.now();
    if (now - _last < 80) return;
    _last = now;
    try {
      const ac  = ensureCtx();
      const buf = ensureBuf(ac);
      const src = ac.createBufferSource();
      const g   = ac.createGain();
      src.buffer = buf;
      g.gain.value = 0.08;
      src.connect(g);
      g.connect(ac.destination);
      src.start();
    } catch (_) { /* silent */ }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setup();
  }
})();
