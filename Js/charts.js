/* ============================================================
   DeepForensics — charts.js
   Gráficos minimalistas em canvas puro (sem dependências).
   Usado para: histograma, linha de speedup/eficiência e barras.
   ============================================================ */

(function () {
  const PAD = { l: 42, r: 12, t: 18, b: 30 };

  function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0f3460';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return ctx;
  }

  function drawAxes(ctx, W, H, xLabels, yMax, yLabel) {
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t);
    ctx.lineTo(PAD.l, H - PAD.b);
    ctx.lineTo(W - PAD.r, H - PAD.b);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '10px "Space Mono", monospace';

    // y ticks
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const yv = (yMax * i) / yTicks;
      const y  = H - PAD.b - ((H - PAD.b - PAD.t) * i) / yTicks;
      ctx.fillText(yv.toFixed(yMax >= 10 ? 0 : 2).padStart(4), 4, y + 3);
      ctx.strokeStyle = 'rgba(255,255,255,0.07)';
      ctx.beginPath();
      ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
    }

    // x labels
    const innerW = W - PAD.l - PAD.r;
    xLabels.forEach((lbl, i) => {
      const x = PAD.l + (innerW * (i + 0.5)) / xLabels.length;
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.textAlign = 'center';
      ctx.fillText(String(lbl), x, H - PAD.b + 14);
    });
    ctx.textAlign = 'start';

    if (yLabel) {
      ctx.save();
      ctx.translate(12, PAD.t + (H - PAD.b - PAD.t) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.textAlign = 'center';
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
  }

  function pointX(i, n, W) {
    const innerW = W - PAD.l - PAD.r;
    return PAD.l + (innerW * (i + 0.5)) / n;
  }
  function pointY(v, yMax, H) {
    const innerH = H - PAD.b - PAD.t;
    return H - PAD.b - (innerH * v) / yMax;
  }

  /* ── Linhas (multi-série) ──
     series: [{ name, color, values }]
     xLabels: array de string (1 por ponto) */
  function drawLines(canvas, xLabels, series, opts = {}) {
    const W = canvas.width, H = canvas.height;
    const ctx = clearCanvas(canvas);
    const yMax = opts.yMax || Math.max(1.0, ...series.flatMap(s => s.values));
    drawAxes(ctx, W, H, xLabels, yMax, opts.yLabel || '');

    series.forEach(s => {
      ctx.strokeStyle = s.color;
      ctx.fillStyle   = s.color;
      ctx.lineWidth   = 2;
      if (s.dashed) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);
      ctx.beginPath();
      s.values.forEach((v, i) => {
        const x = pointX(i, xLabels.length, W);
        const y = pointY(v, yMax, H);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // pontos + labels
      s.values.forEach((v, i) => {
        const x = pointX(i, xLabels.length, W);
        const y = pointY(v, yMax, H);
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
        if (!s.dashed) {
          ctx.fillStyle = '#fff';
          ctx.font = '9px "Space Mono", monospace';
          ctx.textAlign = 'center';
          ctx.fillText(v.toFixed(2), x, y - 8);
          ctx.fillStyle = s.color;
        }
      });
    });

    // legenda
    let lx = PAD.l + 6, ly = PAD.t + 10;
    ctx.font = '11px "DM Sans", sans-serif';
    series.forEach(s => {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = 'white';
      ctx.fillText(s.name, lx + 14, ly + 1);
      lx += 14 + ctx.measureText(s.name).width + 14;
    });
  }

  /* ── Barras simples ──
     bars: [{ label, value, color? }] */
  function drawBars(canvas, bars, opts = {}) {
    const W = canvas.width, H = canvas.height;
    const ctx = clearCanvas(canvas);
    const yMax = opts.yMax || Math.max(0.001, ...bars.map(b => b.value)) * 1.15;
    drawAxes(ctx, W, H, bars.map(b => b.label), yMax, opts.yLabel || '');

    const innerW = W - PAD.l - PAD.r;
    const bw = (innerW / bars.length) * 0.6;
    bars.forEach((b, i) => {
      const x = pointX(i, bars.length, W) - bw / 2;
      const y = pointY(b.value, yMax, H);
      ctx.fillStyle = b.color || '#74b9ff';
      ctx.fillRect(x, y, bw, H - PAD.b - y);
      ctx.fillStyle = '#fff';
      ctx.font = '10px "Space Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText((opts.format || (v => v.toFixed(2)))(b.value), x + bw/2, y - 4);
    });
    ctx.textAlign = 'start';
  }

  /* ── Histograma (2 séries empilhadas / lado-a-lado) ──
     series: [{name, color, values}]  values = array de scores 0..100 */
  function drawHistogram(canvas, series, opts = {}) {
    const W = canvas.width, H = canvas.height;
    const ctx = clearCanvas(canvas);
    const bins = opts.bins || 20;
    const hist = series.map(s => {
      const h = new Array(bins).fill(0);
      s.values.forEach(v => {
        const b = Math.min(bins - 1, Math.max(0, Math.floor((v / 100) * bins)));
        h[b]++;
      });
      return h;
    });
    const yMax = Math.max(1, ...hist.flat()) * 1.1;
    const xLabels = ['0', '20', '40', '60', '80', '100'];
    drawAxes(ctx, W, H, xLabels, yMax, 'Qtde');

    // zonas
    const zones = [
      {a:0,  b:20,  c:'rgba(192,57,43,0.18)'},
      {a:20, b:40,  c:'rgba(230,126,34,0.18)'},
      {a:40, b:60,  c:'rgba(241,196,15,0.16)'},
      {a:60, b:80,  c:'rgba(41,128,185,0.18)'},
      {a:80, b:100, c:'rgba(39,174,96,0.18)'},
    ];
    const innerW = W - PAD.l - PAD.r;
    zones.forEach(z => {
      const x1 = PAD.l + (innerW * z.a / 100);
      const x2 = PAD.l + (innerW * z.b / 100);
      ctx.fillStyle = z.c;
      ctx.fillRect(x1, PAD.t, x2 - x1, H - PAD.b - PAD.t);
    });

    const bw = (innerW / bins) * 0.42;
    hist.forEach((h, sIdx) => {
      ctx.fillStyle = series[sIdx].color;
      h.forEach((v, i) => {
        const cx = PAD.l + (innerW * (i + 0.5)) / bins;
        const x = cx + (sIdx === 0 ? -bw : 0);
        const y = pointY(v, yMax, H);
        ctx.fillRect(x, y, bw, H - PAD.b - y);
      });
    });

    // legenda
    let lx = PAD.l + 6, ly = PAD.t + 10;
    ctx.font = '11px "DM Sans", sans-serif';
    series.forEach(s => {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = 'white';
      ctx.fillText(`${s.name} (${s.values.length})`, lx + 14, ly + 1);
      lx += 14 + ctx.measureText(`${s.name} (${s.values.length})`).width + 14;
    });
  }

  window.Charts = { drawLines, drawBars, drawHistogram };
})();
