/* ============================================================
   DeepForensics — scroll-animation.js
   Aplica rotateX/scale/translateY conforme o usuário rola um
   container marcado com  data-scroll-anim.
   ============================================================ */

(function () {
  const els = document.querySelectorAll('[data-scroll-anim]');
  if (!els.length) return;

  const targets = Array.from(els).map(c => ({
    container: c,
    header   : c.querySelector('.scroll-anim-header'),
    card     : c.querySelector('.scroll-anim-card'),
  })).filter(t => t.card);
  if (!targets.length) return;

  let mobile = window.innerWidth <= 768;
  window.addEventListener('resize', () => { mobile = window.innerWidth <= 768; update(); });

  function lerp(a, b, t) { return a + (b - a) * t; }

  function progressFor(rect, vh) {
    /* 0 → topo do container chegando na base da viewport
       1 → topo passou bem além do topo da viewport */
    const total = rect.height + vh;
    const p = (vh - rect.top) / total;
    return Math.min(1, Math.max(0, p));
  }

  function update() {
    const vh = window.innerHeight;
    for (const t of targets) {
      const rect = t.container.getBoundingClientRect();
      /* só atualiza se elemento está visível ou próximo */
      if (rect.bottom < -vh || rect.top > vh * 2) continue;

      const p = progressFor(rect, vh);

      const scaleFrom = mobile ? 0.7  : 1.05;
      const scaleTo   = mobile ? 0.9  : 1.0;
      const rotate    = lerp(20, 0, p);
      const scale     = lerp(scaleFrom, scaleTo, p);
      const trans     = lerp(0, -100, p);

      t.card.style.transform = `rotateX(${rotate.toFixed(2)}deg) scale(${scale.toFixed(3)})`;
      if (t.header) t.header.style.transform = `translateY(${trans.toFixed(1)}px)`;
    }
  }

  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { update(); ticking = false; });
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  document.addEventListener('DOMContentLoaded', update);
  /* primeira pintada */
  update();
})();
