---
name: benchmark-cp-embutido
description: Benchmark de CP roda por requisição sobre o upload do usuário (single=4 análises, lote=N imagens); modos Dataset e Benchmark ocultos na UI
metadata:
  type: project
---

Os modos "Dataset (Au + Sp)" e "Benchmark CP" foram **ocultados da UI** ([index.html](index.html): tabs removidos, painéis comentados; [Js/upload.js](Js/upload.js): blocos MODO 3/4 dentro de `if (false)`). O dataset serve só para treino do Random Forest. O benchmark de CP roda **EMBUTIDO e por requisição, sobre o que o usuário enviou** (não sobre o dataset):

- **Imagem única** → paralelizar as **4 análises** (VP/ELA/Ruído/Escala) da imagem entre workers. `benchmark_analises_api(path)` em [Algorithm/detector_unificado.py](Algorithm/detector_unificado.py); tarefa picklável `_task_analise`. `escopo='analises'`, máx. 4 workers.
- **Lote** → distribuir as **N imagens enviadas** entre workers (`benchmark_paralelo_api` nos paths do upload, via `_bench_lote` em [Algorithm/app.py](Algorithm/app.py)). `escopo='imagens'`.
- `/api/analyze` anexa `payload.benchmark`; `/api/batch` idem. Front: `renderBenchmark`+`renderBenchmarkEmbutido` ([Js/results.js](Js/results.js)) ajustam rótulo/nota conforme `escopo` e reusam `#bench-section` via `showSections`.

**Why:** o usuário pediu o benchmark relacionado às imagens enviadas, não ao dataset. Decisão: lote = 1 benchmark ao final sobre todas (não média por imagem); single = paralelizar as 4 análises.

**How to apply:** speedup vem **<1** (overhead domina): pipeline rápido (~0,03s/imagem; 4 análises ≈0,045s) e `spawn` no Windows reimporta cv2/numpy/matplotlib por worker. Single fica ~0,03–0,08 (4 tarefas minúsculas); lote encosta/passa de 1 com ~6+ imagens. Honesto (regime Amdahl/overhead). Roda a cada requisição (sem cache, pois depende do upload) → adiciona ~5–8s por análise. Ver [[detector-ml-random-forest]] e [[werkzeug-trunca-respostas-grandes]].
