---
name: detector-ml-random-forest
description: Random Forest treinado sobre os 4 scores (VP/ELA/Ruído/Escala) substitui a soma de pesos; predição por upload
metadata:
  type: project
---

O classificador principal passou a ser um **Random Forest** ([Algorithm/detector_ml.py](Algorithm/detector_ml.py)), não mais a soma de pesos fixos do `detector_unificado.py`. Decisões do usuário (2026-06-09): features = **apenas os 4 scores** `[score_vp, score_ela, score_ruido, score_escala]`; e o RF **substitui** o score unificado.

- Treino offline gera `Algorithm/modelo_rf.pkl` (joblib): `python Algorithm/detector_ml.py --treinar` ou `POST /api/ml/train`. Labels Au=0, Sp=1.
- Predição por **upload**: `POST /api/ml/predict` (multipart `image`) → classe/probabilidade. `GET /api/ml/status` mostra metadados.
- `analisar_para_api` (usado por `/api/analyze`) agora usa o RF quando há `.pkl`; mantém a soma de pesos como **fallback** (campo `score_pesos`) e expõe `classificador` + bloco `ml`. Import do RF é lazy (evita import circular).

**Why:** aprender a fronteira de decisão em vez de pesos manuais.

**How to apply:** com só ~50 Au + 50 Sp e 4 features saturadas, acurácia/CV ficam ~50% (quase aleatório) — para melhorar, ampliar o dataset e/ou adicionar features brutas (ela_mean, cv_ruido, n_linhas...). Ver [[werkzeug-trunca-respostas-grandes]].
