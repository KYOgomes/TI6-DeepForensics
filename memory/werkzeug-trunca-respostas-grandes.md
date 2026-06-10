---
name: werkzeug-trunca-respostas-grandes
description: Servidor dev do Werkzeug truncava respostas >~9-10MB no Windows — RESOLVIDO usando waitress em app.py
metadata:
  type: project
---

O servidor de desenvolvimento do Werkzeug (`app.run`) **truncava respostas grandes** no Windows de forma intermitente: o `Content-Length` vinha correto mas só parte do corpo chegava ao cliente. Reproduzia em `/api/analyze` com imagem grande — os 8 visuais base64 passam de ~10MB e a resposta chegava cortada no meio de um base64, virando JSON inválido (flaky: às vezes a mesma imagem de ~1MB passava, às vezes cortava).

**RESOLVIDO:** [Algorithm/app.py](Algorithm/app.py) agora serve via **waitress** (servidor WSGI real) no `__main__`, com fallback para `app.run` se waitress não estiver instalado. `waitress>=3.0` está no requirements. Com waitress, respostas de 10.6MB chegam íntegras.

**Why:** limitação conhecida do socket send do Werkzeug dev server com corpos grandes; não tinha relação com o pipeline nem com o classificador.

**How to apply:** rode `python Algorithm/app.py` normalmente (usa waitress). Se reaparecer truncamento, confirme que waitress está instalado. Ver [[detector-ml-random-forest]] e [[benchmark-cp-embutido]].
