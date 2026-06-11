# Memory Index

- [Detector ML Random Forest](detector-ml-random-forest.md) — RF sobre os 4 scores substitui a soma de pesos; predição por upload
- [Werkzeug trunca respostas grandes](werkzeug-trunca-respostas-grandes.md) — truncava visuais >~10MB no Windows; resolvido com waitress em app.py
- [Benchmark CP embutido](benchmark-cp-embutido.md) — benchmark roda no dataset e acompanha single/lote; modos Dataset/Benchmark ocultos na UI
- [Módulo distribuído](modulo-distribuido.md) — paralelismo entre nós WireGuard; cada nó analisa uma fatia das imagens; worker/coordenador em Algorithm/distributed.py
