# DeepForensics

**Detecção de adulteração em imagens** — pipeline unificado com 4 algoritmos (Vanishing Points + ELA + Ruído + Escala), pesos adaptativos por imagem e benchmark de computação paralela.

> Trabalho Interdisciplinar VI · PUC Minas · 2026
> Caio Gloria · Daniel Magalhães · Felipe Faria · João Paulo · Lucas Rodrigues

---

## Visão geral

| Componente                  | Stack                                                       |
| --------------------------- | ----------------------------------------------------------- |
| Algoritmo de detecção       | Python · OpenCV · NumPy · Pillow · matplotlib               |
| Backend HTTP                | Flask + flask-cors                                          |
| Front-end                   | HTML + CSS + JS vanilla (sem React/build tools)             |
| Computação paralela         | `multiprocessing.Pool` + benchmark forte e fraca            |
| Dataset                     | CASIA ITDE (amostra de 50 + 50 incluída em `Dataset/*/sample/`) |

## Pipeline

Para cada imagem o detector roda quatro análises em paralelo e combina os scores:

| Algoritmo            | O que mede                                            |
| -------------------- | ----------------------------------------------------- |
| **Vanishing Points** | Canny + HoughLinesP + RANSAC; divergência angular     |
| **ELA**              | Error Level Analysis com recompressão JPEG q=75       |
| **Ruído**            | Coeficiente de variação do ruído entre blocos 32×32   |
| **Escala**           | Blob detection + razão altura/perspectiva esperada    |

O score final (0–100) cai em 5 zonas: **Manipulada / Alta chance / Inconclusiva / Consistência média / Consistente**.

## Estrutura do projeto

```
DeepForensics/
├── Algorithm/
│   ├── detector_unificado.py   ← pipeline + funções de análise (CLI e API)
│   ├── app.py                   ← servidor Flask que serve o front e expõe a API REST
│   └── requirements.txt
├── Js/
│   ├── api.js                   ← cliente da API
│   ├── results.js               ← renderização dos 4 modos de resultado
│   ├── upload.js                ← UI: tabs, uploads, drag&drop, status
│   ├── charts.js                ← gráficos canvas puro
│   ├── theme-toggler.js         ← dark/light com SVG morph
│   └── scroll-animation.js      ← efeito de scroll na seção #demo
├── css/style.css
├── index.html                   ← UI principal com 4 modos
├── Fotos/                       ← imagens de exemplo (img1.jpg … img6.jpg)
└── Dataset/
    ├── Au/sample/               ← 50 imagens autênticas (CASIA)
    └── Tp/sample/               ← 50 imagens manipuladas (CASIA)
```

## Como rodar

### Pré-requisitos

- **Python 3.10+** — baixe em [python.org](https://python.org)
  - No **Windows**, marque **"Add Python to PATH"** durante a instalação
  - No **Mac**, pode instalar também via Homebrew: `brew install python`
- **Git** — [git-scm.com](https://git-scm.com)

---

### 🍎 Mac / Linux

Abra o **Terminal** e cole os comandos abaixo em sequência:

```bash
# 1. Clone o repositório e entre na pasta
git clone https://github.com/KYOgomes/TI6-DeepForensics.git
cd TI6-DeepForensics

# 2. Crie o ambiente virtual
python3 -m venv .venv

# 3. Ative o ambiente virtual
source .venv/bin/activate

# 4. Instale as dependências
pip install -r Algorithm/requirements.txt

# 5. Suba o servidor
python Algorithm/app.py
```

Abra **http://127.0.0.1:5000** no navegador.

> **⚠️ Toda vez que abrir um terminal novo**, ative o ambiente virtual antes de rodar o servidor:
> ```bash
> cd TI6-DeepForensics
> source .venv/bin/activate
> python Algorithm/app.py
> ```

---

### 🪟 Windows

Abra o **Prompt de Comando (cmd)** e cole os comandos abaixo em sequência:

```cmd
:: 1. Clone o repositório e entre na pasta
git clone https://github.com/KYOgomes/TI6-DeepForensics.git
cd TI6-DeepForensics

:: 2. Crie o ambiente virtual
python -m venv .venv

:: 3. Ative o ambiente virtual
.venv\Scripts\activate

:: 4. Instale as dependências
pip install -r Algorithm/requirements.txt

:: 5. Suba o servidor
python Algorithm/app.py
```

Abra **http://127.0.0.1:5000** no navegador.

> **⚠️ Se usar PowerShell** em vez do cmd e aparecer erro de permissão, rode antes:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> E use `.venv\Scripts\Activate.ps1` no lugar de `.venv\Scripts\activate`.

> **⚠️ Toda vez que abrir um terminal novo**, ative o ambiente virtual antes de rodar o servidor:
> ```cmd
> cd TI6-DeepForensics
> .venv\Scripts\activate
> python Algorithm/app.py
> ```

---

### 📝 Observações

- **Não use o Live Server do VS Code** — o próprio Flask já serve o front-end. Acesse sempre via **http://127.0.0.1:5000**.
- O terminal vai mostrar `(.venv)` no início da linha quando o ambiente virtual estiver ativo.
- Para parar o servidor, pressione `Ctrl + C` no terminal.

---

### Dataset completo (opcional)

O repositório inclui só uma amostra de 50+50 em `Dataset/Au/sample/` e `Dataset/Tp/sample/`. Para usar o CASIA completo, baixe o ITDE 2.0 e coloque as imagens em `Dataset/Au/` e `Dataset/Tp/`. O backend usa as pastas principais por padrão e cai automaticamente nas `sample/` se as principais estiverem vazias.

## Modos da UI

| Aba             | O que faz                                                    |
| --------------- | ------------------------------------------------------------ |
| **Imagem única** | Upload + análise completa com 8 visualizações (linhas, VP, ELA, ruído, blobs…) |
| **Lote**         | Múltiplas imagens · toggle paralelo/sequencial · tabela ordenada por score |
| **Dataset**      | Au + Sp · F1, acurácia, precisão, recall, matriz de confusão, histograma |
| **Benchmark CP** | Roda em 1/2/4/N workers · speedup, eficiência, overhead, escalabilidade fraca |

## API REST

| Método | Endpoint                | Descrição                                      |
| ------ | ----------------------- | ---------------------------------------------- |
| `GET`  | `/api/health`           | Healthcheck                                    |
| `GET`  | `/api/info`             | Limiares, CPUs, paths dos datasets             |
| `POST` | `/api/analyze`          | Imagem única (multipart: `image=<file>`)       |
| `POST` | `/api/batch`            | Lote (multipart: `images=<file>*`, `mode=seq\|par`) |
| `POST` | `/api/dataset`          | Au + Sp (multipart: `au=<file>*`, `sp=<file>*`)|
| `POST` | `/api/dataset-local`    | Avalia `Dataset/Au` + `Dataset/Tp` locais      |
| `POST` | `/api/benchmark`        | Benchmark CP em upload                         |
| `POST` | `/api/benchmark-local`  | Benchmark CP em pasta local                    |

## CLI do detector (uso direto sem servidor)

> Ative o ambiente virtual antes (`source .venv/bin/activate` no Mac / `.venv\Scripts\activate` no Windows).

```bash
# Imagem única
python Algorithm/detector_unificado.py Fotos/img1.jpg

# Dataset rotulado (com benchmark CP)
python Algorithm/detector_unificado.py --dataset --au Dataset/Au/sample --sp Dataset/Tp/sample

# Pasta sem rótulos
python Algorithm/detector_unificado.py --pasta Fotos
```

Cada modo gera um PNG de saída (`Resultado_*.jpg`, `Metricas_cp.jpg`) na raiz.
