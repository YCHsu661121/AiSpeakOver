# 🎙 AiSpeakOver

即時語音識別 + 本地 AI 翻譯的網頁應用。對著麥克風說話，自動翻譯為目標語言。

## 架構

```
瀏覽器 (Web Speech API 或 MediaRecorder)
    │
    ▼
nginx HTTPS 反向代理  (:4445)
    │
    ▼
FastAPI 後端  (:8000)
  ├── /api/translate  →  Ollama LLM  (:11434)
  └── /api/transcribe →  faster-whisper  (:8000 內部)
```

> 麥克風需要 HTTPS，nginx 會自動產生自簽憑證。

## 功能

- 即時語音辨識，可選擇兩種模式：
  - **🌐 Web Speech**（瀏覽器原生，低延遲，需網路）
  - **🤫 Whisper 本地**（faster-whisper，完全離線，中日韓準確度更高）
- Ollama 本地 LLM 串流翻譯，不需要雲端 API
- 支援 5 種語言互譯：繁體中文、簡體中文、English、日本語、한국어
- UI 直接切換模型 / 從 Ollama library 下載模型
- Docker Compose 一鍵部署

## 語音辨識模式

| 模式 | 延遲 | 離線 | 瀏覽器限制 |
|---|---|---|---|
| 🌐 Web Speech | 極低（即時串流） | 否，送 Google/Microsoft | Chrome / Edge 專用 |
| 🤫 Whisper 本地 | 稍高（每句傳送） | 是 | 任何支援 MediaRecorder 的瀏覽器 |

切換方式：右上角下拉選單選擇 **Whisper (本地)**。Whisper 以無聲偵測（VAD）為分段依據：偵測到靜音 1.5 秒後自動送出辨識。

## 快速開始

### 前置需求

- Docker + Docker Compose
- 支援 Web Speech API 的瀏覽器（Chrome / Edge）

### 啟動

```bash
docker compose up -d
```

第一次啟動時，`ollama-init` 容器會自動拉取預設模型（`qwen2.5:1.5b`，約 1.1 GB）。

### 開啟網頁

```
https://localhost:4445
```

> 瀏覽器會警告「不安全的連線」（自簽憑證），點選「繼續前往」即可。

## 設定

編輯 `config.yaml`，重啟容器後生效：

```yaml
# Ollama 服務位址
ollama_base_url: "http://ollama:11434"

# 預設翻譯模型
default_model: "qwen2.5:1.5b"

# 預設語言對
default_source_lang: "zh-TW"   # zh-TW / zh-CN / en / ja / ko
default_target_lang: "en"
```

也可以透過環境變數覆蓋：

| 環境變數 | 說明 |
|---|---|
| `OLLAMA_BASE_URL` | Ollama 服務位址 |

### 推薦模型

| 模型 | 大小 | 說明 |
|---|---|---|
| `qwen2.5:1.5b` | 1.1 GB | 小巧快速，支援中╱日╱韓↔英 |
| `qwen2.5:3b`   | 2.0 GB | 品質較佳，多語言 |
| `qwen2.5:7b`   | 4.7 GB | 高品質翻譯 |
| `gemma3:1b`    | 0.8 GB | 超小，英文為主 |
| `llama3.2:3b`  | 2.0 GB | 平衡速度與品質 |

## GPU 加速（Whisper，選用）

修改 `docker-compose.yml` 中 `whisper` 服務，將映像由 `latest-cpu` 改為 `latest-cuda`，並加入 GPU 區塊：

```yaml
  whisper:
    image: fedirz/faster-whisper-server:latest-cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## GPU 加速（Ollama，選用）

編輯 `docker-compose.yml`，取消 `ollama` 服務的 GPU 區塊注釋：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

需要先安裝 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

## 使用外部 Ollama

若已有運行中的 Ollama 實例，修改 `config.yaml`：

```yaml
ollama_base_url: "http://<host>:11434"
```

並移除 `docker-compose.yml` 中的 `ollama` 與 `ollama-init` 服務。

## 專案結構

```
├── config.yaml          # 主要設定檔
├── docker-compose.yml   # 容器編排
├── Dockerfile           # FastAPI 映像
├── app/
│   ├── main.py          # FastAPI 後端（翻譯 API + 靜態檔案）
│   ├── requirements.txt
│   └── static/          # 前端（HTML / CSS / JS）
└── nginx/
    ├── nginx.conf        # HTTPS 反向代理設定
    └── gen-cert.sh       # 自簽憑證產生腳本
```

## API 端點

| 方法 | 路徑 | 說明 |
|---|---|---|
| `GET` | `/api/config` | 目前設定值 |
| `GET` | `/api/models` | 已安裝模型列表 |
| `GET` | `/api/library` | 推薦模型列表 |
| `POST` | `/api/pull` | 下載模型（SSE 串流進度） |
| `POST` | `/api/translate` | 翻譯文字（SSE 串流回應） |
| `POST` | `/api/transcribe` | 音訊辨識 → 文字（代理至 faster-whisper） |

## 常見問題

**Whisper 辨識速度慢**  
CPU 模式下 `small` 模型每段約 1–2 秒，可改為 `tiny` 或 `base` 加快，或啟用 GPU 加速。

**Whisper 模型尚未下載**  
第一次啟動 `whisper` 容器時會自動從 HuggingFace 下載模型（`Systran/faster-whisper-small` 約 244 MB），需稍候。

**麥克風無法使用**  
瀏覽器需要 HTTPS 才能存取麥克風，請確認使用 `https://localhost:4445`。

**Web Speech API 不支援**  
請使用 Chrome 或 Edge 瀏覽器。

**翻譯速度很慢**  
改用較小的模型（如 `qwen2.5:1.5b`），或啟用 GPU 加速。

**憑證警告**  
這是正常現象（自簽憑證），在瀏覽器警告頁面點選「進階」→「繼續前往」即可。

## 技術棧

- **後端**：FastAPI + uvicorn + httpx
- **LLM**：Ollama（本地推論）
- **前端**：原生 HTML / CSS / JavaScript，Web Speech API
- **代理**：nginx（TLS 終止 + HTTP→HTTPS 重導）
- **容器**：Docker Compose
