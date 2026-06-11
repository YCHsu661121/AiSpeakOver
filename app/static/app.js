// ── State ──────────────────────────────────────────────────────────────────
let recognition   = null;
let isListening   = false;
let restartTimer  = null;let _micStream    = null;         // raw MediaStream (kept for Whisper re-init)
let sttMode       = 'webspeech';  // 'webspeech' | 'whisper'let activeAbort   = null;   // AbortController for current SSE fetch
let currentModel  = '';
let sourceLang    = 'zh-TW';
let targetLang    = 'en';
let isPulling     = false;  // true while auto-downloading a model
let dualMode      = false;  // dual-speaker (diarize) mode

const LANG_LABELS = {
  'zh-TW': '中文（繁體）',
  'zh-CN': '中文（简体）',
  'en':    'English',
  'ja':    '日本語',
  'ko':    '한국어',
};

// Web Speech API uses slightly different locale tags
const STT_LOCALE = {
  'zh-TW': 'zh-TW',
  'zh-CN': 'zh-CN',
  'en':    'en-US',
  'ja':    'ja-JP',
  'ko':    'ko-KR',
};

// Whisper language codes (ISO 639-1)
const WHISPER_LANG = { 'zh-TW': 'zh', 'zh-CN': 'zh', 'en': 'en', 'ja': 'ja', 'ko': 'ko' };

// Whisper VAD config
const VOL_THRESHOLD  = 3;      // RMS (0–100) below this = silence
const SILENCE_MS     = 1500;   // ms of silence → flush
const MAX_SPEECH_MS  = 10000;  // ms of continuous speech → force flush

// Whisper state
let _whisperRecorder  = null;
let _whisperChunks    = [];
let _whisperVadId     = null;   // setInterval ID
let _whisperSilTimer  = null;   // silence timeout handle
let _whisperForceTimer= null;   // max-duration force-flush handle
let _whisperHasSpeech = false;
let _whisperActive    = false;

// ── Debug log ──────────────────────────────────────────────────────────────
function dbg(msg, level = 'info') {
  const log = document.getElementById('debugLog');
  if (!log) return;
  const row  = document.createElement('div');
  row.className = 'dbg-row';
  const now  = new Date();
  const ts   = now.toTimeString().slice(0, 8) + '.' + String(now.getMilliseconds()).padStart(3, '0');
  row.innerHTML =
    `<span class="dbg-time">${ts}</span>` +
    `<span class="dbg-tag ${level}">${level.toUpperCase()}</span>` +
    `<span class="dbg-msg">${String(msg).replace(/</g, '&lt;')}</span>`;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}
function clearDebugLog() { const l = document.getElementById('debugLog'); if (l) l.innerHTML = ''; }
function toggleDebugPanel() {
  const p = document.getElementById('debugPanel');
  if (!p) return;
  p.style.display = (p.style.display === 'none') ? 'flex' : 'none';
}

// 拖曳 debug 視窗
(function initDebugDrag() {
  document.addEventListener('DOMContentLoaded', () => {
    const panel = document.getElementById('debugPanel');
    const header = panel && panel.querySelector('.debug-header');
    if (!panel || !header) return;
    let ox = 0, oy = 0, dragging = false;
    header.addEventListener('mousedown', e => {
      if (e.target.tagName === 'BUTTON') return;
      dragging = true;
      const r = panel.getBoundingClientRect();
      ox = e.clientX - r.left;
      oy = e.clientY - r.top;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
      e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      panel.style.left = (e.clientX - ox) + 'px';
      panel.style.top  = (e.clientY - oy) + 'px';
    });
    document.addEventListener('mouseup', () => { dragging = false; });
  });
})();

// Ctrl+Shift+D 鍵盤快捷鍵
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.shiftKey && e.key === 'D') {
    e.preventDefault();
    const p = document.getElementById('debugPanel');
    if (p) p.hidden = !p.hidden;
  }
});

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Force debug panel visible
  const dbgPanel = document.getElementById('debugPanel');
  if (dbgPanel) { dbgPanel.removeAttribute('hidden'); dbgPanel.style.display = 'flex'; }

  await loadConfig();
  await loadModels();
  checkOllamaHealth();   // 非阻塞，背景執行
  setupControls();
  initSpeech();
  // 行動裝置 HTTP 下 mediaDevices 可能不存在，改為手動點擊才啟動
  if (window.isSecureContext && navigator.mediaDevices) {
    await requestMicAndStart();
  } else {
    showMicPrompt();  // 直接顯示「點此授權」按鈕
  }
});

// 請求麥克風權限，成功就同時啟動音量計 + 語音辨識
function micErrMsg(e) {
  const n = e.name || '';
  if (n === 'NotFoundError' || n === 'DevicesNotFoundError')
    return '找不到麥克風裝置，請確認麥克風已插入並未被其他程式佔用';
  if (n === 'NotAllowedError' || n === 'PermissionDeniedError')
    return '麥克風權限被拒絕，請在瀏覽器網址列允許麥克風存取';
  if (n === 'NotReadableError' || n === 'TrackStartError')
    return '麥克風已被其他應用程式佔用，請關閉後再試';
  if (n === 'OverconstrainedError')
    return '麥克風不支援所要求的音訊格式';
  return '麥克風錯誤：' + (e.message || n || e);
}

async function requestMicAndStart() {
  // mediaDevices 在 HTTP 非 localhost 下不存在
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showMicPrompt();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    _micStream = stream;
    startVolMonitor(stream);
    if (sttMode === 'whisper') initWhisperMode(stream);
    else startListening();
  } catch (e) {
    dbg('getUserMedia failed: ' + e.name + ' — ' + e.message, 'error');
    showMicPrompt(e);
  }
}

function showMicPrompt(err) {
  setBadge(false, true);
  const bar = document.getElementById('statusBar');
  bar.className = 'statusbar error';

  // HTTP 非 localhost：無法使用麥克風，顯示說明
  if (!window.isSecureContext || !navigator.mediaDevices) {
    bar.textContent = '⚠️ 需要 HTTPS 才能使用麥克風，請改用 https:// 或 localhost 開啟';
    return;
  }

  // 已知錯誤（NotFoundError 等）：直接顯示說明，不顯示按鈕重試也沒意義
  if (err && err.name !== 'NotAllowedError' && err.name !== 'PermissionDeniedError') {
    bar.textContent = '⚠️ ' + micErrMsg(err);
    return;
  }

  // HTTPS / localhost：顯示授權按鈕
  bar.innerHTML = '';
  const msg = document.createTextNode('麥克風未授權——');
  const btn = document.createElement('button');
  btn.textContent = '🎙 點此授權麥克風';
  btn.style.cssText = 'background:var(--accent);color:#fff;border:none;border-radius:6px;padding:6px 18px;font-size:.85rem;cursor:pointer;margin-left:8px';
  btn.onclick = async () => {
    bar.textContent = '請求麥克風權限…';
    bar.className = 'statusbar';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      _micStream = stream;
      startVolMonitor(stream);
      if (sttMode === 'whisper') initWhisperMode(stream);
      else startListening();
    } catch (e) {
      dbg('mic retry failed: ' + e.name + ' — ' + e.message, 'error');
      bar.textContent = '⚠️ ' + micErrMsg(e);
      bar.className = 'statusbar error';
    }
  };
  bar.appendChild(msg);
  bar.appendChild(btn);
}

// ── Config / Models ────────────────────────────────────────────────────────
async function loadConfig() {
  try {
    const cfg = await fetch('/api/config').then(r => r.json());
    sourceLang   = cfg.default_source_lang || 'zh-TW';
    targetLang   = cfg.default_target_lang || 'en';
    currentModel = cfg.default_model       || '';
    sttMode      = cfg.default_stt         || 'whisper';
    dbg(`config loaded: stt=${sttMode}  model=${currentModel}  ${cfg.default_source_lang}→${cfg.default_target_lang}`, 'ok');
    document.getElementById('sourceLang').value = sourceLang;
    document.getElementById('targetLang').value = targetLang;
    document.getElementById('sttToggle').value  = sttMode;
    updateLangLabels();
  } catch (e) {
    setStatus('無法載入設定：' + e.message, 'error');
  }
}

async function loadModels() {
  try {
    const data   = await fetch('/api/models').then(r => r.json());
    const sel    = document.getElementById('modelSelect');
    sel.innerHTML = '';

    if (!data.models || data.models.length === 0) {
      sel.innerHTML = '<option value="">無可用模型</option>';
      setStatus('未偵測到已安裝模型，請選擇一個下載', 'error');
      await showModelPicker();
      return;
    }

    data.models.forEach(m => {
      const opt = new Option(m, m);
      if (m === currentModel) opt.selected = true;
      sel.appendChild(opt);
    });

    // If config model not found, fall back to first
    if (!sel.value) sel.selectedIndex = 0;
    currentModel = sel.value;
    _updateModelBadge();

    sel.addEventListener('change', () => { currentModel = sel.value; _updateModelBadge(); });
  } catch (e) {
    setStatus('無法載入模型：' + e.message, 'error');
  }
}

function _updateModelBadge() {
  let badge = document.getElementById('modelBadge');
  if (!badge) {
    badge = document.createElement('span');
    badge.id = 'modelBadge';
    badge.style.cssText = 'font-size:.7rem;color:#888;margin-left:6px;white-space:nowrap;';
    const brand = document.querySelector('.brand');
    if (brand) brand.appendChild(badge);
  }
  badge.textContent = currentModel ? `・${currentModel}` : '・未選模型';
  badge.style.color = currentModel ? '#8f8' : '#f88';
}
async function checkOllamaHealth() {
  let dot = document.getElementById('ollamaDot');
  if (!dot) {
    dot = document.createElement('span');
    dot.id = 'ollamaDot';
    dot.title = 'Ollama 連線狀態';
    dot.style.cssText = 'font-size:.75rem;margin-left:8px;cursor:pointer;';
    dot.onclick = () => checkOllamaHealth();
    const brand = document.querySelector('.brand');
    if (brand) brand.appendChild(dot);
  }
  dot.textContent = '⏳';
  try {
    const h = await fetch('/api/health').then(r => r.json());
    if (h.ollama) {
      dot.textContent = '🟢';
      dot.title = `Ollama OK · 模型: ${(h.ollama_models || []).join(', ') || '(無)'}`;
      dbg(`Ollama OK  模型: ${(h.ollama_models || []).join(', ')}`, 'ok');
    } else {
      dot.textContent = '🔴';
      dot.title = `Ollama 無法連線: ${h.ollama_error || ''}`;
      dbg(`Ollama 無法連線: ${h.ollama_error || ''}`, 'error');
      setStatus('⚠️ Ollama 無法連線，翻譯不可用', 'error');
    }
  } catch (e) {
    dot.textContent = '🔴';
    dot.title = 'Ollama 連線失敗';
    dbg('Ollama health check failed: ' + e.message, 'error');
  }
}
async function showModelPicker() {
  const overlay  = document.getElementById('modelPicker');
  const listEl   = document.getElementById('pickerList');
  const customIn = document.getElementById('customModelInput');
  const customBtn= document.getElementById('customPullBtn');

  overlay.hidden = false;

  // Load curated list
  try {
    const data = await fetch('/api/library').then(r => r.json());
    listEl.innerHTML = '';
    data.models.forEach(m => {
      const row = document.createElement('div');
      row.className = 'picker-row' + (m.default ? ' picker-row--default' : '');
      row.innerHTML = `
        <div class="picker-row-info">
          <span class="picker-name">${m.name}</span>
          <span class="picker-size">${m.size}</span>
          <span class="picker-note">${m.note}</span>
        </div>
        <button class="picker-dl-btn" data-model="${m.name}">下載</button>`;
      listEl.appendChild(row);
    });
    listEl.querySelectorAll('.picker-dl-btn').forEach(btn => {
      btn.addEventListener('click', () => startPickerPull(btn.dataset.model));
    });
  } catch (e) {
    listEl.innerHTML = '<div class="picker-loading">載入清單失敗：' + e.message + '</div>';
  }

  // Custom model input
  customBtn.onclick = () => {
    const name = customIn.value.trim();
    if (name) startPickerPull(name);
  };
  customIn.addEventListener('keydown', e => {
    if (e.key === 'Enter') { const n = customIn.value.trim(); if (n) startPickerPull(n); }
  });
}

function startPickerPull(model) {
  document.getElementById('modelPicker').hidden = true;
  autoPullModel(model);
}

async function autoPullModel(model) {
  const sel = document.getElementById('modelSelect');
  sel.disabled = true;
  isPulling = true;

  try {
    const resp = await fetch('/api/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;
        try {
          const obj = JSON.parse(raw);
          if (obj.error) {
            isPulling = false;
            setStatus('下載失敗：' + obj.error, 'error');
            sel.disabled = false;
            return;
          }
          if (obj.status === 'done') break;

          // Show progress: "pulling manifest", "downloading xx%", etc.
          let msg = '下載 ' + model + '：' + (obj.status || '');
          if (obj.total && obj.completed) {
            const pct = Math.round((obj.completed / obj.total) * 100);
            msg += ` ${pct}%`;
          }
          setStatus(msg, 'listening');
        } catch (_) { /* skip bad JSON */ }
      }
    }

    isPulling = false;
    setStatus('模型下載完成，重新載入模型清單…', 'listening');
    sel.disabled = false;
    // Re-fetch models now that pull is done
    await loadModels();
  } catch (e) {
    isPulling = false;
    setStatus('下載模型失敗：' + e.message, 'error');
    sel.disabled = false;
  }
}

// ── Language controls ──────────────────────────────────────────────────────
function setupControls() {
  document.getElementById('sourceLang').addEventListener('change', e => {
    sourceLang = e.target.value;
    updateLangLabels();
    restartListening();
  });
  document.getElementById('targetLang').addEventListener('change', e => {
    targetLang = e.target.value;
    updateLangLabels();
  });
  document.getElementById('swapBtn').addEventListener('click', () => {
    [sourceLang, targetLang] = [targetLang, sourceLang];
    document.getElementById('sourceLang').value = sourceLang;
    document.getElementById('targetLang').value = targetLang;
    updateLangLabels();
    restartListening();
  });
  document.getElementById('sttToggle').addEventListener('change', e => setSttMode(e.target.value));
  document.getElementById('modeToggle').addEventListener('click', toggleDualMode);
  document.getElementById('dualLangA').addEventListener('change', e => {
    dualSpeaker.a.lang = e.target.value;
    updateDualLabels();
  });
  document.getElementById('dualLangB').addEventListener('change', e => {
    dualSpeaker.b.lang = e.target.value;
    updateDualLabels();
  });
  document.getElementById('dualRecordToggle').addEventListener('click', toggleDualRecord);
  document.getElementById('dualReset').addEventListener('click', resetDiarization);
}

function updateLangLabels() {
  document.getElementById('srcLangLabel').textContent = LANG_LABELS[sourceLang] || sourceLang;
  document.getElementById('tgtLangLabel').textContent = LANG_LABELS[targetLang] || targetLang;
}

// ── Speech Recognition ────────────────────────────────────────────────────
function initSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setStatus('此瀏覽器不支援語音識別，請改用 Chrome 或 Edge', 'error');
    setBadge(false, true);
    return;
  }

  recognition = new SR();
  recognition.continuous     = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    isListening = true;
    setBadge(true);
    setStatus('🎙 正在監聽…', 'listening');
  };

  recognition.onend = () => {
    isListening = false;
    if (dualMode) { setBadge(false); return; }  // dual mode handles its own state
    setBadge(false);
    // Auto-restart after brief pause
    if (!restartTimer) {
      restartTimer = setTimeout(() => {
        restartTimer = null;
        startListening();
      }, 600);
    }
  };

  recognition.onerror = e => {
    if (e.error === 'no-speech') return;  // normal silence, ignore
    if (e.error === 'not-allowed') {
      if (!isPulling) showMicPrompt();
      setBadge(false, true);
      return;
    }
    if (!isPulling) setStatus('語音識別錯誤：' + e.error, 'error');
  };

let _interimTimer   = null;   // debounce timer
let _forceTimer     = null;   // max-interval forced flush (3 s)
let _lastSentText   = '';     // last text sent to translation

  recognition.onresult = e => {
    let interim = '';
    let finals  = '';

    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finals  += t;
      else                       interim += t;
    }

    // ── Dual-mode routing ─────────────────────────────────────────────────
    if (dualMode && _dualActiveSpeaker) {
      const sp = _dualActiveSpeaker;
      if (finals) {
        const speechEl = document.getElementById('dual-speech-' + sp);
        const span = document.createElement('span');
        span.className = 'sentence';
        span.textContent = finals + ' ';
        speechEl.appendChild(span);
        speechEl.scrollTop = speechEl.scrollHeight;
        const tgtLang = sp === 'a' ? dualSpeaker.b.lang : dualSpeaker.a.lang;
        dualTranslate(finals, dualSpeaker[sp].lang, tgtLang,
                      document.getElementById('dual-trans-' + sp));
      }
      return;  // don't fall through to single-mode logic
    }

    document.getElementById('interimText').textContent = interim;

    if (finals) {
      // Final: cancel timers, translate only if different from last sent
      clearTimeout(_interimTimer);
      clearTimeout(_forceTimer);
      _interimTimer = _forceTimer = null;
      appendFinal(finals);
      document.getElementById('interimText').textContent = '';
      if (finals.trim() !== _lastSentText) {
        _lastSentText = finals.trim();
        triggerTranslation(finals);
      }
    } else if (interim.trim().length > 2) {
      // Interim debounce: reset 600 ms timer
      clearTimeout(_interimTimer);
      _interimTimer = setTimeout(() => {
        _interimTimer = null;
        clearTimeout(_forceTimer);
        _forceTimer = null;
        if (interim.trim() !== _lastSentText) {
          _lastSentText = interim.trim();
          triggerTranslation(interim.trim());
        }
      }, 600);

      // Force flush every 3 s even if still speaking
      if (!_forceTimer) {
        _forceTimer = setTimeout(() => {
          _forceTimer = null;
          clearTimeout(_interimTimer);
          _interimTimer = null;
          const cur = document.getElementById('interimText').textContent.trim();
          if (cur.length > 2 && cur !== _lastSentText) {
            _lastSentText = cur;
            triggerTranslation(cur);
          }
        }, 3000);
      }
    }
  };
}

// ── Mic volume monitor (Web Audio API) ────────────────────────────────────
let _audioCtx   = null;
let _analyser   = null;
let _volRafId   = null;

function startVolMonitor(stream) {
  const canvas = document.getElementById('volCanvas');
  if (!canvas) return;
  try {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    _analyser = _audioCtx.createAnalyser();
    _analyser.fftSize = 256;
    _audioCtx.createMediaStreamSource(stream).connect(_analyser);

    const ctx = canvas.getContext('2d');
    const buf = new Uint8Array(_analyser.frequencyBinCount);
    const W = canvas.width, H = canvas.height;

    function draw() {
      _volRafId = requestAnimationFrame(draw);
      _analyser.getByteFrequencyData(buf);

      // RMS → dB
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      const db  = rms > 0 ? 20 * Math.log10(rms / 255) : -60;
      document.getElementById('volLabel').textContent = db.toFixed(1) + ' dB';

      // Bar chart
      ctx.clearRect(0, 0, W, H);
      const barW  = W / buf.length * 2;  // use first half
      const half  = Math.floor(buf.length / 2);
      for (let i = 0; i < half; i++) {
        const v = buf[i] / 255;
        const h = v * H;
        // colour: green → yellow → red
        const r = Math.min(255, Math.round(v * 2 * 255));
        const g = Math.min(255, Math.round((1 - v) * 2 * 255));
        ctx.fillStyle = `rgb(${r},${g},30)`;
        ctx.fillRect(i * barW, H - h, barW - 1, h);
      }
    }
    draw();
  } catch (e) {
    document.getElementById('volLabel').textContent = '錯誤';
  }
}

function startListening() {
  if (!recognition) return;
  recognition.lang = STT_LOCALE[sourceLang] || sourceLang;
  try { recognition.start(); } catch (_) { /* already running */ }
}

function restartListening() {
  if (isListening) {
    recognition.stop();   // onend → auto-restart with new lang
  } else {
    startListening();
  }
}

function appendFinal(text) {
  const el   = document.getElementById('finalText');
  const span = document.createElement('span');
  span.className   = 'sentence';
  span.textContent = text + ' ';
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

// ── Whisper STT (faster-whisper via backend proxy) ────────────────────────

function initWhisperMode(stream) {
  _whisperActive    = true;
  _whisperChunks    = [];
  _whisperHasSpeech = false;

  const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']
    .find(m => MediaRecorder.isTypeSupported(m)) || '';

  _whisperRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  _whisperRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) _whisperChunks.push(e.data);
  };
  _whisperRecorder.start(200);  // collect a chunk every 200 ms

  _startVad();
  setBadge(true);
  setStatus((sttMode === 'nemo' ? '🔥 NeMo' : '🤫 Whisper') + ' 監聽中…', 'listening');
}

function _startVad() {
  clearInterval(_whisperVadId);
  _whisperVadId = setInterval(() => {
    if (!_whisperActive || !_analyser) return;

    const buf = new Uint8Array(_analyser.frequencyBinCount);
    _analyser.getByteTimeDomainData(buf);

    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / buf.length) * 100;

    if (rms > VOL_THRESHOLD) {
      _whisperHasSpeech = true;
      clearTimeout(_whisperSilTimer);
      _whisperSilTimer = null;
      if (!_whisperForceTimer) {
        _whisperForceTimer = setTimeout(() => {
          _whisperForceTimer = null;
          flushWhisperChunk();
        }, MAX_SPEECH_MS);
      }
    } else if (_whisperHasSpeech && !_whisperSilTimer) {
      _whisperSilTimer = setTimeout(() => {
        _whisperSilTimer = null;
        clearTimeout(_whisperForceTimer);
        _whisperForceTimer = null;
        flushWhisperChunk();
      }, SILENCE_MS);
    }
  }, 100);
}

function _stopVad() {
  clearInterval(_whisperVadId);  _whisperVadId = null;
  clearTimeout(_whisperSilTimer);  _whisperSilTimer = null;
  clearTimeout(_whisperForceTimer); _whisperForceTimer = null;
}

async function flushWhisperChunk() {
  if (!_whisperHasSpeech || _whisperChunks.length === 0) return;

  const chunks = [..._whisperChunks];
  _whisperChunks    = [];
  _whisperHasSpeech = false;

  const mimeType = _whisperRecorder?.mimeType || 'audio/webm';
  const blob = new Blob(chunks, { type: mimeType });
  if (blob.size < 1000) { dbg(`blob too small (${blob.size}B), skip`, 'warn'); return; }

  const form = new FormData();
  form.append('audio', blob, 'audio.webm');
  // In dual mode: no language hint (Whisper auto-detects); in single: use source lang
  if (!dualMode) form.append('language', WHISPER_LANG[sourceLang] || 'zh');
  form.append('backend', sttMode === 'nemo' ? 'nemo' : 'whisper');
  if (dualMode) form.append('diarize', 'true');

  const modeLabel = sttMode === 'nemo' ? 'NeMo' : 'Whisper';
  const modeIcon  = sttMode === 'nemo' ? '🔥' : '🤫';
  dbg(`→ POST /api/transcribe  backend=${sttMode}  size=${blob.size}B  dual=${dualMode}`, 'info');
  setStatus(`⏳ ${modeLabel} 識別中…`);
  try {
    const resp = await fetch('/api/transcribe', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.error) { dbg(`transcribe error: ${data.error}`, 'error'); setStatus('識別錯誤：' + data.error, 'error'); return; }
    const text = (data.text || '').trim();
    dbg(`← transcribe  text="${text}"  speaker_id=${data.speaker_id ?? 'n/a'}`, text ? 'ok' : 'warn');

    if (dualMode) {
      if (!text) { setStatus('🟢 講者辨識中…', 'listening'); return; }
      const sp      = (data.speaker_id ?? 0) === 0 ? 'a' : 'b';
      const srcLang = dualSpeaker[sp].lang;
      const tgtLang = sp === 'a' ? dualSpeaker.b.lang : dualSpeaker.a.lang;
      dbg(`講者 ${sp.toUpperCase()}  src=${srcLang}  tgt=${tgtLang}`, 'info');
      updateSpkBadge(sp);
      const speechEl = document.getElementById('dual-speech-' + sp);
      const span = document.createElement('span');
      span.className = 'sentence';
      span.textContent = text + ' ';
      speechEl.appendChild(span);
      speechEl.scrollTop = speechEl.scrollHeight;
      if (!currentModel) {
        setStatus('⚠️ 尚未選擇翻譯模型，請在頂部下拉選單選擇', 'error');
      } else {
        dualTranslate(text, srcLang, tgtLang, document.getElementById('dual-trans-' + sp));
      }
    } else {
      if (!text) { setStatus(`${modeIcon} ${modeLabel} 監聽中…`, 'listening'); return; }
      appendFinal(text);
      document.getElementById('interimText').textContent = '';
      if (!currentModel) {
        setStatus('⚠️ 尚未選擇翻譯模型，請在頂部下拉選單選擇', 'error');
      } else if (text !== _lastSentText) {
        _lastSentText = text;
        triggerTranslation(text);
      } else {
        setStatus(`${modeIcon} ${modeLabel} 監聽中…`, 'listening');
      }
    }
  } catch (e) {
    dbg(`transcribe fetch failed: ${e.message}`, 'error');
    setStatus('識別失敗：' + e.message, 'error');
  }
}

function stopWhisperMode() {
  _whisperActive = false;
  _stopVad();
  _whisperChunks    = [];
  _whisperHasSpeech = false;
  if (_whisperRecorder && _whisperRecorder.state !== 'inactive') _whisperRecorder.stop();
  _whisperRecorder = null;
}

function setSttMode(mode) {
  sttMode = mode;
  document.getElementById('sttToggle').value = mode;
  if (mode === 'webspeech') {
    stopWhisperMode();
    if (!recognition) initSpeech();
    startListening();
  } else {
    clearTimeout(restartTimer);
    restartTimer = null;
    if (recognition) try { recognition.stop(); } catch (_) {}
    // Quick health-check so the user knows immediately if the backend isn't running
    checkSttBackend(mode).then(ok => {
      if (!ok) return;  // error already shown
      if (_micStream) initWhisperMode(_micStream);
      else setStatus('麥克風尚未連接', 'error');
    });
  }
}

async function checkSttBackend(mode) {
  try {
    const health = await fetch('/api/stt/health').then(r => r.json());
    const status = mode === 'nemo' ? (health.nemo_status || (health.nemo ? 'ready' : 'unreachable')) : (health.whisper ? 'ready' : 'unreachable');
    dbg(`stt/health ← ${mode}=${status}${health.nemo_error ? '  err:' + health.nemo_error : ''}`, health[mode] ? 'ok' : 'warn');
    if (health[mode] === false) {
      const label = mode === 'nemo' ? 'NeMo' : 'Whisper';
      if (mode === 'nemo' && health.nemo_status === 'loading') {
        setStatus('⏳ NeMo 模型載入中，請稍後再試…', 'listening');
        return false;
      }
      if (mode === 'nemo' && health.nemo_error) {
        setStatus(`⚠️ NeMo 模型載入失敗：${health.nemo_error}`, 'error');
        return false;
      }
      setStatus(
        `⚠️ ${label} 容器未啟動。` +
        (mode === 'nemo' ? ' 請以 docker compose --profile nemo up -d 重新啟動。' : ''),
        'error'
      );
      return false;
    }
  } catch (_) { /* health endpoint unreachable — let the transcribe call fail naturally */ }
  return true;
}

// ── Translation (SSE) ──────────────────────────────────────────────────────
async function triggerTranslation(text) {
  if (!text.trim()) return;
  if (!currentModel) { setStatus('⚠️ 尚未選擇翻譯模型，請在頂部下拉選單選擇', 'error'); return; }
  dbg(`→ translate  model=${currentModel}  ${sourceLang}→${targetLang}  "${text.slice(0,60)}"`, 'info');

  // Cancel any in-flight request
  if (activeAbort) activeAbort.abort();
  const ctrl    = new AbortController();
  activeAbort   = ctrl;

  const outEl   = document.getElementById('translationOut');
  const block   = document.createElement('div');
  block.className = 'trans-block';
  outEl.appendChild(block);
  outEl.scrollTop = outEl.scrollHeight;

  setStatus('⏳ 翻譯中…');

  try {
    const resp = await fetch('/api/translate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text, source_lang: sourceLang, target_lang: targetLang, model: currentModel }),
      signal:  ctrl.signal,
    });

    dbg(`← /api/translate  HTTP ${resp.status}`, resp.ok ? 'ok' : 'error');
    if (!resp.ok) {
      const errText = await resp.text();
      dbg(`translate HTTP error: ${errText}`, 'error');
      setStatus(`翻譯失敗 (HTTP ${resp.status})`, 'error');
      return;
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let gotContent = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();  // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') { dbg('translate DONE', 'ok'); break; }
        try {
          const parsed = JSON.parse(payload);
          if (parsed.error) { dbg(`translate stream error: ${parsed.error}`, 'error'); setStatus('翻譯錯誤：' + parsed.error, 'error'); break; }
          if (parsed.content) {
            if (!gotContent) dbg('translate first token received', 'ok');
            gotContent = true;
            block.textContent += parsed.content;
            outEl.scrollTop = outEl.scrollHeight;
          }
        } catch (_) {}
      }
    }

    if (!gotContent) dbg('translate: no content received from Ollama', 'warn');
    setStatus('🎙 正在監聽…', 'listening');
  } catch (e) {
    dbg(`translate error: ${e.message}`, 'error');
    if (e.name !== 'AbortError') setStatus('翻譯失敗：' + e.message, 'error');
  }
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function clearAll() {
  document.getElementById('finalText').innerHTML    = '';
  document.getElementById('interimText').textContent = '';
  document.getElementById('translationOut').innerHTML = '';
}

function copyResult() {
  const text = document.getElementById('translationOut').textContent.trim();
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => setStatus('已複製到剪貼簿', 'ok'));
}

function setBadge(active, err = false) {
  const badge = document.getElementById('micBadge');
  const label = document.getElementById('micLabel');
  badge.className = 'mic-badge' + (active ? ' listening' : err ? ' error' : '');
  label.textContent = active ? '監聽中' : err ? '無法使用' : '已暫停';
}

function setStatus(msg, type = '') {
  const bar = document.getElementById('statusBar');
  bar.textContent = msg;
  bar.className   = 'statusbar ' + type;
}

// ── Dual-speaker mode ─────────────────────────────────────────────────────

const LANG_LABELS_DUAL = {
  'zh-TW': '中文（繁體）', 'zh-CN': '中文（简体）',
  'en': 'English', 'ja': '日本語', 'ko': '한국어',
};

function toggleDualMode() {
  dualMode = !dualMode;
  document.getElementById('panelsSingle').hidden  =  dualMode;
  document.getElementById('panelsDual').hidden    = !dualMode;
  document.getElementById('langPairSingle').style.display = dualMode ? 'none' : '';
  document.getElementById('modeToggle').classList.toggle('active', dualMode);
  document.getElementById('modeToggle').textContent = dualMode ? '⇌ 單人' : '⇌ 雙向';

  if (dualMode) {
    // Dual mode requires Whisper/NeMo — auto-switch if Web Speech selected
    if (sttMode === 'webspeech') {
      setSttMode('whisper');
      document.getElementById('sttToggle').value = 'whisper';
    }
    clearTimeout(restartTimer); restartTimer = null;
    if (recognition) try { recognition.stop(); } catch (_) {}
    stopWhisperMode();
    updateDualLabels();
    setStatus('🟢 雙向模式：按「開始識別」啟動', 'listening');
  } else {
    if (_dualIsRecording) { stopWhisperMode(); _dualIsRecording = false; }
    _updateDualRecBtn();
    if (sttMode === 'whisper' || sttMode === 'nemo') {
      if (_micStream) initWhisperMode(_micStream);
    } else {
      startListening();
    }
  }
}

function _updateDualRecBtn() {
  const btn = document.getElementById('dualRecordToggle');
  if (!btn) return;
  btn.textContent = _dualIsRecording ? '⏹ 停止' : '▶ 開始識別';
  btn.classList.toggle('recording', _dualIsRecording);
}

function toggleDualRecord() {
  if (!dualMode) return;
  if (_dualIsRecording) {
    stopWhisperMode();
    _dualIsRecording = false;
    setStatus('🟢 雙向模式：按「開始識別」啟動', 'listening');
  } else {
    if (!_micStream) { setStatus('麥克風尚未連接', 'error'); return; }
    initWhisperMode(_micStream);
    _dualIsRecording = true;
    setStatus('🟢 講者辨識中…', 'listening');
  }
  _updateDualRecBtn();
}

async function resetDiarization() {
  try { await fetch('/api/diarize/reset', { method: 'POST' }); } catch (_) {}
  clearDual('a');
  clearDual('b');
  document.getElementById('spkBadgeA').classList.remove('active');
  document.getElementById('spkBadgeB').classList.remove('active');
  setStatus('🔄 講者狀態已重設', 'listening');
}

let _spkBadgeTimer = null;
function updateSpkBadge(sp) {
  const a = document.getElementById('spkBadgeA');
  const b = document.getElementById('spkBadgeB');
  a.classList.toggle('active', sp === 'a');
  b.classList.toggle('active', sp === 'b');
  clearTimeout(_spkBadgeTimer);
  _spkBadgeTimer = setTimeout(() => {
    a.classList.remove('active');
    b.classList.remove('active');
  }, 2500);
}

function updateDualLabels() {
  document.getElementById('dualTgtLabelA').textContent =
    LANG_LABELS_DUAL[dualSpeaker.b.lang] || dualSpeaker.b.lang;
  document.getElementById('dualTgtLabelB').textContent =
    LANG_LABELS_DUAL[dualSpeaker.a.lang] || dualSpeaker.a.lang;
}

async function dualTranslate(text, srcLang, tgtLang, outEl) {
  if (!text) return;
  if (!currentModel) { setStatus('⚠️ 尚未選擇翻譯模型，請在頂部下拉選單選擇', 'error'); return; }
  const block = document.createElement('div');
  block.className = 'trans-block';
  outEl.appendChild(block);
  outEl.scrollTop = outEl.scrollHeight;
  setStatus('⏳ 翻譯中…');
  try {
    const resp = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, source_lang: srcLang, target_lang: tgtLang, model: currentModel }),
    });
    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') break;
        try {
          const p = JSON.parse(payload);
          if (p.content) { block.textContent += p.content; outEl.scrollTop = outEl.scrollHeight; }
        } catch (_) {}
      }
    }
    setStatus('🟢 講者辨識中…', 'listening');
  } catch (e) {
    if (e.name !== 'AbortError') setStatus('翻譯失敗：' + e.message, 'error');
  }
}

function clearDual(sp) {
  document.getElementById('dual-speech-' + sp).innerHTML = '';
  document.getElementById('dual-trans-'  + sp).innerHTML = '';
}


// (PTT functions removed — replaced by automatic diarization)

