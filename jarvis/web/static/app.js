/**
 * Jarvis Web GUI Controller (Dual-Engine Voice: Real-Time Web Speech + 16kHz PCM WAV Fallback)
 */

let chatHistory = [];
let currentLang = 'pt-BR'; // 'pt-BR' or 'en-US'
let isRecording = false;

// TTS (Text-to-Speech) State
let ttsEnabled = true;
let isSpeaking = false;

// Web Speech API instances
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let finalSpeechTranscript = '';

// Web Audio API fallback instances
let audioContext = null;
let mediaStream = null;
let audioInput = null;
let scriptProcessor = null;
let recordedPcmBuffers = [];
let recordingSampleRate = 16000;

document.addEventListener('DOMContentLoaded', () => {
  initUI();
  initLanguageSelector();
  initSpeechRecognition();
  initTTS();
  checkHealth();
  refreshDemands();
  loadDemandProjects();
  loadMCPTools();
  refreshTelemetry();
  setInterval(checkHealth, 30000);
  setInterval(refreshTelemetry, 15000);
});

function initUI() {
  const userInput = document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');
  const btnMic = document.getElementById('btn-mic');
  const btnStopRecording = document.getElementById('btn-stop-recording');
  const btnTts = document.getElementById('tts-toggle-btn');

  // Auto-grow textarea
  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 128) + 'px';
  });

  // Enter to send, Shift+Enter for newline
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  btnSend.addEventListener('click', () => sendMessage());
  btnMic.addEventListener('click', toggleVoiceDictation);
  btnStopRecording.addEventListener('click', stopVoiceDictation);
  if (btnTts) {
    btnTts.addEventListener('click', toggleTTS);
  }

  // Tabs
  document.getElementById('tab-btn-darkhub').addEventListener('click', () => switchTab('darkhub'));
  document.getElementById('tab-btn-mcp').addEventListener('click', () => switchTab('mcp'));
  const btnTelemetry = document.getElementById('tab-btn-telemetry');
  if (btnTelemetry) {
    btnTelemetry.addEventListener('click', () => switchTab('telemetry'));
  }

  // Demand Form
  document.getElementById('form-demand').addEventListener('submit', handleDemandSubmit);
}

function initLanguageSelector() {
  const btnPt = document.getElementById('lang-btn-pt');
  const btnEn = document.getElementById('lang-btn-en');

  btnPt.addEventListener('click', () => setLanguage('pt-BR'));
  btnEn.addEventListener('click', () => setLanguage('en-US'));
}

function setLanguage(lang) {
  currentLang = lang;
  const btnPt = document.getElementById('lang-btn-pt');
  const btnEn = document.getElementById('lang-btn-en');

  if (lang === 'pt-BR') {
    btnPt.className = 'px-2 py-1 rounded bg-emerald-600 text-white font-semibold transition';
    btnEn.className = 'px-2 py-1 rounded text-slate-400 hover:text-white transition';
  } else {
    btnEn.className = 'px-2 py-1 rounded bg-emerald-600 text-white font-semibold transition';
    btnPt.className = 'px-2 py-1 rounded text-slate-400 hover:text-white transition';
  }

  if (recognition) {
    recognition.lang = currentLang;
  }

  const indicator = document.getElementById('transcription-indicator');
  indicator.innerText = `Idioma de voz: ${lang === 'pt-BR' ? 'Português (Brasil)' : 'English (US)'}`;
  setTimeout(() => { indicator.innerText = ''; }, 2500);
}

function initSpeechRecognition() {
  if (!SpeechRecognition) {
    console.warn('Web Speech API not supported natively in this browser; falling back to Web Audio PCM WAV.');
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = currentLang;

  recognition.onstart = () => {
    isRecording = true;
    finalSpeechTranscript = '';
    document.getElementById('recording-banner').classList.remove('hidden');
    document.getElementById('btn-mic').classList.add('bg-red-600', 'text-white', 'recording-pulse');
    document.getElementById('transcription-indicator').innerText = `🎙️ Ouvindo em tempo real (${currentLang})...`;
  };

  recognition.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalSpeechTranscript += event.results[i][0].transcript + ' ';
      } else {
        interim += event.results[i][0].transcript;
      }
    }

    const fullLiveText = (finalSpeechTranscript + interim).trim();
    const userInput = document.getElementById('user-input');
    userInput.value = fullLiveText;
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 128) + 'px';
  };

  recognition.onerror = (event) => {
    console.warn('SpeechRecognition error:', event.error);
    if (event.error === 'not-allowed') {
      alert('Permissão de microfone negada. Por favor, autorize o microfone no navegador.');
      stopVoiceDictation();
    } else if (event.error === 'network') {
      console.info('Web Speech network issue; switching to local PCM WAV fallback.');
      stopVoiceDictation();
      startWebAudioRecordingFallback();
    }
  };

  recognition.onend = () => {
    if (isRecording) {
      // If still supposed to be recording, restart seamlessly
      try {
        recognition.start();
      } catch (e) {
        stopVoiceDictation();
      }
    } else {
      stopVoiceDictation();
    }
  };
}

function toggleVoiceDictation() {
  if (isRecording) {
    stopVoiceDictation();
  } else {
    startVoiceDictation();
  }
}

function startVoiceDictation() {
  finalSpeechTranscript = document.getElementById('user-input').value.trim();
  if (finalSpeechTranscript) finalSpeechTranscript += ' ';

  if (SpeechRecognition && recognition) {
    try {
      recognition.lang = currentLang;
      recognition.start();
      return;
    } catch (err) {
      console.warn('Could not start Web Speech, falling back to Web Audio WAV:', err);
    }
  }

  // Fallback to Web Audio WAV encoder
  startWebAudioRecordingFallback();
}

function stopVoiceDictation() {
  isRecording = false;

  if (recognition) {
    try { recognition.stop(); } catch (e) {}
  }

  if (mediaStream) {
    stopWebAudioRecordingFallback();
  }

  document.getElementById('recording-banner').classList.add('hidden');
  document.getElementById('btn-mic').classList.remove('bg-red-600', 'text-white', 'recording-pulse');
  document.getElementById('transcription-indicator').innerText = '';
}

// ==============================================================================
// Web Audio API (PCM 16kHz WAV Fallback - No FFmpeg required on server)
// ==============================================================================

async function startWebAudioRecordingFallback() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    audioInput = audioContext.createMediaStreamSource(mediaStream);
    
    // Script processor for capturing PCM float32 samples
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    recordedPcmBuffers = [];

    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const channelData = e.inputBuffer.getChannelData(0);
      recordedPcmBuffers.push(new Float32Array(channelData));
    };

    audioInput.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    isRecording = true;
    document.getElementById('recording-banner').classList.remove('hidden');
    document.getElementById('btn-mic').classList.add('bg-red-600', 'text-white', 'recording-pulse');
    document.getElementById('transcription-indicator').innerText = '🎙️ Gravando áudio WAV PCM...';
  } catch (err) {
    alert(`Erro ao acessar microfone: ${err.message}`);
    stopVoiceDictation();
  }
}

async function stopWebAudioRecordingFallback() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (audioInput) {
    audioInput.disconnect();
    audioInput = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  if (recordedPcmBuffers.length === 0) return;

  // Flatten Float32Array buffers
  let totalLength = recordedPcmBuffers.reduce((acc, b) => acc + b.length, 0);
  let mergedPcm = new Float32Array(totalLength);
  let offset = 0;
  for (let b of recordedPcmBuffers) {
    mergedPcm.set(b, offset);
    offset += b.length;
  }

  // Encode as standard 16-bit Mono 16kHz WAV
  const wavBlob = encode16BitWav(mergedPcm, 16000);
  recordedPcmBuffers = [];

  // Upload WAV directly to backend
  uploadWavAudio(wavBlob);
}

function encode16BitWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw PCM) */
  view.setUint16(20, 1, true);
  /* channel count (mono) */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  // Write 16-bit PCM samples
  let index = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    index += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

async function uploadWavAudio(wavBlob) {
  const indicator = document.getElementById('transcription-indicator');
  indicator.innerText = 'Transcrevendo áudio WAV...';

  const formData = new FormData();
  formData.append('file', wavBlob, 'recording.wav');
  formData.append('language', currentLang === 'en-US' ? 'en' : 'pt');

  try {
    const res = await fetch(`/api/audio/transcribe?language=${currentLang === 'en-US' ? 'en' : 'pt'}`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.text) {
      indicator.innerText = `🎙️ [${data.engine_used}]: "${data.text.substring(0, 45)}..."`;
      const input = document.getElementById('user-input');
      input.value = (input.value ? input.value + ' ' : '') + data.text;
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 128) + 'px';
    } else {
      indicator.innerText = `⚠️ ${data.fallback_reason || 'Nenhum áudio detectado'}`;
    }
  } catch (err) {
    indicator.innerText = `⚠️ Falha na transcrição: ${err.message}`;
  } finally {
    setTimeout(() => { indicator.innerText = ''; }, 6000);
  }
}

// ==============================================================================
// Chat & Messaging Logic
// ==============================================================================

async function sendMessage(overrideText) {
  const userInput = document.getElementById('user-input');
  const text = (overrideText || userInput.value).trim();
  if (!text) return;

  userInput.value = '';
  userInput.style.height = 'auto';

  // Render user message bubble
  appendMessage('user', text);

  // Get selected model & provider
  const selectorVal = document.getElementById('model-selector').value;
  const [model, provider] = selectorVal.split('|');

  // Loading indicator bubble
  const loadingId = 'loading-' + Date.now();
  appendLoadingBubble(loadingId);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: chatHistory,
        model: model,
        provider: provider,
      }),
    });

    removeLoadingBubble(loadingId);

    if (!res.ok) {
      const err = await res.json();
      appendMessage('assistant', `⚠️ Erro ao processar resposta: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    appendMessage('assistant', data.response_text, {
      model: data.model_used,
      provider: data.provider_used,
      tools: data.tools_executed,
      latency: data.latency_ms,
      tokens_prompt: data.tokens_prompt,
      tokens_completion: data.tokens_completion,
      cost_usd: data.cost_usd,
    });

    // Update history
    chatHistory.push({ role: 'user', content: text });
    chatHistory.push({ role: 'assistant', content: data.response_text });

    // Refresh telemetry immediately after interaction
    refreshTelemetry();

    // Speak concise executive summary in Portuguese (Dual-Channel Output)
    speakResponse(data.response_text);

  } catch (err) {
    removeLoadingBubble(loadingId);
    appendMessage('assistant', `⚠️ Falha de comunicação com o servidor: ${err.message}`);
  }
}

function appendMessage(role, text, meta) {
  const stream = document.getElementById('messages-stream');
  const msgDiv = document.createElement('div');
  const isUser = role === 'user';

  msgDiv.className = `flex gap-3 max-w-3xl mx-auto ${isUser ? 'justify-end' : 'justify-start'}`;

  const formattedContent = marked.parse(text);

  let metaBadge = '';
  if (meta && !isUser) {
    const toolsExecutedHtml = (meta.tools && meta.tools.length > 0)
      ? meta.tools.map(t => `<span class="px-1.5 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-800 text-[10px] font-mono">⚡ ${t.tool}</span>`).join(' ')
      : '';
    const totalTokens = (meta.tokens_prompt || 0) + (meta.tokens_completion || 0);
    const tokensHtml = totalTokens > 0
      ? `<span title="Prompt: ${meta.tokens_prompt || 0} | Completion: ${meta.tokens_completion || 0}">• 🔤 ${totalTokens} tok</span>`
      : '';
    let costText = '';
    if (meta.cost_usd !== undefined && meta.cost_usd !== null) {
      costText = meta.cost_usd === 0 ? '$0.00 (Local)' : `$${meta.cost_usd.toFixed(4)}`;
    }
    const costHtml = costText ? `<span class="text-emerald-400 font-semibold">• 💰 ${costText}</span>` : '';

    metaBadge = `
      <div class="flex flex-wrap items-center gap-2 mt-2 pt-2 border-t border-slate-800/80 text-[10px] text-slate-500 font-mono">
        <span>${meta.model}</span>
        ${meta.latency ? `<span>• ${meta.latency}ms</span>` : ''}
        ${tokensHtml}
        ${costHtml}
        ${toolsExecutedHtml}
      </div>
    `;
  }

  if (isUser) {
    msgDiv.innerHTML = `
      <div class="max-w-xl bg-emerald-600/90 text-white px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm shadow-md">
        <div class="leading-relaxed whitespace-pre-wrap">${escapeHtml(text)}</div>
      </div>
    `;
  } else {
    msgDiv.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-surface-850 border border-slate-700 flex items-center justify-center text-emerald-400 font-bold shrink-0 text-xs">
        J
      </div>
      <div class="max-w-2xl bg-surface-850 border border-slate-800 text-slate-200 px-4 py-3 rounded-2xl rounded-tl-sm text-sm shadow-sm">
        <div class="prose prose-invert prose-sm max-w-none leading-relaxed text-slate-200">${formattedContent}</div>
        ${metaBadge}
      </div>
    `;
  }

  stream.appendChild(msgDiv);
  stream.scrollTop = stream.scrollHeight;
}

function appendLoadingBubble(id) {
  const stream = document.getElementById('messages-stream');
  const div = document.createElement('div');
  div.id = id;
  div.className = 'flex gap-3 max-w-3xl mx-auto justify-start';
  div.innerHTML = `
    <div class="w-8 h-8 rounded-lg bg-surface-850 border border-slate-700 flex items-center justify-center text-emerald-400 font-bold shrink-0 text-xs">
      J
    </div>
    <div class="bg-surface-850 border border-slate-800 px-4 py-3 rounded-2xl rounded-tl-sm text-xs text-slate-400 flex items-center gap-2">
      <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
      <span>Processando resposta e ferramentas...</span>
    </div>
  `;
  stream.appendChild(div);
  stream.scrollTop = stream.scrollHeight;
}

function removeLoadingBubble(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function quickPrompt(text) {
  sendMessage(text);
}

function switchTab(tab) {
  const tabDarkhub = document.getElementById('tab-darkhub');
  const tabMcp = document.getElementById('tab-mcp');
  const tabTelemetry = document.getElementById('tab-telemetry');
  const btnDarkhub = document.getElementById('tab-btn-darkhub');
  const btnMcp = document.getElementById('tab-btn-mcp');
  const btnTelemetry = document.getElementById('tab-btn-telemetry');

  // Hide all containers
  tabDarkhub.classList.add('hidden');
  tabMcp.classList.add('hidden');
  if (tabTelemetry) tabTelemetry.classList.add('hidden');

  const activeClass = 'flex-1 py-1.5 px-2 rounded-lg bg-slate-800 text-white font-medium border border-slate-700 text-center truncate';
  const inactiveClass = 'flex-1 py-1.5 px-2 rounded-lg text-slate-400 hover:text-white transition text-center truncate';

  btnDarkhub.className = inactiveClass;
  btnMcp.className = inactiveClass;
  if (btnTelemetry) btnTelemetry.className = inactiveClass;

  if (tab === 'darkhub') {
    tabDarkhub.classList.remove('hidden');
    btnDarkhub.className = activeClass;
  } else if (tab === 'mcp') {
    tabMcp.classList.remove('hidden');
    btnMcp.className = activeClass;
  } else if (tab === 'telemetry') {
    if (tabTelemetry) tabTelemetry.classList.remove('hidden');
    if (btnTelemetry) btnTelemetry.className = activeClass;
    refreshTelemetry();
  }
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // DarkHub
    const darkhubDot = document.getElementById('darkhub-dot');
    const darkhubText = document.getElementById('darkhub-text');
    if (data.darkhub_online) {
      darkhubDot.className = 'w-2 h-2 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400/50';
      darkhubText.innerText = 'DarkHub: Online';
    } else {
      darkhubDot.className = 'w-2 h-2 rounded-full bg-amber-400';
      darkhubText.innerText = 'DarkHub: Standby';
    }

    // Ollama
    const ollamaDot = document.getElementById('ollama-dot');
    const ollamaText = document.getElementById('ollama-text');
    if (data.ollama_online) {
      ollamaDot.className = 'w-2 h-2 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400/50';
      ollamaText.innerText = 'Ollama: Local ($0)';
    } else {
      ollamaDot.className = 'w-2 h-2 rounded-full bg-slate-500';
      ollamaText.innerText = 'Ollama: Offline';
    }

    document.getElementById('mcp-count').innerText = data.tools_count || '0';
  } catch (err) {
    document.getElementById('darkhub-dot').className = 'w-2 h-2 rounded-full bg-red-400';
    document.getElementById('darkhub-text').innerText = 'DarkHub: Desconectado';
  }
}

async function loadDemandProjects() {
  const select = document.getElementById('demand-project');
  if (!select) return;
  try {
    const res = await fetch('/api/darkfac/projects');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const projects = await res.json();
    if (!Array.isArray(projects) || projects.length === 0) return;
    const previous = select.value;
    select.innerHTML = projects
      .map(project => '<option value="' + escapeHtml(project.id) + '">' +
        escapeHtml(project.name) + ' (' + escapeHtml(project.id) + ')</option>')
      .join('');
    select.value = projects.some(project => project.id === previous)
      ? previous
      : projects.some(project => project.id === 'jarvis')
        ? 'jarvis'
        : projects[0].id;
  } catch (err) {
    console.warn('Could not load Dark Factory projects:', err);
  }
}

async function refreshDemands() {
  const container = document.getElementById('demands-list');
  try {
    const res = await fetch('/api/darkfac/demands');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const demands = await res.json();

    if (demands.length === 0) {
      container.innerHTML = '<div class="p-3 text-slate-500 text-center text-[11px]">Nenhuma demanda no backlog.</div>';
      return;
    }

    container.innerHTML = demands.slice(0, 10).map(d => `
      <div class="p-2.5 rounded-xl bg-surface-850 hover:bg-slate-800/80 border border-slate-800 transition">
        <div class="flex items-center justify-between mb-1">
          <span class="font-mono text-[10px] text-emerald-400 font-semibold">${d.id}</span>
          <span class="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-slate-800 text-slate-300">${d.status}</span>
        </div>
        <div class="text-slate-200 font-medium text-xs truncate">${escapeHtml(d.title)}</div>
        <div class="text-[10px] text-slate-500 mt-1 font-mono">Projeto: ${d.project_id}</div>
      </div>
    `).join('');
  } catch (err) {
    container.innerHTML = `<div class="p-3 text-red-400 text-center text-[11px]">DarkHub offline ou inacessível.</div>`;
  }
}

async function loadMCPTools() {
  const container = document.getElementById('mcp-tools-list');
  try {
    const res = await fetch('/api/mcp/tools');
    if (!res.ok) return;
    const tools = await res.json();

    container.innerHTML = tools.map(t => `
      <div class="p-2.5 rounded-xl bg-surface-850 border border-slate-800 text-xs">
        <div class="font-mono text-emerald-400 font-semibold text-[11px]">${t.name}</div>
        <div class="text-slate-400 text-[11px] mt-1 leading-snug">${escapeHtml(t.description)}</div>
        <div class="text-[10px] text-slate-500 font-mono mt-1.5">Servidor: ${t.server_name}</div>
      </div>
    `).join('');
  } catch (err) {
    container.innerHTML = `<div class="p-3 text-slate-500 text-center text-[11px]">Falha ao carregar MCP tools.</div>`;
  }
}

function openNewDemandModal() {
  document.getElementById('modal-demand').classList.remove('hidden');
}

function closeNewDemandModal() {
  document.getElementById('modal-demand').classList.add('hidden');
}

let isSubmittingDemand = false;

async function handleDemandSubmit(e) {
  e.preventDefault();
  if (isSubmittingDemand) return;

  const proj = document.getElementById('demand-project').value;
  const title = document.getElementById('demand-title').value.trim();
  const problem = document.getElementById('demand-problem').value.trim();

  if (!title) return;

  const submitBtn = e.target.querySelector('button[type="submit"]');
  const originalBtnText = submitBtn ? submitBtn.innerText : 'Registrar Demanda';

  try {
    isSubmittingDemand = true;
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerText = 'Registrando...';
      submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
    }

    const res = await fetch('/api/darkfac/demands', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project_id: proj,
        title: title,
        problem_statement: problem,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      closeNewDemandModal();
      document.getElementById('demand-title').value = '';
      document.getElementById('demand-problem').value = '';
      refreshDemands();
      appendMessage(
        'assistant',
        'Demanda ' + data.demand_id + ' aceita no projeto ' + data.project_id +
        '; execução ' + data.run_id + ' e job inicial ' + data.initial_job_id + ' enfileirados.'
      );
    } else {
      const err = await res.json();
      alert(`Erro ao registrar demanda: ${JSON.stringify(err)}`);
    }
  } catch (err) {
    alert(`Erro de rede ao enviar demanda: ${err.message}`);
  } finally {
    isSubmittingDemand = false;
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerText = originalBtnText;
      submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
  }
}

// ==============================================================================
// Text-to-Speech (TTS) Neural Voice Synthesis Engine (Dual-Channel Output)
// ==============================================================================

function initTTS() {
  updateTTSButtonUI();
  if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      // Warm up and cache voices
      window.speechSynthesis.getVoices();
    };
  }
}

function toggleTTS() {
  if (isSpeaking) {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isSpeaking = false;
    updateTTSButtonUI();
    return;
  }
  ttsEnabled = !ttsEnabled;
  updateTTSButtonUI();
}

function updateTTSButtonUI() {
  const btn = document.getElementById('tts-toggle-btn');
  const icon = document.getElementById('tts-icon');
  const label = document.getElementById('tts-label');
  if (!btn) return;

  if (isSpeaking) {
    btn.className = 'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-teal-600 border border-teal-400 text-white text-xs font-mono recording-pulse transition cursor-pointer shadow-md shadow-teal-500/20';
    icon.innerText = '🔊';
    label.innerText = 'Falando...';
  } else if (ttsEnabled) {
    btn.className = 'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-950/70 border border-emerald-800/80 text-emerald-400 text-xs font-mono hover:bg-emerald-900/60 transition cursor-pointer';
    icon.innerText = '🔊';
    label.innerText = 'Voz: Ativa';
  } else {
    btn.className = 'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-400 text-xs font-mono hover:text-white transition cursor-pointer';
    icon.innerText = '🔇';
    label.innerText = 'Voz: Mudo';
  }
}

/**
 * Extracts a concise executive summary from markdown to avoid audio fatigue and long recitations.
 * Strips code blocks, links, tables, and headers, selecting only the first 1-2 key sentences.
 */
function extractSpokenSummary(markdownText) {
  if (!markdownText) return '';

  // 1. Remove code blocks
  let text = markdownText.replace(/```[\s\S]*?```/g, ' [código omitido no áudio] ');
  // 2. Remove inline code
  text = text.replace(/`([^`]+)`/g, '$1');
  // 3. Remove images
  text = text.replace(/!\[.*?\]\(.*?\)/g, '');
  // 4. Remove links: [text](url) -> text
  text = text.replace(/\[(.*?)\]\(.*?\)/g, '$1');
  // 5. Remove markdown headers
  text = text.replace(/^#+\s+/gm, '');
  // 6. Remove bold/italics
  text = text.replace(/(\*\*|__)(.*?)\1/g, '$2');
  text = text.replace(/(\*|_)(.*?)\1/g, '$2');
  // 7. Remove blockquotes, table separators and list bullets
  text = text.replace(/^>\s+/gm, '');
  text = text.replace(/^\|.*?\|$/gm, '');
  text = text.replace(/^[\*\-\+]\s+/gm, '');
  text = text.replace(/^\d+\.\s+/gm, '');
  // 8. Remove HTML tags
  text = text.replace(/<[^>]*>/g, '');
  // 9. Clean excessive whitespaces and newlines
  text = text.replace(/\s+/g, ' ').trim();

  if (!text) return '';

  // 10. Split into sentences to take only the first 1 or 2 concise sentences
  const sentenceRegex = /[^.!?]+[.!?]+/g;
  const matches = text.match(sentenceRegex);

  let summary = '';
  if (matches && matches.length > 0) {
    summary = matches[0].trim();
    if (matches.length > 1 && (summary.length + matches[1].trim().length) <= 220) {
      summary += ' ' + matches[1].trim();
    }
    // If there is extensive follow-up text, append natural closure if not already present
    if (matches.length > 2 && !summary.toLowerCase().includes('tela') && !summary.toLowerCase().includes('detalhes')) {
      summary += ' Os detalhes completos estão na tela.';
    }
  } else {
    // Truncate at ~160 chars on word boundary if no punctuation
    if (text.length > 160) {
      summary = text.substring(0, 160).replace(/\s\S*$/, '') + '... Os detalhes completos estão na tela.';
    } else {
      summary = text;
    }
  }

  return summary;
}

function speakResponse(markdownText) {
  if (!ttsEnabled || !('speechSynthesis' in window)) return;

  const spokenText = extractSpokenSummary(markdownText);
  if (!spokenText) return;

  try {
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(spokenText);
    utterance.lang = 'pt-BR';
    utterance.rate = 1.05;
    utterance.pitch = 1.0;

    // Pick best natural Portuguese voice
    const voices = window.speechSynthesis.getVoices();
    if (voices && voices.length > 0) {
      const ptVoices = voices.filter(v => v.lang && (v.lang === 'pt-BR' || v.lang === 'pt_BR' || v.lang.startsWith('pt')));
      const preferred = ptVoices.find(v => v.name.includes('Francisca') || v.name.includes('Natural')) ||
                        ptVoices.find(v => v.name.includes('Antonio')) ||
                        ptVoices.find(v => v.name.includes('Google')) ||
                        ptVoices[0];
      if (preferred) {
        utterance.voice = preferred;
      }
    }

    utterance.onstart = () => {
      isSpeaking = true;
      updateTTSButtonUI();
    };

    utterance.onend = () => {
      isSpeaking = false;
      updateTTSButtonUI();
    };

    utterance.onerror = (e) => {
      console.warn('SpeechSynthesis event error:', e);
      isSpeaking = false;
      updateTTSButtonUI();
    };

    window.speechSynthesis.speak(utterance);
  } catch (err) {
    console.warn('SpeechSynthesis playback failed:', err);
    isSpeaking = false;
    updateTTSButtonUI();
  }
}

function formatUsdAmount(val) {
  if (val === undefined || val === null || isNaN(val)) return '0.00';
  const num = Number(val);
  if (num > 0 && num < 0.01) return num.toFixed(3);
  return num.toFixed(2);
}

async function refreshTelemetry() {
  try {
    const [summaryRes, budgetRes] = await Promise.all([
      fetch('/api/telemetry/summary'),
      fetch('/api/telemetry/budget'),
    ]);

    if (!summaryRes.ok || !budgetRes.ok) return;

    const summary = await summaryRes.json();
    const budgetData = await budgetRes.json();
    const status = budgetData.status || {};
    const policy = budgetData.policy || {};

    const dailySpentNum = Number(status.daily_spent_usd ?? status.daily_spend_usd ?? 0.0);
    const dailyLimitNum = Number(status.daily_limit_usd ?? policy.daily_limit_usd ?? 2.0);
    const totalSavingsNum = Number(summary.total_savings_usd ?? 0.0);
    const promptTok = Number(summary.prompt_tokens ?? summary.total_prompt_tokens ?? 0);
    const compTok = Number(summary.completion_tokens ?? summary.total_completion_tokens ?? 0);
    const totalTok = Number(summary.total_tokens ?? (promptTok + compTok));

    // 1. Header Badges
    const budgetDot = document.getElementById('budget-dot');
    const budgetText = document.getElementById('budget-text');
    const savingsText = document.getElementById('savings-text');

    if (budgetText) {
      budgetText.innerText = `Gasto: $${formatUsdAmount(dailySpentNum)} / $${formatUsdAmount(dailyLimitNum)}`;
    }

    if (budgetDot) {
      if (status.status === 'exceeded') {
        budgetDot.className = 'w-2 h-2 rounded-full bg-red-500 animate-ping';
      } else if (status.status === 'warning') {
        budgetDot.className = 'w-2 h-2 rounded-full bg-amber-400';
      } else {
        budgetDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
      }
    }

    if (savingsText) {
      savingsText.innerText = `Economia: $${formatUsdAmount(totalSavingsNum)}`;
    }

    // 2. Sidebar Card Details
    const cardStatus = document.getElementById('card-budget-status');
    const cardBar = document.getElementById('card-budget-bar');
    const cardDailySpend = document.getElementById('card-daily-spend');
    const cardDailyLimit = document.getElementById('card-daily-limit');

    if (cardStatus) {
      cardStatus.innerText = (status.status || 'ok').toUpperCase();
      if (status.status === 'exceeded') {
        cardStatus.className = 'px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-red-950 text-red-400 border border-red-800';
      } else if (status.status === 'warning') {
        cardStatus.className = 'px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-amber-950 text-amber-400 border border-amber-800';
      } else {
        cardStatus.className = 'px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-emerald-950 text-emerald-400 border border-emerald-800';
      }
    }

    if (cardBar) {
      let pct = 0;
      if (dailyLimitNum > 0) {
        pct = Math.min(100, Math.round((dailySpentNum / dailyLimitNum) * 100));
      } else if (dailySpentNum > 0) {
        pct = 100;
      }
      cardBar.style.width = `${pct}%`;
      if (pct >= 100) {
        cardBar.className = 'bg-red-500 h-2 rounded-full transition-all duration-300';
      } else if (pct >= 80) {
        cardBar.className = 'bg-amber-400 h-2 rounded-full transition-all duration-300';
      } else {
        cardBar.className = 'bg-emerald-500 h-2 rounded-full transition-all duration-300';
      }
    }

    if (cardDailySpend) {
      cardDailySpend.innerText = `$${formatUsdAmount(dailySpentNum)} gasto`;
    }
    if (cardDailyLimit) {
      cardDailyLimit.innerText = `Teto: $${formatUsdAmount(dailyLimitNum)}`;
    }

    // 3. Grid Metrics
    const cardTokens = document.getElementById('card-total-tokens');
    const cardRatio = document.getElementById('card-tokens-ratio');
    const cardSavings = document.getElementById('card-total-savings');

    if (cardTokens) {
      cardTokens.innerText = totalTok.toLocaleString();
    }
    if (cardRatio) {
      cardRatio.innerText = `Prompt: ${promptTok.toLocaleString()} | Comp: ${compTok.toLocaleString()}`;
    }
    if (cardSavings) {
      cardSavings.innerText = `$${formatUsdAmount(totalSavingsNum)}`;
    }

    // 4. Breakdown by model
    const modelsContainer = document.getElementById('telemetry-models-breakdown');
    if (modelsContainer && summary.by_model) {
      const modelKeys = Object.keys(summary.by_model);
      if (modelKeys.length === 0) {
        modelsContainer.innerHTML = '<div class="p-2 rounded-lg bg-surface-850 border border-slate-800 text-[11px] text-slate-400 text-center">Sem dados de consumo ainda.</div>';
      } else {
        modelsContainer.innerHTML = modelKeys.map(m => {
          const stats = summary.by_model[m];
          const costVal = Number(stats.cost_usd || 0);
          const costStr = costVal === 0 ? '$0.00' : `$${formatUsdAmount(costVal)}`;
          return `
            <div class="p-2 rounded-lg bg-surface-850 border border-slate-800 text-[11px] flex items-center justify-between">
              <div class="truncate max-w-[150px]">
                <div class="text-slate-300 font-medium truncate" title="${escapeHtml(m)}">${escapeHtml(m)}</div>
                <div class="text-[10px] text-slate-500 font-mono">${(stats.tokens || 0).toLocaleString()} tok (${stats.calls || 0}x)</div>
              </div>
              <div class="text-right font-mono text-emerald-400 font-semibold">${costStr}</div>
            </div>
          `;
        }).join('');
      }
    }

    // 5. Circuit Breaker & Fallback Indicators
    const cardCbStatus = document.getElementById('card-cb-status');
    const cardCbFallback = document.getElementById('card-cb-fallback');
    const isCbTripped = Boolean(status.circuit_breaker_active);
    const isCbArmed = Boolean(policy.enforce_circuit_breaker ?? policy.circuit_breaker_enabled ?? true);

    if (cardCbStatus) {
      if (isCbTripped) {
        cardCbStatus.innerText = '🔴 Disparado (Bloqueio Nuvem)';
        cardCbStatus.className = 'font-mono text-red-400 font-semibold';
      } else if (isCbArmed) {
        cardCbStatus.innerText = '🟢 Pronto (Armado)';
        cardCbStatus.className = 'font-mono text-emerald-400';
      } else {
        cardCbStatus.innerText = '⚪ Desativado';
        cardCbStatus.className = 'font-mono text-slate-400';
      }
    }

    if (cardCbFallback) {
      const fallbackActive = Boolean(policy.auto_fallback_to_local ?? policy.fallback_to_local_on_limit ?? true);
      cardCbFallback.innerText = fallbackActive ? '🟢 Ativado' : '⚪ Desativado';
      cardCbFallback.className = fallbackActive ? 'font-mono text-emerald-400' : 'font-mono text-slate-400';
    }

  } catch (err) {
    console.warn('Telemetry refresh failed:', err);
  }
}

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
