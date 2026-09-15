/**
 * Jarvis Web GUI Controller (Dual-Engine Voice: Real-Time Web Speech + 16kHz PCM WAV Fallback)
 */

let chatHistory = [];
let currentLang = 'pt-BR'; // 'pt-BR' or 'en-US'
let isRecording = false;

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
  checkHealth();
  refreshDemands();
  loadMCPTools();
  setInterval(checkHealth, 30000);
});

function initUI() {
  const userInput = document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');
  const btnMic = document.getElementById('btn-mic');
  const btnStopRecording = document.getElementById('btn-stop-recording');

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

  // Tabs
  document.getElementById('tab-btn-darkhub').addEventListener('click', () => switchTab('darkhub'));
  document.getElementById('tab-btn-mcp').addEventListener('click', () => switchTab('mcp'));

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
    });

    // Update history
    chatHistory.push({ role: 'user', content: text });
    chatHistory.push({ role: 'assistant', content: data.response_text });

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
    metaBadge = `
      <div class="flex items-center gap-2 mt-2 pt-2 border-t border-slate-800/80 text-[10px] text-slate-500 font-mono">
        <span>${meta.model}</span>
        ${meta.latency ? `<span>• ${meta.latency}ms</span>` : ''}
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
  const btnDarkhub = document.getElementById('tab-btn-darkhub');
  const btnMcp = document.getElementById('tab-btn-mcp');

  if (tab === 'darkhub') {
    tabDarkhub.classList.remove('hidden');
    tabMcp.classList.add('hidden');
    btnDarkhub.className = 'flex-1 py-1.5 px-3 rounded-lg bg-slate-800 text-white font-medium border border-slate-700';
    btnMcp.className = 'flex-1 py-1.5 px-3 rounded-lg text-slate-400 hover:text-white transition';
  } else {
    tabDarkhub.classList.add('hidden');
    tabMcp.classList.remove('hidden');
    btnMcp.className = 'flex-1 py-1.5 px-3 rounded-lg bg-slate-800 text-white font-medium border border-slate-700';
    btnDarkhub.className = 'flex-1 py-1.5 px-3 rounded-lg text-slate-400 hover:text-white transition';
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

async function handleDemandSubmit(e) {
  e.preventDefault();
  const proj = document.getElementById('demand-project').value;
  const title = document.getElementById('demand-title').value.trim();
  const problem = document.getElementById('demand-problem').value.trim();

  if (!title) return;

  try {
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
      closeNewDemandModal();
      document.getElementById('demand-title').value = '';
      document.getElementById('demand-problem').value = '';
      refreshDemands();
      appendMessage('assistant', `✅ **Demanda registrada com sucesso no DarkHub!**\n- **Título**: ${title}\n- **Projeto**: \`${proj}\``);
    } else {
      const err = await res.json();
      alert(`Erro ao registrar demanda: ${JSON.stringify(err)}`);
    }
  } catch (err) {
    alert(`Erro de rede ao enviar demanda: ${err.message}`);
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
