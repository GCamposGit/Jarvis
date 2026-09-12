/**
 * Jarvis Web GUI Controller (Vanilla JS + Web Audio API)
 */

let chatHistory = [];
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

document.addEventListener('DOMContentLoaded', () => {
  initUI();
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

  btnSend.addEventListener('click', sendMessage);
  btnMic.addEventListener('click', toggleRecording);
  btnStopRecording.addEventListener('click', stopRecording);

  // Tabs
  document.getElementById('tab-btn-darkhub').addEventListener('click', () => switchTab('darkhub'));
  document.getElementById('tab-btn-mcp').addEventListener('click', () => switchTab('mcp'));

  // Demand Form
  document.getElementById('form-demand').addEventListener('submit', handleDemandSubmit);
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

// ==============================================================================
// Web Audio API & MediaRecorder
// ==============================================================================

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstop = uploadRecordedAudio;
    mediaRecorder.start();
    isRecording = true;

    document.getElementById('recording-banner').classList.remove('hidden');
    document.getElementById('btn-mic').classList.add('bg-red-600', 'text-white', 'recording-pulse');
  } catch (err) {
    alert(`Erro ao acessar microfone: ${err.message}`);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;

    document.getElementById('recording-banner').classList.add('hidden');
    document.getElementById('btn-mic').classList.remove('bg-red-600', 'text-white', 'recording-pulse');
  }
}

async function uploadRecordedAudio() {
  const indicator = document.getElementById('transcription-indicator');
  indicator.innerText = 'Transcrevendo áudio...';

  const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
  const formData = new FormData();
  formData.append('file', audioBlob, 'mic_recording.webm');

  try {
    const res = await fetch('/api/audio/transcribe', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.text) {
      indicator.innerText = `🎙️ [${data.engine_used}]: "${data.text.substring(0, 40)}..."`;
      // Put in input and send
      document.getElementById('user-input').value = data.text;
      sendMessage();
    } else {
      indicator.innerText = `⚠️ Nenhum texto reconhecido (${data.fallback_reason || 'baixa confiança'}).`;
    }
  } catch (err) {
    indicator.innerText = `⚠️ Falha na transcrição: ${err.message}`;
  } finally {
    setTimeout(() => { indicator.innerText = ''; }, 6000);
  }
}

// ==============================================================================
// DarkHub Demands Management
// ==============================================================================

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
