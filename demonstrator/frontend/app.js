/**
 * TenSafe Finance Demonstrator — Client JS
 *
 * WebSocket streaming chat with real-time HE metrics,
 * comparison mode, and settings management.
 */

// ---- State ----
let ws = null;
let isStreaming = false;
let compareMode = false;
let splitMode = false;
let splitClient = null;  // SplitInferenceClient instance
let wsRetryDelay = 1000;
const WS_MAX_RETRY = 30000;

const settings = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 50,
  max_tokens: 256,
  use_he: true,
};

// ---- DOM refs ----
const $messages   = document.getElementById("messages");
const $input      = document.getElementById("input-msg");
const $btnSend    = document.getElementById("btn-send");
const $btnCompare = document.getElementById("btn-compare");
const $btnSettings = document.getElementById("btn-settings");
const $badge      = document.getElementById("badge-status");
const $metricsBar = document.getElementById("metrics-bar");
const $heBreak    = document.getElementById("he-breakdown");
const $cmpPanel   = document.getElementById("compare-panel");
const $settings   = document.getElementById("settings-drawer");

// Metric displays
const $mToks    = document.getElementById("m-toks");
const $mLatency = document.getElementById("m-latency");
const $mHeOps   = document.getElementById("m-heops");
const $mRot     = document.getElementById("m-rot");
const $heEnc    = document.getElementById("he-enc");
const $heComp   = document.getElementById("he-comp");
const $heDec    = document.getElementById("he-dec");
const $heNet    = document.getElementById("he-net");
const $heSlots  = document.getElementById("he-slots");
const $heCtB    = document.getElementById("he-ctbytes");

// Pipeline stages
const $pipePlain   = document.getElementById("pipe-plaintext");
const $pipeEnc     = document.getElementById("pipe-encrypt");
const $pipeComp    = document.getElementById("pipe-compute");
const $pipeDec     = document.getElementById("pipe-decrypt");
const $pipeResp    = document.getElementById("pipe-response");
const $pipeTPlain  = document.getElementById("pipe-t-plain");
const $pipeTEnc    = document.getElementById("pipe-t-enc");
const $pipeTComp   = document.getElementById("pipe-t-comp");
const $pipeTDec    = document.getElementById("pipe-t-dec");
const $pipeTResp   = document.getElementById("pipe-t-resp");
const pipeArrows   = document.querySelectorAll(".pipe-arrow");

// Routing pills
const $routeBanking    = document.getElementById("route-banking");
const $routeInvestment = document.getElementById("route-investment");
const $routeShared     = document.getElementById("route-shared");
const $routeBankingCt  = document.getElementById("route-banking-count");
const $routeInvestCt   = document.getElementById("route-investment-count");
const $routeSharedCt   = document.getElementById("route-shared-count");
let pipelineFadeTimer  = null;

// ======================================================================
// WebSocket
// ======================================================================

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/api/v1/chat/stream`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    wsRetryDelay = 1000;  // Reset backoff on successful connect
    $badge.textContent = "Online";
    $badge.className = "badge online";
    console.log("WS connected");
  };

  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      handleChunk(data);
    } catch (e) {
      console.error("Failed to parse WS message:", e, ev.data?.slice(0, 200));
    }
  };

  ws.onclose = () => {
    $badge.textContent = "Reconnecting...";
    $badge.className = "badge";
    isStreaming = false;
    setTimeout(connectWS, wsRetryDelay);
    wsRetryDelay = Math.min(wsRetryDelay * 2, WS_MAX_RETRY);
  };

  ws.onerror = () => {
    ws.close();
  };
}

// ======================================================================
// Chat
// ======================================================================

let currentAssistantEl = null;
let currentTextEl = null;

function sendMessage() {
  const text = $input.value.trim();
  if (!text || isStreaming) return;

  if (compareMode) {
    sendCompare(text);
    return;
  }

  // Route through split client if split mode is on
  if (splitMode) {
    sendSplit(text);
    return;
  }

  // User bubble
  appendMsg("user", text);
  $input.value = "";

  // Pipeline + routing reset
  pipelineOnSend();
  resetRouting();

  // Assistant bubble (streaming target)
  const { msgEl, textEl } = appendMsg("assistant", "");
  currentAssistantEl = msgEl;
  currentTextEl = textEl;
  isStreaming = true;

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    appendMsg("assistant", "Connection lost. Reconnecting...");
    isStreaming = false;
    return;
  }

  ws.send(JSON.stringify({
    query: text,
    max_tokens: settings.max_tokens,
    temperature: settings.temperature,
    top_p: settings.top_p,
    top_k: settings.top_k,
    use_he: settings.use_he,
  }));
}

function handleChunk(data) {
  if (data.type === "input_info") {
    pipelineOnInputInfo(data);
    routingOnInputInfo(data);
    // Show input encryption badge
    if (currentAssistantEl && data.encrypted) {
      const badge = document.createElement("div");
      badge.className = "enc-badge encrypted";
      badge.innerHTML =
        `&#x1f512; Input Encrypted | CKKS | ${data.simd_slots} slots | ${fmtBytes(data.ciphertext_bytes)}` +
        `<span class="expert-tag">Expert: ${data.active_expert}</span>`;
      currentAssistantEl.prepend(badge);
    }
    return;
  }

  if (data.type === "token" && currentTextEl) {
    currentTextEl.textContent += data.token;
    updateMetrics(data.aggregate);
    updateHEBreakdown(data.metrics);
    pipelineOnToken(data);
    scrollToBottom();
    return;
  }

  if (data.type === "error") {
    isStreaming = false;
    // Show error in the assistant bubble
    if (currentTextEl) {
      currentTextEl.textContent = data.message || "Unknown error";
      currentTextEl.style.color = "var(--error)";
    } else {
      appendMsg("assistant", "⚠ " + (data.message || "Unknown error"));
    }
    // If it's a privacy budget error, show reset hint
    if (data.message && data.message.includes("Privacy budget")) {
      const hint = document.createElement("div");
      hint.className = "enc-badge";
      hint.style.background = "#3a2211";
      hint.style.color = "var(--warn)";
      hint.innerHTML = "Budget: ε=" + (data.epsilon_spent ?? "?").toFixed?.(2) +
        "/" + (data.max_epsilon ?? "?") +
        " — Open Settings → Reset Privacy Budget";
      if (currentAssistantEl) currentAssistantEl.appendChild(hint);
    }
    currentAssistantEl = null;
    currentTextEl = null;
    return;
  }

  if (data.type === "done") {
    isStreaming = false;
    updateMetrics(data.aggregate);
    pipelineOnDone(data);
    routingOnDone(data);

    // Output badge
    if (currentAssistantEl && data.aggregate.encryption_active) {
      const badge = document.createElement("div");
      badge.className = "enc-badge decrypted";
      const experts = Object.entries(data.aggregate.expert_distribution || {});
      const expertStr = experts.map(([k, v]) => `${k}:${v}`).join(" ");
      badge.innerHTML =
        `&#x1f513; Output Decrypted | ${data.aggregate.total_tokens} tokens | ` +
        `${data.aggregate.tokens_per_second.toFixed(1)} tok/s` +
        `<span class="expert-tag">${expertStr}</span>`;
      currentAssistantEl.appendChild(badge);
    }

    currentAssistantEl = null;
    currentTextEl = null;
  }
}

function appendMsg(role, text) {
  const msgEl = document.createElement("div");
  msgEl.className = `msg ${role}`;
  const textEl = document.createElement("span");
  textEl.textContent = text;
  msgEl.appendChild(textEl);
  $messages.appendChild(msgEl);
  scrollToBottom();
  return { msgEl, textEl };
}

function scrollToBottom() {
  const chat = document.getElementById("chat-area");
  chat.scrollTop = chat.scrollHeight;
}

// ======================================================================
// Metrics
// ======================================================================

function updateMetrics(agg) {
  if (!agg) return;
  $mToks.textContent = agg.tokens_per_second ? agg.tokens_per_second.toFixed(1) : "--";
  $mLatency.textContent = agg.avg_latency_ms ? agg.avg_latency_ms.toFixed(0) : "--";
  $mHeOps.textContent = agg.total_he_operations || 0;
  $mRot.textContent = agg.total_rotations || 0;
}

function updateHEBreakdown(m) {
  if (!m) return;
  $heEnc.textContent  = m.encrypt_ms ? m.encrypt_ms.toFixed(1) + " ms" : "--";
  $heComp.textContent = m.compute_ms ? m.compute_ms.toFixed(1) + " ms" : "--";
  $heDec.textContent  = m.decrypt_ms ? m.decrypt_ms.toFixed(1) + " ms" : "--";
  $heNet.textContent  = m.network_ms ? m.network_ms.toFixed(1) + " ms" : "--";
  $heSlots.textContent = m.simd_slots_used || "--";
  $heCtB.textContent  = m.ciphertext_bytes ? fmtBytes(m.ciphertext_bytes) : "--";
}

// ======================================================================
// Compare mode
// ======================================================================

async function sendCompare(text) {
  appendMsg("user", text);
  $input.value = "";

  document.getElementById("cmp-base").textContent = "Generating...";
  document.getElementById("cmp-adapted").textContent = "Generating...";
  document.getElementById("cmp-base-stats").textContent = "";
  document.getElementById("cmp-adapted-stats").textContent = "";

  try {
    const res = await fetch("/api/v1/chat/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        max_tokens: settings.max_tokens,
        temperature: settings.temperature,
      }),
    });
    const data = await res.json();

    document.getElementById("cmp-base").textContent = data.base.response || "(empty)";
    document.getElementById("cmp-base-stats").textContent =
      `${data.base.tokens} tok | ${data.base.tok_s} tok/s | ${data.base.time_ms} ms | No encryption`;

    document.getElementById("cmp-adapted").textContent = data.adapted.response || "(empty)";
    document.getElementById("cmp-adapted-stats").textContent =
      `${data.adapted.tokens} tok | ${data.adapted.tok_s} tok/s | ${data.adapted.time_ms} ms | ` +
      `${data.adapted.he_operations} HE ops | 0 rotations`;
  } catch (e) {
    document.getElementById("cmp-base").textContent = "Error: " + e.message;
  }
}

// ======================================================================
// Settings
// ======================================================================

function initSettings() {
  const bind = (id, key, vId) => {
    const el = document.getElementById(id);
    const vEl = document.getElementById(vId);
    el.addEventListener("input", () => {
      settings[key] = parseFloat(el.value);
      vEl.textContent = el.value;
    });
  };
  bind("s-temp", "temperature", "v-temp");
  bind("s-topp", "top_p", "v-topp");
  bind("s-topk", "top_k", "v-topk");
  bind("s-maxt", "max_tokens", "v-maxt");

  document.getElementById("s-he").addEventListener("change", (e) => {
    settings.use_he = e.target.checked;
  });

  document.getElementById("s-split").addEventListener("change", (e) => {
    splitMode = e.target.checked;
    if (splitMode) {
      $badge.textContent = "Split Mode";
      $badge.className = "badge split";
    } else {
      $badge.textContent = ws && ws.readyState === 1 ? "Online" : "Disconnected";
      $badge.className = ws && ws.readyState === 1 ? "badge online" : "badge";
    }
  });
}

// ======================================================================
// Pipeline visualization
// ======================================================================

function resetPipeline() {
  [$pipePlain, $pipeEnc, $pipeComp, $pipeDec, $pipeResp].forEach(el => {
    el.classList.remove("active", "done");
  });
  [$pipeTPlain, $pipeTEnc, $pipeTComp, $pipeTDec, $pipeTResp].forEach(el => {
    el.textContent = "";
  });
  pipeArrows.forEach(a => a.classList.remove("lit"));
  if (pipelineFadeTimer) clearTimeout(pipelineFadeTimer);
}

function pipelineOnSend() {
  resetPipeline();
  $pipePlain.classList.add("active");
  $pipeTPlain.textContent = "sending";
  pipeArrows[0].classList.add("lit");
}

function pipelineOnInputInfo(data) {
  $pipePlain.classList.remove("active");
  $pipePlain.classList.add("done");
  $pipeTPlain.textContent = data.input_tokens + " tok";

  $pipeEnc.classList.add("active");
  $pipeTEnc.textContent = data.encrypt_time_ms ? data.encrypt_time_ms.toFixed(1) + " ms" : "";
  pipeArrows[1].classList.add("lit");
}

function pipelineOnToken(data) {
  $pipeEnc.classList.remove("active");
  $pipeEnc.classList.add("done");

  $pipeComp.classList.add("active");
  if (data.metrics && data.metrics.compute_ms) {
    $pipeTComp.textContent = data.metrics.compute_ms.toFixed(1) + " ms";
  }

  $pipeDec.classList.add("active");
  if (data.metrics && data.metrics.decrypt_ms) {
    $pipeTDec.textContent = data.metrics.decrypt_ms.toFixed(1) + " ms";
  }
  pipeArrows[2].classList.add("lit");
  pipeArrows[3].classList.add("lit");
}

function pipelineOnDone(data) {
  [$pipeComp, $pipeDec].forEach(el => {
    el.classList.remove("active");
    el.classList.add("done");
  });
  $pipeResp.classList.add("done");
  if (data.aggregate && data.aggregate.total_tokens) {
    $pipeTResp.textContent = data.aggregate.total_tokens + " tok";
  }
  if (data.aggregate && data.aggregate.total_compute_ms) {
    $pipeTComp.textContent = data.aggregate.total_compute_ms.toFixed(0) + " ms";
  }
  if (data.aggregate && data.aggregate.total_decrypt_ms) {
    $pipeTDec.textContent = data.aggregate.total_decrypt_ms.toFixed(0) + " ms";
  }
  pipeArrows.forEach(a => a.classList.add("lit"));

  pipelineFadeTimer = setTimeout(() => {
    [$pipePlain, $pipeEnc, $pipeComp, $pipeDec, $pipeResp].forEach(el => {
      el.classList.remove("active");
    });
  }, 3000);
}


// ======================================================================
// Expert routing display
// ======================================================================

function resetRouting() {
  $routeBanking.classList.remove("active");
  $routeInvestment.classList.remove("active");
  $routeBankingCt.textContent = "";
  $routeInvestCt.textContent = "";
  $routeSharedCt.textContent = "";
}

function routingOnInputInfo(data) {
  resetRouting();
  const expert = data.active_expert || "";
  if (expert.includes("banking")) {
    $routeBanking.classList.add("active");
  } else if (expert.includes("investment")) {
    $routeInvestment.classList.add("active");
  }
}

function routingOnDone(data) {
  if (!data.aggregate || !data.aggregate.expert_distribution) return;
  const dist = data.aggregate.expert_distribution;
  if (dist.banking_expert) $routeBankingCt.textContent = dist.banking_expert;
  if (dist.investment_expert) $routeInvestCt.textContent = dist.investment_expert;
  if (dist.shared_attention) $routeSharedCt.textContent = dist.shared_attention;
}


// ======================================================================
// GateLink-Split Inference Mode
// ======================================================================

const $splitStatus  = document.getElementById("split-status");
const $splitStage   = document.getElementById("split-stage");
const $splitBar     = document.getElementById("split-progress-bar");
const $splitDetail  = document.getElementById("split-detail");

function splitProgress(stage, frac, detail) {
  $splitStatus.classList.remove("hidden");
  $splitStage.textContent = stage;
  if (typeof frac === "number") {
    $splitBar.style.width = Math.round(frac * 100) + "%";
  }
  if (detail) $splitDetail.textContent = detail;
}

async function initSplitClient() {
  if (splitClient && splitClient.ready) return;
  splitClient = new SplitInferenceClient();
  await splitClient.initialize((stage, frac, detail) => {
    splitProgress(stage, frac, detail || stage);
  });
  $splitStatus.classList.add("hidden");
}

async function sendSplit(text) {
  appendMsg("user", text);
  $input.value = "";
  isStreaming = true;

  // Pipeline + routing reset
  pipelineOnSend();
  resetRouting();

  // Initialize client if needed (downloads weights first time)
  try {
    await initSplitClient();
  } catch (e) {
    appendMsg("assistant", "Split client init failed: " + e.message);
    isStreaming = false;
    return;
  }

  // Create assistant bubble
  const { msgEl, textEl } = appendMsg("assistant", "");
  currentAssistantEl = msgEl;
  currentTextEl = textEl;

  // Show split input badge
  const expert = splitClient.routeExpert(text);
  const inputIds = splitClient.tokenizer.encode(text);
  const inBadge = document.createElement("div");
  inBadge.className = "enc-badge encrypted";
  const dpLabel = splitClient.dpEps > 0
    ? ` + DP noise (\u03b5=${splitClient.dpEps})`
    : " (no DP noise)";
  inBadge.innerHTML =
    `&#x1f512; Split Inference | Client embed${dpLabel} | ` +
    `${inputIds.length} tokens` +
    `<span class="expert-tag">Expert: ${expert}</span>`;
  currentAssistantEl.prepend(inBadge);

  // Update pipeline visualization
  $pipePlain.classList.remove("active"); $pipePlain.classList.add("done");
  $pipeTPlain.textContent = inputIds.length + " tok";
  $pipeEnc.classList.add("active");
  $pipeTEnc.textContent = "DP \u03b5=" + splitClient.dpEps;
  pipeArrows[0].classList.add("lit");
  pipeArrows[1].classList.add("lit");

  // Update routing
  if (expert.includes("banking")) $routeBanking.classList.add("active");
  else if (expert.includes("investment")) $routeInvestment.classList.add("active");

  // Run split inference
  let totalHeOps = 0;
  try {
    const result = await splitClient.generate(
      text,
      {
        maxTokens: settings.max_tokens,
        temperature: settings.temperature,
        topP: settings.top_p,
        topK: settings.top_k,
        useHE: settings.use_he,
      },
      // onToken callback
      (tokenText, metrics) => {
        currentTextEl.textContent += tokenText;
        totalHeOps += metrics.heOps || 0;

        // Update metrics display
        $mToks.textContent = metrics.tokPerSec;
        $mLatency.textContent = metrics.stepMs;
        $mHeOps.textContent = totalHeOps;
        $mRot.textContent = "0";

        // Update HE breakdown
        $heEnc.textContent = metrics.encryptMs ? metrics.encryptMs.toFixed(1) + " ms" : "--";
        $heComp.textContent = metrics.computeMs ? metrics.computeMs.toFixed(1) + " ms" : "--";
        $heDec.textContent = metrics.decryptMs ? metrics.decryptMs.toFixed(1) + " ms" : "--";
        $heNet.textContent = metrics.serverMs ? (metrics.serverMs - (metrics.encryptMs||0) - (metrics.computeMs||0) - (metrics.decryptMs||0)).toFixed(0) + " ms" : "--";

        // Update pipeline
        $pipeEnc.classList.remove("active"); $pipeEnc.classList.add("done");
        $pipeComp.classList.add("active");
        $pipeTComp.textContent = metrics.computeMs ? metrics.computeMs.toFixed(0) + " ms" : "";
        $pipeDec.classList.add("active");
        $pipeTDec.textContent = metrics.decryptMs ? metrics.decryptMs.toFixed(0) + " ms" : "";
        pipeArrows[2].classList.add("lit");
        pipeArrows[3].classList.add("lit");

        // Split-specific: show LM head projection time
        splitProgress("project", null, "LM head: " + metrics.projectMs + " ms");

        scrollToBottom();
      },
      // onStatus callback
      (stage, detail) => {
        splitProgress(stage, null, detail);
      }
    );

    // Done — final badges and cleanup
    $splitStatus.classList.add("hidden");

    // Pipeline done
    [$pipeComp, $pipeDec].forEach(el => { el.classList.remove("active"); el.classList.add("done"); });
    $pipeResp.classList.add("done");
    $pipeTResp.textContent = result.totalTokens + " tok";
    pipeArrows.forEach(a => a.classList.add("lit"));

    // Output badge
    const outBadge = document.createElement("div");
    outBadge.className = "enc-badge decrypted";
    outBadge.innerHTML =
      `&#x1f513; Split Complete | ${result.totalTokens} tokens | ` +
      `${result.tokPerSec} tok/s | Client LM head + sampling` +
      `<span class="expert-tag">${result.expert}</span>`;
    currentAssistantEl.appendChild(outBadge);

  } catch (e) {
    const errEl = document.createElement("div");
    errEl.className = "enc-badge";
    errEl.style.background = "#3a1111";
    errEl.textContent = "Split error: " + e.message;
    currentAssistantEl.appendChild(errEl);
    $splitStatus.classList.add("hidden");
  }

  isStreaming = false;
  currentAssistantEl = null;
  currentTextEl = null;
}


// ======================================================================
// Utilities
// ======================================================================

function fmtBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

// ======================================================================
// Event binding
// ======================================================================

$btnSend.addEventListener("click", sendMessage);
$input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

$btnCompare.addEventListener("click", () => {
  compareMode = !compareMode;
  $cmpPanel.classList.toggle("hidden", !compareMode);
  $btnCompare.style.background = compareMode ? "var(--accent)" : "";
  $btnCompare.style.color = compareMode ? "#000" : "";
});

$btnSettings.addEventListener("click", () => {
  $settings.classList.remove("hidden");
  // Refresh budget display
  fetch("/api/v1/metrics").then(r => r.json()).then(d => {
    const dp = d.differential_privacy || {};
    const el = document.getElementById("v-budget");
    if (el) {
      el.textContent = dp.dp_total_epsilon_spent != null
        ? dp.dp_total_epsilon_spent.toFixed(2) + " / " + (dp.dp_max_epsilon || 10)
        : "--";
      el.style.color = (dp.dp_budget_remaining || 0) <= 0 ? "var(--error)" : "var(--accent)";
    }
  }).catch(() => {});
});
document.getElementById("btn-close-settings").addEventListener("click", () => {
  $settings.classList.add("hidden");
});

document.getElementById("btn-reset-budget").addEventListener("click", () => {
  fetch("/api/v1/privacy/reset", { method: "POST" })
    .then(r => r.json())
    .then(d => {
      const el = document.getElementById("v-budget");
      if (el) {
        el.textContent = "0.00 / " + (d.max_epsilon || 10);
        el.style.color = "var(--accent)";
      }
      alert("Privacy budget reset! You can send queries again.");
    })
    .catch(e => alert("Reset failed: " + e.message));
});

$metricsBar.addEventListener("click", () => {
  $heBreak.classList.toggle("collapsed");
  $heBreak.classList.toggle("expanded");
});

// ======================================================================
// Boot
// ======================================================================

initSettings();
connectWS();

// Fetch initial metrics
fetch("/api/v1/metrics")
  .then(r => r.json())
  .then(d => {
    if (d.he_active) {
      $badge.textContent = "HE Active";
      $badge.className = "badge he";
    }
  })
  .catch(() => {});
