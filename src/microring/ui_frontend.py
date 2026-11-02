"""Clean, Guided Step-by-Step Conversational UI for `MicroRing Studio v3.1`.

Designed for clarity and zero visual overload:
  - Left Panel: Spacious Guided Chatbot asking ONE clear question at a time with clickable cards/chips.
  - Top Bar: 6-Step Progress Tracker + optional `⚙ Expert Controls` slide-over drawer (hidden by default).
  - Right Panel: Contextual Live Stage showing ONLY the visual relevant to the current step:
      1) `overview` — Workflow Entry Cards + Live Design Blueprint & Cross-Section SVG
      2) `mode`     — 2D Vector Mode |E| Heatmap + Single-Mode Width Cutoff Curve
      3) `mrr`      — 5 Key KPIs (including Propagation Loss dB/cm <-> Q_int) + Spectrum + Coupler/Sweep Curve
      4) `gds`      — Interactive 2D GDSII Polygon & Sub-nm Gap Caliper Canvas + Mask Export
      5) `fit`      — Lab Spectrum CMT Fitter & Propagation Loss (`dB/cm`) Extractor -> 1-Click Design Handoff
"""
from __future__ import annotations

HTML_APP_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MicroRing Studio — Guided Photonics Microring Designer</title>
<style>
:root {
  --bg-app: #0b101b;
  --bg-panel: #111827;
  --bg-card: #162033;
  --bg-card-hover: #1d2a44;
  --bg-input: #0d1424;
  --border: #24324b;
  --border-accent: #38bdf8;
  --text-main: #f1f5f9;
  --text-sub: #cbd5e1;
  --text-muted: #8b9dc3;
  --cyan: #38bdf8;
  --blue: #2563eb;
  --green: #34d399;
  --amber: #fb923c;
  --red: #f87171;
  --purple: #a78bfa;
  --font-ui: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Arial, sans-serif;
  --font-mono: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--font-ui);
  background: var(--bg-app);
  color: var(--text-main);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}

/* ==================== TOP HEADER & STEP PROGRESS ==================== */
header {
  height: 58px;
  background: #0d1424;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  gap: 16px;
  flex-shrink: 0;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-logo {
  width: 32px; height: 32px; border-radius: 9px;
  background: linear-gradient(135deg, #0ea5e9, #6366f1);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 15px; color: #fff;
  box-shadow: 0 2px 10px rgba(14,165,233,0.35);
}
.brand-title { font-size: 15.5px; font-weight: 700; letter-spacing: 0.2px; }
.hw-pill {
  font-size: 11px; padding: 3px 9px; border-radius: 999px;
  background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.35);
  color: var(--green); font-family: var(--font-mono);
}

.stepper {
  display: flex; align-items: center; gap: 6px;
  background: rgba(22,32,51,0.75); padding: 5px 10px;
  border-radius: 999px; border: 1px solid var(--border);
}
.step-item {
  display: flex; align-items: center; gap: 6px;
  font-size: 12px; color: var(--text-muted);
  padding: 4px 10px; border-radius: 999px;
  transition: all 0.18s ease;
  white-space: nowrap;
}
.step-num {
  width: 19px; height: 19px; border-radius: 50%;
  background: #1e293b; color: var(--text-muted);
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700;
}
.step-item.active {
  background: rgba(56,189,248,0.16);
  color: #fff; font-weight: 600;
}
.step-item.active .step-num { background: var(--cyan); color: #08101d; }
.step-item.done { color: #a7f3d0; }
.step-item.done .step-num { background: rgba(52,211,153,0.25); color: var(--green); }
.step-sep { color: #334155; font-size: 11px; }

.header-actions { display: flex; align-items: center; gap: 8px; }
.lang-switch {
  display: inline-flex; align-items: center;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 8px; padding: 2px; gap: 2px;
}
.lang-btn {
  border: none; background: transparent; color: var(--text-muted);
  font-size: 11.5px; font-weight: 700; padding: 4px 9px;
  border-radius: 6px; cursor: pointer; transition: all 0.14s ease;
}
.lang-btn:hover { color: var(--text-main); }
.lang-btn.active { background: var(--cyan); color: #07101e; }

/* ==================== BUTTONS ==================== */
.btn {
  padding: 7px 13px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-card);
  color: var(--text-main); font-size: 12.5px; font-weight: 600;
  cursor: pointer; transition: all 0.15s ease;
  display: inline-flex; align-items: center; gap: 6px;
  text-decoration: none;
}
.btn:hover { background: var(--bg-card-hover); border-color: #475569; }
.btn.primary { background: #0284c7; border-color: #38bdf8; color: #fff; }
.btn.primary:hover { background: #0369a1; }
.btn.green { background: #059669; border-color: #34d399; color: #fff; }
.btn.green:hover { background: #047857; }
.btn.amber { background: #d97706; border-color: #fb923c; color: #fff; }
.btn.subtle { background: transparent; color: var(--text-sub); }
.btn.sm { padding: 5px 10px; font-size: 11.5px; border-radius: 7px; }

/* ==================== MAIN TWO-COLUMN STUDIO ==================== */
.studio-body {
  flex: 1; display: flex; overflow: hidden;
}

/* LEFT: GUIDED CHATBOT ASSISTANT */
.chat-col {
  width: 45%; min-width: 430px; max-width: 640px;
  background: var(--bg-panel);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
}
.chat-scroll {
  flex: 1; overflow-y: auto;
  padding: 22px 24px 14px 24px;
  display: flex; flex-direction: column; gap: 16px;
}
.msg-row { display: flex; gap: 12px; align-items: flex-start; animation: fadeIn 0.18s ease; }
.msg-row.user { flex-direction: row-reverse; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

.avatar {
  width: 30px; height: 30px; border-radius: 8px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700;
}
.msg-row.assistant .avatar { background: linear-gradient(135deg, #0ea5e9, #6366f1); color: #fff; }
.msg-row.user .avatar { background: #1e293b; color: #94a3b8; border: 1px solid var(--border); }

.bubble {
  max-width: 88%; padding: 13px 16px; border-radius: 13px;
  font-size: 13.8px; line-height: 1.62;
}
.msg-row.assistant .bubble {
  background: var(--bg-card); border: 1px solid var(--border); color: var(--text-main);
  border-top-left-radius: 4px;
}
.msg-row.user .bubble {
  background: #1d4ed8; color: #fff; border-top-right-radius: 4px;
}
.bubble p + p { margin-top: 8px; }
.bubble ul { margin: 8px 0 4px 18px; }
.bubble li { margin-bottom: 4px; }
.bubble code {
  background: rgba(8,13,24,0.75); padding: 1.5px 6px; border-radius: 5px;
  font-family: var(--font-mono); font-size: 12.2px; color: #7dd3fc;
}
.msg-row.user .bubble code { background: rgba(255,255,255,0.18); color: #fff; }

/* GUIDED CHOICE CHIPS */
.guided-choices-wrap {
  padding: 10px 24px 12px 24px;
  border-top: 1px solid rgba(36,50,75,0.65);
  background: rgba(13,20,36,0.55);
}
.choices-caption {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px;
  color: var(--text-muted); margin-bottom: 8px; font-weight: 600;
}
.chips-grid {
  display: flex; flex-wrap: wrap; gap: 7px;
}
.chip-btn {
  padding: 8px 13px; border-radius: 10px;
  background: #17233a; border: 1px solid #2b3e63;
  color: #e2e8f0; font-size: 12.8px; font-weight: 500;
  cursor: pointer; transition: all 0.14s ease;
  text-align: left;
}
.chip-btn:hover {
  background: #1e3354; border-color: var(--cyan); color: #fff;
  transform: translateY(-1px);
}

/* CHAT COMPOSER */
.chat-composer {
  padding: 12px 20px 16px 20px;
  background: #0d1424; border-top: 1px solid var(--border);
}
.composer-tools {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px; gap: 8px;
}
.composer-left-tools { display: flex; gap: 6px; }
.composer-box { display: flex; gap: 8px; align-items: center; }
.composer-box input {
  flex: 1; background: var(--bg-card); border: 1px solid var(--border);
  color: var(--text-main); border-radius: 10px; padding: 11px 14px;
  font-size: 13.5px; outline: none; transition: border-color 0.15s;
}
.composer-box input:focus { border-color: var(--cyan); }

/* ==================== RIGHT: CONTEXTUAL LIVE STAGE ==================== */
.stage-col {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
  background: var(--bg-app);
}
.stage-topbar {
  height: 46px; padding: 0 22px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  background: rgba(17,24,39,0.65); flex-shrink: 0;
}
.stage-pills { display: flex; gap: 5px; }
.stage-pill {
  padding: 5px 12px; border-radius: 7px; border: 1px solid transparent;
  background: transparent; color: var(--text-muted);
  font-size: 12px; font-weight: 600; cursor: pointer;
}
.stage-pill:hover { color: var(--text-main); background: rgba(255,255,255,0.04); }
.stage-pill.active {
  background: var(--bg-card); border-color: var(--border); color: var(--cyan);
}
.stage-status { font-size: 11.5px; color: var(--text-muted); font-family: var(--font-mono); }

.stage-content {
  flex: 1; overflow-y: auto; padding: 22px;
}
.stage-view { display: none; flex-direction: column; gap: 16px; max-width: 1040px; margin: 0 auto; }
.stage-view.active { display: flex; }

/* CARDS & GRIDS */
.card {
  background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px;
}
.card-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px;
}
.card-title { font-size: 13.5px; font-weight: 700; color: #e2e8f0; }
.card-sub { font-size: 11.5px; color: var(--text-muted); font-family: var(--font-mono); }

/* WORKFLOW ENTRY CARDS (STAGE 1) */
.workflow-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
}
.wf-card {
  background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; cursor: pointer;
  transition: all 0.16s ease; display: flex; flex-direction: column; gap: 8px;
}
.wf-card:hover { border-color: var(--cyan); background: var(--bg-card); transform: translateY(-2px); }
.wf-icon { font-size: 22px; }
.wf-title { font-size: 13.5px; font-weight: 700; color: #fff; }
.wf-desc { font-size: 12px; color: var(--text-muted); line-height: 1.5; }

/* LIVE BLUEPRINT SPEC GRID */
.blueprint-grid {
  display: grid; grid-template-columns: 1.25fr 1fr; gap: 16px; align-items: center;
}
.spec-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.spec-box {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 9px; padding: 10px 12px;
}
.spec-lbl { font-size: 10.5px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.4px; }
.spec-val { font-size: 13.5px; font-weight: 700; color: var(--cyan); margin-top: 3px; font-family: var(--font-mono); }

/* KPI STRIP */
.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
.kpi-row.four { grid-template-columns: repeat(4, 1fr); }
.kpi-card {
  background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 11px 13px;
}
.kpi-lbl { font-size: 10.5px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
.kpi-val { font-size: 16.5px; font-weight: 800; color: #fff; font-family: var(--font-mono); margin-top: 4px; }
.kpi-val.cyan { color: var(--cyan); }
.kpi-val.green { color: var(--green); }
.kpi-val.amber { color: var(--amber); }
.kpi-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; font-family: var(--font-mono); }

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
canvas.plot-cv { width: 100%; height: 250px; border-radius: 8px; background: #070b14; display: block; }
canvas.plot-cv.tall { height: 300px; }

.warn-banner {
  display: none; padding: 10px 14px; border-radius: 9px;
  background: rgba(245,158,11,0.14); border: 1px solid rgba(251,146,60,0.5);
  color: #fed7aa; font-size: 12.2px; line-height: 1.45;
}

/* ACTION BAR */
.action-bar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 10px 14px; background: var(--bg-panel);
  border: 1px solid var(--border); border-radius: 10px;
}

/* FORM INPUTS (USED IN FITTER & EXPERT DRAWER) */
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 11px; color: var(--text-muted); font-weight: 600; }
.field input, .field select {
  background: var(--bg-input); border: 1px solid var(--border);
  color: var(--text-main); border-radius: 7px; padding: 6px 9px;
  font-size: 12.2px; font-family: var(--font-mono); outline: none;
}
.field input:focus, .field select:focus { border-color: var(--cyan); }

/* TABLE */
table.clean-tbl { width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--font-mono); }
table.clean-tbl th, table.clean-tbl td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }
table.clean-tbl th { color: var(--text-muted); font-size: 11px; }
table.clean-tbl tr:hover { background: rgba(56,189,248,0.08); }

/* ==================== EXPERT DRAWER (HIDDEN BY DEFAULT!) ==================== */
.drawer-backdrop {
  display: none; position: fixed; inset: 0;
  background: rgba(4,8,16,0.65); z-index: 90;
}
.drawer-backdrop.open { display: block; }
.expert-drawer {
  position: fixed; top: 0; right: -560px; width: 540px; max-width: 94vw; height: 100vh;
  background: var(--bg-panel); border-left: 1px solid var(--border);
  z-index: 100; transition: right 0.22s ease;
  display: flex; flex-direction: column;
  box-shadow: -12px 0 40px rgba(0,0,0,0.55);
}
.expert-drawer.open { right: 0; }
.drawer-header {
  padding: 16px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.drawer-body {
  flex: 1; overflow-y: auto; padding: 18px 20px;
  display: flex; flex-direction: column; gap: 18px;
}
.expert-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }

@media (max-width: 1100px) {
  .stepper { display: none; }
  .workflow-grid, .two-col, .blueprint-grid { grid-template-columns: 1fr; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<!-- ==================== TOP HEADER ==================== -->
<header>
  <div class="brand">
    <div class="brand-logo">M</div>
    <div>
      <div class="brand-title">MicroRing Studio</div>
    </div>
    <span class="hw-pill" id="hw-badge">Detecting...</span>
  </div>

  <!-- 6-Step Guided Progress Bar -->
  <div class="stepper" id="wizard-stepper">
    <div class="step-item active" data-step="1"><span class="step-num">1</span><span data-i18n="step1">Goal</span></div>
    <span class="step-sep">›</span>
    <div class="step-item" data-step="2"><span class="step-num">2</span><span data-i18n="step2">Waveband</span></div>
    <span class="step-sep">›</span>
    <div class="step-item" data-step="3"><span class="step-num">3</span><span data-i18n="step3">Material</span></div>
    <span class="step-sep">›</span>
    <div class="step-item" data-step="4"><span class="step-num">4</span><span data-i18n="step4">Stack &amp; Mode</span></div>
    <span class="step-sep">›</span>
    <div class="step-item" data-step="5"><span class="step-num">5</span><span data-i18n="step5">Loss &amp; Ring</span></div>
    <span class="step-sep">›</span>
    <div class="step-item" data-step="6"><span class="step-num">6</span><span data-i18n="step6">Optimize &amp; GDS</span></div>
  </div>

  <div class="header-actions">
    <div class="lang-switch" role="group" aria-label="Language">
      <button class="lang-btn active" id="lang-en" onclick="setLanguage('en')">EN</button>
      <button class="lang-btn" id="lang-zh" onclick="setLanguage('zh')">中文</button>
    </div>
    <button class="btn subtle sm" onclick="sendChatPrompt('reset')" data-i18n="btn_reset">↻ Start Over</button>
    <button class="btn sm" onclick="toggleExpertDrawer(true)" data-i18n="btn_expert">⚙ Expert Controls</button>
  </div>
</header>

<!-- ==================== MAIN STUDIO ==================== -->
<div class="studio-body">

  <!-- LEFT: GUIDED CONVERSATIONAL ASSISTANT -->
  <section class="chat-col">
    <div class="chat-scroll" id="chat-messages"></div>

    <div class="guided-choices-wrap">
      <div class="choices-caption" data-i18n="choices_caption">Suggested Next Step (Click or type below)</div>
      <div class="chips-grid" id="chat-chips"></div>
    </div>

    <div class="chat-composer">
      <div class="composer-tools">
        <div class="composer-left-tools">
          <button class="btn sm" onclick="document.getElementById('gds-file-input').click()" data-i18n="btn_up_gds_short">📐 Upload .GDS</button>
          <button class="btn sm" onclick="openSpectrumFitterDirect()" data-i18n="btn_ext_loss_short">📈 Extract Loss (.CSV)</button>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
          <select id="ai-mode-select" onchange="onAiModeChange()" style="width:auto;padding:4px 8px;font-size:11px;border-radius:7px;">
            <option value="offline" data-i18n="ai_opt_offline">⚡ Offline · Built-in Copilot</option>
            <option value="ollama" data-i18n="ai_opt_ollama">🖥 Offline · Local Ollama / MLX</option>
            <option value="claude" data-i18n="ai_opt_claude">🟣 Online · Claude (CLI / API)</option>
            <option value="codex" data-i18n="ai_opt_codex">🟢 Online · Codex / OpenAI (CLI / API)</option>
          </select>
          <span id="ai-mode-badge" style="font-size:10px;color:#34d399;display:none;white-space:nowrap;"></span>
          <button class="btn sm subtle" id="btn-ai-key" style="display:none;padding:3px 7px;font-size:10.5px;" onclick="toggleAiKeyBox()">🔑 Key</button>
        </div>
      </div>
      <div id="ai-key-bar" style="display:none;margin-bottom:7px;padding:6px 8px;background:rgba(15,23,42,0.85);border:1px solid var(--border);border-radius:8px;font-size:11px;">
        <div style="display:flex;gap:6px;align-items:center;">
          <input id="ai-api-key-input" type="password" placeholder="Optional API Key (kept in memory only; leave blank if claude/codex CLI is signed in)" oninput="saveSessionApiKey(this.value)" style="flex:1;padding:5px 8px;font-size:11px;"/>
          <button class="btn sm subtle" onclick="toggleAiKeyBox()">OK</button>
        </div>
      </div>
      <div class="composer-box">
        <input id="chat-input" type="text" data-i18n-placeholder="chat_placeholder" placeholder="Type an answer or spec (e.g. 'loss = 0.8 dB/cm, FSR 150 GHz, pulley')..." onkeydown="if(event.key==='Enter') sendUserChat()"/>
        <button class="btn primary" id="chat-send-btn" onclick="sendUserChat()" data-i18n="btn_send">Send →</button>
      </div>
    </div>
  </section>

  <!-- RIGHT: CLEAN CONTEXTUAL LIVE STAGE -->
  <main class="stage-col">
    <div class="stage-topbar">
      <div class="stage-pills">
        <button class="stage-pill active" id="pill-overview" onclick="setStageView('overview')" data-i18n="pill_overview">📋 Blueprint</button>
        <button class="stage-pill" id="pill-mode" onclick="setStageView('mode')" data-i18n="pill_mode">🔬 2D Mode</button>
        <button class="stage-pill" id="pill-mrr" onclick="setStageView('mrr')" data-i18n="pill_mrr">📊 Ring Spectrum</button>
        <button class="stage-pill" id="pill-gds" onclick="setStageView('gds')" data-i18n="pill_gds">📐 GDS Mask</button>
        <button class="stage-pill" id="pill-fit" onclick="setStageView('fit')" data-i18n="pill_fit">📈 Loss &amp; Q Fit</button>
        <button class="stage-pill" id="pill-fab" onclick="setStageView('fab'); if(!lastFabResult) runFabToleranceUI();" data-i18n="pill_fab">🧪 Fab &amp; Dose</button>
        <button class="stage-pill" id="pill-tuning" onclick="setStageView('tuning'); if(!lastTuningResult) runActiveTuningUI();" data-i18n="pill_tuning">⚡ Tuning &amp; IC</button>
        <button class="stage-pill" id="pill-io" onclick="setStageView('io'); if(!lastIoResult && !lastHighQResult) runIoCouplerUI();" data-i18n="pill_io">🔌 I/O &amp; High-Q</button>
        <button class="stage-pill" id="pill-mzi" onclick="setStageView('mzi'); if(!lastMziResult) runMziUI();" data-i18n="pill_mzi">🔀 MZI</button>
      </div>
      <span class="stage-status" id="spec-status" data-i18n="status_ready">Ready · 400 nm – 14 µm</span>
    </div>

    <div class="stage-content">

      <!-- ==================== STAGE 1: OVERVIEW & LIVE BLUEPRINT ==================== -->
      <div class="stage-view active" id="view-overview">
        <div class="workflow-grid">
          <div class="wf-card" onclick="sendChatPrompt(currentLang==='zh'?'开始逐步设计微环':'Start step-by-step microring design')">
            <div class="wf-icon">✨</div>
            <div class="wf-title" data-i18n="wf1_title">1. Design a New Ring Step-by-Step</div>
            <div class="wf-desc" data-i18n-html="wf1_desc">Answer one simple question at a time — waveband, material, wafer stack, propagation loss (<code>dB/cm</code>), and target FSR.</div>
          </div>
          <div class="wf-card" onclick="openSpectrumFitterDirect()">
            <div class="wf-icon">📈</div>
            <div class="wf-title" data-i18n-html="wf2_title">2. Extract Loss (<code>dB/cm</code>) from Spectrum</div>
            <div class="wf-desc" data-i18n-html="wf2_desc">Upload your measured <code>.CSV</code> trace (linear or <code>dB</code>) to fit <code>Q_int</code>, propagation loss <code>α [dB/cm]</code>, mode splitting &amp; <code>n_g</code>.</div>
          </div>
          <div class="wf-card" onclick="document.getElementById('gds-file-input').click()">
            <div class="wf-icon">📐</div>
            <div class="wf-title" data-i18n-html="wf3_title">3. Upload &amp; Analyze a <code>.GDS</code> Mask</div>
            <div class="wf-desc" data-i18n-html="wf3_desc">Inspect existing <code>.GDS</code> / <code>.OAS</code> layouts with automatic polygon gap calipers and simulate their optical spectrum.</div>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="bp_card_title">Live Design Blueprint (Updates as You Chat)</span>
            <span class="card-sub" id="bp-stage-badge" data-i18n="bp_badge">Guided Setup</span>
          </div>
          <div class="blueprint-grid">
            <div class="spec-list">
              <div class="spec-box"><div class="spec-lbl" data-i18n="bp_lbl_wl">Wavelength &amp; Pol</div><div class="spec-val" id="bp-wl">1550 nm · TE0</div></div>
              <div class="spec-box"><div class="spec-lbl" data-i18n="bp_lbl_mat">Core Platform</div><div class="spec-val" id="bp-mat">Si3N4</div></div>
              <div class="spec-box"><div class="spec-lbl" data-i18n="bp_lbl_stack">Wafer Stack</div><div class="spec-val" id="bp-stack">H=400 nm, Strip</div></div>
              <div class="spec-box"><div class="spec-lbl" data-i18n="bp_lbl_loss">Propagation Loss α</div><div class="spec-val" id="bp-loss">Foundry Default</div></div>
              <div class="spec-box"><div class="spec-lbl" data-i18n="bp_lbl_top">Topology</div><div class="spec-val" id="bp-top">Point Circular</div></div>
              <div class="spec-box"><div class="spec-lbl" data-i18n="bp_lbl_geom">Radius &amp; Gap</div><div class="spec-val" id="bp-geom">R=40 µm · g=250 nm</div></div>
            </div>
            <div id="svg-xsec-overview" style="display:flex;justify-content:center;"></div>
          </div>
        </div>
      </div>

      <!-- ==================== STAGE 2: 2D EIGENMODE & WIDTH CUTOFF ==================== -->
      <div class="stage-view" id="view-mode">
        <div class="kpi-row four">
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mode_lbl_sm">Single-Mode Max Width</div>
            <div class="kpi-val green" id="mode-kpi-sm">--</div>
            <div class="kpi-sub">TE1 / TM1 Cutoff</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mode_lbl_wbus">Recommended Bus W</div>
            <div class="kpi-val cyan" id="mode-kpi-wbus">--</div>
            <div class="kpi-sub">Strictly Single-Mode</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mode_lbl_wring">Recommended Ring W</div>
            <div class="kpi-val" id="mode-kpi-wring">--</div>
            <div class="kpi-sub">Low Sidewall Loss</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mode_lbl_neff">Fundamental Mode</div>
            <div class="kpi-val amber" id="mode-kpi-neff">--</div>
            <div class="kpi-sub" id="mode-kpi-ng">n_g = --</div>
          </div>
        </div>

        <div class="two-col">
          <div class="card">
            <div class="card-head">
              <span class="card-title" data-i18n="mode_card_field">2D Vector Eigenmode Field</span>
              <div style="display:flex;gap:6px;align-items:center;">
                <select id="mode-comp" onchange="renderSelectedMode()" style="background:#0d1424;color:#e2e8f0;border:1px solid var(--border);border-radius:6px;padding:3px 7px;font-size:11.5px;">
                  <option value="E_abs">|E| Total</option>
                  <option value="Ex_abs">|E_x| Radial (TE)</option>
                  <option value="Ey_abs">|E_y| Vertical (TM)</option>
                </select>
                <span class="card-sub" id="mode-meta"></span>
              </div>
            </div>
            <canvas class="plot-cv" id="cv-mode"></canvas>
            <table class="clean-tbl" id="mode-table" style="margin-top:8px;">
              <thead><tr><th>Mode</th><th>n_eff</th><th>TE %</th><th>Core Γ</th><th>A_eff [µm²]</th></tr></thead>
              <tbody></tbody>
            </table>
          </div>

          <div class="card">
            <div class="card-head">
              <span class="card-title" data-i18n="mode_card_cutoff">Waveguide Width Dispersion &amp; Cutoff</span>
              <span class="card-sub" id="width-cutoff-badge">2D FEM Sweep</span>
            </div>
            <canvas class="plot-cv tall" id="cv-width-sweep"></canvas>
          </div>
        </div>
      </div>

      <!-- ==================== STAGE 3: MRR SPECTRUM & COUPLER SYNTHESIS ==================== -->
      <div class="stage-view" id="view-mrr">
        <div id="wl-warning-bar" class="warn-banner"></div>

        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mrr_lbl_fsr">Free Spectral Range</div>
            <div class="kpi-val cyan" id="kpi-fsr">--</div>
            <div class="kpi-sub" id="kpi-fsr-nm">-- nm</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mrr_lbl_loss">Prop. Loss α &amp; Q_int</div>
            <div class="kpi-val green" id="kpi-loss">-- dB/cm</div>
            <div class="kpi-sub" id="kpi-qint">Q_int --</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mrr_lbl_ql">Loaded Q Factor</div>
            <div class="kpi-val" id="kpi-ql">--</div>
            <div class="kpi-sub" id="kpi-fwhm">FWHM -- pm</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mrr_lbl_er">Extinction Ratio</div>
            <div class="kpi-val amber" id="kpi-er">-- dB</div>
            <div class="kpi-sub" id="kpi-regime">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mrr_lbl_cc">Critical Coupling</div>
            <div class="kpi-val green" id="kpi-cc">--</div>
            <div class="kpi-sub" id="kpi-eo-eff">Target for α</div>
          </div>
        </div>

        <div class="action-bar">
          <button class="btn green sm" onclick="sendChatPrompt(currentLang==='zh'?'临界耦合':'Apply critical coupling')" data-i18n="mrr_btn_cc">⚡ Apply Critical Coupling</button>
          <button class="btn sm" onclick="setStageView('io'); runIoCouplerUI();" data-i18n="mrr_btn_io">🔌 Butt / GC &amp; High-Q</button>
          <button class="btn amber sm" onclick="setStageView('tuning'); runActiveTuningUI();" data-i18n="mrr_btn_tune">🎛️ Active Tuning &amp; INTERCONNECT</button>
          <button class="btn sm" onclick="setStageView('fab'); runFabToleranceUI();" data-i18n="mrr_btn_fab">🧪 Fab / PR Dose Error</button>
          <button class="btn sm" onclick="sendChatPrompt(currentLang==='zh'?'扫描间距':'Sweep gap')" data-i18n="mrr_btn_sw_gap">📏 Sweep Gap</button>
          <button class="btn sm" onclick="sendChatPrompt(currentLang==='zh'?'扫描损耗':'Sweep propagation loss')" data-i18n="mrr_btn_sw_loss">📉 Sweep Loss (dB/cm)</button>
          <button class="btn primary sm" onclick="sendChatPrompt(currentLang==='zh'?'导出单个 GDS 版图':'Export single GDS layout')" data-i18n="mrr_btn_gds">💾 Export .GDS Mask</button>
          <button class="btn sm" onclick="exportSpectrumCSV()" data-i18n="mrr_btn_csv">📥 Download Spectrum .CSV</button>
        </div>

        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="mrr_card_spec">Broadband Transmission Spectrum</span>
            <span class="card-sub" id="kpi-qext">Through &amp; Drop Ports</span>
          </div>
          <canvas class="plot-cv" id="cv-spectrum"></canvas>
        </div>

        <div class="two-col">
          <div class="card">
            <div class="card-head">
              <span class="card-title" id="sweep-plot-title" data-i18n="mrr_card_coupler">Coupler Critical-Coupling Synthesis</span>
              <span class="card-sub" id="sweep-sub">Orange line = Critical Coupling</span>
            </div>
            <canvas class="plot-cv" id="cv-coupler"></canvas>
          </div>
          <div class="card">
            <div class="card-head">
              <span class="card-title" data-i18n="mrr_card_schem">Resonator &amp; Cross-Section Schematic</span>
              <span class="card-sub" id="kpi-shift">Δλ = 0.0 pm</span>
            </div>
            <div id="svg-topdown" style="display:flex;justify-content:center;margin-bottom:8px;"></div>
            <div id="svg-xsec" style="display:flex;justify-content:center;"></div>
          </div>
        </div>
      </div>

      <!-- ==================== STAGE 4: GDS UPLOAD & MASK CREATOR ==================== -->
      <div class="stage-view" id="view-gds">
        <div class="action-bar" style="justify-content:space-between;">
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <input type="file" id="gds-file-input" accept=".gds,.gds2,.oas" style="display:none" onchange="handleGdsFileSelect(this.files[0])"/>
            <button class="btn primary sm" onclick="document.getElementById('gds-file-input').click()" data-i18n="gds_btn_up">📂 Upload .GDS / .OAS</button>
            <button class="btn green sm" onclick="createSingleGdsUI()" data-i18n="gds_btn_single">💾 Generate Single .GDS</button>
            <button class="btn sm" onclick="createSweepArrayGdsUI()" data-i18n="gds_btn_array">🗂️ Generate 5-Ring Sweep Array .GDS</button>
          </div>
          <div id="gds-download-links" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;"></div>
        </div>

        <div class="card" id="gds-extracted-box" style="font-size:12.8px;color:var(--text-sub);" data-i18n-html="gds_intro">
          Upload an existing <code>.GDS</code> / <code>.OAS</code> layout or click <strong>Generate Single .GDS</strong> above to inspect polygons and automatic sub-nm coupler gap calipers.
        </div>

        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="gds_viewer_title">Interactive 2D GDSII Polygon &amp; Gap Caliper Viewer (Scroll Zoom · Drag Pan)</span>
            <div style="display:flex;gap:6px;align-items:center;">
              <div id="gds-layer-pills" style="display:flex;gap:4px;"></div>
              <button class="btn sm" onclick="resetGdsCamera()" data-i18n="gds_btn_reset">Reset Zoom</button>
            </div>
          </div>
          <canvas class="plot-cv tall" id="cv-gds" style="height:370px;cursor:grab;"></canvas>
        </div>
      </div>

      <!-- ==================== STAGE 5: LAB SPECTRUM LOSS (dB/cm) & Q EXTRACTOR ==================== -->
      <div class="stage-view" id="view-fit">
        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n-html="fit_card_title">Extract Waveguide Propagation Loss (<code>dB/cm</code>) &amp; Q from Measured Spectrum</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
              <input type="file" id="csv-fit-input" accept=".csv,.txt,.tsv" style="display:none" onchange="loadLabCsvFile(this.files[0])"/>
              <button class="btn primary sm" onclick="document.getElementById('csv-fit-input').click()" data-i18n="fit_btn_up">📂 Upload Lab .CSV / .TXT</button>
              <button class="btn sm" onclick="loadDemoSpectrum('split_doublet')" data-i18n="fit_demo_doublet">Demo: Split Doublet</button>
              <button class="btn sm" onclick="loadDemoSpectrum('lorentzian')" data-i18n="fit_demo_single">Demo: Single Dip</button>
              <button class="btn sm" onclick="loadDemoSpectrum('multi_fsr')" data-i18n="fit_demo_mfsr">Demo: Multi-FSR (Auto n_g + Loss)</button>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:repeat(4,1fr) auto;gap:10px;align-items:end;">
            <div class="field">
              <label data-i18n="fit_lbl_model">CMT Lineshape Model</label>
              <select id="fit-model" onchange="runCmtFitUI()">
                <option value="auto">Auto (Lorentzian vs Doublet BIC)</option>
                <option value="split_doublet">Backscattering Split Doublet</option>
                <option value="lorentzian">Single Lorentzian</option>
                <option value="fano">Fano + Etalon Background</option>
                <option value="multi_fsr">Broadband Multi-FSR (Auto n_g &amp; α)</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="fit_lbl_regime">Coupling Regime</label>
              <select id="fit-regime" onchange="runCmtFitUI()">
                <option value="">Auto / Report Both Branches</option>
                <option value="under">Known Under-Coupled</option>
                <option value="over">Known Over-Coupled</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="fit_lbl_ng">Group Index n_g (for single dip)</label>
              <input id="fit-ng" type="number" step="0.01" value="2.10" onchange="runCmtFitUI()"/>
            </div>
            <div class="field">
              <label data-i18n="fit_lbl_lrt">Ring Length L_rt [µm] (2πR + 2L_s)</label>
              <input id="fit-lrt" type="number" step="1" value="263.89" onchange="runCmtFitUI()"/>
            </div>
            <button class="btn green" onclick="runCmtFitUI()" data-i18n="fit_btn_run">Run Fit</button>
          </div>
        </div>

        <div class="kpi-row four">
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fit_kpi1">Extracted Prop. Loss α</div>
            <div class="kpi-val green" id="fit-kpi-loss">-- dB/cm</div>
            <div class="kpi-sub" id="fit-kpi-loss-branches">From Q_int &amp; n_g</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fit_kpi2">Intrinsic Q_int</div>
            <div class="kpi-val cyan" id="fit-kpi-qint">--</div>
            <div class="kpi-sub" id="fit-kpi-rtloss">Round-trip loss --</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fit_kpi3">Loaded Q_L &amp; FWHM</div>
            <div class="kpi-val" id="fit-kpi-ql">--</div>
            <div class="kpi-sub" id="fit-kpi-fwhm">FWHM -- pm</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fit_kpi4">Model &amp; Splitting</div>
            <div class="kpi-val amber" id="fit-kpi-split">--</div>
            <div class="kpi-sub" id="fit-kpi-r2">R² = --</div>
          </div>
        </div>

        <div class="action-bar" style="justify-content:space-between;">
          <div id="fit-results-box" style="font-size:12.4px;color:var(--text-sub);line-height:1.5;" data-i18n-html="fit_intro">
            Upload a two-column <code>wavelength_nm, transmission</code> file (linear power OR <code>dB</code>/<code>dBm</code>) to extract propagation loss <code>α [dB/cm]</code> and intrinsic <code>Q_int</code>.
          </div>
          <button class="btn green sm" id="btn-use-extracted-loss" style="display:none;" onclick="applyExtractedLossToDesign()">
            🎯 Design New Ring with This Loss →
          </button>
        </div>

        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="fit_plot_title">Measured Spectrum vs. Analytical CMT Fit</span>
            <span class="card-sub" data-i18n="fit_plot_sub">Supports Linear Power &amp; dB/dBm Traces</span>
          </div>
          <canvas class="plot-cv tall" id="cv-fit"></canvas>
        </div>
      </div>

      <!-- ==================== STAGE 6: CLEANROOM LITHOGRAPHY & PR DOSE TOLERANCE ==================== -->
      <div class="stage-view" id="view-fab">
        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="fab_card_title">Cleanroom Lithography Over-Dose / Under-Dose (Positive vs. Negative PR) &amp; CD Bias Analysis</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
              <button class="btn primary sm" onclick="runFabToleranceUI()" data-i18n="fab_btn_run">▶ Analyze Fab Error</button>
              <button class="btn green sm" onclick="exportFabBiasGdsUI()" data-i18n="fab_btn_gds">🗂️ Export 5-Cell Pre-Biased .GDS</button>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:2.1fr repeat(4,1fr);gap:10px;align-items:end;">
            <div class="field">
              <label data-i18n="fab_lbl_resist">Photoresist / Lithography Polarity</label>
              <select id="fab-resist" onchange="onFabResistChange()">
                <option value="positive_pr" data-i18n="fab_opt_pos">Positive PR (ZEP / CSAR / PMMA / KrF-ArF DUV)</option>
                <option value="negative_pr" data-i18n="fab_opt_neg">Negative PR (HSQ / ma-N / SU-8 / Lift-Off)</option>
                <option value="direct_cd" data-i18n="fab_opt_cd">Direct Foundry CD Bias (ΔCD [nm])</option>
              </select>
            </div>
            <div class="field">
              <label id="fab-lbl-span" data-i18n="fab_lbl_dose">Max Dose Error [±%]</label>
              <input id="fab-dose-pct" type="number" step="1" value="15"/>
            </div>
            <div class="field">
              <label data-i18n="fab_lbl_slope">CD Slope [nm / %]</label>
              <input id="fab-slope" type="number" step="0.2" value="2.0"/>
            </div>
            <div class="field">
              <label data-i18n="fab_lbl_prox">Gap Proximity η</label>
              <input id="fab-prox" type="number" step="0.05" value="1.05"/>
            </div>
            <div class="field">
              <label data-i18n="fab_lbl_dh">Film ΔH [±nm]</label>
              <input id="fab-dh" type="number" step="1" value="0"/>
            </div>
          </div>
          <div id="fab-mech-banner" style="margin-top:10px;padding:9px 12px;border-radius:8px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.28);font-size:12px;color:#cbd5e1;line-height:1.5;">
            Select <strong>Positive PR</strong> or <strong>Negative PR</strong> and click <strong>▶ Analyze Fab Error</strong> to evaluate simultaneous waveguide width <code>W ± ΔCD</code> and anti-correlated gap <code>g ∓ η·ΔCD</code> shifts via 2D FEM + Spatial CMT.
          </div>
          <div id="fab-drc-banner" class="warn-banner" style="margin-top:8px;"></div>
        </div>

        <div class="kpi-row four">
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fab_kpi1">ER ≥ 15 dB Yield Window</div>
            <div class="kpi-val green" id="fab-kpi-win">-- nm</div>
            <div class="kpi-sub" id="fab-kpi-win-sub">Dose window: -- %</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fab_kpi2">Over-Dose (+Max) Corner</div>
            <div class="kpi-val amber" id="fab-kpi-over">-- dB</div>
            <div class="kpi-sub" id="fab-kpi-over-sub">ΔCD -- nm · gap -- nm</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fab_kpi3">Under-Dose (-Max) Corner</div>
            <div class="kpi-val cyan" id="fab-kpi-under">-- dB</div>
            <div class="kpi-sub" id="fab-kpi-under-sub">ΔCD -- nm · gap -- nm</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="fab_kpi4">Resonance CD Sensitivity</div>
            <div class="kpi-val" id="fab-kpi-slope">-- pm/nm</div>
            <div class="kpi-sub" id="fab-kpi-heater">Heater trim: -- K / 10nm</div>
          </div>
        </div>

        <div class="two-col">
          <div class="card">
            <div class="card-head">
              <span class="card-title" data-i18n="fab_plot1_title">Extinction Ratio &amp; Resonance Shift vs. Exposure Dose / CD</span>
              <span class="card-sub" data-i18n="fab_plot1_sub">Green Band = ER ≥ 15 dB Yield Window</span>
            </div>
            <canvas class="plot-cv" id="cv-fab-er"></canvas>
          </div>
          <div class="card">
            <div class="card-head">
              <span class="card-title" data-i18n="fab_plot2_title">3-Corner Overlaid Resonance Dips</span>
              <span class="card-sub" data-i18n="fab_plot2_sub">Under-Dose vs. Nominal vs. Over-Dose</span>
            </div>
            <canvas class="plot-cv" id="cv-fab-dips"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="fab_tbl_title">Corner-by-Corner Lithography &amp; Coupling Breakdown</span>
            <span class="card-sub" id="fab-opc-sub">Anti-Correlated W_fab = W + ΔCD, gap_fab = gap - η·ΔCD</span>
          </div>
          <table class="clean-tbl" id="fab-table">
            <thead>
              <tr>
                <th>Dose Error</th><th>ΔCD [nm]</th><th>W_ring [nm]</th><th>Gap_fab [nm]</th>
                <th>ER [dB]</th><th>Loaded Q_L</th><th>Δλ_res [nm]</th><th>Regime / DRC</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <!-- ==================== STAGE 7: 2D-FEM ACTIVE TUNING (TO / EO / SI-PN) & LUMERICAL INTERCONNECT ==================== -->
      <div class="stage-view" id="view-tuning">
        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="tune_card_title">2D-FEM Active Tuning (TO / EO / Si PN-PIN) &amp; Ansys Lumerical INTERCONNECT Export (Zero COMSOL)</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
              <button class="btn primary sm" onclick="runActiveTuningUI()" data-i18n="tune_btn_run">▶ Solve 2D-FEM Tuning &amp; Spectra</button>
              <button class="btn green sm" onclick="exportInterconnectUI()" data-i18n="tune_btn_ic">📦 Export Lumerical INTERCONNECT (.ZIP)</button>
              <span id="ic-download-slot"></span>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1.55fr 1.25fr 1.2fr;gap:10px;align-items:end;">
            <div class="field">
              <label data-i18n="tune_lbl_mech">Physical Tuning Mechanism</label>
              <select id="tune-mech" onchange="onTuneControlChange('mech')">
                <option value="auto" data-i18n="tune_opt_auto">Auto-Select by Platform (Si PN / Pockels EO / TO Heater)</option>
                <option value="si_pn_depletion" data-i18n="tune_opt_pn">Silicon Lateral PN Depletion (Soref-Bennett + 2D FEM)</option>
                <option value="pockels_eo" data-i18n="tune_opt_eo">Anisotropic Pockels EO (2D Laplace FEM / Direct E)</option>
                <option value="thermo_optic" data-i18n="tune_opt_to">Thermo-Optic Microheater (2D Fourier FEM / Direct ΔT)</option>
                <option value="si_pin_injection" data-i18n="tune_opt_pin">Silicon PIN Forward Injection (Carrier Plasma + FCA)</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_drive">Input Drive / Physical Field</label>
              <select id="tune-drive" onchange="onTuneControlChange('drive')">
                <option value="auto" data-i18n="tune_drv_auto">Auto Default Drive</option>
                <option value="voltage_v" data-i18n="tune_drv_v">Voltage V [V] (2D Geometry-Coupled)</option>
                <option value="current_ma" data-i18n="tune_drv_i">Current I [mA] (2D Geometry-Coupled)</option>
                <option value="power_mw" data-i18n="tune_drv_p">Heater Power P [mW] (2D Fourier FEM)</option>
                <option value="delta_temp_k" data-i18n="tune_drv_t">Direct Temperature Shift ΔT [K]</option>
                <option value="e_field_v_per_um" data-i18n="tune_drv_e">Direct Electric Field E [V/µm]</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_biases">Bias / T / E Sweep Values</label>
              <input id="tune-biases" type="text" value="" placeholder="Auto (e.g. 0, -2, -5 or 0, 20, 50)"/>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;align-items:end;margin-top:10px;">
            <div class="field">
              <label data-i18n="tune_lbl_hgap">Heater Gap [µm]</label>
              <input id="tune-hgap" type="number" step="0.1" value="0.90"/>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_box">BOX Oxide [µm]</label>
              <input id="tune-box" type="number" step="0.2" value="2.00"/>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_egap">EO Gap G [µm]</label>
              <input id="tune-egap" type="number" step="0.25" placeholder="Auto"/>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_doping">Si PN N_A=N_D [cm⁻³]</label>
              <input id="tune-doping" type="text" value="5e17"/>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_pnoff">PN Offset [nm]</label>
              <input id="tune-pnoff" type="number" step="5" value="25"/>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_fact">Active Fraction</label>
              <input id="tune-fact" type="number" step="0.05" min="0.1" max="1.0" value="0.78"/>
            </div>
          </div>

          <div id="tune-geom-banner" style="margin-top:10px;padding:9px 12px;border-radius:8px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.28);font-size:12px;color:#cbd5e1;line-height:1.5;">
            Pure-Python 2D-FEM multiphysics (Fourier heat conduction, anisotropic Laplace electrostatics, or Debye-screened Silicon PN/PIN junction) feeds spatial <code>n(x,y) + i·k(x,y)</code> directly into the 2D vector Maxwell eigenmode solver with zero COMSOL dependency.
          </div>
        </div>

        <div class="kpi-row four">
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="tune_kpi1">Device Tuning Slope</div>
            <div class="kpi-val cyan" id="tune-kpi-slope">--</div>
            <div class="kpi-sub" id="tune-kpi-fslope">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="tune_kpi2">π-Phase-Shift Figure</div>
            <div class="kpi-val green" id="tune-kpi-pi">--</div>
            <div class="kpi-sub" id="tune-kpi-pi-sub">V_π·L / P_π / ΔT_π / E_π</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="tune_kpi3">3-dB Modulation BW</div>
            <div class="kpi-val amber" id="tune-kpi-bw">--</div>
            <div class="kpi-sub" id="tune-kpi-bw-sub">RC / Thermal Cutoff</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="tune_kpi4">2D-FEM Geometry Coupling</div>
            <div class="kpi-val" id="tune-kpi-geom">--</div>
            <div class="kpi-sub" id="tune-kpi-geom-sub">Cross-Section PDE</div>
          </div>
        </div>

        <div class="two-col">
          <div class="card">
            <div class="card-head">
              <span class="card-title" data-i18n="tune_plot1_title">Multi-State Transmission Spectra (V / I / ΔT / E)</span>
              <span class="card-sub" id="tune-plot1-sub">Perturbed 2D-FEM Eigenmode + CMT</span>
            </div>
            <canvas class="plot-cv" id="cv-tune-spec"></canvas>
          </div>
          <div class="card">
            <div class="card-head">
              <span class="card-title" data-i18n="tune_plot2_title">Resonance Shift Δλ [pm] &amp; Active Loss α [dB/cm]</span>
              <span class="card-sub" id="tune-plot2-sub">Solid Cyan = Δλ · Dashed Amber = α</span>
            </div>
            <canvas class="plot-cv" id="cv-tune-shift"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="tune_tbl_title">State-by-State Perturbed 2D-FEM Eigenmode &amp; Resonance Breakdown</span>
            <span class="card-sub" id="tune-tbl-sub">Lumerical INTERCONNECT SPAR + OMM Ready</span>
          </div>
          <table class="clean-tbl" id="tune-table">
            <thead>
              <tr>
                <th>Drive State</th><th>Core Field / Junction</th><th>Re(n_eff)</th><th>Δn_eff [×10⁻⁴]</th>
                <th>Total α [dB/cm]</th><th>Δλ_res [pm]</th><th>ER [dB]</th><th>Loaded Q_L</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <!-- ==================== STAGE 8: CHIP I/O COUPLERS (BUTT / 2D-FDFD GC) & HIGH-Q FDTD CONVERGENCE ==================== -->
      <div class="stage-view" id="view-io">
        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="io_card_title">License-Free Photonic IC Chip I/O (2D-FDFD Grating &amp; 2D-FEM Butt Coupler) + High-Q FDTD Convergence</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
              <button class="btn primary sm" onclick="runIoCouplerUI()" data-i18n="io_btn_run">▶ Solve I/O or High-Q</button>
              <button class="btn sm" id="btn-io-gds-single" onclick="exportIoGdsUI(false)" data-i18n="io_btn_gds">💾 Export Coupler .GDS</button>
              <button class="btn green sm" id="btn-io-gds-loop" onclick="exportIoGdsUI(true)" data-i18n="io_btn_loop">🔁 127 µm FA + Ring Loopback .GDS</button>
              <button class="btn amber sm" id="btn-io-ic" onclick="exportIoInterconnectUI()" data-i18n="io_btn_ic">📦 INTERCONNECT SPAR (.ZIP)</button>
              <button class="btn green sm" id="btn-hq-cutout" style="display:none;" onclick="exportCouplerCutoutUI()" data-i18n="io_btn_cutout">✂️ Export 0.8 ps Coupler Cutout (.GDS+.lsf)</button>
              <span id="io-download-slot"></span>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1.8fr 1.5fr repeat(4,1fr);gap:10px;align-items:end;">
            <div class="field">
              <label data-i18n="io_lbl_mode">Solver Mode (Zero Lumerical / COMSOL / RSoft)</label>
              <select id="io-mode" onchange="onIoModeChange()">
                <option value="grating_coupler" data-i18n="io_opt_gc">Surface Grating Coupler (Bragg + Native 2D Full-Wave FDFD)</option>
                <option value="butt_edge" data-i18n="io_opt_butt">Butt / Edge Coupler (2D Vector FEM + Poynting Overlap + EME)</option>
                <option value="high_q_fdtd" data-i18n="io_opt_hq">High-Q Ring-Down Convergence vs Truncated 3D-FDTD</option>
              </select>
            </div>
            <div class="field" id="io-wrap-fiber">
              <label data-i18n="io_lbl_fiber">Target Optical Fiber / Laser Diode</label>
              <select id="io-fiber">
                <option value="SMF28_1550">SMF-28 (1550 nm, MFD 10.4 µm)</option>
                <option value="Lensed_2p5um">Conical Lensed Fiber (MFD 2.5 µm)</option>
                <option value="DFB_Laser_Butt">III-V DFB Laser Facet (2.4 × 1.2 µm)</option>
                <option value="UHNA_3">UHNA-3 High-NA Fiber (MFD 3.3 µm)</option>
                <option value="Lensed_4p0um">Lensed Fiber (MFD 4.0 µm)</option>
                <option value="SMF28_1310">SMF-28 O-Band (1310 nm, MFD 9.2 µm)</option>
                <option value="SMF_Vis780">780HP Visible Fiber (MFD 5.0 µm)</option>
                <option value="MidIR_InF3">Mid-IR InF3 Fiber (MFD 13.5 µm)</option>
              </select>
            </div>
            <div class="field" id="io-wrap-p1">
              <label id="io-lbl-p1">Etch Depth [µm]</label>
              <input id="io-p1" type="number" step="0.01" placeholder="Auto (e.g. 0.07)"/>
            </div>
            <div class="field" id="io-wrap-p2">
              <label id="io-lbl-p2">Fiber Tilt [deg]</label>
              <input id="io-p2" type="number" step="1" value="10"/>
            </div>
            <div class="field" id="io-wrap-p3">
              <label id="io-lbl-p3">BOX Oxide [µm]</label>
              <input id="io-p3" type="number" step="0.2" value="2.0"/>
            </div>
            <div class="field" id="io-wrap-p4">
              <label id="io-lbl-p4">Teeth Profile</label>
              <select id="io-p4">
                <option value="apod_focus">Apodized + Focusing</option>
                <option value="unif_focus">Uniform + Focusing</option>
                <option value="apod_linear">Apodized + Linear</option>
              </select>
            </div>
          </div>

          <div id="io-banner" style="margin-top:10px;padding:9px 12px;border-radius:8px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.28);font-size:12px;color:#cbd5e1;line-height:1.5;">
            Synthesizes Bragg / Gaussian-apodized teeth and solves full-wave 2D Maxwell Frequency-Domain Finite-Difference (<code>2D FDFD</code> with <code>UPML</code>) across etched grooves, top oxide, <code>BOX</code>, and bottom Si substrate.
          </div>
        </div>

        <div class="kpi-row four">
          <div class="kpi-card">
            <div class="kpi-lbl" id="io-kpi1-lbl">Peak Insertion Loss</div>
            <div class="kpi-val green" id="io-kpi1-val">--</div>
            <div class="kpi-sub" id="io-kpi1-sub">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" id="io-kpi2-lbl">1-dB / 3-dB Bandwidth</div>
            <div class="kpi-val cyan" id="io-kpi2-val">--</div>
            <div class="kpi-sub" id="io-kpi2-sub">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" id="io-kpi3-lbl">Synthesized Geometry</div>
            <div class="kpi-val amber" id="io-kpi3-val">--</div>
            <div class="kpi-sub" id="io-kpi3-sub">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" id="io-kpi4-lbl">Return Loss / Directionality</div>
            <div class="kpi-val" id="io-kpi4-val">--</div>
            <div class="kpi-sub" id="io-kpi4-sub">--</div>
          </div>
        </div>

        <div class="two-col">
          <div class="card">
            <div class="card-head">
              <span class="card-title" id="io-plot1-title">Broadband Coupling Insertion Loss IL(λ) &amp; Reflection S11</span>
              <span class="card-sub" id="io-plot1-sub">2D Full-Wave Maxwell FDFD</span>
            </div>
            <canvas class="plot-cv" id="cv-io-main"></canvas>
          </div>
          <div class="card">
            <div class="card-head">
              <span class="card-title" id="io-plot2-title">2D Full-Wave Radiated Electric Field |E(z, y)|</span>
              <span class="card-sub" id="io-plot2-sub">Guided Launch → Upward Tilted SMF Beam</span>
            </div>
            <canvas class="plot-cv" id="cv-io-aux"></canvas>
          </div>
        </div>
      </div>

      <!-- ==================== STAGE 9: MACH-ZEHNDER INTERFEROMETER (Y-BRANCH / DC / MMI) & MODULATOR ==================== -->
      <div class="stage-view" id="view-mzi">
        <div class="card">
          <div class="card-head">
            <span class="card-title" data-i18n="mzi_card_title">2D-FEM + EME Mach-Zehnder Interferometer (Y-Branch / DC / MMI) &amp; Active Modulator (Zero 3D FDTD)</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
              <button class="btn primary sm" onclick="runMziUI()" data-i18n="mzi_btn_run">▶ Solve MZI &amp; Tuning</button>
              <button class="btn sm" onclick="exportMziGdsUI()" data-i18n="mzi_btn_gds">💾 Export MZI .GDS</button>
              <button class="btn amber sm" onclick="exportMziInterconnectUI()" data-i18n="mzi_btn_ic">📦 INTERCONNECT (.ZIP)</button>
              <span id="mzi-download-slot"></span>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1.9fr repeat(5,1fr);gap:10px;align-items:end;">
            <div class="field">
              <label data-i18n="mzi_lbl_splitter">50:50 Splitter / Combiner (2D-FEM Supermode / EME)</label>
              <select id="mzi-splitter" onchange="onMziSplitterChange()">
                <option value="directional_coupler" data-i18n="mzi_opt_dc">2×2 Directional Coupler (Twin-Core 2D-FEM Supermodes)</option>
                <option value="mmi_2x2" data-i18n="mzi_opt_mmi2">2×2 MMI Coupler (Multi-Mode 2D-FEM EME + Self-Imaging)</option>
                <option value="mmi_1x2" data-i18n="mzi_opt_mmi1">1×2 MMI Splitter (Symmetric 2D-FEM EME)</option>
                <option value="y_branch" data-i18n="mzi_opt_y">1×2 Adiabatic Y-Branch (2D-FEM Apex Supermode Overlap)</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="mzi_lbl_l1">Arm 1 Length L₁ [µm]</label>
              <input id="mzi-l1" type="number" step="50" value="1000"/>
            </div>
            <div class="field">
              <label data-i18n="mzi_lbl_dl">Imbalance ΔL [µm]</label>
              <input id="mzi-dl" type="number" step="5" value="80"/>
            </div>
            <div class="field">
              <label data-i18n="mzi_lbl_fsr">Target FSR [nm]</label>
              <input id="mzi-fsr" type="number" step="0.5" placeholder="Auto from ΔL"/>
            </div>
            <div class="field">
              <label id="mzi-lbl-sp-param">DC Gap [µm]</label>
              <input id="mzi-sp-param" type="number" step="0.02" placeholder="Auto 50:50"/>
            </div>
            <div class="field">
              <label data-i18n="mzi_lbl_ramzi">Ring-Assisted (RAMZI)</label>
              <select id="mzi-ramzi">
                <option value="off">Standard MZI</option>
                <option value="on">RAMZI (+ All-Pass Ring)</option>
              </select>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1.7fr 1.2fr 1.2fr 1.3fr 1fr;gap:10px;align-items:end;margin-top:9px;">
            <div class="field">
              <label data-i18n="mzi_lbl_mech">Active Phase-Shifter Physics</label>
              <select id="mzi-mech">
                <option value="auto" data-i18n="tune_opt_auto">Auto-Select by Platform (Si PN / Pockels EO / TO Heater)</option>
                <option value="passive_only" data-i18n="mzi_opt_passive">Passive MZI Spectrum Only (Fastest)</option>
                <option value="si_pn_depletion" data-i18n="tune_opt_pn">Silicon Lateral PN Depletion (Soref-Bennett + 2D FEM)</option>
                <option value="pockels_eo" data-i18n="tune_opt_eo">Anisotropic Pockels EO (2D Laplace FEM / Direct E)</option>
                <option value="thermo_optic" data-i18n="tune_opt_to">Thermo-Optic Microheater (2D Fourier FEM / Direct ΔT)</option>
                <option value="si_pin_injection" data-i18n="tune_opt_pin">Silicon PIN Forward Injection (Carrier Plasma + FCA)</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="mzi_lbl_scheme">Electrode Drive Scheme</label>
              <select id="mzi-scheme">
                <option value="auto">Auto (Push-Pull EO/PN · Single-Arm TO)</option>
                <option value="push_pull">Push-Pull Differential (±Δφ/2)</option>
                <option value="single_arm">Single-Arm Unbalanced</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_drive">Input Drive / Physical Field</label>
              <select id="mzi-drive">
                <option value="auto" data-i18n="tune_drv_auto">Auto Default Drive</option>
                <option value="voltage_v" data-i18n="tune_drv_v">Voltage V [V] (2D Geometry-Coupled)</option>
                <option value="power_mw" data-i18n="tune_drv_p">Heater Power P [mW] (2D Fourier FEM)</option>
                <option value="current_ma" data-i18n="tune_drv_i">Current I [mA] (2D Geometry-Coupled)</option>
                <option value="delta_temp_k" data-i18n="tune_drv_t">Direct Temperature Shift ΔT [K]</option>
                <option value="e_field_v_per_um" data-i18n="tune_drv_e">Direct Electric Field E [V/µm]</option>
              </select>
            </div>
            <div class="field">
              <label data-i18n="tune_lbl_biases">Bias / T / E Sweep Values</label>
              <input id="mzi-biases" type="text" placeholder="Auto (e.g. 0, -2, -4 or 0, 25, 50)"/>
            </div>
            <div class="field">
              <label data-i18n="mzi_lbl_egap">EO / Heater Gap [µm]</label>
              <input id="mzi-egap" type="number" step="0.1" placeholder="Auto"/>
            </div>
          </div>

          <div id="mzi-banner" style="margin-top:10px;padding:9px 12px;border-radius:8px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.28);font-size:12px;color:#cbd5e1;line-height:1.5;">
            Predicts complete MZI broadband spectra &amp; modulator performance without 3D FDTD: <strong>DC</strong> via mirror-symmetric Twin-Core 2D-FEM Even/Odd Supermodes, <strong>MMI</strong> via Multi-Mode 2D-FEM Eigenmode Expansion (<code>EME</code> + complex Poynting taper overlap), and <strong>Y-Branch</strong> via 2D-FEM waist-narrowed apex supermode overlap.
          </div>
        </div>

        <div class="kpi-row four">
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mzi_kpi1">50:50 Splitter Ratio &amp; IL</div>
            <div class="kpi-val green" id="mzi-kpi1-val">--</div>
            <div class="kpi-sub" id="mzi-kpi1-sub">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mzi_kpi2">MZI FSR &amp; Passive ER</div>
            <div class="kpi-val cyan" id="mzi-kpi2-val">--</div>
            <div class="kpi-sub" id="mzi-kpi2-sub">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mzi_kpi3">Modulator V_π·L / P_π</div>
            <div class="kpi-val amber" id="mzi-kpi3-val">--</div>
            <div class="kpi-sub" id="mzi-kpi3-sub">--</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-lbl" data-i18n="mzi_kpi4">Non-Cavity 3-dB BW &amp; Dyn. ER</div>
            <div class="kpi-val" id="mzi-kpi4-val">--</div>
            <div class="kpi-sub" id="mzi-kpi4-sub">--</div>
          </div>
        </div>

        <div class="two-col">
          <div class="card">
            <div class="card-head">
              <span class="card-title" id="mzi-plot1-title" data-i18n="mzi_plot1_title">Broadband MZI Bar &amp; Cross Transmission Spectra</span>
              <span class="card-sub" id="mzi-plot1-sub">2D-FEM Supermode / EME + Dispersive Arms</span>
            </div>
            <canvas class="plot-cv" id="cv-mzi-spec"></canvas>
          </div>
          <div class="card">
            <div class="card-head">
              <span class="card-title" id="mzi-plot2-title">Switching Transfer Curve T(u) @ λ₀</span>
              <div style="display:flex;gap:5px;align-items:center;">
                <button class="btn sm subtle" id="btn-mzi-aux-tf" style="display:none;padding:2px 7px;font-size:10.5px;" onclick="setMziAuxMode('transfer')">T(V/P)</button>
                <button class="btn sm subtle" id="btn-mzi-aux-mmi" style="display:none;padding:2px 7px;font-size:10.5px;" onclick="setMziAuxMode('mmi_field')">MMI |E(x,z)|</button>
                <span class="card-sub" id="mzi-plot2-sub">Bar &amp; Cross Power</span>
              </div>
            </div>
            <canvas class="plot-cv" id="cv-mzi-aux"></canvas>
          </div>
        </div>

        <div class="card" id="mzi-table-card">
          <div class="card-head">
            <span class="card-title" data-i18n="mzi_tbl_title">State-by-State MZI Phase Shift, Arm Loss &amp; Port Extinction Breakdown</span>
            <span class="card-sub" id="mzi-tbl-sub">Lumerical INTERCONNECT SPAR + Active Arm OMM Ready</span>
          </div>
          <table class="clean-tbl" id="mzi-table">
            <thead>
              <tr>
                <th>Drive State</th><th>Δn_eff (Arm 1 / Arm 2)</th><th>Differential Δφ [π rad]</th>
                <th>Arm α₁ / α₂ [dB/cm]</th><th>Power [mW]</th><th>Bar @ λ₀ [dB]</th><th>Cross @ λ₀ [dB]</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

    </div>
  </main>
</div>

<!-- ==================== SLIDE-OVER EXPERT DRAWER (HIDDEN BY DEFAULT) ==================== -->
<div class="drawer-backdrop" id="drawer-backdrop" onclick="toggleExpertDrawer(false)"></div>
<aside class="expert-drawer" id="expert-drawer">
  <div class="drawer-header">
    <div>
      <div style="font-size:15px;font-weight:700;" data-i18n="exp_title">⚙ Expert Parameter Workbench</div>
      <div style="font-size:11.5px;color:var(--text-muted);" data-i18n="exp_sub">Direct numerical override, 24-material dispersion &amp; hardware benchmark</div>
    </div>
    <button class="btn sm" onclick="toggleExpertDrawer(false)" data-i18n="exp_close">✕ Close</button>
  </div>

  <div class="drawer-body">
    <div class="card">
      <div class="card-head">
        <span class="card-title" data-i18n="exp_override">Manual Geometry, Waveband &amp; Loss Override</span>
        <button class="btn primary sm" onclick="runManualMRR(); toggleExpertDrawer(false);" data-i18n="exp_solve">▶ Solve &amp; Update</button>
      </div>
      <div class="expert-grid">
        <div class="field"><label>Waveband Preset</label>
          <select id="cfg-band" onchange="applyWavebandPresetUI(this.value)">
            <option value="custom">-- Custom λ --</option>
            <option value="532">Visible 532 nm</option>
            <option value="780">Quantum 780 nm</option>
            <option value="1310">O-Band 1310 nm</option>
            <option value="1550" selected>C-Band 1550 nm</option>
            <option value="2000">SWIR 2000 nm</option>
            <option value="4200">Mid-IR 4200 nm</option>
            <option value="7500">LWIR 7500 nm</option>
          </select>
        </div>
        <div class="field"><label>Wavelength [nm]</label><input id="cfg-wl" type="number" step="1" value="1550"/></div>
        <div class="field"><label>Core Material</label><select id="cfg-material"></select></div>
        <div class="field"><label>Topology</label><select id="cfg-topology"></select></div>
        <div class="field"><label>Bottom Clad</label>
          <select id="cfg-clad"><option value="SiO2">SiO2</option><option value="CaF2">CaF2</option><option value="MgF2">MgF2</option><option value="Al2O3">Sapphire</option><option value="Si">Si</option></select>
        </div>
        <div class="field"><label>Top Clad</label>
          <select id="cfg-topclad"><option value="SiO2">SiO2</option><option value="Air">Air</option><option value="CaF2">CaF2</option><option value="PMMA">PMMA</option></select>
        </div>
        <div class="field"><label>Polarization</label><select id="cfg-pol"><option value="TE0">TE0</option><option value="TM0">TM0</option></select></div>
        <div class="field"><label>H_film [µm]</label><input id="cfg-hfilm" type="number" step="0.01" value="0.40"/></div>
        <div class="field"><label>H_slab [µm]</label><input id="cfg-hslab" type="number" step="0.01" value="0.00"/></div>
        <div class="field"><label>Sidewall [deg]</label><input id="cfg-sw" type="number" step="1" value="85"/></div>
        <div class="field"><label>W_ring [µm]</label><input id="cfg-wring" type="number" step="0.02" value="1.20"/></div>
        <div class="field"><label>W_bus [µm]</label><input id="cfg-wbus" type="number" step="0.02" value="1.00"/></div>
        <div class="field"><label>Radius R_1 [µm]</label><input id="cfg-r1" type="number" step="1" value="40"/></div>
        <div class="field"><label>Gap [µm]</label><input id="cfg-gap" type="number" step="0.005" value="0.25"/></div>
        <div class="field"><label>Pulley θ [deg]</label><input id="cfg-theta" type="number" step="1" value="25"/></div>
        <div class="field"><label>Straight L_s [µm]</label><input id="cfg-ls" type="number" step="1" value="12"/></div>
        <div class="field"><label>Prop. Loss [dB/cm]</label><input id="cfg-loss" type="number" step="0.05" placeholder="Auto"/></div>
        <div class="field"><label>Span [nm]</label><input id="cfg-span" type="number" step="1" value="12"/></div>
        <div class="field"><label>Heater ΔT [K]</label><input id="cfg-dt" type="number" step="1" value="0"/></div>
        <div class="field"><label>Bias V [V]</label><input id="cfg-volt" type="number" step="1" value="0"/></div>
        <div class="field"><label>Ring-2 R_2 [µm]</label><input id="cfg-r2" type="number" step="0.5" value="42.5"/></div>
      </div>
      <!-- Hidden helper inputs for 2D mode & sweep array compatibility -->
      <input type="hidden" id="mode-w" value="1.20"/>
      <input type="hidden" id="mode-r" value="40"/>
      <input type="hidden" id="mode-backend" value="fem"/>
      <input type="hidden" id="gds-sw-param" value="gap_thru_um"/>
      <input type="hidden" id="gds-sw-min" value="0.16"/>
      <input type="hidden" id="gds-sw-max" value="0.32"/>
      <input type="hidden" id="gds-sw-n" value="5"/>
    </div>

    <div class="card">
      <div class="card-head">
        <span class="card-title">24-Material Sellmeier Dispersion (380 – 10,000 nm)</span>
        <select id="mat-browser-sel" onchange="loadMaterialCurve()" style="background:#0d1424;color:#e2e8f0;border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:11.5px;"></select>
      </div>
      <canvas class="plot-cv" id="cv-mat" style="height:190px;"></canvas>
      <div id="mat-props-text" style="margin-top:8px;font-size:11.5px;color:var(--text-muted);font-family:var(--font-mono);"></div>
    </div>

    <div class="card">
      <div class="card-head">
        <span class="card-title">Local Compute Engine</span>
        <button class="btn sm" onclick="runBenchmarkUI()">⚡ Benchmark</button>
      </div>
      <div id="hw-detail-box" style="font-size:12px;color:var(--text-sub);line-height:1.6;"></div>
    </div>
  </div>
</aside>

<script>
// ==================== GLOBAL STATE & BILINGUAL I18N ====================
let currentLang = (typeof localStorage !== 'undefined' && localStorage.getItem('microring_lang')) || 'en';
let copilotSession = null;
let catalogMaterials = [];
let catalogTopologies = {};
let lastMrrResult = null;
let lastModesData = null;
let selectedModeIdx = 0;
let lastWidthSweep = null;
let currentGdsView = null;
let labSpectrumData = null;
let lastExtractedLossDbCm = null;
let lastFabResult = null;
let lastTuningResult = null;
let lastIoResult = null;
let lastHighQResult = null;
let ioBusy = false;
let lastMziResult = null;
let mziBusy = false;
let mziAuxMode = 'auto';
let aiBackendStatus = { offline: true, ollama_available: false, claude_cli_available: false, claude_env_key: false, codex_cli_available: false, codex_env_key: false };

const UI_I18N = {
  en: {
    step1: 'Goal', step2: 'Waveband', step3: 'Material', step4: 'Stack & Mode', step5: 'Loss & Ring', step6: 'Optimize & GDS',
    btn_reset: '↻ Start Over', btn_expert: '⚙ Expert Controls',
    choices_caption: 'Suggested Next Step (Click or type below)',
    btn_up_gds_short: '📐 Upload .GDS', btn_ext_loss_short: '📈 Extract Loss (.CSV)', lbl_ollama: 'Local LLM Tip',
    ai_opt_offline: '⚡ Offline · Built-in Copilot',
    ai_opt_ollama: '🖥 Offline · Local Ollama / MLX',
    ai_opt_claude: '🟣 Online · Claude (CLI / API)',
    ai_opt_codex: '🟢 Online · Codex / OpenAI (CLI / API)',
    chat_placeholder: "Type an answer or spec (e.g. 'loss = 0.8 dB/cm, FSR 150 GHz, pulley')...",
    btn_send: 'Send →',
    pill_overview: '📋 Blueprint', pill_mode: '🔬 2D Mode', pill_mrr: '📊 Ring Spectrum', pill_gds: '📐 GDS Mask', pill_fit: '📈 Loss & Q Fit', pill_fab: '🧪 Fab & Dose', pill_tuning: '⚡ Tuning & IC', pill_io: '🔌 Butt/GC & High-Q', pill_mzi: '🔀 MZI',
    status_ready: 'Ready · 400 nm – 14 µm',
    wf1_title: '1. Design a New Ring Step-by-Step',
    wf1_desc: 'Answer one simple question at a time — waveband, material, wafer stack, propagation loss (<code>dB/cm</code>), and target FSR.',
    wf2_title: '2. Extract Loss (<code>dB/cm</code>) from Spectrum',
    wf2_desc: 'Upload your measured <code>.CSV</code> trace (linear or <code>dB</code>) to fit <code>Q_int</code>, propagation loss <code>α [dB/cm]</code>, mode splitting &amp; <code>n_g</code>.',
    wf3_title: '3. Upload &amp; Analyze a <code>.GDS</code> Mask',
    wf3_desc: 'Inspect existing <code>.GDS</code> / <code>.OAS</code> layouts with automatic polygon gap calipers and simulate their optical spectrum.',
    bp_card_title: 'Live Design Blueprint (Updates as You Chat)', bp_badge: 'Guided Setup',
    bp_lbl_wl: 'Wavelength & Pol', bp_lbl_mat: 'Core Platform', bp_lbl_stack: 'Wafer Stack',
    bp_lbl_loss: 'Propagation Loss α', bp_lbl_top: 'Topology', bp_lbl_geom: 'Radius & Gap',
    mode_lbl_sm: 'Single-Mode Max Width', mode_lbl_wbus: 'Recommended Bus W', mode_lbl_wring: 'Recommended Ring W', mode_lbl_neff: 'Fundamental Mode',
    mode_card_field: '2D Vector Eigenmode Field', mode_card_cutoff: 'Waveguide Width Dispersion & Cutoff',
    mrr_lbl_fsr: 'Free Spectral Range', mrr_lbl_loss: 'Prop. Loss α & Q_int', mrr_lbl_ql: 'Loaded Q Factor', mrr_lbl_er: 'Extinction Ratio', mrr_lbl_cc: 'Critical Coupling',
    mrr_btn_cc: '⚡ Apply Critical Coupling', mrr_btn_io: '🔌 Butt / GC & High-Q', mrr_btn_tune: '🎛️ Active Tuning & INTERCONNECT', mrr_btn_fab: '🧪 Fab / PR Dose Error', mrr_btn_sw_gap: '📏 Sweep Gap', mrr_btn_sw_loss: '📉 Sweep Loss (dB/cm)',
    mrr_btn_gds: '💾 Export .GDS Mask', mrr_btn_csv: '📥 Download Spectrum .CSV',
    mrr_card_spec: 'Broadband Transmission Spectrum', mrr_card_coupler: 'Coupler Critical-Coupling Synthesis', mrr_card_schem: 'Resonator & Cross-Section Schematic',
    gds_btn_up: '📂 Upload .GDS / .OAS', gds_btn_single: '💾 Generate Single .GDS', gds_btn_array: '🗂️ Generate 5-Ring Sweep Array .GDS',
    gds_intro: 'Upload an existing <code>.GDS</code> / <code>.OAS</code> layout or click <strong>Generate Single .GDS</strong> above to inspect polygons and automatic sub-nm coupler gap calipers.',
    gds_viewer_title: 'Interactive 2D GDSII Polygon & Gap Caliper Viewer (Scroll Zoom · Drag Pan)', gds_btn_reset: 'Reset Zoom',
    fit_card_title: 'Extract Waveguide Propagation Loss (<code>dB/cm</code>) &amp; Q from Measured Spectrum',
    fit_btn_up: '📂 Upload Lab .CSV / .TXT', fit_demo_doublet: 'Demo: Split Doublet', fit_demo_single: 'Demo: Single Dip', fit_demo_mfsr: 'Demo: Multi-FSR (Auto n_g + Loss)',
    fit_lbl_model: 'CMT Lineshape Model', fit_lbl_regime: 'Coupling Regime', fit_lbl_ng: 'Group Index n_g (for single dip)', fit_lbl_lrt: 'Ring Length L_rt [µm] (2πR + 2L_s)',
    fit_btn_run: 'Run Fit',
    fit_kpi1: 'Extracted Prop. Loss α', fit_kpi2: 'Intrinsic Q_int', fit_kpi3: 'Loaded Q_L & FWHM', fit_kpi4: 'Model & Splitting',
    fit_intro: 'Upload a two-column <code>wavelength_nm, transmission</code> file (linear power OR <code>dB</code>/<code>dBm</code>) to extract propagation loss <code>α [dB/cm]</code> and intrinsic <code>Q_int</code>.',
    fit_plot_title: 'Measured Spectrum vs. Analytical CMT Fit', fit_plot_sub: 'Supports Linear Power & dB/dBm Traces',
    exp_title: '⚙ Expert Parameter Workbench', exp_sub: 'Direct numerical override, 24-material dispersion & hardware benchmark',
    exp_close: '✕ Close', exp_override: 'Manual Geometry, Waveband & Loss Override', exp_solve: '▶ Solve & Update',
    fab_card_title: 'Cleanroom Lithography Over-Dose / Under-Dose (Positive vs. Negative PR) & CD Bias Analysis',
    fab_btn_run: '▶ Analyze Fab Error', fab_btn_gds: '🗂️ Export 5-Cell Pre-Biased .GDS',
    fab_lbl_resist: 'Photoresist / Lithography Polarity',
    fab_opt_pos: 'Positive PR (ZEP / CSAR / PMMA / KrF-ArF DUV)',
    fab_opt_neg: 'Negative PR (HSQ / ma-N / SU-8 / Lift-Off)',
    fab_opt_cd: 'Direct Foundry CD Bias (ΔCD [nm])',
    fab_lbl_dose: 'Max Dose Error [±%]', fab_lbl_slope: 'CD Slope [nm / %]', fab_lbl_prox: 'Gap Proximity η', fab_lbl_dh: 'Film ΔH [±nm]',
    fab_kpi1: 'ER ≥ 15 dB Yield Window', fab_kpi2: 'Over-Dose (+Max) Corner', fab_kpi3: 'Under-Dose (-Max) Corner', fab_kpi4: 'Resonance CD Sensitivity',
    fab_plot1_title: 'Extinction Ratio & Resonance Shift vs. Exposure Dose / CD', fab_plot1_sub: 'Green Band = ER ≥ 15 dB Yield Window',
    fab_plot2_title: '3-Corner Overlaid Resonance Dips', fab_plot2_sub: 'Under-Dose vs. Nominal vs. Over-Dose',
    fab_tbl_title: 'Corner-by-Corner Lithography & Coupling Breakdown',
    tune_card_title: '2D-FEM Active Tuning (TO / EO / Si PN-PIN) & Ansys Lumerical INTERCONNECT Export (Zero COMSOL)',
    tune_btn_run: '▶ Solve 2D-FEM Tuning & Spectra', tune_btn_ic: '📦 Export Lumerical INTERCONNECT (.ZIP)',
    tune_lbl_mech: 'Physical Tuning Mechanism', tune_lbl_drive: 'Input Drive / Physical Field', tune_lbl_biases: 'Bias / T / E Sweep Values',
    tune_opt_auto: 'Auto-Select by Platform (Si PN / Pockels EO / TO Heater)',
    tune_opt_pn: 'Silicon Lateral PN Depletion (Soref-Bennett + 2D FEM)',
    tune_opt_eo: 'Anisotropic Pockels EO (2D Laplace FEM / Direct E)',
    tune_opt_to: 'Thermo-Optic Microheater (2D Fourier FEM / Direct ΔT)',
    tune_opt_pin: 'Silicon PIN Forward Injection (Carrier Plasma + FCA)',
    tune_drv_auto: 'Auto Default Drive', tune_drv_v: 'Voltage V [V] (2D Geometry-Coupled)', tune_drv_i: 'Current I [mA] (2D Geometry-Coupled)',
    tune_drv_p: 'Heater Power P [mW] (2D Fourier FEM)', tune_drv_t: 'Direct Temperature Shift ΔT [K]', tune_drv_e: 'Direct Electric Field E [V/µm]',
    tune_lbl_hgap: 'Heater Gap [µm]', tune_lbl_box: 'BOX Oxide [µm]', tune_lbl_egap: 'EO Gap G [µm]',
    tune_lbl_doping: 'Si PN N_A=N_D [cm⁻³]', tune_lbl_pnoff: 'PN Offset [nm]', tune_lbl_fact: 'Active Fraction',
    tune_kpi1: 'Device Tuning Slope', tune_kpi2: 'π-Phase-Shift Figure', tune_kpi3: '3-dB Modulation BW', tune_kpi4: '2D-FEM Geometry Coupling',
    tune_plot1_title: 'Multi-State Transmission Spectra (V / I / ΔT / E)',
    tune_plot2_title: 'Resonance Shift Δλ [pm] & Active Loss α [dB/cm]',
    tune_tbl_title: 'State-by-State Perturbed 2D-FEM Eigenmode & Resonance Breakdown',
    io_card_title: 'License-Free Photonic IC Chip I/O (2D-FDFD Grating & 2D-FEM Butt Coupler) + High-Q FDTD Convergence',
    io_btn_run: '▶ Solve I/O or High-Q',
    io_btn_gds: '💾 Export Coupler .GDS',
    io_btn_loop: '🔁 127 µm FA + Ring Loopback .GDS',
    io_btn_ic: '📦 INTERCONNECT SPAR (.ZIP)',
    io_btn_cutout: '✂️ Export 0.8 ps Coupler Cutout (.GDS+.lsf)',
    io_lbl_mode: 'Solver Mode (Zero Lumerical / COMSOL / RSoft)',
    io_opt_gc: 'Surface Grating Coupler (Bragg + Native 2D Full-Wave FDFD)',
    io_opt_butt: 'Butt / Edge Coupler (2D Vector FEM + Poynting Overlap + EME)',
    io_opt_hq: 'High-Q Ring-Down Convergence vs Truncated 3D-FDTD',
    io_lbl_fiber: 'Target Optical Fiber / Laser Diode',
    mzi_card_title: '2D-FEM + EME Mach-Zehnder Interferometer (Y-Branch / DC / MMI) & Active Modulator (Zero 3D FDTD)',
    mzi_btn_run: '▶ Solve MZI & Tuning', mzi_btn_gds: '💾 Export MZI .GDS', mzi_btn_ic: '📦 INTERCONNECT (.ZIP)',
    mzi_lbl_splitter: '50:50 Splitter / Combiner (2D-FEM Supermode / EME)',
    mzi_opt_dc: '2×2 Directional Coupler (Twin-Core 2D-FEM Supermodes)',
    mzi_opt_mmi2: '2×2 MMI Coupler (Multi-Mode 2D-FEM EME + Self-Imaging)',
    mzi_opt_mmi1: '1×2 MMI Splitter (Symmetric 2D-FEM EME)',
    mzi_opt_y: '1×2 Adiabatic Y-Branch (2D-FEM Apex Supermode Overlap)',
    mzi_lbl_l1: 'Arm 1 Length L₁ [µm]', mzi_lbl_dl: 'Imbalance ΔL [µm]', mzi_lbl_fsr: 'Target FSR [nm]',
    mzi_lbl_ramzi: 'Ring-Assisted (RAMZI)', mzi_lbl_mech: 'Active Phase-Shifter Physics',
    mzi_opt_passive: 'Passive MZI Spectrum Only (Fastest)', mzi_lbl_scheme: 'Electrode Drive Scheme',
    mzi_lbl_egap: 'EO / Heater Gap [µm]',
    mzi_kpi1: '50:50 Splitter Ratio & IL', mzi_kpi2: 'MZI FSR & Passive ER',
    mzi_kpi3: 'Modulator V_π·L / P_π', mzi_kpi4: 'Non-Cavity 3-dB BW & Dyn. ER',
    mzi_plot1_title: 'Broadband MZI Bar & Cross Transmission Spectra',
    mzi_tbl_title: 'State-by-State MZI Phase Shift, Arm Loss & Port Extinction Breakdown'
  },
  zh: {
    step1: '选择目标', step2: '工作波段', step3: '芯层材料', step4: '叠层与模式', step5: '损耗与微环', step6: '耦合优化与版图',
    btn_reset: '↻ 重新开始', btn_expert: '⚙ 专家参数',
    choices_caption: '下一步引导选项（点击按钮或在下方输入）',
    btn_up_gds_short: '📐 上传 .GDS', btn_ext_loss_short: '📈 提取损耗 (.CSV)', lbl_ollama: '本地大模型建议',
    ai_opt_offline: '⚡ 离线模式 · 内置物理引擎',
    ai_opt_ollama: '🖥 离线模式 · 本地 Ollama / MLX',
    ai_opt_claude: '🟣 在线模式 · Claude (CLI / API)',
    ai_opt_codex: '🟢 在线模式 · Codex / OpenAI (CLI / API)',
    chat_placeholder: '输入您的需求或参数（例如：损耗 0.8 dB/cm, FSR 150 GHz, 滑轮耦合）...',
    btn_send: '发送 →',
    pill_overview: '📋 设计蓝图', pill_mode: '🔬 二维模式', pill_mrr: '📊 微环光谱', pill_gds: '📐 GDS 版图', pill_fit: '📈 损耗与Q拟合', pill_fab: '🧪 工艺与剂量', pill_tuning: '⚡ 调谐与 IC', pill_io: '🔌 端面/光栅与高Q', pill_mzi: '🔀 MZI 调制',
    status_ready: '就绪 · 400 nm – 14 µm',
    wf1_title: '1. 逐步引导设计新微环',
    wf1_desc: '每次只问一个清晰问题——工作波段、波导材料、晶圆叠层、传播损耗 (<code>dB/cm</code>) 与目标 FSR。',
    wf2_title: '2. 从实测光谱提取损耗 (<code>dB/cm</code>)',
    wf2_desc: '上传实验 <code>.CSV</code> 透射谱（支持线性功率或 <code>dB</code>/<code>dBm</code>），自动拟合 <code>Q_int</code>、传播损耗 <code>α [dB/cm]</code>、背向散射双峰劈裂与 <code>n_g</code>。',
    wf3_title: '3. 上传并解析 <code>.GDS</code> 版图',
    wf3_desc: '读取已有 <code>.GDS</code> / <code>.OAS</code> 掩膜版图，自动亚纳米测量耦合间距并计算透射光谱。',
    bp_card_title: '实时设计参数蓝图（随对话自动更新）', bp_badge: '分步引导模式',
    bp_lbl_wl: '工作波长与偏振', bp_lbl_mat: '波导芯层平台', bp_lbl_stack: '晶圆垂直叠层',
    bp_lbl_loss: '波导传播损耗 α', bp_lbl_top: '微环拓扑结构', bp_lbl_geom: '半径与耦合间距',
    mode_lbl_sm: '单模最大截止宽度', mode_lbl_wbus: '推荐直波导宽 W_bus', mode_lbl_wring: '推荐微环宽 W_ring', mode_lbl_neff: '基模有效折射率',
    mode_card_field: '二维全矢量本征模场分布', mode_card_cutoff: '波导宽度色散与单模截止曲线',
    mrr_lbl_fsr: '自由光谱范围 FSR', mrr_lbl_loss: '传播损耗 α 与本征 Q', mrr_lbl_ql: '有载品质因数 Q_L', mrr_lbl_er: '谐振消光比 ER', mrr_lbl_cc: '临界耦合目标参数',
    mrr_btn_cc: '⚡ 一键应用临界耦合', mrr_btn_io: '🔌 端面/光栅耦合与高Q', mrr_btn_tune: '🎛️ 有源调谐与 INTERCONNECT', mrr_btn_fab: '🧪 正/负胶曝光容差', mrr_btn_sw_gap: '📏 扫描耦合间距', mrr_btn_sw_loss: '📉 扫描传播损耗 (dB/cm)',
    mrr_btn_gds: '💾 导出 .GDS 掩膜版图', mrr_btn_csv: '📥 下载光谱 .CSV',
    mrr_card_spec: '宽谱透射与下载光谱', mrr_card_coupler: '耦合器临界耦合综合曲线', mrr_card_schem: '微环拓扑与波导截面示意图',
    gds_btn_up: '📂 上传 .GDS / .OAS', gds_btn_single: '💾 生成单器件 .GDS', gds_btn_array: '🗂️ 生成 5 环容差阵列 .GDS',
    gds_intro: '请上传已有 <code>.GDS</code> / <code>.OAS</code> 版图，或点击上方 <strong>生成单器件 .GDS</strong> 查看多边形与耦合间距自动卡尺。',
    gds_viewer_title: '交互式 2D GDSII 多边形与耦合间距卡尺预览（滚轮缩放 · 拖拽平移）', gds_btn_reset: '重置缩放',
    fit_card_title: '从实测透射谱提取波导传播损耗 (<code>dB/cm</code>) 与本征 Q 值',
    fit_btn_up: '📂 上传实验 .CSV / .TXT', fit_demo_doublet: '示例：高Q双峰劈裂', fit_demo_single: '示例：单谐振谷', fit_demo_mfsr: '示例：多谐振峰 (自动提取 n_g + 损耗)',
    fit_lbl_model: '耦合模线型模型', fit_lbl_regime: '耦合状态分支', fit_lbl_ng: '群折射率 n_g (单峰拟合用)', fit_lbl_lrt: '微环周长 L_rt [µm] (2πR + 2L_s)',
    fit_btn_run: '运行拟合',
    fit_kpi1: '提取波导传播损耗 α', fit_kpi2: '本征品质因数 Q_int', fit_kpi3: '有载 Q_L 与线宽', fit_kpi4: '拟合模型与模式劈裂',
    fit_intro: '上传两列 <code>wavelength_nm, transmission</code> 数据（支持线性透过率或 <code>dB</code>/<code>dBm</code>），一键提取波导传播损耗 <code>α [dB/cm]</code> 与 <code>Q_int</code>。',
    fit_plot_title: '实验测量光谱 vs. 解析耦合模 (CMT) 拟合曲线', fit_plot_sub: '自动兼容线性功率与 dB/dBm 光谱',
    exp_title: '⚙ 专家参数工作台', exp_sub: '直接数值覆盖、24 种跨波段材料色散库与硬件算力基准测试',
    exp_close: '✕ 关闭', exp_override: '手动几何、波长与损耗覆盖', exp_solve: '▶ 求解并更新',
    fab_card_title: '超净间光刻过曝光 / 欠曝光（正胶 vs. 负胶）与线宽偏置容差分析',
    fab_btn_run: '▶ 运行工艺误差分析', fab_btn_gds: '🗂️ 导出 5 单元预偏置补偿 .GDS',
    fab_lbl_resist: '光刻胶极性 / 曝光工艺模式',
    fab_opt_pos: '正性光刻胶 Positive PR (ZEP / CSAR / PMMA / DUV 正胶)',
    fab_opt_neg: '负性光刻胶 Negative PR (HSQ / ma-N / SU-8 / 剥离反转)',
    fab_opt_cd: '代工厂线宽偏置直接扫描 (ΔCD [nm])',
    fab_lbl_dose: '最大剂量误差 [±%]', fab_lbl_slope: '线宽斜率 [nm / %]', fab_lbl_prox: '窄缝邻近因子 η', fab_lbl_dh: '膜厚误差 ΔH [±nm]',
    fab_kpi1: 'ER ≥ 15 dB 良率窗口', fab_kpi2: '过曝光 (+Max) 极值点', fab_kpi3: '欠曝光 (-Max) 极值点', fab_kpi4: '谐振波长线宽灵敏度',
    fab_plot1_title: '消光比与谐振漂移随曝光剂量 / 线宽变化曲线', fab_plot1_sub: '绿色阴影 = ER ≥ 15 dB 良率窗口',
    fab_plot2_title: '欠曝光 / 标称 / 过曝光三态谐振峰线型对比', fab_plot2_sub: '中心对齐相对失谐对比',
    fab_tbl_title: '逐曝光梯度工艺线宽、耦合间距与光学指标明细表',
    tune_card_title: '纯自研 2D-FEM 多物理场有源调谐 (TO / EO / 硅 PN-PIN) 与 Lumerical INTERCONNECT 导出 (零 COMSOL)',
    tune_btn_run: '▶ 计算 2D-FEM 调谐与多态光谱', tune_btn_ic: '📦 导出 Lumerical INTERCONNECT (.ZIP)',
    tune_lbl_mech: '物理调谐机制', tune_lbl_drive: '驱动量 / 物理场输入', tune_lbl_biases: '扫描电压 / 电流 / 温升 / 电场序列',
    tune_opt_auto: '按材料自动匹配 (硅 PN 耗尽 / Pockels 电光 / 热光微加热器)',
    tune_opt_pn: '硅横向 PN 结反向耗尽 (Soref-Bennett + 2D FEM)',
    tune_opt_eo: '各向异性 Pockels 电光调制 (2D Laplace 静电场 / 直接电场)',
    tune_opt_to: '热光微加热器调谐 (2D Fourier 热传导有限元 / 直接温升 ΔT)',
    tune_opt_pin: '硅 PIN 结正向载流子注入 (等离子体色散 + FCA)',
    tune_drv_auto: '默认物理激励', tune_drv_v: '电压 V [V] (关联截面几何)', tune_drv_i: '电流 I [mA] (关联截面几何)',
    tune_drv_p: '加热功率 P [mW] (2D 热有限元)', tune_drv_t: '直接输入温度变化 ΔT [K]', tune_drv_e: '直接输入电场强度 E [V/µm]',
    tune_lbl_hgap: '加热器间距 [µm]', tune_lbl_box: '埋氧层 BOX [µm]', tune_lbl_egap: '电光电极间距 G [µm]',
    tune_lbl_doping: '硅 PN 掺杂浓度 [cm⁻³]', tune_lbl_pnoff: 'PN 结偏移 [nm]', tune_lbl_fact: '有源区占空比',
    tune_kpi1: '器件调谐效率斜率', tune_kpi2: 'π 相移关键指标', tune_kpi3: '3-dB 电光/热光带宽', tune_kpi4: '2D-FEM 截面多物理耦合',
    tune_plot1_title: '多偏置 / 多温度 / 多电场透射光谱簇',
    tune_plot2_title: '谐振漂移 Δλ [pm] 与有源模态损耗 α [dB/cm]',
    tune_tbl_title: '逐状态微扰 2D-FEM 本征模与谐振指标明细表',
    io_card_title: '零商业软件光子芯片 I/O 耦合 (2D-FDFD 光栅与 2D-FEM 端面耦合器) + 超高 Q 值 FDTD 收敛引擎',
    io_btn_run: '▶ 求解耦合器或高Q收敛',
    io_btn_gds: '💾 导出耦合器 .GDS',
    io_btn_loop: '🔁 127 µm 光纤阵列 + 微环回环 .GDS',
    io_btn_ic: '📦 INTERCONNECT S参数 (.ZIP)',
    io_btn_cutout: '✂️ 导出单次耦合区切片 (.GDS+.lsf)',
    io_lbl_mode: '求解模式 (零 Lumerical / COMSOL / RSoft)',
    io_opt_gc: '表面光栅耦合器 (Bragg 变迹综合 + 原生二维全波 FDFD)',
    io_opt_butt: '端面模斑转换耦合器 (2D 全矢量 FEM + 复坡印廷重叠 + EME)',
    io_opt_hq: '超高 Q 值谐振振铃收敛诊断 vs 截断 3D-FDTD',
    io_lbl_fiber: '目标光纤 / 半导体激光器模斑',
    mzi_card_title: '零 3D-FDTD 马赫-曾德尔干涉仪 (Y 分支 / DC / MMI) 与高速调制器分析 (2D-FEM + EME)',
    mzi_btn_run: '▶ 求解 MZI 与有源调制', mzi_btn_gds: '💾 导出 MZI .GDS', mzi_btn_ic: '📦 INTERCONNECT (.ZIP)',
    mzi_lbl_splitter: '50:50 分束/合束器 (2D-FEM 超模 / 多模干涉 EME)',
    mzi_opt_dc: '2×2 定向耦合器 DC (双芯对称 2D-FEM 奇偶超模)',
    mzi_opt_mmi2: '2×2 多模干涉仪 MMI (多模 2D-FEM EME + 自成像)',
    mzi_opt_mmi1: '1×2 对称 MMI 分束器 (2D-FEM EME)',
    mzi_opt_y: '1×2 绝热 Y 分支 (2D-FEM 分叉尖端超模重叠)',
    mzi_lbl_l1: '第一臂长 L₁ [µm]', mzi_lbl_dl: '臂长差 ΔL [µm]', mzi_lbl_fsr: '目标 FSR [nm]',
    mzi_lbl_ramzi: '微环辅助 (RAMZI)', mzi_lbl_mech: '相移臂有源物理机制',
    mzi_opt_passive: '仅计算无源 MZI 光谱 (最快)', mzi_lbl_scheme: '电极驱动拓扑',
    mzi_lbl_egap: '电极/加热器间距 [µm]',
    mzi_kpi1: '50:50 分光比与附加损耗', mzi_kpi2: 'MZI 自由光谱范围与消光比',
    mzi_kpi3: '调制器 V_π·L / P_π 指标', mzi_kpi4: '非谐振 3-dB 带宽与动态 ER',
    mzi_plot1_title: '宽谱 MZI 直通 (Bar) 与交叉 (Cross) 透射光谱',
    mzi_tbl_title: '逐驱动态相移臂 Δn_eff、相移量与端口消光比明细表'
  }
};

function applyUiTranslations() {
  const dict = UI_I18N[currentLang] || UI_I18N.en;
  document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
  document.getElementById('lang-en').classList.toggle('active', currentLang === 'en');
  document.getElementById('lang-zh').classList.toggle('active', currentLang === 'zh');

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (dict[k]) el.textContent = dict[k];
  });
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const k = el.getAttribute('data-i18n-html');
    if (dict[k] && !(el.id === 'fit-results-box' && lastExtractedLossDbCm !== null) && !(el.id === 'gds-extracted-box' && currentGdsView)) {
      el.innerHTML = dict[k];
    }
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const k = el.getAttribute('data-i18n-placeholder');
    if (dict[k]) el.placeholder = dict[k];
  });
  if (lastExtractedLossDbCm !== null) {
    const useBtn = document.getElementById('btn-use-extracted-loss');
    if (useBtn) {
      useBtn.textContent = currentLang === 'zh'
        ? `🎯 用 α = ${Number(lastExtractedLossDbCm).toFixed(2)} dB/cm 设计新微环 →`
        : `🎯 Design Ring with α = ${Number(lastExtractedLossDbCm).toFixed(2)} dB/cm →`;
    }
  }
  if (copilotSession && copilotSession.config) syncFormFromConfig(copilotSession.config);
}

async function setLanguage(lang) {
  currentLang = (lang === 'zh') ? 'zh' : 'en';
  try { localStorage.setItem('microring_lang', currentLang); } catch (_) {}
  applyUiTranslations();

  if (!copilotSession || (copilotSession.stage === 'welcome' && (copilotSession.history || []).length <= 1)) {
    await initChat();
    return;
  }
  copilotSession.lang = currentLang;
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: copilotSession, message: '__switch_lang__', lang: currentLang })
    });
    const data = await r.json();
    if (!data.error) {
      copilotSession = data.session;
      renderChips(data.chips);
    }
  } catch (_) {}
}

let gdsCam = { scale: 1.0, offsetX: 0, offsetY: 0, dragging: false, lastX: 0, lastY: 0, hiddenLayers: {} };
const LAYER_COLORS = {
  '1/0': '#38bdf8', '2/0': '#0284c7', '3/0': '#a78bfa', '4/0': '#f43f5e',
  '10/0': '#f59e0b', '11/0': '#eab308', '40/0': '#fb7185', '65/0': '#34d399', '66/0': '#cbd5e1', '68/0': '#475569'
};

// ==================== UI NAVIGATION & STEPPER ====================
function setStageView(viewName) {
  document.querySelectorAll('.stage-view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.stage-pill').forEach(p => p.classList.remove('active'));
  const vEl = document.getElementById('view-' + viewName);
  const pEl = document.getElementById('pill-' + viewName);
  if (vEl) vEl.classList.add('active');
  if (pEl) pEl.classList.add('active');

  setTimeout(() => {
    if (viewName === 'mrr' && lastMrrResult) renderMrrResult(lastMrrResult);
    if (viewName === 'mode') {
      if (lastModesData) renderSelectedMode();
      if (lastWidthSweep) drawWidthSweepPlot(lastWidthSweep);
    }
    if (viewName === 'gds' && currentGdsView) drawGdsCanvas();
    if (viewName === 'fit' && labSpectrumData) runCmtFitUI();
    if (viewName === 'fab' && lastFabResult) renderFabResult(lastFabResult);
    if (viewName === 'tuning' && lastTuningResult) renderTuningResult(lastTuningResult);
    if (viewName === 'io') {
      const m = document.getElementById('io-mode').value;
      if (m === 'high_q_fdtd' && lastHighQResult) renderHighQResult(lastHighQResult);
      else if (m !== 'high_q_fdtd' && lastIoResult) renderIoResult(lastIoResult);
      else if (!ioBusy) runIoCouplerUI();
    }
    if (viewName === 'mzi') {
      if (lastMziResult) renderMziResult(lastMziResult);
      else if (!mziBusy) runMziUI();
    }
  }, 25);
}

function updateStepper(stepNum) {
  const s = Number(stepNum) || 1;
  document.querySelectorAll('#wizard-stepper .step-item').forEach(el => {
    const n = Number(el.dataset.step);
    const badge = el.querySelector('.step-num');
    el.classList.remove('active', 'done');
    if (n < s) {
      el.classList.add('done');
      badge.textContent = '✓';
    } else if (n === s) {
      el.classList.add('active');
      badge.textContent = String(n);
    } else {
      badge.textContent = String(n);
    }
  });
}

function toggleExpertDrawer(open) {
  document.getElementById('drawer-backdrop').classList.toggle('open', open);
  document.getElementById('expert-drawer').classList.toggle('open', open);
  if (open) setTimeout(loadMaterialCurve, 50);
}

function openSpectrumFitterDirect() {
  setStageView('fit');
  if (!labSpectrumData) loadDemoSpectrum('split_doublet');
}

// ==================== CLEAN MARKDOWN RENDERER ====================
function formatMarkdown(md) {
  let s = md
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\$\\Rightarrow\$/g, '⇒')
    .replace(/\$\\iff\$/g, '⇔')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  const lines = s.split('\n');
  let html = '', inList = false;
  for (const ln of lines) {
    if (ln.trim().startsWith('- ')) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + ln.trim().slice(2) + '</li>';
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      if (ln.trim() === '') html += '<div style="height:6px;"></div>';
      else html += '<div>' + ln + '</div>';
    }
  }
  if (inList) html += '</ul>';
  return html;
}

function appendChatBubble(role, text) {
  const box = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = role === 'assistant' ? 'M' : (currentLang === 'zh' ? '我' : 'You');
  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.innerHTML = formatMarkdown(text);
  row.appendChild(av);
  row.appendChild(bub);
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
  return row;
}

function renderChips(chips) {
  const wrap = document.getElementById('chat-chips');
  wrap.innerHTML = '';
  (chips || []).forEach(c => {
    const b = document.createElement('button');
    b.className = 'chip-btn';
    b.textContent = c.label;
    b.onclick = () => {
      if (c.ui_action === 'open_gds_picker') {
        document.getElementById('gds-file-input').click();
      } else {
        sendChatPrompt(c.prompt);
      }
    };
    wrap.appendChild(b);
  });
}

// ==================== SYNC FORM & LIVE BLUEPRINT ====================
function syncFormFromConfig(cfg) {
  if (!cfg) return;
  const setVal = (id, v) => { const el = document.getElementById(id); if (el && v !== undefined && v !== null) el.value = v; };
  setVal('cfg-topology', cfg.topology);
  setVal('cfg-material', cfg.core_material);
  setVal('cfg-clad', cfg.clad_material || 'SiO2');
  setVal('cfg-topclad', cfg.top_clad_material);
  setVal('cfg-pol', cfg.polarization);
  setVal('cfg-wl', cfg.wavelength_nm);
  setVal('cfg-hfilm', cfg.film_height_um);
  setVal('cfg-hslab', cfg.slab_height_um);
  setVal('cfg-sw', cfg.sidewall_deg);
  setVal('cfg-wring', cfg.ring_width_um);
  setVal('cfg-wbus', cfg.bus_width_um);
  setVal('cfg-r1', cfg.ring_radius_um);
  setVal('cfg-gap', cfg.gap_thru_um);
  setVal('cfg-ls', cfg.straight_length_um);
  setVal('cfg-theta', cfg.pulley_angle_thru_deg);
  setVal('cfg-dt', cfg.delta_temp_k);
  setVal('cfg-volt', cfg.voltage_v);
  setVal('cfg-r2', cfg.ring2_radius_um);
  setVal('cfg-span', cfg.span_nm);
  document.getElementById('cfg-loss').value = (cfg.alpha_loss_db_per_cm !== null && cfg.alpha_loss_db_per_cm !== undefined) ? cfg.alpha_loss_db_per_cm : '';
  setVal('mode-w', cfg.ring_width_um);
  setVal('mode-r', cfg.ring_radius_um);

  // Update Blueprint Summary Cards
  const hNm = Math.round((cfg.film_height_um || 0.4) * 1000);
  const sNm = Math.round((cfg.slab_height_um || 0) * 1000);
  document.getElementById('bp-wl').textContent = `${cfg.wavelength_nm} nm · ${cfg.polarization}`;
  document.getElementById('bp-mat').textContent = `${cfg.core_material} (${cfg.top_clad_material}/${cfg.clad_material||'SiO2'})`;
  document.getElementById('bp-stack').textContent = sNm > 0
    ? (currentLang === 'zh' ? `H=${hNm} nm, 平板=${sNm} nm` : `H=${hNm} nm, Slab=${sNm} nm`)
    : (currentLang === 'zh' ? `H=${hNm} nm, 条形波导` : `H=${hNm} nm, Strip`);
  document.getElementById('bp-loss').textContent = (cfg.alpha_loss_db_per_cm !== null && cfg.alpha_loss_db_per_cm !== undefined)
    ? `${cfg.alpha_loss_db_per_cm} dB/cm`
    : (currentLang === 'zh' ? '工艺默认损耗' : 'Foundry Typical');
  document.getElementById('bp-top').textContent = (catalogTopologies[cfg.topology] && catalogTopologies[cfg.topology].name) || cfg.topology;
  document.getElementById('bp-geom').textContent = `R=${cfg.ring_radius_um} µm · g=${Math.round((cfg.gap_thru_um||0.25)*1000)} nm`;

  // Also keep Fitter L_rt synced to current ring circumference!
  const lRt = 2 * Math.PI * (cfg.ring_radius_um || 40) + (cfg.topology && cfg.topology.startsWith('racetrack') ? 2 * (cfg.straight_length_um || 0) : 0);
  if (!labSpectrumData) document.getElementById('fit-lrt').value = lRt.toFixed(2);

  renderSchematicsSVG(cfg);
}

function readConfigFromForm() {
  const num = (id, fb) => { const v = parseFloat(document.getElementById(id).value); return Number.isFinite(v) ? v : fb; };
  const lossRaw = document.getElementById('cfg-loss').value;
  return {
    topology: document.getElementById('cfg-topology').value || 'point_circular_ap',
    core_material: document.getElementById('cfg-material').value || 'Si3N4',
    clad_material: document.getElementById('cfg-clad').value || 'SiO2',
    top_clad_material: document.getElementById('cfg-topclad').value || 'SiO2',
    polarization: document.getElementById('cfg-pol').value || 'TE0',
    wavelength_nm: num('cfg-wl', 1550),
    film_height_um: num('cfg-hfilm', 0.40),
    slab_height_um: num('cfg-hslab', 0.0),
    sidewall_deg: num('cfg-sw', 85),
    ring_width_um: num('cfg-wring', 1.20),
    bus_width_um: num('cfg-wbus', 1.00),
    ring_radius_um: num('cfg-r1', 40.0),
    gap_thru_um: num('cfg-gap', 0.25),
    gap_drop_um: num('cfg-gap', 0.25),
    straight_length_um: num('cfg-ls', 12.0),
    pulley_angle_thru_deg: num('cfg-theta', 25.0),
    pulley_angle_drop_deg: num('cfg-theta', 25.0),
    delta_temp_k: num('cfg-dt', 0.0),
    voltage_v: num('cfg-volt', 0.0),
    ring2_radius_um: num('cfg-r2', 42.5),
    span_nm: num('cfg-span', 12.0),
    alpha_loss_db_per_cm: lossRaw === '' ? null : parseFloat(lossRaw)
  };
}

function applyWavebandPresetUI(val) {
  if (val === 'custom') return;
  const wl = parseFloat(val);
  document.getElementById('cfg-wl').value = wl;
  let curMat = document.getElementById('cfg-material').value;
  if (wl < 1050 && (curMat === 'Si' || curMat === 'Ge' || curMat === 'a-Si')) {
    curMat = 'Si3N4'; document.getElementById('cfg-material').value = 'Si3N4';
  } else if (wl > 5500 && curMat !== 'Ge' && curMat !== 'Diamond') {
    curMat = 'Ge'; document.getElementById('cfg-material').value = 'Ge';
  }
  if (wl > 3650) {
    document.getElementById('cfg-clad').value = (wl > 5500 && curMat === 'Ge') ? 'Si' : 'CaF2';
    document.getElementById('cfg-topclad').value = 'Air';
  } else {
    document.getElementById('cfg-clad').value = 'SiO2';
    document.getElementById('cfg-topclad').value = 'SiO2';
  }
  const r = wl / 1550.0;
  document.getElementById('cfg-hfilm').value = (curMat === 'Si' && wl >= 1200 && wl <= 1680) ? 0.22 : Math.max(0.14, +(0.38 * Math.pow(r, 0.88)).toFixed(2));
  document.getElementById('cfg-wring').value = Math.max(0.28, +(((curMat === 'Si') ? 0.48 : 1.05) * Math.pow(r, 0.88)).toFixed(2));
  document.getElementById('cfg-wbus').value = Math.max(0.24, +(((curMat === 'Si') ? 0.42 : 0.90) * Math.pow(r, 0.88)).toFixed(2));
  document.getElementById('cfg-gap').value = Math.max(0.12, +(0.24 * Math.pow(r, 0.65)).toFixed(2));
  document.getElementById('cfg-span').value = Math.max(2.0, +(12.0 * Math.pow(r, 1.80)).toFixed(1));
}

// ==================== GUIDED CHATBOT API ====================
async function initChat() {
  const r = await fetch(`/api/chat_init?lang=${encodeURIComponent(currentLang)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lang: currentLang })
  });
  const data = await r.json();
  copilotSession = data.session;
  document.getElementById('chat-messages').innerHTML = '';
  appendChatBubble('assistant', data.assistant_message);
  renderChips(data.chips);
  updateStepper(copilotSession.wizard_step || 1);
  syncFormFromConfig(copilotSession.config);
}

async function sendUserChat() {
  const inp = document.getElementById('chat-input');
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = '';
  await sendChatPrompt(txt);
}

function saveSessionApiKey(val) {
  const mode = document.getElementById('ai-mode-select').value;
  try { sessionStorage.setItem('microring_key_' + mode, val.trim()); } catch (_) {}
  updateAiModeBadge();
}

function getSessionApiKey(mode) {
  try { return sessionStorage.getItem('microring_key_' + mode) || ''; } catch (_) { return ''; }
}

function toggleAiKeyBox(forceOpen) {
  const bar = document.getElementById('ai-key-bar');
  const nextShow = (typeof forceOpen === 'boolean') ? forceOpen : (bar.style.display === 'none');
  bar.style.display = nextShow ? 'block' : 'none';
  if (nextShow) {
    const mode = document.getElementById('ai-mode-select').value;
    const inp = document.getElementById('ai-api-key-input');
    inp.value = getSessionApiKey(mode);
    inp.placeholder = mode === 'claude'
      ? (currentLang === 'zh' ? '输入 Anthropic API Key (仅存于内存；若已登录 claude CLI 可留空)' : 'Anthropic sk-ant-... key (memory only; leave blank if claude CLI is signed in)')
      : (currentLang === 'zh' ? '输入 OpenAI API Key (仅存于内存；若已登录 codex CLI 可留空)' : 'OpenAI sk-... key (memory only; leave blank if codex CLI is signed in)');
  }
}

function updateAiModeBadge() {
  const mode = document.getElementById('ai-mode-select').value;
  const badge = document.getElementById('ai-mode-badge');
  const keyBtn = document.getElementById('btn-ai-key');
  if (mode === 'offline') {
    badge.style.display = 'none';
    keyBtn.style.display = 'none';
    document.getElementById('ai-key-bar').style.display = 'none';
  } else if (mode === 'ollama') {
    keyBtn.style.display = 'none';
    document.getElementById('ai-key-bar').style.display = 'none';
    badge.style.display = 'inline';
    badge.style.color = aiBackendStatus.ollama_available ? '#34d399' : '#f59e0b';
    badge.textContent = aiBackendStatus.ollama_available ? '✓ Local Ready' : '127.0.0.1';
  } else if (mode === 'claude') {
    keyBtn.style.display = 'inline-flex';
    badge.style.display = 'inline';
    const ready = aiBackendStatus.claude_cli_available || aiBackendStatus.claude_env_key || !!getSessionApiKey('claude');
    badge.style.color = ready ? '#34d399' : '#f59e0b';
    badge.textContent = aiBackendStatus.claude_cli_available ? '✓ Claude CLI' : (ready ? '✓ Key Set' : 'Key / CLI');
  } else if (mode === 'codex') {
    keyBtn.style.display = 'inline-flex';
    badge.style.display = 'inline';
    const ready = aiBackendStatus.codex_cli_available || aiBackendStatus.codex_env_key || !!getSessionApiKey('codex');
    badge.style.color = ready ? '#34d399' : '#f59e0b';
    badge.textContent = aiBackendStatus.codex_cli_available ? '✓ Codex CLI' : (ready ? '✓ Key Set' : 'Key / CLI');
  }
}

function onAiModeChange() {
  const mode = document.getElementById('ai-mode-select').value;
  updateAiModeBadge();
  if (mode === 'claude' && !aiBackendStatus.claude_cli_available && !aiBackendStatus.claude_env_key && !getSessionApiKey('claude')) {
    toggleAiKeyBox(true);
  } else if (mode === 'codex' && !aiBackendStatus.codex_cli_available && !aiBackendStatus.codex_env_key && !getSessionApiKey('codex')) {
    toggleAiKeyBox(true);
  } else {
    toggleAiKeyBox(false);
  }
}

async function sendChatPrompt(promptText) {
  const btn = document.getElementById('chat-send-btn');
  btn.disabled = true;
  const aiMode = document.getElementById('ai-mode-select').value || 'offline';
  const aiKey = getSessionApiKey(aiMode);
  if (promptText.toLowerCase() !== 'reset') appendChatBubble('user', promptText);
  const thinking = appendChatBubble('assistant', currentLang === 'zh' ? '正在计算当前步骤...' : 'Working on your step...');
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session: copilotSession,
        message: promptText,
        lang: currentLang,
        ai_mode: aiMode,
        ai_api_key: aiKey,
        use_ollama: aiMode === 'ollama'
      })
    });
    const data = await r.json();
    thinking.remove();
    if (data.error) {
      appendChatBubble('assistant', '**Solver Notice**: `' + data.error + '`');
      return;
    }
    copilotSession = data.session;
    if (promptText.toLowerCase() === 'reset') document.getElementById('chat-messages').innerHTML = '';
    appendChatBubble('assistant', data.assistant_message);
    renderChips(data.chips);
    syncFormFromConfig(copilotSession.config);
    updateStepper(data.wizard_step || copilotSession.wizard_step || 1);

    if (data.ui_view) setStageView(data.ui_view);
    if (data.auto_demo_fit && !labSpectrumData) loadDemoSpectrum('split_doublet');

    if (data.width_sweep_result) {
      lastWidthSweep = data.width_sweep_result;
      drawWidthSweepPlot(data.width_sweep_result);
      solve2DModesUI();
    }
    if (data.mrr_result) {
      lastMrrResult = data.mrr_result;
      renderMrrResult(data.mrr_result);
      document.getElementById('spec-status').textContent = currentLang === 'zh'
        ? `求解完成 (${data.mrr_result.elapsed_ms} ms)`
        : `Solved in ${data.mrr_result.elapsed_ms} ms`;
    }
    if (data.mrr_sweep_result) {
      drawMrrSweepPlot(data.mrr_sweep_result);
    }
    if (data.gds_export) {
      showGdsExportResult(data.gds_export);
    }
    if (data.fab_result) {
      lastFabResult = data.fab_result;
      document.getElementById('fab-resist').value = data.fab_result.resist_tone || 'positive_pr';
      renderFabResult(data.fab_result);
    }
    if (data.tuning_result) {
      lastTuningResult = data.tuning_result;
      renderTuningResult(data.tuning_result);
    }
    if (data.io_result) {
      lastIoResult = data.io_result;
      document.getElementById('io-mode').value = data.io_result.coupler_type || 'grating_coupler';
      onIoModeChange(true);
      renderIoResult(data.io_result);
    }
    if (data.high_q_result) {
      lastHighQResult = data.high_q_result;
      document.getElementById('io-mode').value = 'high_q_fdtd';
      onIoModeChange(true);
      renderHighQResult(data.high_q_result);
    }
    if (data.mzi_result) {
      lastMziResult = data.mzi_result;
      const mCfg = data.mzi_result.config || {};
      if (mCfg.splitter_type) {
        document.getElementById('mzi-splitter').value = mCfg.splitter_type;
        onMziSplitterChange(true);
      }
      if (mCfg.arm1_length_um) document.getElementById('mzi-l1').value = mCfg.arm1_length_um;
      if (mCfg.delta_length_um !== undefined) document.getElementById('mzi-dl').value = mCfg.delta_length_um;
      document.getElementById('mzi-ramzi').value = mCfg.ramzi_enabled ? 'on' : 'off';
      if (data.mzi_result.active_tuning) {
        document.getElementById('mzi-mech').value = data.mzi_result.active_tuning.mechanism || 'auto';
        document.getElementById('mzi-scheme').value = data.mzi_result.active_tuning.drive_scheme || 'auto';
      } else {
        document.getElementById('mzi-mech').value = 'passive_only';
      }
      renderMziResult(data.mzi_result);
    }
    if (data.interconnect_export && data.interconnect_export.zip_download_url) {
      showInterconnectDownload(data.interconnect_export);
    }
  } catch (err) {
    thinking.remove();
    appendChatBubble('assistant', 'Request failed: `' + err + '`');
  } finally {
    btn.disabled = false;
  }
}

// ==================== RETINA CANVAS PLOTTING ====================
function setupCanvas(id) {
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 2;
  const rect = cv.getBoundingClientRect();
  const w = Math.max(rect.width, 320);
  const h = Math.max(rect.height, 200);
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { cv, ctx, w, h };
}

function drawLinePlot(canvasId, seriesList, xLabel, yLabel, vLineX = null) {
  const { ctx, w, h } = setupCanvas(canvasId);
  ctx.fillStyle = '#070b14';
  ctx.fillRect(0, 0, w, h);
  const pad = { l: 54, r: 18, t: 22, b: 36 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  seriesList.forEach(s => {
    for (let i = 0; i < s.x.length; i++) {
      const xv = s.x[i], yv = s.y[i];
      if (yv === null || yv === undefined || !Number.isFinite(yv)) continue;
      if (xv < xmin) xmin = xv; if (xv > xmax) xmax = xv;
      if (yv < ymin) ymin = yv; if (yv > ymax) ymax = yv;
    }
  });
  if (!Number.isFinite(xmin)) return;
  if (Math.abs(ymax - ymin) < 1e-9) { ymax += 1; ymin -= 1; }
  const ym = 0.08 * (ymax - ymin); ymin -= ym; ymax += ym;

  const sx = x => pad.l + ((x - xmin) / (xmax - xmin || 1)) * pw;
  const sy = y => pad.t + ph - ((y - ymin) / (ymax - ymin || 1)) * ph;

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.1;
  ctx.strokeRect(pad.l, pad.t, pw, ph);

  ctx.fillStyle = '#94a3b8';
  ctx.font = '10.5px Arial, sans-serif';
  for (let i = 0; i <= 4; i++) {
    const tx = pad.l + (i / 4) * pw;
    const vx = xmin + (i / 4) * (xmax - xmin);
    ctx.beginPath(); ctx.moveTo(tx, pad.t + ph); ctx.lineTo(tx, pad.t + ph - 5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx, pad.t); ctx.lineTo(tx, pad.t + 5); ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(vx.toFixed(Math.abs(xmax - xmin) < 0.8 ? 3 : 2), tx, h - 19);

    const ty = pad.t + ph - (i / 4) * ph;
    const vy = ymin + (i / 4) * (ymax - ymin);
    ctx.beginPath(); ctx.moveTo(pad.l, ty); ctx.lineTo(pad.l + 5, ty); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l + pw, ty); ctx.lineTo(pad.l + pw - 5, ty); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(vy.toFixed(Math.abs(ymax - ymin) < 2 ? 2 : 1), pad.l - 7, ty + 3.5);
  }

  if (vLineX !== null && vLineX >= xmin && vLineX <= xmax) {
    const vx = sx(vLineX);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#fb923c';
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(vx, pad.t); ctx.lineTo(vx, pad.t + ph); ctx.stroke();
    ctx.restore();
  }

  seriesList.forEach((s, idx) => {
    ctx.strokeStyle = s.color || '#38bdf8';
    ctx.lineWidth = s.width || 1.75;
    if (s.dashed) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.x.length; i++) {
      const yv = s.y[i];
      if (yv === null || yv === undefined || !Number.isFinite(yv)) { started = false; continue; }
      const px = sx(s.x[i]), py = sy(yv);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    if (s.dots) {
      ctx.fillStyle = s.color || '#38bdf8';
      for (let i = 0; i < s.x.length; i++) {
        if (s.y[i] === null || !Number.isFinite(s.y[i])) continue;
        ctx.beginPath(); ctx.arc(sx(s.x[i]), sy(s.y[i]), 3.0, 0, Math.PI * 2); ctx.fill();
      }
    }
    if (s.label) {
      ctx.fillStyle = s.color;
      ctx.textAlign = 'left';
      ctx.font = 'bold 10.5px Arial';
      ctx.fillText(s.label, pad.l + 10 + idx * 135, pad.t + 15);
    }
  });

  ctx.fillStyle = '#cbd5e1';
  ctx.textAlign = 'center';
  ctx.font = '11px Arial';
  ctx.fillText(xLabel, pad.l + pw / 2, h - 4);
  ctx.save();
  ctx.translate(14, pad.t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

// ==================== MRR SIMULATION & SCHEMATICS ====================
async function runManualMRR() {
  const cfg = readConfigFromForm();
  if (copilotSession) copilotSession.config = Object.assign(copilotSession.config || {}, cfg);
  syncFormFromConfig(cfg);
  setStageView('mrr');
  document.getElementById('spec-status').textContent = 'Solving 2D FEM + CMT...';
  const r = await fetch('/api/design_mrr', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg)
  });
  const data = await r.json();
  if (data.error) {
    document.getElementById('spec-status').textContent = 'Error: ' + data.error;
    return;
  }
  lastMrrResult = data;
  renderMrrResult(data);
  document.getElementById('spec-status').textContent = `Solved in ${data.elapsed_ms} ms`;
}

function renderMrrResult(res) {
  const m = res.metrics;
  const cc = res.critical_coupling;
  const wBar = document.getElementById('wl-warning-bar');
  const warns = res.wavelength_warnings || [];
  if (warns.length > 0) {
    wBar.style.display = 'block';
    wBar.textContent = '⚠ ' + warns.join(' | ');
  } else {
    wBar.style.display = 'none';
  }

  document.getElementById('kpi-fsr').textContent = `${m.FSR_GHz.toFixed(2)} GHz`;
  document.getElementById('kpi-fsr-nm').textContent = `${m.FSR_nm.toFixed(3)} nm`;
  document.getElementById('kpi-loss').textContent = `${m.alpha_loss_db_per_cm.toFixed(2)} dB/cm`;
  document.getElementById('kpi-qint').textContent = `Q_int ${Math.round(m.Q_int).toLocaleString()}`;
  document.getElementById('kpi-ql').textContent = Math.round(m.Q_loaded).toLocaleString();
  document.getElementById('kpi-fwhm').textContent = `FWHM ${m.FWHM_pm.toFixed(2)} pm`;
  document.getElementById('kpi-er').textContent = `${m.ER_thru_dB.toFixed(1)} dB`;
  document.getElementById('kpi-regime').textContent = cc.regime;
  document.getElementById('kpi-cc').textContent = `${cc.recommended_value} ${cc.unit}`;
  document.getElementById('kpi-eo-eff').textContent = `n_eff=${m.neff_ring.toFixed(4)} · n_g=${m.ng_ring.toFixed(4)}`;
  document.getElementById('kpi-qext').textContent = `Q_ext = ${Math.round(m.Q_ext_thru).toLocaleString()}`;
  document.getElementById('kpi-shift').textContent = `Tuning Δλ = ${(m.tuning_shift_pm || 0).toFixed(1)} pm`;

  // Sync n_g into spectrum fitter for convenience!
  document.getElementById('fit-ng').value = m.ng_ring.toFixed(4);
  document.getElementById('fit-lrt').value = m.roundtrip_length_um.toFixed(2);

  const sp = res.spectrum;
  const series = [{ x: sp.wavelength_nm, y: sp.thru_dB, color: '#38bdf8', label: 'Through [dB]' }];
  if (sp.drop_dB && sp.drop_dB.some(v => v > -59.5)) {
    series.push({ x: sp.wavelength_nm, y: sp.drop_dB, color: '#fb923c', label: 'Drop [dB]' });
  }
  drawLinePlot('cv-spectrum', series, 'Wavelength [nm]', 'Transmission [dB]');

  if (res.coupler_curve) {
    const c = res.coupler_curve;
    document.getElementById('sweep-plot-title').textContent = 'Coupler Critical-Coupling Synthesis';
    document.getElementById('sweep-sub').textContent = `Target: ${cc.recommended_value} ${cc.unit}`;
    drawLinePlot(
      'cv-coupler',
      [{ x: c.x_values, y: c.log10_Q_ext, color: '#34d399', label: 'log10(Q_ext)' }],
      `${c.x_param} [${c.x_unit}]`,
      'log10(Q_ext)',
      cc.recommended_value
    );
  }
  renderSchematicsSVG(res.config || readConfigFromForm());
}

function drawMrrSweepPlot(sw) {
  document.getElementById('sweep-plot-title').textContent = `Sweep: ${sw.param_name} [${sw.unit}]`;
  document.getElementById('sweep-sub').textContent = `Best ER = ${Math.max(...sw.ER_thru_dB).toFixed(1)} dB`;
  drawLinePlot(
    'cv-coupler',
    [{ x: sw.values, y: sw.ER_thru_dB, color: '#fb923c', dots: true, label: 'Extinction Ratio [dB]' }],
    `${sw.param_name} [${sw.unit}]`,
    'Through ER [dB]',
    sw.values[sw.best_critical_coupling_index]
  );
}

function renderSchematicsSVG(cfg) {
  const t = cfg.topology || 'point_circular_ap';
  const is4 = t.endsWith('_ad');
  const isRace = t.startsWith('racetrack');
  const isPulley = t.startsWith('pulley');
  const isVernier = t === 'vernier_crow_ad';

  let ringSvg = '';
  if (isVernier) {
    ringSvg = `<circle cx="150" cy="60" r="26" fill="none" stroke="#38bdf8" stroke-width="4.5"/>
               <circle cx="212" cy="60" r="28" fill="none" stroke="#a78bfa" stroke-width="4.5"/>
               <line x1="35" y1="95" x2="335" y2="95" stroke="#34d399" stroke-width="3.5"/>
               <line x1="35" y1="24" x2="335" y2="24" stroke="#fb923c" stroke-width="3.5"/>`;
  } else if (isRace) {
    ringSvg = `<rect x="115" y="32" width="135" height="56" rx="28" fill="none" stroke="#38bdf8" stroke-width="4.5"/>
               <line x1="35" y1="96" x2="335" y2="96" stroke="#34d399" stroke-width="3.5"/>` +
               (is4 ? `<line x1="35" y1="24" x2="335" y2="24" stroke="#fb923c" stroke-width="3.5"/>` : '');
  } else if (isPulley) {
    ringSvg = `<circle cx="185" cy="60" r="32" fill="none" stroke="#38bdf8" stroke-width="4.5"/>
               <path d="M 40 104 L 132 104 Q 185 88 238 104 L 330 104" fill="none" stroke="#34d399" stroke-width="3.5"/>` +
               (is4 ? `<path d="M 40 18 L 132 18 Q 185 34 238 18 L 330 18" fill="none" stroke="#fb923c" stroke-width="3.5"/>` : '');
  } else {
    ringSvg = `<circle cx="185" cy="60" r="32" fill="none" stroke="#38bdf8" stroke-width="4.5"/>
               <line x1="35" y1="99" x2="335" y2="99" stroke="#34d399" stroke-width="3.5"/>` +
               (is4 ? `<line x1="35" y1="21" x2="335" y2="21" stroke="#fb923c" stroke-width="3.5"/>` : '');
  }

  const topSvgHtml = `
    <svg width="370" height="122" viewBox="0 0 370 122">
      <rect width="370" height="122" rx="8" fill="#070b14"/>
      ${ringSvg}
      <text x="12" y="114" fill="#94a3b8" font-size="10.5" font-family="monospace">${t} · R=${cfg.ring_radius_um} µm · gap=${Math.round((cfg.gap_thru_um||0.25)*1000)} nm</text>
    </svg>`;
  const elTop = document.getElementById('svg-topdown');
  if (elTop) elTop.innerHTML = topSvgHtml;

  const hSlabRatio = Math.min(0.75, (cfg.slab_height_um || 0) / Math.max(cfg.film_height_um || 0.4, 0.05));
  const coreH = 44, slabH = Math.round(coreH * hSlabRatio);
  const xsecSvgHtml = `
    <svg width="370" height="126" viewBox="0 0 370 126">
      <rect width="370" height="126" rx="8" fill="#070b14"/>
      <rect x="20" y="84" width="330" height="28" fill="#1e293b"/>
      <rect x="20" y="16" width="330" height="68" fill="${cfg.top_clad_material==='Air'?'#0b1324':'#172554'}" opacity="0.72"/>
      ${slabH > 0 ? `<rect x="35" y="${84 - slabH}" width="300" height="${slabH}" fill="#0284c7"/>` : ''}
      <polygon points="135,84 235,84 227,${84 - coreH} 143,${84 - coreH}" fill="#38bdf8" stroke="#bae6fd" stroke-width="1"/>
      <text x="185" y="${84 - coreH/2 + 4}" text-anchor="middle" fill="#070b14" font-weight="bold" font-size="11">${cfg.core_material}</text>
      <text x="12" y="118" fill="#94a3b8" font-size="10.5" font-family="monospace">W=${cfg.ring_width_um} µm · H=${Math.round((cfg.film_height_um||0.4)*1000)} nm · slab=${Math.round((cfg.slab_height_um||0)*1000)} nm · ${cfg.top_clad_material}/${cfg.clad_material||'SiO2'}</text>
    </svg>`;
  ['svg-xsec', 'svg-xsec-overview'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = xsecSvgHtml;
  });
}

// ==================== 2D EIGENMODE & WIDTH CUTOFF ====================
async function solve2DModesUI() {
  const cfg = readConfigFromForm();
  const payload = {
    core_material: cfg.core_material,
    clad_material: cfg.clad_material,
    top_clad_material: cfg.top_clad_material,
    w_bottom_um: cfg.ring_width_um,
    h_film_um: cfg.film_height_um,
    h_slab_um: cfg.slab_height_um,
    sidewall_deg: cfg.sidewall_deg,
    wavelength_nm: cfg.wavelength_nm,
    bend_radius_um: 0,
    solver_backend: 'fem',
    num_modes: 4
  };
  const r = await fetch('/api/solve_mode', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await r.json();
  if (data.error) return;
  lastModesData = data;
  selectedModeIdx = 0;
  const tbody = document.querySelector('#mode-table tbody');
  tbody.innerHTML = '';
  data.modes.forEach((m, i) => {
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.onclick = () => { selectedModeIdx = i; renderSelectedMode(); };
    tr.innerHTML = `<td>M${i}</td><td>${m.n_eff_real.toFixed(5)}</td><td>${(m.te_fraction*100).toFixed(1)}%</td><td>${(m.core_confinement*100).toFixed(1)}%</td><td>${m.a_eff_um2.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });
  renderSelectedMode();
}

function infernoColor(t) {
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.pow(c, 0.65) * 1.25));
  const g = Math.round(255 * Math.pow(c, 1.85));
  const b = Math.round(255 * (c < 0.5 ? c * 1.4 : Math.max(0, 1.4 - c * 1.5)));
  return [r, g, b];
}

function renderSelectedMode() {
  if (!lastModesData || !lastModesData.modes.length) return;
  const m = lastModesData.modes[selectedModeIdx];
  const comp = document.getElementById('mode-comp').value;
  const grid = m[comp];
  document.getElementById('mode-meta').textContent = `n_eff = ${m.n_eff_real.toFixed(5)} (${(m.te_fraction*100).toFixed(0)}% TE)`;

  const { ctx, w, h } = setupCanvas('cv-mode');
  ctx.fillStyle = '#050811';
  ctx.fillRect(0, 0, w, h);
  const pad = { l: 36, r: 14, t: 14, b: 24 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const ny = grid.length, nx = grid[0].length;
  let vmax = 1e-9;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (grid[j][i] > vmax) vmax = grid[j][i];

  const off = document.createElement('canvas');
  off.width = nx; off.height = ny;
  const octx = off.getContext('2d');
  const img = octx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const [r, g, b] = infernoColor(grid[ny - 1 - j][i] / vmax);
      const k = 4 * (j * nx + i);
      img.data[k] = r; img.data[k+1] = g; img.data[k+2] = b; img.data[k+3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.drawImage(off, pad.l, pad.t, pw, ph);

  const xs = m.x_um, ys = m.y_um;
  const xmin = xs[0], xmax = xs[xs.length - 1], ymin = ys[0], ymax = ys[ys.length - 1];
  ctx.strokeStyle = 'rgba(255,255,255,0.88)';
  ctx.lineWidth = 1.4;
  (lastModesData.polygon_outlines_um || []).forEach(poly => {
    ctx.beginPath();
    poly.forEach(([px, py], idx) => {
      const cx = pad.l + ((px - xmin) / (xmax - xmin)) * pw;
      const cy = pad.t + ph - ((py - ymin) / (ymax - ymin)) * ph;
      if (idx === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    });
    ctx.closePath();
    ctx.stroke();
  });
}

function drawWidthSweepPlot(sw) {
  document.getElementById('width-cutoff-badge').textContent = `Cutoff W_max = ${sw.single_mode_max_width_um} µm`;
  document.getElementById('mode-kpi-sm').textContent = `${sw.single_mode_max_width_um} µm`;
  document.getElementById('mode-kpi-wbus').textContent = `${sw.recommended_bus_width_um} µm`;
  document.getElementById('mode-kpi-wring').textContent = `${sw.recommended_ring_width_um} µm`;

  const idx = Math.min(sw.widths_um.length - 1, Math.max(0, Math.floor(sw.widths_um.length / 2)));
  document.getElementById('mode-kpi-neff').textContent = `n_eff ${sw.neff_fundamental[idx].toFixed(4)}`;
  document.getElementById('mode-kpi-ng').textContent = `n_g = ${sw.ng_fundamental[idx].toFixed(4)}`;
  document.getElementById('fit-ng').value = sw.ng_fundamental[idx].toFixed(4);

  const series = [
    { x: sw.widths_um, y: sw.neff_fundamental, color: '#38bdf8', dots: true, label: 'n_eff (Fund.)' },
    { x: sw.widths_um, y: sw.ng_fundamental, color: '#34d399', dots: true, label: 'n_g (Group)' }
  ];
  if (sw.neff_higher_order.some(v => v !== null)) {
    series.push({ x: sw.widths_um, y: sw.neff_higher_order, color: '#f87171', dashed: true, dots: true, label: 'Higher-Order Cutoff' });
  }
  drawLinePlot('cv-width-sweep', series, 'Waveguide Bottom Width W [µm]', 'Effective / Group Index', sw.single_mode_max_width_um);
}

// ==================== GDSII UPLOAD & EXPORT ====================
function handleGdsFileSelect(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    const b64 = reader.result.split(',')[1] || reader.result;
    const r = await fetch('/api/upload_gds', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name, content_b64: b64, session: copilotSession, lang: currentLang })
    });
    const data = await r.json();
    if (data.error) { alert('GDS Parse Error: ' + data.error); return; }
    copilotSession = data.session;
    appendChatBubble('assistant', data.assistant_message);
    renderChips(data.chips);
    syncFormFromConfig(copilotSession.config);
    updateStepper(data.wizard_step || 4);
    setStageView('gds');
    setGdsView(data.gds_info);
  };
  reader.readAsDataURL(file);
}

function setGdsView(gdsInfo) {
  currentGdsView = gdsInfo;
  gdsCam.hiddenLayers = {};
  const ext = gdsInfo.extracted_geometry;
  if (ext.is_non_ring) {
    document.getElementById('gds-extracted-box').innerHTML = `
      <div style="line-height:1.6;">
        <strong>${currentLang === 'zh' ? '文件' : 'File'}</strong>: <code>${gdsInfo.file_name}</code> (Cell: <code>${gdsInfo.cell_name}</code>, Layer: <code>${gdsInfo.selected_core_layer.join('/')}</code>) &nbsp;|&nbsp;
        <strong>${currentLang === 'zh' ? '器件类型' : 'Component'}</strong>: <span style="color:#38bdf8;font-weight:700;">${ext.topology_display}</span> (${ext.ports_detected}-Port)<br/>
        <strong>${currentLang === 'zh' ? '版图尺寸' : 'Footprint'}</strong>: <code>${ext.footprint_x_um} × ${ext.footprint_y_um} µm</code>
        ${ext.gap_thru_um > 0 ? ` · <strong>gap</strong> = <strong style="color:#34d399;">${(ext.gap_thru_um*1000).toFixed(1)} nm</strong>` : ''}
      </div>`;
  } else {
    document.getElementById('gds-extracted-box').innerHTML = `
      <div style="line-height:1.6;">
        <strong>${currentLang === 'zh' ? '文件' : 'File'}</strong>: <code>${gdsInfo.file_name}</code> (Cell: <code>${gdsInfo.cell_name}</code>, Layer: <code>${gdsInfo.selected_core_layer.join('/')}</code>) &nbsp;|&nbsp;
        <strong>${currentLang === 'zh' ? '识别拓扑' : 'Detected'}</strong>: <span style="color:#38bdf8;font-weight:700;">${ext.topology_display}</span> (${ext.ports_detected}-Port)<br/>
        <strong>${currentLang === 'zh' ? '多边形测量参数' : 'Measured Polygons'}</strong>: R = <code>${ext.ring_radius_um} µm</code> · W_ring = <code>${ext.ring_width_um} µm</code> · W_bus = <code>${ext.bus_width_um} µm</code> · gap = <strong style="color:#34d399;">${(ext.gap_thru_um*1000).toFixed(1)} nm</strong>
        ${ext.straight_length_um > 0 ? ` · L_s = <code>${ext.straight_length_um} µm</code>` : ''}
        ${ext.topology.includes('pulley') ? ` · θ = <code>${ext.pulley_angle_thru_deg}°</code>` : ''}
      </div>`;
  }

  const pills = document.getElementById('gds-layer-pills');
  pills.innerHTML = '';
  gdsInfo.viewer_layers.forEach(vl => {
    const key = `${vl.layer}/${vl.datatype}`;
    const col = LAYER_COLORS[key] || '#a78bfa';
    const b = document.createElement('button');
    b.className = 'btn sm';
    b.style.borderColor = col; b.style.color = col; b.style.padding = '3px 8px';
    b.textContent = `L ${key}`;
    b.onclick = () => {
      gdsCam.hiddenLayers[key] = !gdsCam.hiddenLayers[key];
      b.style.opacity = gdsCam.hiddenLayers[key] ? '0.35' : '1.0';
      drawGdsCanvas();
    };
    pills.appendChild(b);
  });
  resetGdsCamera();
}

function showGdsExportResult(exp) {
  const dl = document.getElementById('gds-download-links');
  dl.innerHTML = `
    <a class="btn green sm" href="${exp.gds_download_url}" download>⬇ ${currentLang === 'zh' ? '下载 .GDS' : 'Download .GDS'}</a>
    <a class="btn amber sm" href="${exp.lyp_download_url}" download>⬇ ${currentLang === 'zh' ? '下载 KLayout .LYP' : 'Download KLayout .LYP'}</a>`;
  if (exp.gds_view) setGdsView(exp.gds_view);
}

async function createSingleGdsUI() {
  const cfg = readConfigFromForm();
  const r = await fetch('/api/export_gds', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg)
  });
  const data = await r.json();
  if (!data.error) { setStageView('gds'); showGdsExportResult(data); }
}

async function createSweepArrayGdsUI() {
  const cfg = readConfigFromForm();
  const payload = { config: cfg, param_name: 'gap_thru_um', v_min: Math.max(0.10, cfg.gap_thru_um - 0.06), v_max: cfg.gap_thru_um + 0.06, num_devices: 5 };
  const r = await fetch('/api/export_sweep_gds', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await r.json();
  if (!data.error) { setStageView('gds'); showGdsExportResult(data); }
}

function resetGdsCamera() { gdsCam.scale = 1.0; gdsCam.offsetX = 0; gdsCam.offsetY = 0; drawGdsCanvas(); }

function drawGdsCanvas() {
  if (!currentGdsView) return;
  const { ctx, w, h } = setupCanvas('cv-gds');
  ctx.fillStyle = '#060a12';
  ctx.fillRect(0, 0, w, h);

  const [xmin, ymin, xmax, ymax] = currentGdsView.bbox_um;
  const spanX = Math.max(xmax - xmin, 1.0), spanY = Math.max(ymax - ymin, 1.0);
  const baseScale = 0.84 * Math.min(w / spanX, h / spanY) * gdsCam.scale;
  const cx = 0.5 * (xmin + xmax), cy = 0.5 * (ymin + ymax);
  const tx = x => w / 2 + (x - cx) * baseScale + gdsCam.offsetX;
  const ty = y => h / 2 - (y - cy) * baseScale + gdsCam.offsetY;

  currentGdsView.viewer_layers.forEach(vl => {
    const key = `${vl.layer}/${vl.datatype}`;
    if (gdsCam.hiddenLayers[key]) return;
    const col = LAYER_COLORS[key] || '#a78bfa';
    ctx.fillStyle = col + '40';
    ctx.strokeStyle = col;
    ctx.lineWidth = vl.is_core ? 1.4 : 0.9;
    vl.polygons.forEach(pts => {
      if (pts.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(tx(pts[0][0]), ty(pts[0][1]));
      for (let i = 1; i < pts.length; i++) ctx.lineTo(tx(pts[i][0]), ty(pts[i][1]));
      ctx.closePath();
      ctx.fill(); ctx.stroke();
    });
  });

  (currentGdsView.gap_markers || []).forEach(gm => {
    const x1 = tx(gm.ring_pt[0]), y1 = ty(gm.ring_pt[1]);
    const x2 = tx(gm.bus_pt[0]), y2 = ty(gm.bus_pt[1]);
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.fillStyle = '#fecaca'; ctx.font = 'bold 11px monospace';
    ctx.fillText(`gap = ${(gm.gap_um * 1000).toFixed(1)} nm`, 0.5 * (x1 + x2) + 8, 0.5 * (y1 + y2) - 5);
  });
}

(function attachGdsEvents() {
  const cv = document.getElementById('cv-gds');
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    gdsCam.scale = Math.max(0.2, Math.min(40, gdsCam.scale * (e.deltaY < 0 ? 1.18 : 0.85)));
    drawGdsCanvas();
  }, { passive: false });
  cv.addEventListener('mousedown', e => { gdsCam.dragging = true; gdsCam.lastX = e.clientX; gdsCam.lastY = e.clientY; });
  window.addEventListener('mouseup', () => { gdsCam.dragging = false; });
  window.addEventListener('mousemove', e => {
    if (!gdsCam.dragging) return;
    gdsCam.offsetX += (e.clientX - gdsCam.lastX); gdsCam.offsetY += (e.clientY - gdsCam.lastY);
    gdsCam.lastX = e.clientX; gdsCam.lastY = e.clientY;
    drawGdsCanvas();
  });
})();

// ==================== STAGE 5: LAB SPECTRUM LOSS (dB/cm) & Q FITTER ====================
async function loadDemoSpectrum(kind) {
  setStageView('fit');
  const r = await fetch(`/api/demo_spectrum?kind=${kind}`);
  labSpectrumData = await r.json();
  document.getElementById('fit-lrt').value = labSpectrumData.ring_length_um || 250;
  document.getElementById('fit-model').value = (kind === 'multi_fsr') ? 'multi_fsr' : 'auto';
  await runCmtFitUI();
}

function loadLabCsvFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    const lines = reader.result.split(/\r?\n/);
    const wl = [], tr = [];
    for (const ln of lines) {
      if (!ln.trim() || ln.trim().startsWith('#')) continue;
      const parts = ln.trim().split(/[,\s;]+/);
      if (parts.length >= 2) {
        const x = parseFloat(parts[0]), y = parseFloat(parts[1]);
        if (Number.isFinite(x) && Number.isFinite(y)) { wl.push(x); tr.push(y); }
      }
    }
    if (wl.length >= 10) {
      labSpectrumData = { wavelength_nm: wl, transmission: tr };
      setStageView('fit');
      await runCmtFitUI();
    } else {
      alert('Could not parse >=10 numeric (wavelength_nm, transmission) rows from file.');
    }
  };
  reader.readAsText(file);
}

async function runCmtFitUI() {
  if (!labSpectrumData) await loadDemoSpectrum('split_doublet');
  const payload = {
    wavelength_nm: labSpectrumData.wavelength_nm,
    transmission: labSpectrumData.transmission,
    model_type: document.getElementById('fit-model').value,
    known_regime: document.getElementById('fit-regime').value,
    ng_assumed: parseFloat(document.getElementById('fit-ng').value) || 2.10,
    ring_length_um: parseFloat(document.getElementById('fit-lrt').value) || 250
  };
  const r = await fetch('/api/fit_spectrum', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await r.json();
  if (data.error) {
    document.getElementById('fit-results-box').textContent = 'Fit Error: ' + data.error;
    return;
  }

  const box = document.getElementById('fit-results-box');
  const useBtn = document.getElementById('btn-use-extracted-loss');

  if (data.mode === 'multi_fsr') {
    const res = data.result;
    lastExtractedLossDbCm = res.alpha_loss_db_per_cm;
    if (res.ng_extracted) document.getElementById('fit-ng').value = res.ng_extracted.toFixed(4);
    document.getElementById('fit-kpi-loss').textContent = `${(res.alpha_loss_db_per_cm || 0).toFixed(3)} dB/cm`;
    document.getElementById('fit-kpi-loss-branches').textContent = `Under ${res.alpha_loss_under_db_per_cm} / Over ${res.alpha_loss_over_db_per_cm}`;
    document.getElementById('fit-kpi-qint').textContent = Math.round(res.mean_Q_int || 0).toLocaleString();
    document.getElementById('fit-kpi-rtloss').textContent = `Auto n_g = ${(res.ng_extracted || 0).toFixed(4)}`;
    document.getElementById('fit-kpi-ql').textContent = Math.round(res.mean_Q_loaded || 0).toLocaleString();
    document.getElementById('fit-kpi-fwhm').textContent = `FSR = ${res.mean_fsr_ghz.toFixed(2)} GHz`;
    document.getElementById('fit-kpi-split').textContent = `${res.num_resonances} Peaks`;
    document.getElementById('fit-kpi-r2').textContent = `FSR = ${res.mean_fsr_nm.toFixed(3)} nm`;

    box.innerHTML = currentLang === 'zh'
      ? `<strong>多谐振峰自洽提取</strong>：由 FSR 测得群折射率 <code>n_g = ${(res.ng_extracted||0).toFixed(4)}</code> + <code>Q_int = ${Math.round(res.mean_Q_int||0).toLocaleString()}</code> ⇒ 波导传播损耗 <strong><code>α = ${(res.alpha_loss_db_per_cm||0).toFixed(3)} dB/cm</code></strong>（单圈损耗 <code>${(res.roundtrip_loss_db||0).toFixed(4)} dB</code>）。`
      : `<strong>Self-Contained Multi-FSR Extraction</strong>: Measured <code>n_g = ${(res.ng_extracted||0).toFixed(4)}</code> from FSR + <code>Q_int = ${Math.round(res.mean_Q_int||0).toLocaleString()}</code> ⇒ Waveguide Loss <strong><code>α = ${(res.alpha_loss_db_per_cm||0).toFixed(3)} dB/cm</code></strong> (Round-trip loss <code>${(res.roundtrip_loss_db||0).toFixed(4)} dB</code>).`;
    useBtn.style.display = 'inline-flex';
    useBtn.textContent = currentLang === 'zh'
      ? `🎯 用 α = ${(res.alpha_loss_db_per_cm||0).toFixed(2)} dB/cm 设计新微环 →`
      : `🎯 Design Ring with α = ${(res.alpha_loss_db_per_cm||0).toFixed(2)} dB/cm →`;

    const yMeas = res.resonances && labSpectrumData.transmission.some(v => v < -0.25)
      ? labSpectrumData.transmission.map(v => Math.pow(10, v / 10))
      : labSpectrumData.transmission;
    drawLinePlot('cv-fit', [{ x: labSpectrumData.wavelength_nm, y: yMeas, color: '#38bdf8', label: 'Measured Broadband' }], 'Wavelength [nm]', 'Normalized Transmission');
  } else {
    const s = data.summary;
    lastExtractedLossDbCm = s.alpha_loss_db_per_cm;
    const isResolved = s.alpha_loss_selected_db_per_cm !== null && s.alpha_loss_selected_db_per_cm !== undefined;

    document.getElementById('fit-kpi-loss').textContent = `${s.alpha_loss_db_per_cm.toFixed(3)} dB/cm`;
    document.getElementById('fit-kpi-loss-branches').textContent = isResolved
      ? (currentLang === 'zh' ? '分支已唯一确定' : 'Branch Resolved')
      : `Under: ${s.alpha_loss_under_db_per_cm.toFixed(2)} | Over: ${s.alpha_loss_over_db_per_cm.toFixed(2)} dB/cm`;
    document.getElementById('fit-kpi-qint').textContent = Math.round(s.Q_int).toLocaleString();
    document.getElementById('fit-kpi-rtloss').textContent = s.roundtrip_loss_db !== null ? `RT Loss ${s.roundtrip_loss_db.toFixed(4)} dB` : `Q_ext ${Math.round(s.Q_ext).toLocaleString()}`;
    document.getElementById('fit-kpi-ql').textContent = Math.round(s.Q_loaded).toLocaleString();
    document.getElementById('fit-kpi-fwhm').textContent = `FWHM ${s.fwhm_pm.toFixed(2)} pm`;
    document.getElementById('fit-kpi-split').textContent = `${(2 * (s.mu_split_ghz || 0)).toFixed(2)} GHz`;
    document.getElementById('fit-kpi-r2').textContent = `${s.model_type} · R²=${s.r_squared.toFixed(4)}`;

    box.innerHTML = (currentLang === 'zh'
      ? `<strong>${s.model_type.toUpperCase()} 拟合结果</strong>（<code>λ₀ = ${s.lambda0_nm.toFixed(4)} nm</code>，采用 <code>n_g = ${s.ng_used}</code>）：<strong><code>α = ${s.alpha_loss_db_per_cm.toFixed(3)} dB/cm</code></strong> `
      : `<strong>${s.model_type.toUpperCase()} Fit</strong> at <code>λ₀ = ${s.lambda0_nm.toFixed(4)} nm</code> (using <code>n_g = ${s.ng_used}</code>): <strong><code>α = ${s.alpha_loss_db_per_cm.toFixed(3)} dB/cm</code></strong> `) +
      (!isResolved ? `(Under <code>${s.alpha_loss_under_db_per_cm.toFixed(3)} dB/cm</code> vs Over <code>${s.alpha_loss_over_db_per_cm.toFixed(3)} dB/cm</code>) ` : `(${s.coupling_regime_status}) `) +
      `| <code>Q_ext = ${Math.round(s.Q_ext).toLocaleString()}</code>`;

    useBtn.style.display = 'inline-flex';
    useBtn.textContent = currentLang === 'zh'
      ? `🎯 用 α = ${s.alpha_loss_db_per_cm.toFixed(2)} dB/cm 设计新微环 →`
      : `🎯 Design Ring with α = ${s.alpha_loss_db_per_cm.toFixed(2)} dB/cm →`;

    drawLinePlot('cv-fit', [
      { x: labSpectrumData.wavelength_nm, y: data.y_measured_linear || labSpectrumData.transmission, color: '#64748b', label: 'Measured Data' },
      { x: labSpectrumData.wavelength_nm, y: data.y_fit, color: '#fb923c', width: 2.1, label: 'Analytical CMT Fit' }
    ], 'Wavelength [nm]', 'Normalized Transmission');
  }
}

function applyExtractedLossToDesign() {
  if (!lastExtractedLossDbCm) return;
  const lossStr = Number(lastExtractedLossDbCm).toFixed(3);
  document.getElementById('cfg-loss').value = lossStr;
  if (copilotSession && copilotSession.confirmed && copilotSession.confirmed.includes('core_material')) {
    sendChatPrompt(currentLang === 'zh' ? `传播损耗 = ${lossStr} dB/cm, 开始仿真` : `Propagation loss = ${lossStr} dB/cm, simulate`);
  } else {
    if (copilotSession && copilotSession.config) copilotSession.config.alpha_loss_db_per_cm = parseFloat(lossStr);
    sendChatPrompt(currentLang === 'zh' ? '开始逐步设计微环' : 'Start step-by-step microring design');
    setTimeout(() => {
      appendChatBubble('assistant', currentLang === 'zh'
        ? `📌 已自动记录您提取的实验波导损耗 **\`α = ${lossStr} dB/cm\`**！请先选择上方工作波段与材料，随后我将直接按 \`${lossStr} dB/cm\` 综合临界耦合参数。`
        : `📌 Saved your extracted lab loss **\`α = ${lossStr} dB/cm\`**! Pick your wavelength band & material above and I will synthesize critical coupling for \`${lossStr} dB/cm\`.`);
    }, 350);
  }
}

// ==================== EXPERT DRAWER HELPERS ====================
async function loadMaterialCurve() {
  const key = document.getElementById('mat-browser-sel').value || '4H-SiC';
  const r = await fetch(`/api/material_dispersion?key=${encodeURIComponent(key)}`);
  const d = await r.json();
  if (d.error) return;
  const series = [{ x: d.wavelength_nm, y: d.nx, color: '#38bdf8', label: 'n_x (TE)' }];
  if (d.is_anisotropic) series.push({ x: d.wavelength_nm, y: d.ny, color: '#fb923c', dashed: true, label: 'n_y (TM)' });
  drawLinePlot('cv-mat', series, 'Wavelength [nm]', 'Refractive Index n(λ)');
  document.getElementById('mat-props-text').innerHTML =
    `dn/dT = ${d.dn_dt_per_k.toExponential(2)} /K · r_33 = ${d.r33_pm_per_v} pm/V · n_2 = ${d.n2_m2_per_w.toExponential(2)} m²/W<br/>Source: ${d.source}`;
}

async function runBenchmarkUI() {
  const r = await fetch('/api/benchmark');
  const b = await r.json();
  document.getElementById('hw-detail-box').innerHTML += `<div style="color:#34d399;margin-top:6px;">⚡ 100k-pt TMM Spectrum in <strong>${b.elapsed_ms.toFixed(2)} ms</strong> (${b.throughput_mpts_per_s.toFixed(2)} M pts/s)</div>`;
}

function exportSpectrumCSV() {
  if (!lastMrrResult) return;
  const sp = lastMrrResult.spectrum;
  let rows = ['wavelength_nm,thru_transmission_dB,drop_transmission_dB'];
  for (let i = 0; i < sp.wavelength_nm.length; i++) {
    rows.push(`${sp.wavelength_nm[i]},${sp.thru_dB[i]},${sp.drop_dB ? sp.drop_dB[i] : ''}`);
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'microring_spectrum.csv';
  a.click();
}

// ==================== LITHOGRAPHY DOSE & PR FABRICATION TOLERANCE ====================
function onFabResistChange() {
  const r = document.getElementById('fab-resist').value;
  const lbl = document.getElementById('fab-lbl-span');
  if (r === 'negative_pr') {
    document.getElementById('fab-prox').value = '1.15';
    lbl.textContent = currentLang === 'zh' ? '最大剂量误差 [±%]' : 'Max Dose Error [±%]';
    document.getElementById('fab-dose-pct').value = '15';
  } else if (r === 'positive_pr') {
    document.getElementById('fab-prox').value = '1.05';
    lbl.textContent = currentLang === 'zh' ? '最大剂量误差 [±%]' : 'Max Dose Error [±%]';
    document.getElementById('fab-dose-pct').value = '15';
  } else {
    document.getElementById('fab-prox').value = '1.00';
    lbl.textContent = currentLang === 'zh' ? '最大线宽偏置 [±nm]' : 'Max CD Bias [±nm]';
    document.getElementById('fab-dose-pct').value = '30';
  }
  runFabToleranceUI();
}

async function runFabToleranceUI() {
  const cfg = readConfigFromForm();
  const resist = document.getElementById('fab-resist').value || 'positive_pr';
  const spanVal = parseFloat(document.getElementById('fab-dose-pct').value) || 15.0;
  const slope = parseFloat(document.getElementById('fab-slope').value) || 2.0;
  const prox = parseFloat(document.getElementById('fab-prox').value) || 1.05;
  const dh = parseFloat(document.getElementById('fab-dh').value) || 0.0;
  document.getElementById('spec-status').textContent = currentLang === 'zh' ? '正在计算光刻剂量容差...' : 'Solving PR dose corners...';
  try {
    const r = await fetch('/api/fab_tolerance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: cfg,
        resist_tone: resist,
        max_dose_error_pct: spanVal,
        max_cd_bias_nm: spanVal,
        dose_slope_nm_per_pct: slope,
        proximity_gap_factor: prox,
        delta_h_film_nm: dh,
        num_points: 7
      })
    });
    const data = await r.json();
    if (data.error) return;
    lastFabResult = data;
    renderFabResult(data);
    document.getElementById('spec-status').textContent = currentLang === 'zh' ? '工艺误差分析已完成' : 'Fab analysis ready';
  } catch (_) {}
}

async function exportFabBiasGdsUI() {
  const cfg = readConfigFromForm();
  const prox = parseFloat(document.getElementById('fab-prox').value) || 1.05;
  const r = await fetch('/api/export_fab_gds', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config: cfg, proximity_gap_factor: prox })
  });
  const data = await r.json();
  if (data.error) return;
  showGdsExportResult(data);
  setStageView('gds');
}

function renderFabResult(fab) {
  if (!fab || !fab.summary) return;
  const s = fab.summary;
  const uc = s.under_dose_corner;
  const oc = s.over_dose_corner;
  const zh = currentLang === 'zh';

  document.getElementById('fab-mech-banner').innerHTML = zh
    ? `<strong>${fab.resist_name_zh}</strong><br>• ${fab.overdose_mechanism_zh}<br>• ${fab.underdose_mechanism_zh}`
    : `<strong>${fab.resist_name_en}</strong><br>• ${fab.overdose_mechanism_en}<br>• ${fab.underdose_mechanism_en}`;

  const drcEl = document.getElementById('fab-drc-banner');
  if (fab.drc_warnings && fab.drc_warnings.length > 0) {
    drcEl.style.display = 'block';
    drcEl.innerHTML = '⚠️ ' + fab.drc_warnings.join('<br>⚠️ ');
  } else {
    drcEl.style.display = 'none';
  }

  document.getElementById('fab-kpi-win').textContent = `${s.cd_window_er15_span_nm.toFixed(1)} nm`;
  document.getElementById('fab-kpi-win-sub').textContent = fab.resist_tone === 'direct_cd'
    ? `ER≥10 dB: ${s.cd_window_er10_span_nm.toFixed(1)} nm`
    : (zh ? `剂量窗口: ${s.dose_window_er15_span_pct.toFixed(1)}%` : `Dose window: ${s.dose_window_er15_span_pct.toFixed(1)}%`);

  document.getElementById('fab-kpi-over').textContent = `${oc.extinction_ratio_dB.toFixed(1)} dB`;
  document.getElementById('fab-kpi-over-sub').textContent = `ΔCD ${oc.delta_cd_nm >= 0 ? '+' : ''}${oc.delta_cd_nm.toFixed(0)} nm · gap ${(oc.gap_fab_um*1000).toFixed(0)} nm (${oc.coupling_regime})`;

  document.getElementById('fab-kpi-under').textContent = `${uc.extinction_ratio_dB.toFixed(1)} dB`;
  document.getElementById('fab-kpi-under-sub').textContent = `ΔCD ${uc.delta_cd_nm >= 0 ? '+' : ''}${uc.delta_cd_nm.toFixed(0)} nm · gap ${(uc.gap_fab_um*1000).toFixed(0)} nm (${uc.coupling_regime})`;

  document.getElementById('fab-kpi-slope').textContent = `${s.sensitivity_dl_dcd_pm_per_nm >= 0 ? '+' : ''}${s.sensitivity_dl_dcd_pm_per_nm.toFixed(0)} pm/nm`;
  document.getElementById('fab-kpi-heater').textContent = s.heater_dt_for_10nm_blueshift_k !== null
    ? (zh ? `-10nm蓝移热调补偿: +${s.heater_dt_for_10nm_blueshift_k} K` : `-10nm CD heater trim: +${s.heater_dt_for_10nm_blueshift_k} K`)
    : `dλ/dDose = ${s.sensitivity_dl_ddose_pm_per_pct.toFixed(0)} pm/%`;

  const tbody = document.querySelector('#fab-table tbody');
  tbody.innerHTML = '';
  (fab.corners || []).forEach(c => {
    const tr = document.createElement('tr');
    if (Math.abs(c.x_value) < 1e-6) tr.style.background = 'rgba(56,189,248,0.09)';
    const xTxt = fab.resist_tone === 'direct_cd'
      ? `${c.delta_cd_nm >= 0 ? '+' : ''}${c.delta_cd_nm.toFixed(1)} nm`
      : `${c.dose_error_pct >= 0 ? '+' : ''}${c.dose_error_pct.toFixed(1)}%`;
    const flagTxt = (c.drc_flags && c.drc_flags.length > 0) ? ` · <span style="color:#fbbf24;">${c.drc_flags[0]}</span>` : '';
    tr.innerHTML = `
      <td><strong>${xTxt}</strong></td>
      <td>${c.delta_cd_nm >= 0 ? '+' : ''}${c.delta_cd_nm.toFixed(1)}</td>
      <td>${(c.w_ring_fab_um * 1000).toFixed(1)}</td>
      <td>${(c.gap_fab_um * 1000).toFixed(1)}</td>
      <td style="color:${c.extinction_ratio_dB >= 15 ? '#34d399' : (c.extinction_ratio_dB >= 10 ? '#fbbf24' : '#f87171')};font-weight:700;">${c.extinction_ratio_dB.toFixed(1)}</td>
      <td>${Math.round(c.Q_loaded).toLocaleString()}</td>
      <td>${c.delta_wl_res_nm >= 0 ? '+' : ''}${c.delta_wl_res_nm.toFixed(3)}</td>
      <td>${c.coupling_regime}${flagTxt}</td>`;
    tbody.appendChild(tr);
  });

  drawFabErCurve(fab);
  drawFabDipOverlay(fab);
}

function drawFabErCurve(fab) {
  const { ctx, w, h } = setupCanvas('cv-fab-er');
  const pad = { l: 48, r: 52, t: 22, b: 36 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const corners = fab.corners || [];
  if (corners.length < 2) return;

  const xs = corners.map(c => c.x_value);
  const ers = corners.map(c => c.extinction_ratio_dB);
  const dwls = corners.map(c => c.delta_wl_res_nm);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const erMax = Math.max(25, Math.ceil(Math.max(...ers) / 5) * 5 + 3);
  const dwlAbs = Math.max(0.2, Math.max(...dwls.map(Math.abs)) * 1.15);

  const tx = v => pad.l + ((v - xmin) / Math.max(xmax - xmin, 1e-6)) * pw;
  const tyEr = v => pad.t + ph - (Math.max(0, v) / erMax) * ph;
  const tyWl = v => pad.t + ph * (0.5 - (v / (2 * dwlAbs)));

  // Shaded ER >= 15 dB yield band
  const y15 = tyEr(15.0);
  ctx.fillStyle = 'rgba(52,211,153,0.10)';
  ctx.fillRect(pad.l, pad.t, pw, Math.max(0, y15 - pad.t));
  ctx.strokeStyle = 'rgba(52,211,153,0.45)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, y15); ctx.lineTo(pad.l + pw, y15); ctx.stroke();
  ctx.setLineDash([]);

  // Axes border
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1.1;
  ctx.strokeRect(pad.l, pad.t, pw, ph);

  // Zero-dose vertical line
  const x0 = tx(0);
  ctx.strokeStyle = 'rgba(148,163,184,0.35)';
  ctx.beginPath(); ctx.moveTo(x0, pad.t); ctx.lineTo(x0, pad.t + ph); ctx.stroke();

  // Resonance wavelength shift Δλ [nm] curve (dashed amber)
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2.0; ctx.setLineDash([5, 4]);
  ctx.beginPath();
  corners.forEach((c, i) => { const x = tx(c.x_value), y = tyWl(c.delta_wl_res_nm); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke(); ctx.setLineDash([]);

  // Extinction Ratio [dB] curve (solid cyan-green)
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2.5;
  ctx.beginPath();
  corners.forEach((c, i) => { const x = tx(c.x_value), y = tyEr(c.extinction_ratio_dB); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();

  corners.forEach(c => {
    ctx.fillStyle = c.extinction_ratio_dB >= 15 ? '#34d399' : '#f87171';
    ctx.beginPath(); ctx.arc(tx(c.x_value), tyEr(c.extinction_ratio_dB), 4.0, 0, Math.PI * 2); ctx.fill();
  });

  // Labels
  ctx.fillStyle = '#94a3b8'; ctx.font = '11px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  [xmin, 0, xmax].forEach(v => ctx.fillText((v > 0 ? '+' : '') + v.toFixed(0), tx(v), h - 18));
  ctx.fillText(currentLang === 'zh' ? fab.x_label_zh : fab.x_label_en, pad.l + pw / 2, h - 4);

  ctx.textAlign = 'right'; ctx.fillStyle = '#38bdf8';
  [0, 15, erMax].forEach(v => ctx.fillText(v.toFixed(0) + ' dB', pad.l - 5, tyEr(v) + 4));

  ctx.textAlign = 'left'; ctx.fillStyle = '#f59e0b';
  [-dwlAbs, 0, dwlAbs].forEach(v => ctx.fillText((v >= 0 ? '+' : '') + v.toFixed(1) + 'nm', pad.l + pw + 5, tyWl(v) + 4));
}

function drawFabDipOverlay(fab) {
  const { ctx, w, h } = setupCanvas('cv-fab-dips');
  const pad = { l: 46, r: 16, t: 24, b: 36 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const cs = fab.corner_spectra || {};
  const trMin = cs.min_corner, trNom = cs.nominal, trMax = cs.max_corner;
  if (!trNom) return;

  const det = trNom.detuning_pm;
  const xmin = det[0], xmax = det[det.length - 1];
  const allDb = [...(trMin ? trMin.thru_dB : []), ...trNom.thru_dB, ...(trMax ? trMax.thru_dB : [])];
  const ymin = Math.min(-28, Math.floor(Math.min(...allDb) / 5) * 5 - 2);
  const ymax = 1.5;

  const tx = v => pad.l + ((v - xmin) / Math.max(xmax - xmin, 1e-6)) * pw;
  const ty = v => pad.t + ph - ((Math.max(ymin, Math.min(ymax, v)) - ymin) / (ymax - ymin)) * ph;

  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1.1;
  ctx.strokeRect(pad.l, pad.t, pw, ph);

  const series = [
    { data: trMin, col: '#f472b6', lbl: currentLang === 'zh' ? '欠曝光 (-Max)' : 'Under-Dose (-Max)' },
    { data: trNom, col: '#38bdf8', lbl: currentLang === 'zh' ? '标称剂量 (0%)' : 'Nominal (0%)' },
    { data: trMax, col: '#fbbf24', lbl: currentLang === 'zh' ? '过曝光 (+Max)' : 'Over-Dose (+Max)' }
  ];
  series.forEach((s, idx) => {
    if (!s.data) return;
    ctx.strokeStyle = s.col; ctx.lineWidth = idx === 1 ? 2.4 : 1.8;
    ctx.beginPath();
    s.data.detuning_pm.forEach((x, i) => {
      const px = tx(x), py = ty(s.data.thru_dB[i]);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.fillStyle = s.col; ctx.font = '11px -apple-system, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(`● ${s.lbl} (${s.data.dip_wavelength_nm.toFixed(2)} nm)`, pad.l + 8, pad.t + 15 + idx * 14);
  });

  ctx.fillStyle = '#94a3b8'; ctx.textAlign = 'center';
  [xmin, 0, xmax].forEach(v => ctx.fillText(v.toFixed(0) + ' pm', tx(v), h - 18));
  ctx.fillText(currentLang === 'zh' ? '相对各谐振峰中心失谐 Δλ [pm]' : 'Detuning Around Each Corner Dip Δλ [pm]', pad.l + pw / 2, h - 4);
  ctx.textAlign = 'right';
  [0, -15, ymin].forEach(v => ctx.fillText(v.toFixed(0) + ' dB', pad.l - 5, ty(v) + 4));
}

// ==================== STAGE 7: 2D-FEM ACTIVE TUNING & LUMERICAL INTERCONNECT ====================
const TUNE_PALETTE = ['#38bdf8', '#34d399', '#fbbf24', '#fb923c', '#f472b6', '#a78bfa', '#22d3ee'];

function onTuneControlChange(which) {
  const mech = document.getElementById('tune-mech').value;
  const drvEl = document.getElementById('tune-drive');
  let drv = drvEl.value;
  const biasInp = document.getElementById('tune-biases');

  if (which === 'mech') {
    const curMat = document.getElementById('cfg-material').value;
    if ((mech === 'si_pn_depletion' || mech === 'si_pin_injection') && curMat !== 'Si') {
      document.getElementById('cfg-material').value = 'Si';
      document.getElementById('cfg-wl').value = 1550;
      document.getElementById('cfg-hfilm').value = 0.22;
      document.getElementById('cfg-hslab').value = 0.09;
      document.getElementById('cfg-wring').value = 0.48;
      document.getElementById('cfg-wbus').value = 0.44;
      document.getElementById('cfg-r1').value = 20;
      document.getElementById('cfg-gap').value = 0.18;
      syncFormFromConfig(readConfigFromForm());
    } else if (mech === 'pockels_eo' && ['Si3N4', 'Si', 'a-Si', 'Ge', 'SiON', 'TiO2', 'Ta2O5', 'Diamond', 'ChG_As2S3', 'ChG_GeSbSe', 'Polymer_SU8'].includes(curMat)) {
      document.getElementById('cfg-material').value = 'LN_Xcut';
      document.getElementById('cfg-wl').value = 1550;
      document.getElementById('cfg-hfilm').value = 0.60;
      document.getElementById('cfg-hslab').value = 0.30;
      document.getElementById('cfg-wring').value = 1.20;
      document.getElementById('cfg-wbus').value = 1.00;
      document.getElementById('cfg-r1').value = 60;
      document.getElementById('cfg-gap').value = 0.30;
      syncFormFromConfig(readConfigFromForm());
    }
    if (mech === 'si_pn_depletion') { drvEl.value = 'voltage_v'; biasInp.value = '0, -1, -2, -4, -6'; }
    else if (mech === 'si_pin_injection') { drvEl.value = 'current_ma'; biasInp.value = '0, 0.25, 0.5, 1.0, 2.0'; }
    else if (mech === 'pockels_eo') { drvEl.value = 'voltage_v'; biasInp.value = '-15, -7.5, 0, 7.5, 15'; }
    else if (mech === 'thermo_optic') { drvEl.value = 'current_ma'; biasInp.value = '0, 2, 4, 6, 8'; }
    else { drvEl.value = 'auto'; biasInp.value = ''; }
  } else if (which === 'drive') {
    if (drv === 'delta_temp_k') {
      document.getElementById('tune-mech').value = 'thermo_optic';
      biasInp.value = '0, 10, 25, 50, 75';
    } else if (drv === 'e_field_v_per_um') {
      document.getElementById('tune-mech').value = 'pockels_eo';
      const curMat = document.getElementById('cfg-material').value;
      if (['Si3N4', 'Si', 'a-Si', 'Ge', 'SiON', 'TiO2', 'Ta2O5', 'Diamond', 'ChG_As2S3', 'ChG_GeSbSe', 'Polymer_SU8'].includes(curMat)) {
        document.getElementById('cfg-material').value = 'LN_Xcut';
        document.getElementById('cfg-hfilm').value = 0.60;
        document.getElementById('cfg-hslab').value = 0.30;
        document.getElementById('cfg-wring').value = 1.20;
        document.getElementById('cfg-wbus').value = 1.00;
        document.getElementById('cfg-r1').value = 60;
        document.getElementById('cfg-gap').value = 0.30;
        syncFormFromConfig(readConfigFromForm());
      }
      biasInp.value = '-10, -5, 0, 5, 10';
    } else if (drv === 'power_mw') {
      biasInp.value = '0, 5, 10, 20, 30';
    } else if (drv === 'current_ma') {
      biasInp.value = (mech === 'si_pin_injection') ? '0, 0.25, 0.5, 1.0, 2.0' : '0, 2, 4, 6, 8';
    } else if (drv === 'voltage_v') {
      biasInp.value = (mech === 'si_pn_depletion') ? '0, -1, -2, -4, -6' : (mech === 'pockels_eo' ? '-15, -7.5, 0, 7.5, 15' : '0, 1, 2, 3, 4');
    }
  }
  runActiveTuningUI();
}

function buildTuningPayload() {
  const cfg = readConfigFromForm();
  const egapRaw = document.getElementById('tune-egap').value;
  const dopVal = parseFloat(document.getElementById('tune-doping').value) || 5.0e17;
  return {
    config: cfg,
    mechanism: document.getElementById('tune-mech').value || 'auto',
    drive_type: document.getElementById('tune-drive').value || 'auto',
    bias_values: document.getElementById('tune-biases').value || null,
    heater_gap_um: parseFloat(document.getElementById('tune-hgap').value) || 0.90,
    box_thickness_um: parseFloat(document.getElementById('tune-box').value) || 2.00,
    electrode_gap_um: egapRaw === '' ? null : parseFloat(egapRaw),
    na_p_cm3: dopVal,
    nd_n_cm3: dopVal,
    pn_offset_nm: parseFloat(document.getElementById('tune-pnoff').value) || 25.0,
    active_fraction: parseFloat(document.getElementById('tune-fact').value) || 0.78
  };
}

async function runActiveTuningUI() {
  const payload = buildTuningPayload();
  document.getElementById('spec-status').textContent = currentLang === 'zh'
    ? '正在求解 2D-FEM 多物理场与扰动本征模...'
    : 'Solving 2D-FEM multiphysics + perturbed eigenmodes...';
  try {
    const r = await fetch('/api/active_tuning', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (data.error) {
      document.getElementById('tune-geom-banner').innerHTML = `<span style="color:#f87171;">⚠️ ${data.error}</span>`;
      document.getElementById('spec-status').textContent = 'Solver notice';
      return;
    }
    lastTuningResult = data;
    renderTuningResult(data);
    document.getElementById('spec-status').textContent = currentLang === 'zh'
      ? `2D-FEM 调谐完成 (${data.elapsed_ms} ms)`
      : `2D-FEM Tuning in ${data.elapsed_ms} ms`;
  } catch (err) {
    document.getElementById('spec-status').textContent = 'Error: ' + err;
  }
}

function showInterconnectDownload(ic) {
  if (!ic || !ic.zip_download_url) return;
  const html = `<a class="btn amber sm" href="${ic.zip_download_url}" download>⬇ ${ic.zip_name || 'INTERCONNECT.zip'}</a>`;
  const slot = document.getElementById('ic-download-slot');
  if (slot) slot.innerHTML = html;
  const ioSlot = document.getElementById('io-download-slot');
  if (ioSlot && (String(ic.zip_name || '').includes('IO_INTERCONNECT') || document.getElementById('view-io').classList.contains('active'))) {
    ioSlot.innerHTML = html;
  }
}

async function exportInterconnectUI() {
  const payload = buildTuningPayload();
  if (lastTuningResult) payload.tuning_result = lastTuningResult;
  document.getElementById('spec-status').textContent = currentLang === 'zh'
    ? '正在生成 Lumerical INTERCONNECT 模型包...'
    : 'Exporting Lumerical INTERCONNECT bundle...';
  try {
    const r = await fetch('/api/export_interconnect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const ic = await r.json();
    if (ic.error) { alert('INTERCONNECT Export Error: ' + ic.error); return; }
    showInterconnectDownload(ic);
    document.getElementById('spec-status').textContent = currentLang === 'zh'
      ? `已导出 ${ic.zip_name}`
      : `Exported ${ic.zip_name}`;
    if (ic.zip_download_url) {
      const a = document.createElement('a');
      a.href = ic.zip_download_url;
      a.download = ic.zip_name || 'MicroRing_INTERCONNECT.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  } catch (err) {
    alert('Export failed: ' + err);
  }
}

function renderTuningResult(res) {
  if (!res || !res.efficiency) return;
  const eff = res.efficiency;
  const gp = res.geometry_physics || {};
  const pm = res.passive_metrics || {};
  const zh = currentLang === 'zh';

  document.getElementById('tune-kpi-slope').textContent = `${eff.wavelength_slope} ${eff.wavelength_slope_unit}`;
  document.getElementById('tune-kpi-fslope').textContent = `${eff.freq_slope} ${eff.freq_slope_unit}`;
  document.getElementById('tune-kpi-pi').textContent = `${eff.pi_figure} ${eff.pi_figure_unit}`;
  document.getElementById('tune-kpi-pi-sub').textContent = `${eff.pi_figure_label} · L_act = ${gp.active_length_um || '--'} µm`;
  document.getElementById('tune-kpi-bw').textContent = eff.bandwidth_3db !== null && eff.bandwidth_3db !== undefined
    ? `${eff.bandwidth_3db} ${eff.bandwidth_unit}`
    : (zh ? '准静态输入' : 'Direct Field');

  if (eff.mechanism === 'thermo_optic') {
    document.getElementById('tune-kpi-bw-sub').textContent = gp.tau_thermal_us
      ? `τ_th = ${gp.tau_thermal_us} µs · BOX ${gp.box_thickness_um} µm`
      : `dn/dT = ${(gp.dn_dt_core_per_k || 0).toExponential(2)} /K`;
    document.getElementById('tune-kpi-geom').textContent = gp.r_th_core_k_per_mw
      ? `${gp.r_th_core_k_per_mw} K/mW`
      : `dn/dT`;
    document.getElementById('tune-kpi-geom-sub').textContent = gp.heater_resistance_ohm
      ? `R_h = ${gp.heater_resistance_ohm} Ω · gap ${gp.heater_gap_um} µm`
      : (zh ? '直接温升 ΔT 截面扰动' : 'Direct ΔT Cross-Section');
    document.getElementById('tune-geom-banner').innerHTML = zh
      ? `<strong>${gp.mechanism_name_zh}（纯自研 2D Fourier 热传导 FEM + 矢量 Maxwell FEM）</strong>：` +
        (gp.r_th_core_k_per_mw
          ? `波导芯层热阻 <code>R_th = ${gp.r_th_core_k_per_mw} K/mW</code>，<code>${gp.heater_material}</code> 加热器电阻 <code>R_elec = ${gp.heater_resistance_ohm} Ω</code>，埋氧层 <code>BOX = ${gp.box_thickness_um} µm</code>，热响应时间 <code>τ = ${gp.tau_thermal_us} µs</code>。`
          : `直接输入温升 <code>ΔT [K]</code>，结合芯层/包层热光系数 <code>dn/dT = ${(gp.dn_dt_core_per_k||0).toExponential(2)} /K</code> 求解二维微扰本征模。`)
      : `<strong>${gp.mechanism_name_en} (Native 2D Fourier Thermal FEM + Vector Maxwell FEM)</strong>: ` +
        (gp.r_th_core_k_per_mw
          ? `Core thermal resistance <code>R_th = ${gp.r_th_core_k_per_mw} K/mW</code>, <code>${gp.heater_material}</code> heater <code>R_elec = ${gp.heater_resistance_ohm} Ω</code>, <code>BOX = ${gp.box_thickness_um} µm</code>, thermal constant <code>τ = ${gp.tau_thermal_us} µs</code>.`
          : `Direct temperature shift <code>ΔT [K]</code> mapped via core/cladding <code>dn/dT = ${(gp.dn_dt_core_per_k||0).toExponential(2)} /K</code> into 2D vector eigenmodes.`);
  } else if (eff.mechanism === 'pockels_eo') {
    document.getElementById('tune-kpi-bw-sub').textContent = gp.c_total_ff
      ? `C_tot = ${gp.c_total_ff} fF (50 Ω RC)`
      : `r_33 = ${gp.r33_pm_per_v} pm/V`;
    document.getElementById('tune-kpi-geom').textContent = gp.ex_core_v_per_um_per_v !== undefined
      ? `${Math.max(Math.abs(gp.ex_core_v_per_um_per_v||0), Math.abs(gp.ey_core_v_per_um_per_v||0)).toFixed(3)} (V/µm)/V`
      : `r_33 = ${gp.r33_pm_per_v} pm/V`;
    document.getElementById('tune-kpi-geom-sub').textContent = gp.electrode_gap_um
      ? `G = ${gp.electrode_gap_um} µm · C = ${gp.c_elec_ff_per_um} fF/µm`
      : `Axis: ${gp.e_field_axis || 'auto'}`;
    document.getElementById('tune-geom-banner').innerHTML = zh
      ? `<strong>${gp.mechanism_name_zh}（纯自研 2D 各向异性 Laplace 静电场 FEM + 矢量 Maxwell FEM）</strong>：` +
        `电光张量 <code>r_33 = ${gp.r33_pm_per_v} pm/V, r_13 = ${gp.r13_pm_per_v} pm/V</code>` +
        (gp.electrode_gap_um ? `，电极间距 <code>G = ${gp.electrode_gap_um} µm</code>，介电屏蔽因子 <code>${gp.dielectric_screening_factor}</code>，线电容 <code>${gp.c_elec_ff_per_um} fF/µm</code>。` : `，直接输入芯层电场 <code>E [V/µm]</code>。`)
      : `<strong>${gp.mechanism_name_en} (Native 2D Anisotropic Laplace FEM + Vector Maxwell FEM)</strong>: ` +
        `Pockels tensor <code>r_33 = ${gp.r33_pm_per_v} pm/V, r_13 = ${gp.r13_pm_per_v} pm/V</code>` +
        (gp.electrode_gap_um ? `, electrode gap <code>G = ${gp.electrode_gap_um} µm</code>, dielectric screening <code>${gp.dielectric_screening_factor}</code>, line capacitance <code>${gp.c_elec_ff_per_um} fF/µm</code>.` : `, direct electric field <code>E [V/µm]</code>.`);
  } else {
    document.getElementById('tune-kpi-bw-sub').textContent = gp.r_series_ohm
      ? `R_s = ${gp.r_series_ohm} Ω · Slab ${Math.round((gp.slab_height_used_um||0.09)*1000)} nm`
      : `τ_c = ${gp.carrier_lifetime_ns || 1.0} ns`;
    document.getElementById('tune-kpi-geom').textContent = gp.w_dep_0v_nm
      ? `W_dep = ${gp.w_dep_0v_nm} nm`
      : `FCA Δα`;
    document.getElementById('tune-kpi-geom-sub').textContent = gp.cj_0v_ff_per_um
      ? `C_j(0V) = ${gp.cj_0v_ff_per_um} fF/µm · ΔFCA ${eff.fca_relief_db_per_cm || 0} dB/cm`
      : `Max FCA = ${eff.fca_max_db_per_cm || 0} dB/cm`;
    document.getElementById('tune-geom-banner').innerHTML = zh
      ? `<strong>${gp.mechanism_name_zh}（纯自研 2D 硅结耗尽/注入模型 + Soref-Bennett + 矢量 Maxwell FEM）</strong>：` +
        (gp.w_dep_0v_nm
          ? `零偏耗尽区宽度 <code>W_dep(0V) = ${gp.w_dep_0v_nm} nm</code>（内建电势 <code>V_bi = ${gp.v_bi_v} V</code>，结偏移 <code>+${gp.pn_offset_nm} nm</code>），结电容 <code>C_j(0V) = ${gp.cj_0v_ff_per_um} fF/µm</code>，平板串联电阻 <code>R_s = ${gp.r_series_ohm} Ω</code>。`
          : `本征区宽度 <code>W_i = ${gp.intrinsic_width_um} µm</code>，载流子寿命 <code>τ_c = ${gp.carrier_lifetime_ns} ns</code>，同步求解蓝移与自由载流子吸收 (FCA)。`)
      : `<strong>${gp.mechanism_name_en} (Native 2D Si Junction + Soref-Bennett + Vector Maxwell FEM)</strong>: ` +
        (gp.w_dep_0v_nm
          ? `Zero-bias depletion <code>W_dep(0V) = ${gp.w_dep_0v_nm} nm</code> (<code>V_bi = ${gp.v_bi_v} V</code>, offset <code>+${gp.pn_offset_nm} nm</code>), capacitance <code>C_j(0V) = ${gp.cj_0v_ff_per_um} fF/µm</code>, slab resistance <code>R_s = ${gp.r_series_ohm} Ω</code>.`
          : `Intrinsic width <code>W_i = ${gp.intrinsic_width_um} µm</code>, carrier lifetime <code>τ_c = ${gp.carrier_lifetime_ns} ns</code>, solving simultaneous plasma blue-shift &amp; modal FCA.`);
  }

  document.getElementById('tune-plot1-sub').textContent = `FSR = ${(pm.FSR_GHz||0).toFixed(1)} GHz (${(pm.FSR_nm||0).toFixed(3)} nm)`;

  const tbody = document.querySelector('#tune-table tbody');
  tbody.innerHTML = '';
  (res.states || []).forEach((st, idx) => {
    const tr = document.createElement('tr');
    const col = TUNE_PALETTE[idx % TUNE_PALETTE.length];
    let auxTxt = '--';
    if (st.delta_T_core_K !== undefined) auxTxt = `ΔT_core = +${st.delta_T_core_K.toFixed(2)} K (${st.power_mw !== undefined ? st.power_mw.toFixed(1)+' mW' : 'Direct'})`;
    else if (st.w_dep_nm !== undefined) auxTxt = `W_dep = ${st.w_dep_nm.toFixed(1)} nm · C_j = ${st.cj_ff_per_um.toFixed(3)} fF/µm`;
    else if (st.delta_N_cm3 !== undefined) auxTxt = `ΔN = ${st.delta_N_cm3.toExponential(2)} cm⁻³`;
    else if (st.Ex_core_v_per_um !== undefined) auxTxt = `E = (${st.Ex_core_v_per_um.toFixed(2)}, ${st.Ey_core_v_per_um.toFixed(2)}) V/µm`;

    tr.innerHTML = `
      <td><span style="color:${col};font-weight:700;">● ${st.label}</span></td>
      <td>${auxTxt}</td>
      <td>${st.neff_active_real.toFixed(6)}</td>
      <td>${(st.delta_neff_active * 1e4 >= 0 ? '+' : '') + (st.delta_neff_active * 1e4).toFixed(3)}</td>
      <td>${st.alpha_total_roundtrip_db_per_cm.toFixed(2)}</td>
      <td style="font-weight:700;color:#38bdf8;">${(st.resonance_shift_pm >= 0 ? '+' : '') + st.resonance_shift_pm.toFixed(2)}</td>
      <td>${st.ER_thru_dB.toFixed(1)}</td>
      <td>${Math.round(st.Q_loaded).toLocaleString()}</td>`;
    tbody.appendChild(tr);
  });

  drawTuningSpectra(res);
  drawTuningShiftLoss(res);
}

function drawTuningSpectra(res) {
  const sp = res.spectra || {};
  const wl = sp.wavelength_nm || [];
  const curves = sp.curves || [];
  if (!wl.length || !curves.length) return;

  // Focus wavelength window around the tracked resonance + shift span so multi-bias shifts are crystal clear!
  const lam0 = (res.passive_metrics && res.passive_metrics.tracked_resonance_0_nm) || wl[Math.floor(wl.length / 2)];
  const shiftsNm = (res.states || []).map(s => (s.resonance_shift_pm || 0) * 1e-3);
  const sMin = Math.min(0, ...shiftsNm), sMax = Math.max(0, ...shiftsNm);
  const fwhmEstNm = Math.max(0.03, lam0 / Math.max((res.passive_metrics && res.passive_metrics.Q_loaded_0) || 15000, 1500));
  const halfPad = Math.max(4.5 * fwhmEstNm, 0.35 * Math.abs(sMax - sMin) + 2.5 * fwhmEstNm);
  const wLo = Math.max(wl[0], lam0 + sMin - halfPad);
  const wHi = Math.min(wl[wl.length - 1], lam0 + sMax + halfPad);

  const { ctx, w, h } = setupCanvas('cv-tune-spec');
  ctx.fillStyle = '#070b14';
  ctx.fillRect(0, 0, w, h);
  const pad = { l: 50, r: 16, t: 22, b: 36 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

  let ymin = -25, ymax = 1.2;
  curves.forEach(c => {
    for (let i = 0; i < wl.length; i++) {
      if (wl[i] >= wLo && wl[i] <= wHi && c.thru_dB[i] < ymin) ymin = Math.max(-45, Math.floor(c.thru_dB[i] / 5) * 5 - 2);
    }
  });

  const sx = x => pad.l + ((x - wLo) / Math.max(wHi - wLo, 1e-6)) * pw;
  const sy = y => pad.t + ph - ((Math.max(ymin, Math.min(ymax, y)) - ymin) / (ymax - ymin)) * ph;

  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1.1;
  ctx.strokeRect(pad.l, pad.t, pw, ph);

  ctx.fillStyle = '#94a3b8'; ctx.font = '10.5px Arial, sans-serif';
  for (let i = 0; i <= 4; i++) {
    const tx = pad.l + (i / 4) * pw;
    const vx = wLo + (i / 4) * (wHi - wLo);
    ctx.beginPath(); ctx.moveTo(tx, pad.t + ph); ctx.lineTo(tx, pad.t + ph - 5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx, pad.t); ctx.lineTo(tx, pad.t + 5); ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(vx.toFixed(wHi - wLo < 0.6 ? 3 : 2), tx, h - 19);

    const ty = pad.t + ph - (i / 4) * ph;
    const vy = ymin + (i / 4) * (ymax - ymin);
    ctx.beginPath(); ctx.moveTo(pad.l, ty); ctx.lineTo(pad.l + 5, ty); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l + pw, ty); ctx.lineTo(pad.l + pw - 5, ty); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(vy.toFixed(0), pad.l - 6, ty + 3.5);
  }

  curves.forEach((c, idx) => {
    const col = TUNE_PALETTE[idx % TUNE_PALETTE.length];
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.9;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < wl.length; i++) {
      if (wl[i] < wLo || wl[i] > wHi) continue;
      const px = sx(wl[i]), py = sy(c.thru_dB[i]);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.fillStyle = col;
    ctx.font = 'bold 10.5px Arial, sans-serif';
    ctx.textAlign = 'left';
    const colIdx = idx % 3, rowIdx = Math.floor(idx / 3);
    ctx.fillText(`● ${c.label}`, pad.l + 8 + colIdx * Math.floor(pw / 3), pad.t + 14 + rowIdx * 13);
  });

  ctx.fillStyle = '#cbd5e1'; ctx.font = '11px Arial, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(currentLang === 'zh' ? '波长 Wavelength [nm]' : 'Wavelength [nm]', pad.l + pw / 2, h - 4);
  ctx.save(); ctx.translate(14, pad.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('Through [dB]', 0, 0); ctx.restore();
}

function drawTuningShiftLoss(res) {
  const states = (res.states || []).slice().sort((a, b) => a.bias_value - b.bias_value);
  if (states.length < 2) return;
  const { ctx, w, h } = setupCanvas('cv-tune-shift');
  ctx.fillStyle = '#070b14';
  ctx.fillRect(0, 0, w, h);
  const pad = { l: 52, r: 52, t: 22, b: 36 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

  const xs = states.map(s => s.bias_value);
  const shifts = states.map(s => s.resonance_shift_pm);
  const losses = states.map(s => s.alpha_total_roundtrip_db_per_cm);

  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  if (Math.abs(xmax - xmin) < 1e-9) { xmin -= 1; xmax += 1; }
  let smin = Math.min(0, ...shifts), smax = Math.max(0, ...shifts);
  if (Math.abs(smax - smin) < 1e-6) { smin -= 10; smax += 10; }
  const spPad = 0.12 * (smax - smin); smin -= spPad; smax += spPad;

  let lmin = Math.min(...losses), lmax = Math.max(...losses);
  if (Math.abs(lmax - lmin) < 0.05) { lmin = Math.max(0, lmin - 0.5); lmax += 0.5; }
  else { const lp = 0.15 * (lmax - lmin); lmin = Math.max(0, lmin - lp); lmax += lp; }

  const tx = v => pad.l + ((v - xmin) / (xmax - xmin)) * pw;
  const tyS = v => pad.t + ph - ((v - smin) / (smax - smin)) * ph;
  const tyL = v => pad.t + ph - ((v - lmin) / (lmax - lmin)) * ph;

  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1.1;
  ctx.strokeRect(pad.l, pad.t, pw, ph);

  // Zero-shift reference line
  if (0 >= smin && 0 <= smax) {
    ctx.strokeStyle = 'rgba(148,163,184,0.25)';
    ctx.beginPath(); ctx.moveTo(pad.l, tyS(0)); ctx.lineTo(pad.l + pw, tyS(0)); ctx.stroke();
  }

  // Dashed amber: Total roundtrip modal loss [dB/cm]
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.9; ctx.setLineDash([5, 4]);
  ctx.beginPath();
  states.forEach((s, i) => { const x = tx(s.bias_value), y = tyL(s.alpha_total_roundtrip_db_per_cm); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke(); ctx.setLineDash([]);
  states.forEach(s => {
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath(); ctx.arc(tx(s.bias_value), tyL(s.alpha_total_roundtrip_db_per_cm), 3.2, 0, Math.PI * 2); ctx.fill();
  });

  // Solid cyan: Resonance shift Δλ [pm]
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2.4;
  ctx.beginPath();
  states.forEach((s, i) => { const x = tx(s.bias_value), y = tyS(s.resonance_shift_pm); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();
  states.forEach(s => {
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath(); ctx.arc(tx(s.bias_value), tyS(s.resonance_shift_pm), 4.0, 0, Math.PI * 2); ctx.fill();
  });

  ctx.font = '10.5px Arial, sans-serif';
  ctx.fillStyle = '#94a3b8'; ctx.textAlign = 'center';
  [xmin, 0.5 * (xmin + xmax), xmax].forEach(v => ctx.fillText(v.toFixed(Math.abs(xmax - xmin) < 4 ? 2 : 1), tx(v), h - 18));
  const uStr = states[0].bias_unit || '';
  ctx.fillText(`${(res.geometry_physics && res.geometry_physics.drive_type) || 'Drive'} [${uStr}]`, pad.l + pw / 2, h - 4);

  ctx.textAlign = 'right'; ctx.fillStyle = '#38bdf8';
  [smin, 0.5 * (smin + smax), smax].forEach(v => ctx.fillText((v >= 0 ? '+' : '') + v.toFixed(0) + ' pm', pad.l - 5, tyS(v) + 3.5));

  ctx.textAlign = 'left'; ctx.fillStyle = '#f59e0b';
  [lmin, 0.5 * (lmin + lmax), lmax].forEach(v => ctx.fillText(v.toFixed(1) + ' dB/cm', pad.l + pw + 5, tyL(v) + 3.5));
}

// ==================== STAGE 8: CHIP I/O COUPLERS (2D-FDFD GC / 2D-FEM BUTT) & HIGH-Q FDTD CONVERGENCE ====================
function onIoModeChange(skipRun = false) {
  const mode = document.getElementById('io-mode').value;
  const zh = currentLang === 'zh';
  const wrapFiber = document.getElementById('io-wrap-fiber');
  const fibSel = document.getElementById('io-fiber');
  const btnGds = document.getElementById('btn-io-gds-single');
  const btnLoop = document.getElementById('btn-io-gds-loop');
  const btnIc = document.getElementById('btn-io-ic');
  const btnCut = document.getElementById('btn-hq-cutout');

  const lbl1 = document.getElementById('io-lbl-p1'), inp1 = document.getElementById('io-p1');
  const lbl2 = document.getElementById('io-lbl-p2'), inp2 = document.getElementById('io-p2');
  const lbl3 = document.getElementById('io-lbl-p3'), inp3 = document.getElementById('io-p3');
  const lbl4 = document.getElementById('io-lbl-p4'), sel4 = document.getElementById('io-p4');

  if (mode === 'grating_coupler') {
    wrapFiber.style.display = '';
    btnGds.style.display = '';
    btnLoop.style.display = '';
    btnIc.style.display = '';
    btnCut.style.display = 'none';
    if (fibSel.value === 'Lensed_2p5um' || fibSel.value === 'DFB_Laser_Butt') fibSel.value = 'SMF28_1550';
    lbl1.textContent = zh ? '浅刻蚀深度 [µm]' : 'Etch Depth [µm]';
    inp1.value = ''; inp1.step = '0.01'; inp1.placeholder = 'Auto (e.g. 0.07)';
    lbl2.textContent = zh ? '光纤倾角 [°]' : 'Fiber Tilt [deg]';
    inp2.value = '10'; inp2.step = '1';
    lbl3.textContent = zh ? '埋氧层 BOX [µm]' : 'BOX Oxide [µm]';
    inp3.value = '2.0'; inp3.step = '0.2';
    lbl4.textContent = zh ? '光栅齿型' : 'Teeth Profile';
    sel4.innerHTML = zh
      ? '<option value="apod_focus">变迹切趾 + 聚焦弯齿</option><option value="unif_focus">均匀周期 + 聚焦弯齿</option><option value="apod_linear">变迹切趾 + 直齿</option>'
      : '<option value="apod_focus">Apodized + Focusing</option><option value="unif_focus">Uniform + Focusing</option><option value="apod_linear">Apodized + Linear</option>';
  } else if (mode === 'butt_edge') {
    wrapFiber.style.display = '';
    btnGds.style.display = '';
    btnLoop.style.display = '';
    btnIc.style.display = '';
    btnCut.style.display = 'none';
    if (fibSel.value === 'SMF28_1550') fibSel.value = 'Lensed_2p5um';
    lbl1.textContent = zh ? '尖端宽 W_tip [µm]' : 'Tip Width W_tip [µm]';
    inp1.value = ''; inp1.step = '0.01'; inp1.placeholder = zh ? '自动优选' : 'Auto Optimize';
    lbl2.textContent = zh ? '锥长 L_taper [µm]' : 'Taper Len [µm]';
    inp2.value = '180'; inp2.step = '10';
    lbl3.textContent = zh ? '埋氧层 BOX [µm]' : 'BOX Oxide [µm]';
    inp3.value = '2.0'; inp3.step = '0.2';
    lbl4.textContent = zh ? '锥形与增透膜' : 'Taper & Facet';
    sel4.innerHTML = zh
      ? '<option value="adiabatic_ar">绝热锥 + AR 增透膜</option><option value="linear_ar">线性锥 + AR 增透膜</option><option value="adiabatic_air">绝热锥 + 裸端面无膜</option>'
      : '<option value="adiabatic_ar">Adiabatic + AR Coated</option><option value="linear_ar">Linear + AR Coated</option><option value="adiabatic_air">Adiabatic + Bare Facet</option>';
  } else {
    wrapFiber.style.display = 'none';
    btnGds.style.display = 'none';
    btnLoop.style.display = 'none';
    btnIc.style.display = 'none';
    btnCut.style.display = '';
    const cfg = readConfigFromForm();
    lbl1.textContent = zh ? '传播损耗 α [dB/cm]' : 'Prop. Loss α [dB/cm]';
    inp1.value = (cfg.alpha_loss_db_per_cm !== null && cfg.alpha_loss_db_per_cm <= 0.5) ? String(cfg.alpha_loss_db_per_cm) : '0.08';
    inp1.step = '0.02'; inp1.placeholder = '0.08';
    lbl2.textContent = zh ? '微环半径 R [µm]' : 'Radius R [µm]';
    inp2.value = String(cfg.ring_radius_um || 80); inp2.step = '5';
    lbl3.textContent = zh ? '耦合间距 [µm]' : 'Coupler Gap [µm]';
    inp3.value = String(Math.max(cfg.gap_thru_um || 0.35, 0.28)); inp3.step = '0.02';
    lbl4.textContent = zh ? 'FDTD 截断时间 [ps]' : 'FDTD Cutoffs [ps]';
    sel4.innerHTML = '<option value="10, 50, 250, 1000">10 / 50 / 250 / 1000 ps</option><option value="25, 100, 500, 2500">25 / 100 / 500 / 2500 ps</option><option value="5, 20, 100, 400">5 / 20 / 100 / 400 ps</option>';
  }
  if (!skipRun) runIoCouplerUI();
}

function buildIoPayload() {
  const cfg = readConfigFromForm();
  const mode = document.getElementById('io-mode').value;
  const p1Raw = document.getElementById('io-p1').value;
  const p2Val = parseFloat(document.getElementById('io-p2').value);
  const p3Val = parseFloat(document.getElementById('io-p3').value);
  const p4Val = document.getElementById('io-p4').value;

  if (mode === 'high_q_fdtd') {
    const hqCfg = Object.assign({}, cfg, {
      alpha_loss_db_per_cm: Number.isFinite(parseFloat(p1Raw)) ? parseFloat(p1Raw) : 0.08,
      ring_radius_um: Number.isFinite(p2Val) ? p2Val : (cfg.ring_radius_um || 80),
      gap_thru_um: Number.isFinite(p3Val) ? p3Val : 0.36
    });
    return { config: hqCfg, sim_times_ps: p4Val || '10, 50, 250, 1000' };
  }

  let mat = cfg.core_material || 'Si';
  let hUm = cfg.film_height_um || 0.22;
  let wBus = cfg.bus_width_um || 0.48;
  // Note: for default Si3N4 blueprint when opening GC for the first time, Si 220 nm SOI or current platform both work natively!
  if (mode === 'grating_coupler') {
    return {
      coupler_type: 'grating_coupler',
      core_material: mat,
      clad_material: cfg.clad_material || 'SiO2',
      top_clad_material: cfg.top_clad_material || 'SiO2',
      wavelength_nm: cfg.wavelength_nm || 1550,
      polarization: cfg.polarization || 'TE0',
      film_height_um: hUm,
      w_bus_um: wBus,
      etch_depth_um: p1Raw === '' ? null : parseFloat(p1Raw),
      fiber_angle_deg: Number.isFinite(p2Val) ? p2Val : 10.0,
      box_thickness_um: Number.isFinite(p3Val) ? p3Val : 2.0,
      apodized: p4Val !== 'unif_focus',
      focusing: p4Val !== 'apod_linear',
      fiber_preset: document.getElementById('io-fiber').value || 'SMF28_1550',
      ring_config: cfg
    };
  }
  return {
    coupler_type: 'butt_edge',
    core_material: mat,
    clad_material: cfg.clad_material || 'SiO2',
    top_clad_material: cfg.top_clad_material || 'SiO2',
    wavelength_nm: cfg.wavelength_nm || 1550,
    polarization: cfg.polarization || 'TE0',
    film_height_um: hUm,
    slab_height_um: cfg.slab_height_um || 0.0,
    sidewall_deg: cfg.sidewall_deg || 88.0,
    w_bus_um: wBus,
    w_tip_um: p1Raw === '' ? null : parseFloat(p1Raw),
    taper_length_um: Number.isFinite(p2Val) ? p2Val : 180.0,
    box_thickness_um: Number.isFinite(p3Val) ? p3Val : 2.0,
    taper_profile: p4Val === 'linear_ar' ? 'linear' : 'adiabatic',
    ar_coating: p4Val !== 'adiabatic_air',
    fiber_preset: document.getElementById('io-fiber').value || 'Lensed_2p5um',
    ring_config: cfg
  };
}

async function runIoCouplerUI() {
  if (ioBusy) return;
  ioBusy = true;
  const mode = document.getElementById('io-mode').value;
  const payload = buildIoPayload();
  const zh = currentLang === 'zh';
  document.getElementById('spec-status').textContent = mode === 'high_q_fdtd'
    ? (zh ? '正在求解超高 Q 无限圈精确谱 vs FDTD 截断振铃...' : 'Solving exact M->inf High-Q resonance vs truncated FDTD...')
    : (mode === 'grating_coupler'
      ? (zh ? '正在求解 2D 全波 Maxwell FDFD 光栅耦合器...' : 'Solving 2D Full-Wave Maxwell FDFD Grating Coupler...')
      : (zh ? '正在求解 2D 全矢量 FEM 端面模斑耦合器...' : 'Solving 2D Vector FEM Butt/Edge Coupler...'));
  try {
    const url = mode === 'high_q_fdtd' ? '/api/high_q_fdtd' : '/api/io_coupler';
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (data.error) {
      document.getElementById('io-banner').innerHTML = `<span style="color:#f87171;">⚠️ ${data.error}</span>`;
      document.getElementById('spec-status').textContent = 'Solver notice';
      return;
    }
    if (mode === 'high_q_fdtd') {
      lastHighQResult = data;
      renderHighQResult(data);
      document.getElementById('spec-status').textContent = zh
        ? `超高 Q 收敛诊断完成 (Q_L = ${ Number(data.Q_loaded).toLocaleString() })`
        : `High-Q Solved (Q_L = ${ Number(data.Q_loaded).toLocaleString() })`;
    } else {
      lastIoResult = data;
      renderIoResult(data);
      document.getElementById('spec-status').textContent = zh
        ? `I/O 耦合求解完成 (${data.elapsed_ms} ms)`
        : `I/O Coupler Solved in ${data.elapsed_ms} ms`;
    }
  } catch (err) {
    document.getElementById('spec-status').textContent = 'Error: ' + err;
  } finally {
    ioBusy = false;
  }
}

async function exportIoGdsUI(loopback) {
  const payload = buildIoPayload();
  if (lastIoResult && lastIoResult.coupler_type === payload.coupler_type && lastIoResult.core_material === payload.core_material) {
    payload.coupler_result = lastIoResult;
  }
  payload.include_ring_loopback = !!loopback;
  document.getElementById('spec-status').textContent = currentLang === 'zh' ? '正在生成 GDSII 掩膜版图...' : 'Generating Coupler GDSII mask...';
  try {
    const r = await fetch('/api/export_io_gds', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const res = await r.json();
    if (res.error) { alert('GDS Export Error: ' + res.error); return; }
    document.getElementById('io-download-slot').innerHTML =
      `<a class="btn green sm" href="${res.gds_download_url}" download>⬇ .GDS</a> ` +
      `<a class="btn sm" href="${res.lyp_download_url}" download>⬇ .LYP</a>`;
    showGdsExportResult(res);
    setStageView('gds');
  } catch (err) {
    alert('GDS export failed: ' + err);
  }
}

async function exportIoInterconnectUI() {
  const payload = buildIoPayload();
  if (lastIoResult && lastIoResult.coupler_type === payload.coupler_type && lastIoResult.core_material === payload.core_material) {
    payload.coupler_result = lastIoResult;
  }
  document.getElementById('spec-status').textContent = currentLang === 'zh' ? '正在导出耦合器 INTERCONNECT S 参数...' : 'Exporting Coupler INTERCONNECT SPAR...';
  try {
    const r = await fetch('/api/export_io_interconnect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const ic = await r.json();
    if (ic.error) { alert('INTERCONNECT Export Error: ' + ic.error); return; }
    showInterconnectDownload(ic);
    if (ic.zip_download_url) {
      const a = document.createElement('a');
      a.href = ic.zip_download_url;
      a.download = ic.zip_name || 'MicroRing_IO_INTERCONNECT.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  } catch (err) {
    alert('INTERCONNECT export failed: ' + err);
  }
}

async function exportCouplerCutoutUI() {
  const payload = buildIoPayload();
  payload.export_cutout = true;
  document.getElementById('spec-status').textContent = currentLang === 'zh' ? '正在导出单程耦合区 FDTD 切片...' : 'Exporting single-pass Coupler Cutout (.GDS + .lsf)...';
  try {
    const r = await fetch('/api/high_q_fdtd', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (data.error) { alert('Cutout Export Error: ' + data.error); return; }
    lastHighQResult = data;
    renderHighQResult(data);
    if (data.cutout) {
      document.getElementById('io-download-slot').innerHTML =
        `<a class="btn green sm" href="${data.cutout.gds_download_url}" download>⬇ Cutout .GDS</a> ` +
        `<a class="btn amber sm" href="${data.cutout.lsf_download_url}" download>⬇ Lumerical .lsf</a>`;
      if (data.cutout.gds_view) setGdsView(data.cutout.gds_view);
    }
  } catch (err) {
    alert('Cutout export failed: ' + err);
  }
}

function renderIoResult(res) {
  if (!res || !res.metrics) return;
  const zh = currentLang === 'zh';
  const mt = res.metrics;

  if (res.coupler_type === 'grating_coupler') {
    const sy = res.synthesis || {};
    document.getElementById('io-kpi1-lbl').textContent = zh ? '峰值插入损耗 IL' : 'Peak Insertion Loss';
    document.getElementById('io-kpi1-val').textContent = `${mt.peak_il_db.toFixed(2)} dB`;
    document.getElementById('io-kpi1-sub').textContent = `${mt.peak_efficiency_pct.toFixed(1)}% @ ${mt.peak_wavelength_nm.toFixed(1)} nm`;

    document.getElementById('io-kpi2-lbl').textContent = zh ? '1-dB / 3-dB 光学带宽' : '1-dB / 3-dB Bandwidth';
    document.getElementById('io-kpi2-val').textContent = `${mt.bw_1db_nm.toFixed(1)} / ${mt.bw_3db_nm.toFixed(1)} nm`;
    document.getElementById('io-kpi2-sub').textContent = `${res.fiber_label || res.fiber_preset} (${res.fiber_angle_deg}°)`;

    document.getElementById('io-kpi3-lbl').textContent = zh ? '合成光栅周期与刻蚀' : 'Synthesized Grating Teeth';
    document.getElementById('io-kpi3-val').textContent = `Λ = ${(sy.pitch_mean_um * 1000).toFixed(1)} nm`;
    document.getElementById('io-kpi3-sub').textContent = `Etch ${(res.etch_depth_um * 1000).toFixed(0)} nm · ff ${sy.ff_start.toFixed(2)}→${sy.ff_end.toFixed(2)}`;

    document.getElementById('io-kpi4-lbl').textContent = zh ? '向上方向性与反射 S11' : 'Directionality & S11';
    document.getElementById('io-kpi4-val').textContent = `${mt.upward_directionality_pct.toFixed(1)}% | ${mt.s11_db.toFixed(1)} dB`;
    document.getElementById('io-kpi4-sub').textContent = `Sub leak ${mt.substrate_loss_pct.toFixed(1)}% · BOX ${res.box_thickness_um} µm`;

    const warnHtml = (res.warnings && res.warnings.length) ? `<br/><span style="color:#fbbf24;">💡 ${res.warnings[0]}</span>` : '';
    document.getElementById('io-banner').innerHTML = (zh
      ? `<strong>原生 2D 全波 Maxwell FDFD（含 UPML 吸收边界 + Rytov 二阶色散）</strong>：<code>${res.core_material}</code> 膜厚 <code>${(res.film_height_um*1000).toFixed(0)} nm</code>，刻蚀 <code>${(res.etch_depth_um*1000).toFixed(0)} nm</code>，共 <code>${res.num_periods}</code> 齿（长 <code>${sy.grating_length_um} µm</code>，焦距 <code>${res.focal_length_um} µm</code>）。`
      : `<strong>Native 2D Full-Wave Maxwell FDFD (UPML + 2nd-Order Rytov Bloch Synthesis)</strong>: <code>${res.core_material}</code> <code>H = ${(res.film_height_um*1000).toFixed(0)} nm</code>, etch <code>${(res.etch_depth_um*1000).toFixed(0)} nm</code>, <code>${res.num_periods}</code> periods (<code>L = ${sy.grating_length_um} µm</code>, focal <code>${res.focal_length_um} µm</code>).`) + warnHtml;

    document.getElementById('io-plot1-title').textContent = zh ? '宽谱耦合插入损耗 IL(λ) 与波导回波反射 S11(λ)' : 'Broadband Insertion Loss IL(λ) & Reflection S11(λ)';
    document.getElementById('io-plot1-sub').textContent = `2D Full-Wave FDFD (${res.elapsed_ms} ms)`;
    document.getElementById('io-plot2-title').textContent = zh ? '二维全波辐射电场分布 |E(z, y)|' : '2D Full-Wave Radiated Electric Field |E(z, y)|';
    document.getElementById('io-plot2-sub').textContent = zh ? '导模入射 → 向上倾斜光纤束 + BOX 反射' : 'Guided Launch -> Upward Fiber Beam & BOX Interference';

    drawGcSpectrumPlot(res);
    drawGcBeamHeatmap(res);
  } else {
    document.getElementById('io-kpi1-lbl').textContent = zh ? '总端面插入损耗 IL' : 'Total Coupling Loss';
    document.getElementById('io-kpi1-val').textContent = `${mt.total_il_db.toFixed(2)} dB`;
    document.getElementById('io-kpi1-sub').textContent = `${mt.total_efficiency_pct.toFixed(1)}% (Facet ${mt.facet_il_db.toFixed(2)} + Taper ${mt.taper_il_db.toFixed(2)} dB)`;

    document.getElementById('io-kpi2-lbl').textContent = zh ? '最优倒锥尖端宽 W_tip' : 'Optimal Inverse-Taper Tip';
    document.getElementById('io-kpi2-val').textContent = `${(res.w_tip_um * 1000).toFixed(0)} nm`;
    document.getElementById('io-kpi2-sub').textContent = `Chip MFD ${mt.chip_mfd_x_um.toFixed(2)} × ${mt.chip_mfd_y_um.toFixed(2)} µm`;

    document.getElementById('io-kpi3-lbl').textContent = zh ? '1-dB 对准容差 (Δx, Δy)' : '1-dB Alignment Tolerance';
    document.getElementById('io-kpi3-val').textContent = `±${mt.tol_1db_x_um.toFixed(2)} / ±${mt.tol_1db_y_um.toFixed(2)} µm`;
    document.getElementById('io-kpi3-sub').textContent = `3-dB: ±${mt.tol_3db_x_um.toFixed(2)} / ±${mt.tol_3db_y_um.toFixed(2)} µm`;

    document.getElementById('io-kpi4-lbl').textContent = zh ? '回波损耗 RL 与推荐锥长' : 'Return Loss & Adiabatic L';
    document.getElementById('io-kpi4-val').textContent = `${mt.return_loss_db.toFixed(1)} dB | ${res.recommended_taper_length_um.toFixed(0)} µm`;
    document.getElementById('io-kpi4-sub').textContent = `${res.ar_coating ? 'AR Coated' : 'Bare Facet'} · n_tip = ${mt.n_eff_tip.toFixed(4)}`;

    const warnHtml = (res.warnings && res.warnings.length) ? `<br/><span style="color:#fbbf24;">💡 ${res.warnings[0]}</span>` : '';
    document.getElementById('io-banner').innerHTML = (zh
      ? `<strong>原生 2D 全矢量有限元 (FEM) + 复数坡印廷重叠积分 + EME 绝热锥长分析</strong>：对接 <code>${res.fiber_label}</code>（模斑 <code>${res.fiber_mfd_x_um} × ${res.fiber_mfd_y_um} µm</code>），最优尖端宽 <code>W_tip = ${(res.w_tip_optimal_um*1000).toFixed(0)} nm</code>。`
      : `<strong>Native 2D Vector FEM + Complex Poynting Overlap + Adiabatic EME</strong>: Coupling <code>${res.core_material}</code> bus (<code>W = ${res.w_bus_um} µm</code>) to <code>${res.fiber_label}</code> (<code>MFD ${res.fiber_mfd_x_um} × ${res.fiber_mfd_y_um} µm</code>).`) + warnHtml;

    document.getElementById('io-plot1-title').textContent = zh ? '倒锥尖端宽 W_tip 扫描与模斑耦合损耗' : 'Inverse-Taper Tip Width W_tip Optimization Sweep';
    document.getElementById('io-plot1-sub').textContent = `2D Vector FEM + Poynting Overlap (${res.elapsed_ms} ms)`;
    document.getElementById('io-plot2-title').textContent = zh ? '光纤/激光器横向 Δx 与纵向 Δy 对准容差曲线' : 'Lateral (Δx) & Vertical (Δy) Misalignment Penalty';
    document.getElementById('io-plot2-sub').textContent = `1-dB Tol: ±${mt.tol_1db_x_um.toFixed(2)} µm (x), ±${mt.tol_1db_y_um.toFixed(2)} µm (y)`;

    drawButtTipSweepPlot(res);
    drawButtAlignmentPlot(res);
  }
}

function renderHighQResult(res) {
  if (!res || !res.Q_loaded) return;
  const zh = currentLang === 'zh';
  const tr0 = (res.truncated_fdtd_traces && res.truncated_fdtd_traces[1]) || (res.truncated_fdtd_traces && res.truncated_fdtd_traces[0]) || {};

  document.getElementById('io-kpi1-lbl').textContent = zh ? '精确稳态有载 Q 值' : 'Converged Loaded Q';
  document.getElementById('io-kpi1-val').textContent = Number(res.Q_loaded).toLocaleString();
  document.getElementById('io-kpi1-sub').textContent = `FWHM = ${res.FWHM_pm.toFixed(3)} pm · τ_ph = ${res.photon_lifetime_ps.toFixed(1)} ps`;

  document.getElementById('io-kpi2-lbl').textContent = zh ? '3D-FDTD 1e-5 收敛所需时间' : 'Required 1e-5 FDTD Sim Time';
  document.getElementById('io-kpi2-val').textContent = `${res.required_fdtd_sim_time_1e5_ns.toFixed(2)} ns`;
  document.getElementById('io-kpi2-sub').textContent = `${Number(res.required_fdtd_roundtrips_1e5).toLocaleString()} cavity round-trips!`;

  document.getElementById('io-kpi3-lbl').textContent = zh ? '无限圈真实消光比 ER' : 'True Converged ER (M → ∞)';
  document.getElementById('io-kpi3-val').textContent = `${res.true_converged_er_db.toFixed(2)} dB`;
  document.getElementById('io-kpi3-sub').textContent = tr0.fdtd_sim_time_ps
    ? `vs ${tr0.apparent_er_db.toFixed(2)} dB @ ${tr0.fdtd_sim_time_ps} ps FDTD`
    : 'Exact analytical steady state';

  document.getElementById('io-kpi4-lbl').textContent = zh ? '混合单程耦合区 FDTD 提速' : 'Coupler-Cutout Speedup';
  document.getElementById('io-kpi4-val').textContent = `${Number(res.time_domain_speedup_factor).toLocaleString()}×`;
  document.getElementById('io-kpi4-sub').textContent = `Single pass ${res.coupler_only_fdtd_time_ps} ps vs ${res.required_fdtd_sim_time_1e5_ps.toFixed(0)} ps`;

  document.getElementById('io-banner').innerHTML = zh
    ? `<strong>为何高 Q 微环在 Lumerical 3D-FDTD 中极慢且无法收敛？</strong> 腔内光子寿命 <code>τ_ph = ${res.photon_lifetime_ps.toFixed(1)} ps</code>，能量衰减至 <code>1e-5</code> 需 <code>11.51 × τ_ph = ${res.required_fdtd_sim_time_1e5_ns.toFixed(2)} ns</code>（<code>${Number(res.required_fdtd_roundtrips_1e5).toLocaleString()}</code> 圈往返）。提前截断会产生严重 sinc 振铃伪影并使消光比坍塌；点击右上角 <strong>✂️ 导出单次耦合区切片</strong> 可将 3D-FDTD 缩短至 <code>${res.coupler_only_fdtd_time_ps} ps</code>！`
    : `<strong>Why Lumerical 3D-FDTD Stalls on High-Q Rings:</strong> Cavity photon lifetime is <code>τ_ph = ${res.photon_lifetime_ps.toFixed(1)} ps</code>, requiring <code>11.51 × τ_ph = ${res.required_fdtd_sim_time_1e5_ns.toFixed(2)} ns</code> (<code>${Number(res.required_fdtd_roundtrips_1e5).toLocaleString()}</code> round-trips) to reach <code>auto shutoff = 1e-5</code>. Early cutoff causes FFT sinc-ringing &amp; collapsed ER. Click <strong>✂️ Export Coupler Cutout</strong> above for a <code>${res.coupler_only_fdtd_time_ps} ps</code> single-pass 3D-FDTD!`;

  document.getElementById('io-plot1-title').textContent = zh ? '精确稳态谐振谷 (M → ∞) vs 截断 3D-FDTD Sinc 振铃伪影' : 'Exact Converged Dip (M → ∞) vs Truncated 3D-FDTD Ringing';
  document.getElementById('io-plot1-sub').textContent = `λ_res = ${res.resonance_wavelength_nm.toFixed(5)} nm · FWHM = ${res.FWHM_pm.toFixed(3)} pm`;
  document.getElementById('io-plot2-title').textContent = zh ? '表观消光比 ER 与腔内剩余能量 vs FDTD 仿真时长' : 'Apparent ER & Residual Cavity Energy vs FDTD Time';
  document.getElementById('io-plot2-sub').textContent = `True ER = ${res.true_converged_er_db.toFixed(1)} dB`;

  drawHighQResonancePlot(res);
  drawHighQShutoffPlot(res);
}

function drawGcSpectrumPlot(res) {
  const sp = res.spectrum || {};
  const wl = sp.wavelength_nm || [];
  const ilNeg = (sp.il_db || []).map(v => -v);
  const s11 = sp.s11_db || [];
  if (!wl.length) return;
  drawLinePlot('cv-io-main', [
    { x: wl, y: ilNeg, color: '#34d399', label: 'Coupling S21 = -IL [dB]' },
    { x: wl, y: s11, color: '#f59e0b', label: 'Back-Reflection S11 [dB]', dashed: true }
  ], currentLang === 'zh' ? '波长 Wavelength [nm]' : 'Wavelength [nm]', 'S21 / S11 [dB]', res.metrics.peak_wavelength_nm);
}

function drawGcBeamHeatmap(res) {
  const f = res.field_2d;
  const { ctx, w, h } = setupCanvas('cv-io-aux');
  ctx.fillStyle = '#070b14';
  ctx.fillRect(0, 0, w, h);
  if (!f || !f.E_norm || !f.E_norm.length) return;

  const pad = { l: 48, r: 16, t: 18, b: 34 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const zArr = f.z_um, yArr = f.y_um, eMat = f.E_norm;
  const ny = eMat.length, nz = eMat[0].length;

  const off = document.createElement('canvas');
  off.width = nz; off.height = ny;
  const octx = off.getContext('2d');
  const img = octx.createImageData(nz, ny);
  for (let iy = 0; iy < ny; iy++) {
    const row = eMat[ny - 1 - iy];
    for (let iz = 0; iz < nz; iz++) {
      const v = Math.max(0, Math.min(1, row[iz] / 1.05));
      const g = Math.pow(v, 0.65);
      // Perceptually uniform dark-navy -> royal -> cyan -> amber -> white plasma ramp
      let rC, gC, bC;
      if (g < 0.33) {
        const u = g / 0.33;
        rC = 8 + 30 * u; gC = 14 + 95 * u; bC = 32 + 165 * u;
      } else if (g < 0.68) {
        const u = (g - 0.33) / 0.35;
        rC = 38 + 205 * u; gC = 109 + 85 * u; bC = 197 - 125 * u;
      } else {
        const u = (g - 0.68) / 0.32;
        rC = 243 + 12 * u; gC = 194 + 58 * u; bC = 72 + 170 * u;
      }
      const p = 4 * (iy * nz + iz);
      img.data[p] = rC; img.data[p + 1] = gC; img.data[p + 2] = bC; img.data[p + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, pad.l, pad.t, pw, ph);

  const zMin = zArr[0], zMax = zArr[zArr.length - 1];
  const yMin = yArr[0], yMax = yArr[yArr.length - 1];
  const ty = yv => pad.t + ph - ((yv - yMin) / Math.max(yMax - yMin, 1e-6)) * ph;

  // Overlay waveguide core & BOX boundaries (distinct structural white/slate dashed lines vs field colors)
  ctx.strokeStyle = 'rgba(248,250,252,0.72)';
  ctx.lineWidth = 1.1;
  ctx.setLineDash([4, 3]);
  [0.0, res.film_height_um || 0.22, -(res.box_thickness_um || 2.0)].forEach(yv => {
    if (yv >= yMin && yv <= yMax) {
      ctx.beginPath(); ctx.moveTo(pad.l, ty(yv)); ctx.lineTo(pad.l + pw, ty(yv)); ctx.stroke();
    }
  });
  ctx.setLineDash([]);

  ctx.strokeStyle = '#475569'; ctx.lineWidth = 1.1;
  ctx.strokeRect(pad.l, pad.t, pw, ph);
  ctx.fillStyle = '#cbd5e1'; ctx.font = '10.5px Arial, sans-serif';
  ctx.textAlign = 'center';
  [zMin, 0.5 * (zMin + zMax), zMax].forEach(zv => {
    const px = pad.l + ((zv - zMin) / (zMax - zMin)) * pw;
    ctx.fillText(zv.toFixed(1), px, h - 17);
  });
  ctx.fillText('Propagation z [µm]', pad.l + pw / 2, h - 4);
  ctx.textAlign = 'right';
  [yMin, 0.0, yMax].forEach(yv => ctx.fillText(yv.toFixed(1), pad.l - 5, ty(yv) + 3.5));
}

function drawButtTipSweepPlot(res) {
  const ts = res.tip_sweep || [];
  if (!ts.length) return;
  const xs = ts.map(r => r.w_tip_um * 1000.0).reverse();
  const ys = ts.map(r => r.facet_il_db).reverse();
  const mfdX = ts.map(r => r.chip_mfd_x_um).reverse();
  drawLinePlot('cv-io-main', [
    { x: xs, y: ys, color: '#38bdf8', label: 'Facet Coupling Loss IL [dB]' },
    { x: xs, y: mfdX, color: '#34d399', label: 'Expanded Chip MFD_x [µm]', dashed: true }
  ], 'Inverse-Taper Tip Width W_tip [nm]', 'Facet IL [dB] / Chip MFD [µm]', res.w_tip_um * 1000.0);
}

function drawButtAlignmentPlot(res) {
  const al = res.alignment || {};
  if (!al.offset_x_um) return;
  drawLinePlot('cv-io-aux', [
    { x: al.offset_x_um, y: al.il_x_db, color: '#38bdf8', label: 'Lateral Offset Δx IL [dB]' },
    { x: al.offset_y_um, y: al.il_y_db, color: '#f59e0b', label: 'Vertical Offset Δy IL [dB]', dashed: true }
  ], 'Fiber / Laser Misalignment Offset [µm]', 'Coupling Insertion Loss [dB]', 0.0);
}

function drawHighQResonancePlot(res) {
  const det = res.detuning_pm || [];
  const infDb = res.converged_infinite_rt_thru_dB || [];
  if (!det.length) return;
  const palette = ['#f472b6', '#f59e0b', '#a78bfa', '#34d399'];
  const series = (res.truncated_fdtd_traces || []).map((tr, i) => ({
    x: det,
    y: tr.thru_dB,
    color: palette[i % palette.length],
    label: `FDTD @ ${tr.fdtd_sim_time_ps} ps (${tr.completed_roundtrips} RT, ER=${tr.apparent_er_db.toFixed(1)} dB)`,
    dashed: true
  }));
  series.push({
    x: det,
    y: infDb,
    color: '#38bdf8',
    label: `Converged M→∞ (ER=${res.true_converged_er_db.toFixed(1)} dB)`
  });
  drawLinePlot('cv-io-main', series, 'Detuning Around Exact Resonance Δλ [pm]', 'Through Transmission [dB]', 0.0);
}

function drawHighQShutoffPlot(res) {
  const trs = res.truncated_fdtd_traces || [];
  if (!trs.length) return;
  const xs = trs.map(t => t.fdtd_sim_time_ps);
  const ers = trs.map(t => t.apparent_er_db);
  const trueLine = xs.map(() => res.true_converged_er_db);
  drawLinePlot('cv-io-aux', [
    { x: xs, y: ers, color: '#f59e0b', label: 'Truncated FDTD Apparent ER [dB]' },
    { x: xs, y: trueLine, color: '#34d399', label: `True Infinite-RT ER (${res.true_converged_er_db.toFixed(1)} dB)`, dashed: true }
  ], 'FDTD Simulation Cutoff Time t_sim [ps]', 'Extinction Ratio ER [dB]', null);
}

// ==================== STAGE 9: MACH-ZEHNDER INTERFEROMETER (MZI) & MODULATOR ====================
function onMziSplitterChange(skipRun) {
  const st = document.getElementById('mzi-splitter').value;
  const lbl = document.getElementById('mzi-lbl-sp-param');
  const inp = document.getElementById('mzi-sp-param');
  const zh = currentLang === 'zh';
  inp.value = '';
  if (st === 'directional_coupler') {
    lbl.textContent = zh ? 'DC 间距 [µm]' : 'DC Gap [µm]';
    inp.placeholder = 'Auto (~0.20)';
  } else if (st === 'mmi_2x2' || st === 'mmi_1x2') {
    lbl.textContent = zh ? 'MMI 宽 W_MMI [µm]' : 'MMI Width [µm]';
    inp.placeholder = 'Auto 50:50';
  } else {
    lbl.textContent = zh ? 'Y 尖端缝隙 [µm]' : 'Y Apex Gap [µm]';
    inp.placeholder = '0.13 µm';
  }
  mziAuxMode = 'auto';
  if (!skipRun) runMziUI();
}

function buildMziPayload() {
  const cfg = readConfigFromForm();
  const st = document.getElementById('mzi-splitter').value || 'directional_coupler';
  const mechSel = document.getElementById('mzi-mech').value || 'auto';
  const includeActive = mechSel !== 'passive_only';
  const mech = includeActive ? mechSel : 'auto';
  const l1 = parseFloat(document.getElementById('mzi-l1').value);
  const dl = parseFloat(document.getElementById('mzi-dl').value);
  const fsrRaw = parseFloat(document.getElementById('mzi-fsr').value);
  const spParam = parseFloat(document.getElementById('mzi-sp-param').value);
  const egapRaw = parseFloat(document.getElementById('mzi-egap').value);
  const ramziOn = document.getElementById('mzi-ramzi').value === 'on';
  const biasTxt = (document.getElementById('mzi-biases').value || '').trim();

  // Auto-ensure 90 nm slab when Si lateral PN/PIN is selected on strip SOI
  let slabUse = cfg.slab_height_um;
  if (includeActive && (mech === 'si_pn_depletion' || mech === 'si_pin_injection' || (mech === 'auto' && cfg.core_material === 'Si')) && slabUse <= 0.01) {
    slabUse = 0.09;
  }

  const mziCfg = {
    core_material: cfg.core_material,
    clad_material: cfg.clad_material,
    top_clad_material: cfg.top_clad_material,
    wavelength_nm: cfg.wavelength_nm,
    polarization: cfg.polarization,
    film_height_um: cfg.film_height_um,
    slab_height_um: slabUse,
    sidewall_deg: cfg.sidewall_deg,
    arm1_width_um: cfg.ring_width_um,
    arm1_length_um: Number.isFinite(l1) ? l1 : 1000.0,
    delta_length_um: Number.isFinite(dl) ? dl : 80.0,
    target_fsr_nm: (Number.isFinite(fsrRaw) && fsrRaw > 0) ? fsrRaw : null,
    splitter_type: st,
    ramzi_enabled: ramziOn,
    ramzi_ring_radius_um: cfg.ring_radius_um,
    ramzi_ring_gap_um: cfg.gap_thru_um,
    alpha_loss_db_per_cm: cfg.alpha_loss_db_per_cm,
    span_nm: Math.max(cfg.span_nm || 12.0, 20.0)
  };
  if (Number.isFinite(spParam) && spParam > 0) {
    if (st === 'directional_coupler') mziCfg.dc_gap_um = spParam;
    else if (st === 'mmi_2x2' || st === 'mmi_1x2') mziCfg.mmi_width_um = spParam;
    else mziCfg.y_apex_gap_um = spParam;
  }

  return {
    config: mziCfg,
    include_active_tuning: includeActive,
    mechanism: mech,
    drive_scheme: document.getElementById('mzi-scheme').value || 'auto',
    drive_type: document.getElementById('mzi-drive').value || 'auto',
    bias_values: biasTxt || null,
    heater_gap_um: (Number.isFinite(egapRaw) && egapRaw > 0) ? egapRaw : 0.90,
    electrode_gap_um: (Number.isFinite(egapRaw) && egapRaw > 0) ? egapRaw : null
  };
}

async function runMziUI() {
  if (mziBusy) return;
  mziBusy = true;
  const zh = currentLang === 'zh';
  const payload = buildMziPayload();
  document.getElementById('spec-status').textContent = zh
    ? '正在求解 2D-FEM 超模 / EME 分束器与 MZI 调制光谱...'
    : 'Solving 2D-FEM Supermode / EME Splitter & MZI Spectra...';
  try {
    const r = await fetch('/api/design_mzi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (data.error) {
      document.getElementById('mzi-banner').innerHTML = `<span style="color:#f87171;">⚠️ ${data.error}</span>`;
      document.getElementById('spec-status').textContent = 'Solver notice';
      return;
    }
    lastMziResult = data;
    renderMziResult(data);
    document.getElementById('spec-status').textContent = zh
      ? `MZI 求解完成 (${data.elapsed_ms} ms)`
      : `MZI Solved in ${data.elapsed_ms} ms`;
  } catch (err) {
    document.getElementById('spec-status').textContent = 'Error: ' + err;
  } finally {
    mziBusy = false;
  }
}

async function exportMziGdsUI() {
  const payload = buildMziPayload();
  if (lastMziResult && lastMziResult.splitter && lastMziResult.splitter.splitter_type === payload.config.splitter_type && lastMziResult.config.core_material === payload.config.core_material) {
    payload.mzi_result = lastMziResult;
  }
  document.getElementById('spec-status').textContent = currentLang === 'zh' ? '正在导出 MZI .GDS 版图...' : 'Exporting MZI GDSII mask...';
  try {
    const r = await fetch('/api/export_mzi_gds', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const res = await r.json();
    if (res.error) { alert('MZI GDS Error: ' + res.error); return; }
    document.getElementById('mzi-download-slot').innerHTML =
      `<a class="btn green sm" href="${res.gds_download_url}" download>⬇ .GDS</a> ` +
      `<a class="btn sm" href="${res.lyp_download_url}" download>⬇ .LYP</a>`;
    showGdsExportResult(res);
    setStageView('gds');
  } catch (err) {
    alert('MZI GDS export failed: ' + err);
  }
}

async function exportMziInterconnectUI() {
  const payload = buildMziPayload();
  if (lastMziResult && lastMziResult.splitter && lastMziResult.splitter.splitter_type === payload.config.splitter_type && lastMziResult.config.core_material === payload.config.core_material) {
    payload.mzi_result = lastMziResult;
  }
  document.getElementById('spec-status').textContent = currentLang === 'zh' ? '正在导出 Lumerical INTERCONNECT MZI 模型包...' : 'Exporting Lumerical INTERCONNECT MZI bundle...';
  try {
    const r = await fetch('/api/export_mzi_interconnect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const ic = await r.json();
    if (ic.error) { alert('INTERCONNECT Export Error: ' + ic.error); return; }
    document.getElementById('mzi-download-slot').innerHTML =
      `<a class="btn amber sm" href="${ic.zip_download_url}" download>⬇ INTERCONNECT .ZIP</a>`;
    showInterconnectDownload(ic);
    if (ic.zip_download_url) {
      const a = document.createElement('a');
      a.href = ic.zip_download_url;
      a.download = ic.zip_name || 'MicroRing_MZI_INTERCONNECT.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  } catch (err) {
    alert('INTERCONNECT export failed: ' + err);
  }
}

function setMziAuxMode(mode) {
  mziAuxMode = mode;
  if (lastMziResult) renderMziResult(lastMziResult);
}

function renderMziResult(res) {
  if (!res || !res.splitter || !res.metrics) return;
  const zh = currentLang === 'zh';
  const sp = res.splitter;
  const ss = sp.summary || {};
  const mm = res.metrics;
  const act = res.active_tuning;
  const arms = res.arm_eigenmodes || {};

  // KPI 1: Splitter Ratio & Excess Loss
  document.getElementById('mzi-kpi1-val').textContent = `${ss.split_bar_pct_at_wl0.toFixed(1)} : ${ss.split_cross_pct_at_wl0.toFixed(1)}%`;
  document.getElementById('mzi-kpi1-sub').textContent = `IL ${ss.excess_loss_db.toFixed(2)} dB · Imb ${ss.imbalance_at_wl0_db >= 0 ? '+' : ''}${ss.imbalance_at_wl0_db.toFixed(2)} dB`;

  // KPI 2: MZI FSR & Passive ER
  document.getElementById('mzi-kpi2-val').textContent = mm.FSR_nm ? `${mm.FSR_nm.toFixed(2)} nm` : (zh ? '对称平衡臂' : 'Balanced');
  document.getElementById('mzi-kpi2-sub').textContent = `${mm.FSR_GHz ? mm.FSR_GHz.toFixed(1) + ' GHz · ' : ''}ER ${mm.ER_bar_passive_dB.toFixed(1)} dB`;

  // KPI 3 & 4: Active Modulator V_pi*L / P_pi / 3-dB BW / Dynamic ER
  if (act) {
    if (act.v_pi_l_vcm !== null && act.v_pi_l_vcm !== undefined) {
      document.getElementById('mzi-kpi3-val').textContent = `${act.v_pi_l_vcm.toFixed(3)} V·cm`;
      document.getElementById('mzi-kpi3-sub').textContent = `V_π = ${act.v_pi_v.toFixed(2)} V (${act.drive_scheme})`;
    } else if (act.p_pi_mw !== null && act.p_pi_mw !== undefined) {
      document.getElementById('mzi-kpi3-val').textContent = `${act.p_pi_mw.toFixed(1)} mW`;
      document.getElementById('mzi-kpi3-sub').textContent = act.v_pi_v ? `V_π = ${act.v_pi_v.toFixed(2)} V` : `${act.drive_scheme} heater`;
    } else if (act.delta_t_pi_k !== null && act.delta_t_pi_k !== undefined) {
      document.getElementById('mzi-kpi3-val').textContent = `ΔT_π = ${act.delta_t_pi_k.toFixed(1)} K`;
      document.getElementById('mzi-kpi3-sub').textContent = `L_act = ${act.active_length_um.toFixed(0)} µm`;
    } else {
      document.getElementById('mzi-kpi3-val').textContent = act.v_pi_v ? `${act.v_pi_v.toFixed(2)} ${act.drive_unit}` : '--';
      document.getElementById('mzi-kpi3-sub').textContent = `L_act = ${act.active_length_um.toFixed(0)} µm`;
    }
    const bwStr = act.bandwidth_3db ? `${Number(act.bandwidth_3db).toFixed(2)} ${act.bandwidth_unit}` : 'Broadband';
    document.getElementById('mzi-kpi4-val').textContent = bwStr;
    document.getElementById('mzi-kpi4-sub').textContent = `Dyn ER ${act.dynamic_er_bar_db.toFixed(1)} dB · IL_on ${act.on_state_il_db.toFixed(2)} dB`;
  } else {
    document.getElementById('mzi-kpi3-val').textContent = zh ? '无源模式' : 'Passive Only';
    document.getElementById('mzi-kpi3-sub').textContent = `n_eff = ${mm.neff_arm1.toFixed(4)} · n_g = ${mm.ng_arm1.toFixed(3)}`;
    document.getElementById('mzi-kpi4-val').textContent = `IL_min ${mm.IL_min_passive_dB.toFixed(2)} dB`;
    document.getElementById('mzi-kpi4-sub').textContent = `1-dB Split BW ${(ss.bw_1db_imbalance_nm || 0).toFixed(1)} nm`;
  }

  // Banner summary with splitter geometry details
  let geomDetail = '';
  if (sp.splitter_type === 'directional_coupler') {
    geomDetail = `Gap <code>${(ss.dc_gap_um*1000).toFixed(0)} nm</code>, 50:50 Length <code>L_dc = ${ss.dc_length_um} µm</code> (<code>L_π = ${ss.l_pi_um} µm</code>)`;
  } else if (sp.splitter_type === 'mmi_2x2' || sp.splitter_type === 'mmi_1x2') {
    geomDetail = `<code>W_MMI = ${ss.mmi_width_um} µm</code>, Self-Imaging <code>L_MMI = ${ss.mmi_length_um} µm</code> (<code>${ss.guided_modes_solved}</code> guided 2D-FEM modes, capture <code>${ss.input_modal_capture_pct}%</code>)`;
  } else {
    geomDetail = `Stem <code>W = ${ss.stem_width_um} µm</code>, Apex Gap <code>${(ss.y_apex_gap_um*1000).toFixed(0)} nm</code>, Supermode Overlap <code>${ss.apex_supermode_overlap_pct}%</code>`;
  }
  const ramziNote = res.ramzi ? ` | <strong>RAMZI Ring</strong>: <code>R = ${res.ramzi.ring_radius_um} µm</code>, <code>Q_L = ${Math.round(res.ramzi.ring_Q_loaded).toLocaleString()}</code>` : '';
  document.getElementById('mzi-banner').innerHTML = zh
    ? `<strong>${sp.name_zh}</strong>（<code>${ss.solver}</code>）：${geomDetail}；干涉臂 <code>L₁ = ${arms.arm1.length_um} µm</code>, <code>L₂ = ${arms.arm2.length_um} µm</code> (<code>ΔL = ${arms.delta_length_um} µm</code>)${ramziNote}。`
    : `<strong>${sp.name_en}</strong> (<code>${ss.solver}</code>): ${geomDetail} | Arms <code>L₁ = ${arms.arm1.length_um} µm</code>, <code>L₂ = ${arms.arm2.length_um} µm</code> (<code>ΔL = ${arms.delta_length_um} µm</code>)${ramziNote}.`;

  // Left Plot: Broadband MZI Spectra
  drawMziSpectraPlot(res);

  // Right Plot: Switching Transfer Curve OR 2D MMI Self-Imaging |E(x,z)| OR Splitter Bandwidth
  const hasMmiField = !!(sp.field_map && sp.field_map.E_norm && sp.field_map.E_norm.length);
  const hasTransfer = !!(act && act.transfer_curve && act.transfer_curve.drive_values);
  const btnTf = document.getElementById('btn-mzi-aux-tf');
  const btnMmi = document.getElementById('btn-mzi-aux-mmi');
  btnTf.style.display = (hasMmiField && hasTransfer) ? 'inline-flex' : 'none';
  btnMmi.style.display = hasMmiField ? 'inline-flex' : 'none';

  let showMode = mziAuxMode;
  if (showMode === 'auto') {
    showMode = hasTransfer ? 'transfer' : (hasMmiField ? 'mmi_field' : 'splitter_bw');
  } else if (showMode === 'mmi_field' && !hasMmiField) {
    showMode = hasTransfer ? 'transfer' : 'splitter_bw';
  } else if (showMode === 'transfer' && !hasTransfer) {
    showMode = hasMmiField ? 'mmi_field' : 'splitter_bw';
  }

  btnTf.classList.toggle('primary', showMode === 'transfer');
  btnMmi.classList.toggle('primary', showMode === 'mmi_field');

  if (showMode === 'mmi_field') {
    document.getElementById('mzi-plot2-title').textContent = sp.field_map.title;
    document.getElementById('mzi-plot2-sub').textContent = `L_MMI = ${ss.mmi_length_um} µm`;
    drawMmiSelfImagingHeatmap(res);
  } else if (showMode === 'transfer') {
    document.getElementById('mzi-plot2-title').textContent = zh
      ? `调制器开关传递曲线 T(${act.drive_unit}) @ ${res.config.wavelength_nm} nm`
      : `Switching Transfer Curve T(${act.drive_unit}) @ ${res.config.wavelength_nm} nm`;
    document.getElementById('mzi-plot2-sub').textContent = `${act.drive_scheme} · ER ${act.dynamic_er_bar_db.toFixed(1)} dB`;
    drawMziTransferPlot(res);
  } else {
    document.getElementById('mzi-plot2-title').textContent = zh ? '分束器宽谱分光比 P_bar(λ), P_cross(λ)' : 'Broadband Splitter Power Ratio P_bar(λ), P_cross(λ)';
    document.getElementById('mzi-plot2-sub').textContent = `IL = ${ss.excess_loss_db.toFixed(2)} dB`;
    drawSplitterBandwidthPlot(res);
  }

  // State-by-state table
  const tblCard = document.getElementById('mzi-table-card');
  const tbody = document.querySelector('#mzi-table tbody');
  tbody.innerHTML = '';
  if (act && act.states && act.states.length) {
    tblCard.style.display = 'block';
    act.states.forEach(stRow => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${stRow.label}</strong></td>
        <td>${(stRow.delta_neff_arm1*1e4).toFixed(2)} / ${(stRow.delta_neff_arm2*1e4).toFixed(2)} ×10⁻⁴</td>
        <td style="color:#38bdf8;font-weight:700;">${stRow.delta_phase_pi >= 0 ? '+' : ''}${stRow.delta_phase_pi.toFixed(3)} π</td>
        <td>${stRow.alpha_arm1_db_per_cm.toFixed(2)} / ${stRow.alpha_arm2_db_per_cm.toFixed(2)}</td>
        <td>${stRow.power_mw.toFixed(2)}</td>
        <td style="color:#34d399;">${stRow.bar_dB_at_wl0.toFixed(2)} dB</td>
        <td style="color:#f59e0b;">${res.has_cross_port ? stRow.cross_dB_at_wl0.toFixed(2) + ' dB' : 'N/A (1×2)'}</td>`;
      tbody.appendChild(tr);
    });
  } else {
    tblCard.style.display = 'none';
  }
}

function drawMziSpectraPlot(res) {
  const ps = res.passive_spectrum || {};
  const wl = ps.wavelength_nm || [];
  if (!wl.length) return;
  const act = res.active_tuning;
  const palette = ['#38bdf8', '#34d399', '#f59e0b', '#f472b6', '#a78bfa'];
  const series = [];

  if (act && act.spectra && act.spectra.length) {
    act.spectra.slice(0, 5).forEach((cv, i) => {
      series.push({
        x: wl,
        y: cv.bar_dB,
        color: palette[i % palette.length],
        label: `Bar @ ${cv.label} (${cv.delta_phase_pi >= 0 ? '+' : ''}${cv.delta_phase_pi.toFixed(2)}π)`
      });
    });
    if (res.has_cross_port && act.spectra[0]) {
      series.push({
        x: wl,
        y: act.spectra[0].cross_dB,
        color: '#94a3b8',
        label: `Cross @ ${act.spectra[0].label}`,
        dashed: true
      });
    }
    document.getElementById('mzi-plot1-sub').textContent = `${act.mechanism_name_en} (${res.elapsed_ms} ms)`;
  } else {
    series.push({ x: wl, y: ps.bar_dB, color: '#38bdf8', label: 'Bar Port S21 [dB]' });
    if (res.has_cross_port) {
      series.push({ x: wl, y: ps.cross_dB, color: '#f59e0b', label: 'Cross Port S31 [dB]', dashed: true });
    }
    document.getElementById('mzi-plot1-sub').textContent = `2D-FEM + EME (${res.elapsed_ms} ms)`;
  }
  drawLinePlot('cv-mzi-spec', series, currentLang === 'zh' ? '波长 Wavelength [nm]' : 'Wavelength [nm]', 'Transmission [dB]', res.config.wavelength_nm);
}

function drawMziTransferPlot(res) {
  const act = res.active_tuning;
  if (!act || !act.transfer_curve) return;
  const tc = act.transfer_curve;
  const xs = tc.drive_values || [];
  const series = [{ x: xs, y: tc.bar_T, color: '#38bdf8', label: 'Bar Port T_bar (Linear)' }];
  if (res.has_cross_port && tc.cross_T) {
    series.push({ x: xs, y: tc.cross_T, color: '#f59e0b', label: 'Cross Port T_cross (Linear)', dashed: true });
  }
  drawLinePlot('cv-mzi-aux', series, `Drive (${tc.drive_unit})`, 'Normalized Optical Power T', 0.0);
}

function drawSplitterBandwidthPlot(res) {
  const sp = res.splitter || {};
  const wl = sp.wavelength_nm || [];
  if (!wl.length) return;
  drawLinePlot('cv-mzi-aux', [
    { x: wl, y: (sp.power_bar || []).map(v => v * 100), color: '#34d399', label: 'Splitter Bar [%]' },
    { x: wl, y: (sp.power_cross || []).map(v => v * 100), color: '#f59e0b', label: 'Splitter Cross [%]', dashed: true }
  ], 'Wavelength [nm]', 'Split Power Ratio [%]', res.config.wavelength_nm);
}

function drawMmiSelfImagingHeatmap(res) {
  const f = res.splitter && res.splitter.field_map;
  const { ctx, w, h } = setupCanvas('cv-mzi-aux');
  ctx.fillStyle = '#070b14';
  ctx.fillRect(0, 0, w, h);
  if (!f || !f.E_norm || !f.E_norm.length) return;

  const pad = { l: 48, r: 16, t: 18, b: 34 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const zArr = f.z_um, xArr = f.x_um, eMat = f.E_norm; // shape (Nz, Nx)
  const nz = eMat.length, nx = eMat[0].length;

  const off = document.createElement('canvas');
  off.width = nz; off.height = nx;
  const octx = off.getContext('2d');
  const img = octx.createImageData(nz, nx);
  for (let ix = 0; ix < nx; ix++) {
    const srcIx = nx - 1 - ix;
    for (let iz = 0; iz < nz; iz++) {
      const v = Math.max(0, Math.min(1, eMat[iz][srcIx]));
      const g = Math.pow(v, 0.72);
      let rC, gC, bC;
      if (g < 0.33) {
        const u = g / 0.33;
        rC = 8 + 30 * u; gC = 14 + 95 * u; bC = 32 + 165 * u;
      } else if (g < 0.68) {
        const u = (g - 0.33) / 0.35;
        rC = 38 + 205 * u; gC = 109 + 85 * u; bC = 197 - 125 * u;
      } else {
        const u = (g - 0.68) / 0.32;
        rC = 243 + 12 * u; gC = 194 + 58 * u; bC = 72 + 170 * u;
      }
      const p = 4 * (ix * nz + iz);
      img.data[p] = rC; img.data[p + 1] = gC; img.data[p + 2] = bC; img.data[p + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, pad.l, pad.t, pw, ph);

  const zMin = zArr[0], zMax = zArr[zArr.length - 1];
  const xMin = xArr[0], xMax = xArr[xArr.length - 1];
  const ty = xv => pad.t + ph - ((xv - xMin) / Math.max(xMax - xMin, 1e-6)) * ph;

  // Overlay MMI multimode core sidewalls (±W_MMI / 2) in crisp structural dashed white
  const hwMmi = 0.5 * (f.w_mmi_um || 6.0);
  ctx.strokeStyle = 'rgba(248,250,252,0.78)';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([5, 3]);
  [-hwMmi, +hwMmi].forEach(xv => {
    if (xv >= xMin && xv <= xMax) {
      ctx.beginPath(); ctx.moveTo(pad.l, ty(xv)); ctx.lineTo(pad.l + pw, ty(xv)); ctx.stroke();
    }
  });
  ctx.setLineDash([]);

  ctx.strokeStyle = '#475569'; ctx.lineWidth = 1.1;
  ctx.strokeRect(pad.l, pad.t, pw, ph);
  ctx.fillStyle = '#cbd5e1'; ctx.font = '10.5px Arial, sans-serif';
  ctx.textAlign = 'center';
  [zMin, 0.5 * (zMin + zMax), zMax].forEach(zv => {
    const px = pad.l + ((zv - zMin) / Math.max(zMax - zMin, 1e-6)) * pw;
    ctx.fillText(zv.toFixed(1), px, h - 17);
  });
  ctx.fillText('MMI Propagation z [µm]', pad.l + pw / 2, h - 4);
  ctx.textAlign = 'right';
  [-hwMmi, 0.0, +hwMmi].forEach(xv => {
    if (xv >= xMin && xv <= xMax) ctx.fillText(xv.toFixed(1), pad.l - 5, ty(xv) + 3.5);
  });
}

// ==================== BOOTSTRAP ====================
(async function boot() {
  applyUiTranslations();
  const [hwRes, matRes, aiRes] = await Promise.all([
    fetch('/api/hardware'),
    fetch('/api/materials'),
    fetch('/api/ai_backends').catch(() => null),
  ]);
  const hw = await hwRes.json();
  const matData = await matRes.json();
  if (aiRes && aiRes.ok) {
    try { aiBackendStatus = await aiRes.json(); } catch (_) {}
  }
  updateAiModeBadge();

  document.getElementById('hw-badge').textContent = hw.recommended_backend;
  document.getElementById('hw-detail-box').innerHTML = `
    <div><strong>OS / Arch</strong>: ${hw.os_name} (${hw.arch})</div>
    <div><strong>CPU</strong>: ${hw.cpu_brand} (${hw.cpu_cores} cores)</div>
    <div><strong>Accelerator</strong>: ${hw.gpu_name || 'CPU Vectorized'}</div>
    <div><strong>Active Engine</strong>: <span style="color:#38bdf8;">${hw.recommended_backend}</span></div>`;

  catalogMaterials = matData.materials;
  catalogTopologies = matData.topologies;

  const selTop = document.getElementById('cfg-topology');
  Object.entries(catalogTopologies).forEach(([k, v]) => {
    const o = document.createElement('option'); o.value = k; o.textContent = v.name; selTop.appendChild(o);
  });

  ['cfg-material', 'mat-browser-sel'].forEach(id => {
    const el = document.getElementById(id);
    catalogMaterials.forEach(m => {
      if (id === 'cfg-material' && (m.key === 'Air' || m.key === 'SiO2')) return;
      const o = document.createElement('option'); o.value = m.key; o.textContent = `${m.key} — ${m.name}`; el.appendChild(o);
    });
  });
  document.getElementById('mat-browser-sel').value = '4H-SiC';

  await initChat();
})();
</script>
</body>
</html>
"""
